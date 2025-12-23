use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use rten::{Model, Value, ValueView};
use std::path::Path;

/// Encoder cache state for streaming inference
/// The cache maintains temporal context across chunks
#[derive(Default)]
pub struct EncoderCache {
    /// channel cache: [1, 1, 70, 512] - batch=1, 70 frame lookback
    pub cache_last_channel: Array4<f32>,
    /// time cache: [1, 1, 512, 8] - batch=1, fixed 8 time steps
    pub cache_last_time: Array4<f32>,
    /// cache length: [1] with value 0 initially
    pub cache_last_channel_len: Array1<i32>,
}

impl EncoderCache {
    /// 17 layers, batch=1, 70 frame lookback, 512 features
    pub fn new() -> Self {
        Self {
            cache_last_channel: Array4::zeros((17, 1, 70, 512)),
            cache_last_time: Array4::zeros((17, 1, 512, 8)),
            cache_last_channel_len: Array1::from_vec(vec![0i32]),
        }
    }
}

pub struct ParakeetEOUModel {
    encoder: Model,
    decoder_joint: Model,
}

impl ParakeetEOUModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        _exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let encoder_path = model_dir.join("encoder.onnx");
        let decoder_path = model_dir.join("decoder_joint.onnx");

        if !encoder_path.exists() || !decoder_path.exists() {
            return Err(Error::Config(format!(
                "Missing ONNX files in {}. Expected encoder.onnx and decoder_joint.onnx",
                model_dir.display()
            )));
        }

        // Safety: We assume the model will not be modified on disk while in use.
        let encoder = unsafe { Model::load_mmap(&encoder_path) }?;
        let decoder_joint = unsafe { Model::load_mmap(&decoder_path) }?;

        Ok(Self {
            encoder,
            decoder_joint,
        })
    }

    /// Run the stateful encoder with cache
    /// Input: features [1, 128, T], cache state
    /// Output: (encoded [1, 512, T], new_cache)
    pub fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &EncoderCache,
    ) -> Result<(Array3<f32>, EncoderCache)> {
        // Prepare inputs
        let features_layout = features.as_standard_layout();
        let features_value =
            ValueView::from_shape(features_layout.shape(), features_layout.as_slice().unwrap())
                .unwrap();
        let length_value = Value::from_shape([1], vec![length as i32]).unwrap();

        let cache_channel_layout = cache.cache_last_channel.as_standard_layout();
        let cache_channel_value = ValueView::from_shape(
            cache_channel_layout.shape(),
            cache_channel_layout.as_slice().unwrap(),
        )
        .unwrap();

        let cache_time_layout = cache.cache_last_time.as_standard_layout();
        let cache_time_value = ValueView::from_shape(
            cache_time_layout.shape(),
            cache_time_layout.as_slice().unwrap(),
        )
        .unwrap();

        let cache_len_layout = cache.cache_last_channel_len.as_standard_layout();
        let cache_len_value = ValueView::from_shape(
            cache_len_layout.shape(),
            cache_len_layout.as_slice().unwrap(),
        )
        .unwrap();

        let [outputs_out, new_cache_channel, new_cache_time, new_cache_len] = self.encoder.run_n(
            vec![
                (self.encoder.node_id("audio_signal")?, features_value.into()),
                (self.encoder.node_id("length")?, length_value.into()),
                (
                    self.encoder.node_id("cache_last_channel")?,
                    cache_channel_value.into(),
                ),
                (
                    self.encoder.node_id("cache_last_time")?,
                    cache_time_value.into(),
                ),
                (
                    self.encoder.node_id("cache_last_channel_len")?,
                    cache_len_value.into(),
                ),
            ],
            [
                self.encoder.node_id("outputs")?,
                self.encoder.node_id("new_cache_last_channel")?,
                self.encoder.node_id("new_cache_last_time")?,
                self.encoder.node_id("new_cache_last_channel_len")?,
            ],
            None,
        )?;

        // Extract encoder output [1, 512, T]
        let ([b, d, t], data) = outputs_out.into_shape_vec::<f32, 3>()?;
        let encoder_out = Array3::from_shape_vec((b, d, t), data)
            .map_err(|e| Error::Model(format!("Failed to reshape encoder output: {e}")))?;

        // Extract new cache states
        let (ch_shape, ch_data) = new_cache_channel.into_shape_vec::<f32, 4>()?;
        let (tm_shape, tm_data) = new_cache_time.into_shape_vec::<f32, 4>()?;
        let (len_shape, len_data) = new_cache_len.into_shape_vec::<i32, 1>()?;

        // Build new cache with extracted shapes
        let new_cache = EncoderCache {
            cache_last_channel: Array4::from_shape_vec(
                (ch_shape[0], ch_shape[1], ch_shape[2], ch_shape[3]),
                ch_data,
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_last_channel: {e}")))?,

            cache_last_time: Array4::from_shape_vec(
                (tm_shape[0], tm_shape[1], tm_shape[2], tm_shape[3]),
                tm_data,
            )
            .map_err(|e| Error::Model(format!("Failed to reshape cache_last_time: {e}")))?,

            cache_last_channel_len: Array1::from_shape_vec(len_shape[0], len_data)
                .map_err(|e| Error::Model(format!("Failed to reshape cache_len: {e}")))?,
        };

        Ok((encoder_out, new_cache))
    }

    /// Run the stateful decoder
    /// Returns: (logits [1, 1, 1, vocab], new_state_h, new_state_c)
    pub fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>, // [1, 512, 1]
        last_token: &Array2<i32>,    // [1, 1]
        state_h: &Array3<f32>,       // [1, 1, 640]
        state_c: &Array3<f32>,       // [1, 1, 640]
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        // Prepare inputs
        let encoder_frame_layout = encoder_frame.as_standard_layout();
        let encoder_frame_value = ValueView::from_shape(
            encoder_frame_layout.shape(),
            encoder_frame_layout.as_slice().unwrap(),
        )
        .unwrap();

        let last_token_layout = last_token.as_standard_layout();
        let last_token_value = ValueView::from_shape(
            last_token_layout.shape(),
            last_token_layout.as_slice().unwrap(),
        )
        .unwrap();

        let target_len_value = Value::from_shape([1], vec![1i32]).unwrap();

        let state_h_layout = state_h.as_standard_layout();
        let state_h_value =
            ValueView::from_shape(state_h_layout.shape(), state_h_layout.as_slice().unwrap())
                .unwrap();

        let state_c_layout = state_c.as_standard_layout();
        let state_c_value =
            ValueView::from_shape(state_c_layout.shape(), state_c_layout.as_slice().unwrap())
                .unwrap();

        let [logits_out, new_state_h, new_state_c] = self.decoder_joint.run_n(
            vec![
                (
                    self.decoder_joint.node_id("encoder_outputs")?,
                    encoder_frame_value.into(),
                ),
                (
                    self.decoder_joint.node_id("targets")?,
                    last_token_value.into(),
                ),
                (
                    self.decoder_joint.node_id("target_length")?,
                    target_len_value.into(),
                ),
                (
                    self.decoder_joint.node_id("input_states_1")?,
                    state_h_value.into(),
                ),
                (
                    self.decoder_joint.node_id("input_states_2")?,
                    state_c_value.into(),
                ),
            ],
            [
                self.decoder_joint.node_id("outputs")?,
                self.decoder_joint.node_id("output_states_1")?,
                self.decoder_joint.node_id("output_states_2")?,
            ],
            None,
        )?;

        // Extract outputs - logits is 4D [1, 1, 1, vocab]
        let (l_shape, l_data) = logits_out.into_shape_vec::<f32, 4>()?;
        let (_, h_data) = new_state_h.into_shape_vec::<f32, 3>()?;
        let (_, c_data) = new_state_c.into_shape_vec::<f32, 3>()?;

        // Reconstruct Arrays
        // Logits: simplify to [1, 1, vocab]
        let vocab_size = l_shape[3];
        let logits = Array3::from_shape_vec((1, 1, vocab_size), l_data)
            .map_err(|e| Error::Model(format!("Reshape logits failed: {e}")))?;

        // States: [1, 1, 640]
        let new_h = Array3::from_shape_vec((1, 1, 640), h_data)
            .map_err(|e| Error::Model(format!("Reshape state h failed: {e}")))?;

        let new_c = Array3::from_shape_vec((1, 1, 640), c_data)
            .map_err(|e| Error::Model(format!("Reshape state c failed: {e}")))?;

        Ok((logits, new_h, new_c))
    }
}
