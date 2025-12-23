use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array2, Array3};
use rten::{Model, Value, ValueView};
use std::path::{Path, PathBuf};

/// TDT model configs
#[derive(Debug, Clone)]
pub struct TDTModelConfig {
    pub vocab_size: usize,
}

impl TDTModelConfig {
    /// Create config with specified vocab size
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

pub struct ParakeetTDTModel {
    encoder: Model,
    decoder_joint: Model,
    config: TDTModelConfig,
}

impl ParakeetTDTModel {
    /// Load TDT model from directory containing encoder and decoder_joint ONNX files
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing encoder and decoder_joint ONNX files
    /// * `_exec_config` - Execution configuration (unused with rten)
    /// * `vocab_size` - Vocabulary size (number of tokens including blank)
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        _exec_config: ExecutionConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Find encoder and decoder_joint files
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_joint_path = Self::find_decoder_joint(model_dir)?;

        let config = TDTModelConfig::new(vocab_size);

        // Safety: We assume the model will not be modified on disk while in use.
        let encoder = unsafe { Model::load_mmap(&encoder_path) }?;
        let decoder_joint = unsafe { Model::load_mmap(&decoder_joint_path) }?;

        Ok(Self {
            encoder,
            decoder_joint,
            config,
        })
    }
    //file names simply from: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main
    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "encoder-model.onnx",
            "encoder.onnx",
            "encoder-model.int8.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        // fallback
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("encoder") && name.ends_with(".onnx") {
                        return Ok(path);
                    }
                }
            }
        }
        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder_joint(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "decoder_joint-model.onnx",
            "decoder_joint-model.int8.onnx",
            "decoder_joint.onnx",
            "decoder-model.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No decoder_joint model found in {}",
            dir.display()
        )))
    }

    /// Run greedy decoding - returns (token_ids, frame_indices, durations)
    pub fn forward(
        &mut self,
        features: Array2<f32>,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        // Run encoder
        let (encoder_out, encoder_len) = self.run_encoder(&features)?;

        // Run greedy decoding with decoder_joint
        let (tokens, frame_indices, durations) = self.greedy_decode(&encoder_out, encoder_len)?;

        Ok((tokens, frame_indices, durations))
    }

    fn run_encoder(&mut self, features: &Array2<f32>) -> Result<(Array3<f32>, i64)> {
        let batch_size = 1;
        let (time_steps, feature_size) = features.dim();

        // TDT encoder expects (batch, features, time) not (batch, time, features)
        let transposed = features.t();
        let input = transposed
            .to_shape((batch_size, feature_size, time_steps))
            .map_err(|e| Error::Model(format!("Failed to reshape encoder input: {e}")))?;
        let input = input.as_standard_layout();

        let input_value = ValueView::from_shape(input.shape(), input.as_slice().unwrap()).unwrap();
        let length_value = Value::from_shape([1], vec![time_steps as i32]).unwrap();

        let [encoder_out, encoder_lens] = self.encoder.run_n(
            vec![
                (self.encoder.node_id("audio_signal")?, input_value.into()),
                (self.encoder.node_id("length")?, length_value.into()),
            ],
            [
                self.encoder.node_id("outputs")?,
                self.encoder.node_id("encoded_lengths")?,
            ],
            None,
        )?;

        let (out_shape, data) = encoder_out.into_shape_vec::<f32, 3>()?;
        let (_, lens_data) = encoder_lens.into_shape_vec::<i32, 1>()?;

        let encoder_array = Array3::from_shape_vec(out_shape, data)
            .map_err(|e| Error::Model(format!("Failed to create encoder array: {e}")))?;

        // TDT encoder outputs [batch, encoder_dim, time] directly
        Ok((encoder_array, lens_data[0] as i64))
    }

    fn greedy_decode(
        &mut self,
        encoder_out: &Array3<f32>,
        _encoder_len: i64,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        // encoder_out shape: [batch, encoder_dim, time]
        let (_batch, encoder_dim, time_steps) = encoder_out.dim();
        let vocab_size = self.config.vocab_size;
        let max_tokens_per_step = 10;
        let blank_id = vocab_size - 1;

        // States: (num_layers=2, batch=1, hidden_dim=640)
        let mut state_h = Array3::<f32>::zeros((2, 1, 640));
        let mut state_c = Array3::<f32>::zeros((2, 1, 640));

        let mut tokens = Vec::new();
        let mut frame_indices = Vec::new();
        let mut durations = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;
        let mut last_emitted_token = blank_id as i32;

        // Get node IDs once before the loop
        let encoder_outputs_id = self.decoder_joint.node_id("encoder_outputs")?;
        let targets_id = self.decoder_joint.node_id("targets")?;
        let target_length_id = self.decoder_joint.node_id("target_length")?;
        let input_states_1_id = self.decoder_joint.node_id("input_states_1")?;
        let input_states_2_id = self.decoder_joint.node_id("input_states_2")?;
        let outputs_id = self.decoder_joint.node_id("outputs")?;
        let output_states_1_id = self.decoder_joint.node_id("output_states_1")?;
        let output_states_2_id = self.decoder_joint.node_id("output_states_2")?;

        // Frame-by-frame RNN-T/TDT greedy decoding
        while t < time_steps {
            // Get single encoder frame: slice [0, :, t] and reshape to [1, encoder_dim, 1]
            let frame = encoder_out.slice(ndarray::s![0, .., t]).to_owned();
            let frame_reshaped = frame
                .to_shape((1, encoder_dim, 1))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?;
            let frame_reshaped = frame_reshaped.as_standard_layout();

            // Prepare inputs
            let frame_value =
                ValueView::from_shape(frame_reshaped.shape(), frame_reshaped.as_slice().unwrap())
                    .unwrap();
            let targets_value = Value::from_shape([1, 1], vec![last_emitted_token]).unwrap();
            let target_length_value = Value::from_shape([1], vec![1i32]).unwrap();

            let state_h_layout = state_h.as_standard_layout();
            let state_h_value =
                ValueView::from_shape(state_h_layout.shape(), state_h_layout.as_slice().unwrap())
                    .unwrap();
            let state_c_layout = state_c.as_standard_layout();
            let state_c_value =
                ValueView::from_shape(state_c_layout.shape(), state_c_layout.as_slice().unwrap())
                    .unwrap();

            // Run decoder_joint
            let [logits_out, new_state_h, new_state_c] = self.decoder_joint.run_n(
                vec![
                    (encoder_outputs_id, frame_value.into()),
                    (targets_id, targets_value.into()),
                    (target_length_id, target_length_value.into()),
                    (input_states_1_id, state_h_value.into()),
                    (input_states_2_id, state_c_value.into()),
                ],
                [outputs_id, output_states_1_id, output_states_2_id],
                None,
            )?;

            // Extract logits - output is 4D [batch, time, 1, vocab+durations]
            let (_, logits_data) = logits_out.into_shape_vec::<f32, 4>()?;

            // TDT outputs vocab_size + 5 durations
            let vocab_logits: Vec<f32> = logits_data.iter().take(vocab_size).copied().collect();
            let duration_logits: Vec<f32> = logits_data.iter().skip(vocab_size).copied().collect();

            let token_id = vocab_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(blank_id);

            let duration_step = if !duration_logits.is_empty() {
                duration_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            } else {
                0
            };

            // Check if blank token
            if token_id != blank_id {
                // Update states when we emit a token
                let (h_shape, h_data) = new_state_h.into_shape_vec::<f32, 3>()?;
                state_h = Array3::from_shape_vec(h_shape, h_data)
                    .map_err(|e| Error::Model(format!("Failed to update state_h: {e}")))?;

                let (c_shape, c_data) = new_state_c.into_shape_vec::<f32, 3>()?;
                state_c = Array3::from_shape_vec(c_shape, c_data)
                    .map_err(|e| Error::Model(format!("Failed to update state_c: {e}")))?;

                tokens.push(token_id);
                frame_indices.push(t);
                durations.push(duration_step);
                last_emitted_token = token_id as i32;
                emitted_tokens += 1;

                // Don't advance yet - try to emit more tokens from the same frame
            } else {
                // Blank token - advance frame pointer
                // Duration prediction applies when we finally move to next frame after emitting tokens
                if duration_step > 0 && emitted_tokens > 0 {
                    t += duration_step;
                } else {
                    t += 1;
                }
                emitted_tokens = 0;
            }

            // Safety check: if we've emitted too many tokens from the same frame, advance
            if emitted_tokens >= max_tokens_per_step {
                t += 1;
                emitted_tokens = 0;
            }
        }

        Ok((tokens, frame_indices, durations))
    }
}
