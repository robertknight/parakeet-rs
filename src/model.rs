use crate::config::ModelConfig;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::Array2;
use rten::{Model, Value, ValueView};
use std::path::Path;

pub struct ParakeetModel {
    model: Model,
    config: ModelConfig,
}

impl ParakeetModel {
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::from_pretrained_with_config(model_path, ExecutionConfig::default())
    }

    pub fn from_pretrained_with_config<P: AsRef<Path>>(
        model_path: P,
        _exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Use default config (hardcoded constants for Parakeet-CTC-0.6b: please see: json files https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main)
        let config = ModelConfig::default();

        // Safety: We assume the model will not be modified on disk while in use.
        let model = unsafe { Model::load_mmap(model_path) }?;

        Ok(Self { model, config })
    }
    pub fn forward(&mut self, features: Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = 1;
        let time_steps = features.shape()[0];
        let feature_size = features.shape()[1];

        let input = features
            .to_shape((batch_size, time_steps, feature_size))
            .map_err(|e| Error::Model(format!("Failed to reshape input: {e}")))?;
        let input = input.as_standard_layout();

        let input_value = ValueView::from_shape(input.shape(), input.as_slice().unwrap()).unwrap();
        let attention_mask =
            Value::from_shape([batch_size, time_steps], vec![1; batch_size * time_steps]).unwrap();

        let [logits] = self.model.run_n(
            vec![
                (self.model.node_id("input_features")?, input_value.into()),
                (self.model.node_id("attention_mask")?, attention_mask.into()),
            ],
            [self.model.node_id("logits")?],
            None,
        )?;
        let (logits_shape, logits_data) = logits.into_shape_vec::<f32, 3>()?;
        let [batch_size, time_steps_out, vocab_size] = logits_shape;

        if batch_size != 1 {
            return Err(Error::Model(format!(
                "Expected batch size 1, got {batch_size}"
            )));
        }

        let logits_2d = Array2::from_shape_vec((time_steps_out, vocab_size), logits_data)
            .map_err(|e| Error::Model(format!("Failed to create array: {e}")))?;

        Ok(logits_2d)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn pad_token_id(&self) -> usize {
        self.config.pad_token_id
    }
}
