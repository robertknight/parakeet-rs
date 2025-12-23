/// Model execution configuration.
///
/// Note: With the rten backend, these settings are currently not used but
/// the struct is kept for API compatibility.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub intra_threads: usize,
    pub inter_threads: usize,
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = threads;
        self
    }

    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = threads;
        self
    }
}
