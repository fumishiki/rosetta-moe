//! Gradient Checkpointing
//!
//! メモリ最適化: 各 Transformer Block の入力のみ保存し、
//! backward 時に内部 activations を再計算する。
//!
//! メモリ: O(n_layers) vs O(n_layers × seq_len × hidden_dim)

use crate::tensor::Tensor;
use std::collections::HashMap;

/// Checkpoint storage for gradient checkpointing
pub(crate) struct CheckpointStorage {
    /// Block index → saved input tensor
    checkpoints: HashMap<usize, Tensor>,
    /// Whether checkpointing is enabled
    enabled: bool,
}

impl CheckpointStorage {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            checkpoints: HashMap::new(),
            enabled,
        }
    }

    /// Check if checkpointing is enabled
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable checkpointing
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }

    /// Save checkpoint for a block
    pub(crate) fn save(&mut self, block_idx: usize, input: Tensor) {
        if self.enabled {
            self.checkpoints.insert(block_idx, input);
        }
    }

    /// Get checkpoint for a block (for recomputation)
    pub(crate) fn get(&self, block_idx: usize) -> Option<&Tensor> {
        self.checkpoints.get(&block_idx)
    }

    /// Remove checkpoint after use
    pub(crate) fn remove(&mut self, block_idx: usize) -> Option<Tensor> {
        self.checkpoints.remove(&block_idx)
    }

    /// Clear all checkpoints
    pub(crate) fn clear(&mut self) {
        self.checkpoints.clear();
    }

    /// Number of stored checkpoints
    pub(crate) fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Check if empty
    pub(crate) fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}

/// Checkpoint-aware forward pass context
pub(crate) struct CheckpointContext {
    storage: CheckpointStorage,
    /// Segment size for selective checkpointing (every N blocks)
    segment_size: usize,
}

impl CheckpointContext {
    /// Create new context with checkpointing every N blocks
    pub(crate) fn new(segment_size: usize) -> Self {
        Self {
            storage: CheckpointStorage::new(segment_size > 0),
            segment_size: if segment_size > 0 { segment_size } else { 1 },
        }
    }

    /// Disable checkpointing (save all activations)
    pub(crate) fn disabled() -> Self {
        Self {
            storage: CheckpointStorage::new(false),
            segment_size: 1,
        }
    }

    /// Check if this block should be checkpointed
    pub(crate) fn should_checkpoint(&self, block_idx: usize) -> bool {
        self.storage.is_enabled() && (block_idx % self.segment_size == 0)
    }

    /// Save checkpoint if needed
    pub(crate) fn maybe_save(&mut self, block_idx: usize, input: Tensor) {
        if self.should_checkpoint(block_idx) {
            self.storage.save(block_idx, input);
        }
    }

    /// Get checkpoint for recomputation
    pub(crate) fn get_checkpoint(&self, block_idx: usize) -> Option<&Tensor> {
        // Find nearest checkpoint at or before block_idx
        let checkpoint_idx = (block_idx / self.segment_size) * self.segment_size;
        self.storage.get(checkpoint_idx)
    }

    /// Clear all checkpoints after backward pass
    pub(crate) fn clear(&mut self) {
        self.storage.clear();
    }

    /// Get storage reference
    pub(crate) fn storage(&self) -> &CheckpointStorage {
        &self.storage
    }

    /// Get mutable storage reference
    pub(crate) fn storage_mut(&mut self) -> &mut CheckpointStorage {
        &mut self.storage
    }
}

impl Default for CheckpointContext {
    fn default() -> Self {
        // Default: checkpoint every block
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType};

    #[test]
    fn test_checkpoint_storage() {
        let mut storage = CheckpointStorage::new(true);
        assert!(storage.is_enabled());
        assert!(storage.is_empty());

        let tensor = Tensor::zeros(Shape::new(&[2, 3]), DType::F32);
        storage.save(0, tensor);
        assert_eq!(storage.len(), 1);
        assert!(storage.get(0).is_some());
        assert!(storage.get(1).is_none());

        storage.clear();
        assert!(storage.is_empty());
    }

    #[test]
    fn test_checkpoint_context() {
        // Checkpoint every 2 blocks
        let ctx = CheckpointContext::new(2);
        assert!(ctx.should_checkpoint(0));
        assert!(!ctx.should_checkpoint(1));
        assert!(ctx.should_checkpoint(2));
        assert!(!ctx.should_checkpoint(3));
    }

    #[test]
    fn test_disabled_context() {
        let ctx = CheckpointContext::disabled();
        assert!(!ctx.storage().is_enabled());
    }
}
