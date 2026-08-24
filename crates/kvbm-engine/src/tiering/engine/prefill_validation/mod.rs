//! Side-effect-free validation for worker-side remote-prefill admission.

use kvbm_protocols::connector::{FindBlocksRequest, LeaderEngineError};
use kvbm_protocols::disagg::RemotePrefillParams;

pub(super) fn validate_remote_prefill(
    params: &RemotePrefillParams,
    request: &FindBlocksRequest,
    block_size: usize,
) -> Result<(), LeaderEngineError> {
    params
        .validate_for_worker(
            &request.request_id,
            &request.cache,
            &request.sequence_hashes,
            request.total_tokens,
            block_size,
        )
        .map_err(|error| invalid(error.to_string()))
}

fn invalid(reason: impl Into<String>) -> LeaderEngineError {
    LeaderEngineError::InvalidPrefillRequest {
        reason: reason.into(),
    }
}
