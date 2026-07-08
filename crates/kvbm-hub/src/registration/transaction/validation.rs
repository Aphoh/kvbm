//! Registration dependency and shared-config validation.

use std::collections::HashSet;

use crate::protocol::{Feature, FeatureKey, RuntimeConfigSummary};
use crate::server::{HubError, HubServerState};

pub(super) fn validate_register(
    features: &[Feature],
    runtime: Option<&RuntimeConfigSummary>,
    state: &HubServerState,
) -> Result<(), HubError> {
    let declared: HashSet<FeatureKey> = features.iter().map(Feature::key).collect();
    let mut required = crate::features::FeatureConfigRequirements::default();
    for feature in features {
        let key = feature.key();
        let Some(manager) = state.managers().get(&key) else {
            continue;
        };
        for dependency in manager.dependencies() {
            if !declared.contains(dependency) {
                return Err(HubError::bad_request(format!(
                    "Feature::{key} requires Feature::{dependency} to also be declared in the same register request",
                )));
            }
        }
        let requirements = manager.config_requirements();
        required.block_size |= requirements.block_size;
        required.block_layout |= requirements.block_layout;
    }

    let Some(summary) = runtime else {
        for feature in features {
            let key = feature.key();
            if let Some(manager) = state.managers().get(&key)
                && manager.requires_runtime_summary()
            {
                return Err(HubError::bad_request(format!(
                    "Feature::{key} requires a runtime config summary (block_size / max_seq_len / block_layout) in the register request"
                )));
            }
        }
        return Ok(());
    };
    let primary = state.primary_config();

    if required.block_size
        && let Some(expected) = primary.block_size
    {
        check_match("block_size", expected, summary.block_size)?;
    }
    if required.block_layout {
        check_match("block_layout", primary.block_layout, summary.block_layout)?;
    }
    Ok(())
}

fn check_match<T: PartialEq + std::fmt::Debug>(
    field: &str,
    expected: T,
    actual: Option<T>,
) -> Result<(), HubError> {
    match actual {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(HubError::bad_request(format!(
            "{field} mismatch: hub requires {expected:?}, registrant declared {actual:?}"
        ))),
        None => Err(HubError::bad_request(format!(
            "{field} must be declared in the register runtime summary (hub requires {expected:?})"
        ))),
    }
}
