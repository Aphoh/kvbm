// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL backend configuration with Figment support.
//!
//! This module provides configuration extraction for NIXL backends from
//! environment variables with the pattern: `DYN_KVBM_NIXL_BACKEND_<backend>=<value>`

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for NIXL backends.
///
/// Supports extracting backend configurations from environment variables:
/// - `DYN_KVBM_NIXL_BACKEND_UCX=true` - Enable UCX backend with default params
/// - `DYN_KVBM_NIXL_BACKEND_GDS=false` - Explicitly disable GDS backend
/// - `DYN_KVBM_NIXL_BACKEND_GDS_MT__THREAD_COUNT=8` - Enable `GDS_MT` with a
///   `thread_count = "8"` custom param (backend and param are split on the
///   first `__`; backend names may themselves contain single underscores)
/// - Valid boolean values: true/false, 1/0, on/off, yes/no (case-insensitive)
/// - Invalid boolean values (e.g., "maybe", "random") will cause an error
///
/// # Data Structure
///
/// Uses a single HashMap where:
/// - Key presence = backend is enabled
/// - Value (inner HashMap) = backend-specific parameters (empty = defaults)
///
/// # TOML Example
///
/// ```toml
/// [backends.UCX]
/// # UCX with default params (empty map)
///
/// [backends.GDS]
/// threads = "4"
/// buffer_size = "1048576"
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NixlBackendConfig {
    /// Map of backend name (uppercase) -> optional parameters.
    ///
    /// If a backend is present in the map, it's enabled.
    /// The inner HashMap contains optional override parameters.
    /// An empty inner map means use default parameters.
    #[serde(default)]
    backends: HashMap<String, HashMap<String, String>>,
}

impl NixlBackendConfig {
    /// Creates a new configuration with the given backends.
    ///
    /// For an empty configuration with no backends, use [`Default::default()`].
    pub fn new(backends: HashMap<String, HashMap<String, String>>) -> Self {
        Self { backends }
    }

    /// Create configuration from environment variables.
    ///
    /// Extracts backends from `DYN_KVBM_NIXL_BACKEND_<backend>=<value>`
    /// (boolean enable/disable) and
    /// `DYN_KVBM_NIXL_BACKEND_<backend>__<param>=<value>` (custom param,
    /// split on the first `__`) variables. Both forms merge for the same
    /// backend regardless of `std::env::vars()` iteration order; an explicit
    /// `<backend>=false` always wins over a `<backend>__*` param for the same
    /// backend, whichever was visited first.
    ///
    /// # Errors
    /// Returns an error if a boolean-enable value is neither truthy nor
    /// falsey.
    pub fn from_env() -> Result<Self> {
        Self::from_env_pairs(std::env::vars())
    }

    /// Pure variant of [`Self::from_env`] taking an explicit `(key, value)`
    /// iterator instead of reading the process environment. Kept separate so
    /// tests can exercise the parsing logic with synthetic input rather than
    /// mutating global process state.
    fn from_env_pairs(vars: impl Iterator<Item = (String, String)>) -> Result<Self> {
        let mut backends: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut disabled: HashSet<String> = HashSet::new();

        for (key, value) in vars {
            let Some(remainder) = key.strip_prefix("DYN_KVBM_NIXL_BACKEND_") else {
                continue;
            };

            if let Some((backend_part, param_part)) = remainder.split_once("__") {
                // Custom param: DYN_KVBM_NIXL_BACKEND_<backend>__<param>=<value>
                if backend_part.is_empty() || param_part.is_empty() {
                    bail!(
                        "Invalid NIXL backend env var {}: backend and param names must not be \
                         empty (expected DYN_KVBM_NIXL_BACKEND_<backend>__<param>)",
                        key
                    );
                }
                let backend_name = backend_part.to_uppercase();
                let param_name = param_part.to_lowercase();
                backends
                    .entry(backend_name)
                    .or_default()
                    .insert(param_name, value);
            } else {
                // Boolean enable/disable: DYN_KVBM_NIXL_BACKEND_<backend>=<value>
                let backend_name = remainder.to_uppercase();
                match crate::parse_bool(&value) {
                    Ok(true) => {
                        // Preserve any params already collected for this
                        // backend from a `__`-suffixed variable, regardless
                        // of env-var iteration order.
                        backends.entry(backend_name).or_default();
                    }
                    Ok(false) => {
                        // Explicitly disabled; applied as a second pass below
                        // so it wins irrespective of iteration order.
                        disabled.insert(backend_name);
                    }
                    Err(e) => bail!("Invalid value for {}: {}", key, e),
                }
            }
        }

        for backend_name in &disabled {
            backends.remove(backend_name);
        }

        Ok(Self { backends })
    }

    /// Add a backend with default parameters.
    /// Backend name is normalized to uppercase.
    pub fn with_backend(mut self, backend: impl Into<String>) -> Self {
        self.backends
            .insert(backend.into().to_uppercase(), HashMap::new());
        self
    }

    /// Add a backend with custom parameters.
    /// Backend name is normalized to uppercase.
    pub fn with_backend_params(
        mut self,
        backend: impl Into<String>,
        params: HashMap<String, String>,
    ) -> Self {
        self.backends.insert(backend.into().to_uppercase(), params);
        self
    }

    /// Get the list of enabled backend names (uppercase).
    pub fn backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    /// Get parameters for a specific backend.
    /// Backend name is normalized to uppercase for lookup.
    ///
    /// Returns None if the backend is not enabled.
    pub fn backend_params(&self, backend: &str) -> Option<&HashMap<String, String>> {
        self.backends.get(&backend.to_uppercase())
    }

    /// Check if a specific backend is enabled.
    pub fn has_backend(&self, backend: &str) -> bool {
        self.backends.contains_key(&backend.to_uppercase())
    }

    /// Merge another configuration into this one.
    ///
    /// Backends from the other configuration will be added to this one.
    /// If both have the same backend, params from `other` take precedence.
    pub fn merge(mut self, other: NixlBackendConfig) -> Self {
        self.backends.extend(other.backends);
        self
    }

    /// Iterate over all enabled backends and their parameters.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &HashMap<String, String>)> {
        self.backends.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_config_is_empty() {
        let config = NixlBackendConfig::default();
        assert_eq!(config.backends().len(), 0);
    }

    #[test]
    fn test_default_is_empty() {
        let config = NixlBackendConfig::default();
        assert!(config.backends().is_empty()); // default() has no backends
    }

    #[test]
    fn test_with_backend() {
        let config = NixlBackendConfig::default()
            .with_backend("ucx")
            .with_backend("gds_mt");

        assert!(config.has_backend("ucx"));
        assert!(config.has_backend("UCX"));
        assert!(config.has_backend("gds_mt"));
        assert!(config.has_backend("GDS_MT"));
        assert!(!config.has_backend("other"));
    }

    #[test]
    fn test_with_backend_params() {
        let mut params = HashMap::new();
        params.insert("threads".to_string(), "4".to_string());
        params.insert("buffer_size".to_string(), "1048576".to_string());

        let config = NixlBackendConfig::default()
            .with_backend("UCX")
            .with_backend_params("GDS", params);

        // UCX should have empty params
        let ucx_params = config.backend_params("UCX").unwrap();
        assert!(ucx_params.is_empty());

        // GDS should have custom params
        let gds_params = config.backend_params("GDS").unwrap();
        assert_eq!(gds_params.get("threads"), Some(&"4".to_string()));
        assert_eq!(gds_params.get("buffer_size"), Some(&"1048576".to_string()));
    }

    #[test]
    fn test_merge_configs() {
        let config1 = NixlBackendConfig::default().with_backend("ucx");
        let config2 = NixlBackendConfig::default().with_backend("gds");

        let merged = config1.merge(config2);

        assert!(merged.has_backend("ucx"));
        assert!(merged.has_backend("gds"));
    }

    #[test]
    fn test_backend_name_case_insensitive() {
        let config = NixlBackendConfig::default()
            .with_backend("ucx")
            .with_backend("Gds_mt")
            .with_backend("OTHER");

        assert!(config.has_backend("UCX"));
        assert!(config.has_backend("ucx"));
        assert!(config.has_backend("GDS_MT"));
        assert!(config.has_backend("gds_mt"));
        assert!(config.has_backend("OTHER"));
        assert!(config.has_backend("other"));
    }

    #[test]
    fn test_iter() {
        let mut params = HashMap::new();
        params.insert("key".to_string(), "value".to_string());

        let config = NixlBackendConfig::default()
            .with_backend("UCX")
            .with_backend_params("GDS", params);

        let items: Vec<_> = config.iter().collect();
        assert_eq!(items.len(), 2);
    }

    // `from_env()` itself reads the real process environment (racy to test
    // in-process); `from_env_pairs` is the pure, directly-testable core.

    #[test]
    fn test_from_env_pairs_simple_enable() {
        let config = NixlBackendConfig::from_env_pairs(
            [("DYN_KVBM_NIXL_BACKEND_UCX".to_string(), "true".to_string())].into_iter(),
        )
        .unwrap();

        assert!(config.has_backend("UCX"));
        assert!(config.backend_params("UCX").unwrap().is_empty());
    }

    #[test]
    fn test_from_env_pairs_explicit_disable_is_absent() {
        let config = NixlBackendConfig::from_env_pairs(
            [("DYN_KVBM_NIXL_BACKEND_GDS".to_string(), "false".to_string())].into_iter(),
        )
        .unwrap();

        assert!(!config.has_backend("GDS"));
    }

    #[test]
    fn test_from_env_pairs_invalid_boolean_errors() {
        let result = NixlBackendConfig::from_env_pairs(
            [("DYN_KVBM_NIXL_BACKEND_UCX".to_string(), "maybe".to_string())].into_iter(),
        );
        assert!(result.is_err());
    }

    /// An empty backend or param segment either side of the `__` splitter
    /// must be rejected at parse time, naming the offending env var, rather
    /// than deferred to a later `params.set("", ..)` failure that no longer
    /// has the var name in scope.
    #[test]
    fn test_from_env_pairs_empty_backend_or_param_segment_errors() {
        for key in [
            "DYN_KVBM_NIXL_BACKEND_GDS_MT__",
            "DYN_KVBM_NIXL_BACKEND___FOO",
        ] {
            let result =
                NixlBackendConfig::from_env_pairs([(key.to_string(), "8".to_string())].into_iter());
            let err = result.expect_err(&format!("{key} should be rejected"));
            assert!(
                err.to_string().contains(key),
                "error should mention the offending var {key}: {err}"
            );
        }
    }

    /// §5.1b regression test: `GDS_MT` (a backend name containing an
    /// underscore) must be settable, and its `thread_count` custom param
    /// (the double-underscore form) must parse into a lowercase param key.
    #[test]
    fn test_from_env_pairs_gds_mt_thread_count_param() {
        let config = NixlBackendConfig::from_env_pairs(
            [(
                "DYN_KVBM_NIXL_BACKEND_GDS_MT__THREAD_COUNT".to_string(),
                "8".to_string(),
            )]
            .into_iter(),
        )
        .unwrap();

        assert!(config.has_backend("GDS_MT"));
        let params = config.backend_params("GDS_MT").unwrap();
        assert_eq!(params.get("thread_count"), Some(&"8".to_string()));
    }

    #[test]
    fn test_from_env_pairs_param_and_boolean_enable_merge_regardless_of_order() {
        // Param-then-enable and enable-then-param must both preserve the param.
        for pairs in [
            vec![
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT__THREAD_COUNT".to_string(),
                    "8".to_string(),
                ),
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT".to_string(),
                    "true".to_string(),
                ),
            ],
            vec![
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT".to_string(),
                    "true".to_string(),
                ),
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT__THREAD_COUNT".to_string(),
                    "8".to_string(),
                ),
            ],
        ] {
            let config = NixlBackendConfig::from_env_pairs(pairs.into_iter()).unwrap();
            assert!(config.has_backend("GDS_MT"));
            assert_eq!(
                config.backend_params("GDS_MT").unwrap().get("thread_count"),
                Some(&"8".to_string())
            );
        }
    }

    #[test]
    fn test_from_env_pairs_explicit_disable_wins_over_param_regardless_of_order() {
        for pairs in [
            vec![
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT__THREAD_COUNT".to_string(),
                    "8".to_string(),
                ),
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT".to_string(),
                    "false".to_string(),
                ),
            ],
            vec![
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT".to_string(),
                    "false".to_string(),
                ),
                (
                    "DYN_KVBM_NIXL_BACKEND_GDS_MT__THREAD_COUNT".to_string(),
                    "8".to_string(),
                ),
            ],
        ] {
            let config = NixlBackendConfig::from_env_pairs(pairs.into_iter()).unwrap();
            assert!(!config.has_backend("GDS_MT"));
        }
    }

    #[test]
    fn test_from_env_pairs_ignores_unrelated_keys() {
        let config = NixlBackendConfig::from_env_pairs(
            [("SOME_OTHER_VAR".to_string(), "true".to_string())].into_iter(),
        )
        .unwrap();
        assert!(config.backends().is_empty());
    }
}
