// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable cache identity and logical-resource requirements.

mod identity;
mod key;
mod resource;

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;

use kvbm_common::LogicalResourceId;
use serde::{Deserialize, Deserializer, Serialize};

pub use identity::{CacheIdentity, CacheManifestId, CacheScope, ModelIdentity};
pub use key::{BundleKey, BundleKeyError};
use resource::resource_alignment;
pub use resource::{ResourceRequirement, ResourceRole};

/// Version of the canonical manifest encoding hashed by CacheManifest::id.
pub const CACHE_MANIFEST_SCHEMA_VERSION: u16 = 1;

const MANIFEST_DOMAIN: &[u8] = b"kvbm-cache-manifest-v1";

/// Validated, canonical cache ABI description.
///
/// Resources and extension attributes are ordered during construction, so
/// equivalent inputs always produce the same canonical bytes and digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CacheManifest {
    schema_version: u16,
    model: ModelIdentity,
    cache_abi: String,
    resources: Vec<ResourceRequirement>,
    attributes: BTreeMap<String, String>,
    #[serde(skip)]
    alignment_tokens: NonZeroU64,
}

impl CacheManifest {
    pub fn new(
        model: ModelIdentity,
        cache_abi: impl Into<String>,
        resources: Vec<ResourceRequirement>,
        attributes: BTreeMap<String, String>,
    ) -> Result<Self, ManifestError> {
        Self::from_parts(
            CACHE_MANIFEST_SCHEMA_VERSION,
            model,
            cache_abi.into(),
            resources,
            attributes,
        )
    }

    fn from_parts(
        schema_version: u16,
        model: ModelIdentity,
        cache_abi: String,
        mut resources: Vec<ResourceRequirement>,
        attributes: BTreeMap<String, String>,
    ) -> Result<Self, ManifestError> {
        if schema_version != CACHE_MANIFEST_SCHEMA_VERSION {
            return Err(ManifestError::UnsupportedSchemaVersion { schema_version });
        }
        model.validate()?;
        if cache_abi.is_empty() {
            return Err(ManifestError::EmptyCacheAbi);
        }
        if resources.is_empty() {
            return Err(ManifestError::NoResources);
        }
        if attributes.keys().any(String::is_empty) {
            return Err(ManifestError::EmptyAttributeName);
        }

        resources.sort_by_key(ResourceRequirement::resource);
        let mut seen = BTreeSet::new();
        for requirement in &resources {
            if !seen.insert(requirement.resource()) {
                return Err(ManifestError::DuplicateResource {
                    resource: requirement.resource(),
                });
            }
        }
        let alignment_tokens = resource_alignment(&resources)?;

        Ok(Self {
            schema_version,
            model,
            cache_abi,
            resources,
            attributes,
            alignment_tokens,
        })
    }

    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn model(&self) -> &ModelIdentity {
        &self.model
    }

    pub fn cache_abi(&self) -> &str {
        &self.cache_abi
    }

    pub fn resources(&self) -> &[ResourceRequirement] {
        &self.resources
    }

    pub const fn attributes(&self) -> &BTreeMap<String, String> {
        &self.attributes
    }

    /// Domain-separated canonical encoding used for the manifest digest.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_bytes(&mut bytes, MANIFEST_DOMAIN);
        bytes.extend_from_slice(&self.schema_version.to_be_bytes());
        push_str(&mut bytes, self.model.architecture());
        push_str(&mut bytes, self.model.revision());
        bytes.extend_from_slice(self.model.weights_digest());
        push_str(&mut bytes, &self.cache_abi);
        push_len(&mut bytes, self.resources.len());
        for requirement in &self.resources {
            bytes.extend_from_slice(&requirement.resource().0.to_be_bytes());
            bytes.push(requirement.role().canonical_tag());
            bytes.extend_from_slice(&requirement.native_block_tokens().get().to_be_bytes());
        }
        push_len(&mut bytes, self.attributes.len());
        for (name, value) in &self.attributes {
            push_str(&mut bytes, name);
            push_str(&mut bytes, value);
        }
        bytes
    }

    pub fn id(&self) -> CacheManifestId {
        CacheManifestId::from_bytes(*blake3::hash(&self.canonical_bytes()).as_bytes())
    }

    pub fn identity(&self) -> CacheIdentity {
        CacheIdentity::new(self.id(), self.resources.clone(), self.alignment_tokens)
    }
}

#[derive(Deserialize)]
struct CacheManifestWire {
    schema_version: u16,
    model: ModelIdentity,
    cache_abi: String,
    resources: Vec<ResourceRequirement>,
    attributes: BTreeMap<String, String>,
}

impl<'de> Deserialize<'de> for CacheManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CacheManifestWire::deserialize(deserializer)?;
        Self::from_parts(
            wire.schema_version,
            wire.model,
            wire.cache_abi,
            wire.resources,
            wire.attributes,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ManifestError {
    #[error("cache manifest model field {field} is empty")]
    EmptyModelField { field: &'static str },
    #[error("cache manifest ABI is empty")]
    EmptyCacheAbi,
    #[error("cache manifest contains no required resources")]
    NoResources,
    #[error("cache manifest resource {resource:?} has zero native block tokens")]
    ZeroNativeBlockTokens { resource: LogicalResourceId },
    #[error("cache manifest contains duplicate resource {resource:?}")]
    DuplicateResource { resource: LogicalResourceId },
    #[error("cache manifest contains an empty extension-attribute name")]
    EmptyAttributeName,
    #[error("cache manifest resource alignment overflows u64")]
    AlignmentOverflow,
    #[error("unsupported cache manifest schema version {schema_version}")]
    UnsupportedSchemaVersion { schema_version: u16 },
}

fn push_len(bytes: &mut Vec<u8>, len: usize) {
    bytes.extend_from_slice(&(len as u64).to_be_bytes());
}

fn push_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    push_len(bytes, value.len());
    bytes.extend_from_slice(value);
}

fn push_str(bytes: &mut Vec<u8>, value: &str) {
    push_bytes(bytes, value.as_bytes());
}

#[cfg(test)]
mod tests;
