//! Registration-scoped mutation authority.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use axum::http::HeaderMap;
use kvbm_protocols::cache_manifest::RegistrationEpoch;
use velo_ext::{InstanceId, PeerInfo};

use crate::protocol::{Feature, MutationCredential};
use crate::registry::{RegisteredPeer, RegistryIncarnation, RegistryRemoval};

mod lifecycle;
mod transaction;

#[cfg(test)]
mod tests;

pub(crate) use lifecycle::RegistrationLifecycle;
pub(crate) use transaction::{register_instance, unregister_instance};

/// Serializes credential rotation and removal for active registrations.
#[derive(Default)]
pub(crate) struct RegistrationCredentials {
    states: Mutex<HashMap<InstanceId, CredentialState>>,
}

pub(crate) struct RegistrationPermit {
    owner: InstanceId,
    credential: MutationCredential,
    registration_epoch: RegistrationEpoch,
    previous: Option<CommittedRegistration>,
}

pub(crate) struct UnregisterPermit {
    owner: InstanceId,
    registration: CommittedRegistration,
}

enum CredentialState {
    /// Removal observed after hook installation but before its startup
    /// snapshot was synchronized into reservations.
    Removed(RegistryIncarnation),
    Reserved(RegistryIncarnation),
    Active(CommittedRegistration),
    Registering(RegistrationInProgress),
    Unregistering(CommittedRegistration),
    Revoking(RegistryIncarnation),
}

struct RegistrationInProgress {
    credential: MutationCredential,
    registration_epoch: RegistrationEpoch,
    attempted_incarnation: Option<RegistryIncarnation>,
    removed_incarnations: HashSet<RegistryIncarnation>,
}

#[derive(Clone)]
pub(crate) struct CommittedRegistration {
    credential: MutationCredential,
    registration_epoch: RegistrationEpoch,
    peer: PeerInfo,
    features: Vec<Feature>,
    incarnation: RegistryIncarnation,
}

pub(crate) enum AbortRegistrationOutcome {
    Restored,
    Revoked,
}

pub(crate) enum RevocationAction {
    Ignore,
    Cleanup,
}

impl RegistrationCredentials {
    pub(crate) fn authorize(
        &self,
        owner: InstanceId,
        presented: Option<&MutationCredential>,
    ) -> Result<RegistryIncarnation, RegistrationCredentialError> {
        let states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        match states.get(&owner) {
            Some(CredentialState::Active(registration))
                if Some(&registration.credential) == presented =>
            {
                Ok(registration.incarnation)
            }
            Some(CredentialState::Active(_)) => {
                Err(RegistrationCredentialError::Unauthorized { owner })
            }
            Some(CredentialState::Reserved(_)) => {
                Err(RegistrationCredentialError::Unauthorized { owner })
            }
            Some(CredentialState::Removed(_)) => {
                Err(RegistrationCredentialError::NotFound { owner })
            }
            Some(
                CredentialState::Registering(_)
                | CredentialState::Unregistering(_)
                | CredentialState::Revoking(_),
            ) => Err(RegistrationCredentialError::Busy { owner }),
            None => Err(RegistrationCredentialError::NotFound { owner }),
        }
    }

    pub(crate) fn synchronize_reservations(
        &self,
        registrations: Vec<RegisteredPeer>,
    ) -> Result<(), RegistrationCredentialError> {
        let current: HashMap<_, _> = registrations
            .into_iter()
            .map(|registered| (registered.peer().instance_id(), registered.incarnation()))
            .collect();
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        let removed_before_sync: HashSet<_> = states
            .iter()
            .filter_map(|(owner, state)| match state {
                CredentialState::Removed(incarnation)
                    if current.get(owner) == Some(incarnation) =>
                {
                    Some((*owner, *incarnation))
                }
                _ => None,
            })
            .collect();
        states.retain(|owner, state| match state {
            CredentialState::Removed(_) => false,
            CredentialState::Reserved(incarnation) => current.get(owner) == Some(incarnation),
            _ => true,
        });
        for (owner, incarnation) in current {
            if removed_before_sync.contains(&(owner, incarnation)) {
                continue;
            }
            states
                .entry(owner)
                .or_insert(CredentialState::Reserved(incarnation));
        }
        Ok(())
    }

    pub(crate) fn begin_registration(
        &self,
        owner: InstanceId,
        presented: Option<&MutationCredential>,
    ) -> Result<RegistrationPermit, RegistrationCredentialError> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        match states.get(&owner) {
            Some(CredentialState::Active(registration))
                if Some(&registration.credential) != presented =>
            {
                return Err(RegistrationCredentialError::Unauthorized { owner });
            }
            Some(CredentialState::Active(_)) => {}
            Some(CredentialState::Reserved(_)) => {
                return Err(RegistrationCredentialError::Unauthorized { owner });
            }
            Some(CredentialState::Removed(_)) if presented.is_some() => {
                return Err(RegistrationCredentialError::Unauthorized { owner });
            }
            Some(CredentialState::Removed(_)) => {}
            Some(
                CredentialState::Registering(_)
                | CredentialState::Unregistering(_)
                | CredentialState::Revoking(_),
            ) => {
                return Err(RegistrationCredentialError::Busy { owner });
            }
            None if presented.is_some() => {
                return Err(RegistrationCredentialError::Unauthorized { owner });
            }
            None => {}
        }
        let credential = MutationCredential::generate();
        let registration_epoch = RegistrationEpoch::new();
        let previous = match states.get(&owner) {
            Some(CredentialState::Active(registration)) => Some(registration.clone()),
            _ => None,
        };
        states.insert(
            owner,
            CredentialState::Registering(RegistrationInProgress {
                credential: credential.clone(),
                registration_epoch,
                attempted_incarnation: None,
                removed_incarnations: HashSet::new(),
            }),
        );
        Ok(RegistrationPermit {
            owner,
            credential,
            registration_epoch,
            previous,
        })
    }

    pub(crate) fn record_registry_write(
        &self,
        permit: &RegistrationPermit,
        incarnation: RegistryIncarnation,
    ) -> Result<(), RegistrationCredentialError> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        let Some(CredentialState::Registering(progress)) = states.get_mut(&permit.owner) else {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        };
        if progress.credential != permit.credential {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        }
        progress.attempted_incarnation = Some(incarnation);
        if progress.removed_incarnations.contains(&incarnation) {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        }
        Ok(())
    }

    pub(crate) fn commit_registration(
        &self,
        permit: &RegistrationPermit,
        peer: PeerInfo,
        features: Vec<Feature>,
        incarnation: RegistryIncarnation,
    ) -> Result<MutationCredential, RegistrationCredentialError> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        let Some(CredentialState::Registering(progress)) = states.get(&permit.owner) else {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        };
        if progress.credential != permit.credential
            || progress.registration_epoch != permit.registration_epoch
            || progress.attempted_incarnation != Some(incarnation)
            || progress.removed_incarnations.contains(&incarnation)
        {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        }
        let registration = CommittedRegistration {
            credential: permit.credential.clone(),
            registration_epoch: permit.registration_epoch,
            peer,
            features,
            incarnation,
        };
        states.insert(permit.owner, CredentialState::Active(registration));
        Ok(permit.credential.clone())
    }

    pub(crate) fn abort_registration(
        &self,
        permit: RegistrationPermit,
        restored_incarnation: Option<RegistryIncarnation>,
    ) -> Result<AbortRegistrationOutcome, RegistrationCredentialError> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        let Some(CredentialState::Registering(progress)) = states.get(&permit.owner) else {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        };
        if progress.credential != permit.credential {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        }
        let removed_incarnations = progress.removed_incarnations.clone();
        match (permit.previous, restored_incarnation) {
            (Some(mut previous), Some(incarnation))
                if !removed_incarnations.contains(&incarnation) =>
            {
                previous.incarnation = incarnation;
                states.insert(permit.owner, CredentialState::Active(previous));
                Ok(AbortRegistrationOutcome::Restored)
            }
            _ => {
                states.remove(&permit.owner);
                Ok(AbortRegistrationOutcome::Revoked)
            }
        }
    }

    pub(crate) fn fail_closed_registration(
        &self,
        permit: RegistrationPermit,
        reservation: Option<RegistryIncarnation>,
    ) -> Result<(), RegistrationCredentialError> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        let Some(CredentialState::Registering(progress)) = states.get(&permit.owner) else {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        };
        if progress.credential != permit.credential {
            return Err(RegistrationCredentialError::StateChanged {
                owner: permit.owner,
            });
        }
        match reservation {
            Some(incarnation) => {
                states.insert(permit.owner, CredentialState::Reserved(incarnation));
            }
            None => {
                states.remove(&permit.owner);
            }
        }
        Ok(())
    }

    pub(crate) fn begin_unregister(
        &self,
        owner: InstanceId,
        presented: Option<&MutationCredential>,
    ) -> Result<UnregisterPermit, RegistrationCredentialError> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        let Some(CredentialState::Active(registration)) = states.get(&owner) else {
            return match states.get(&owner) {
                Some(CredentialState::Reserved(_)) => {
                    Err(RegistrationCredentialError::Unauthorized { owner })
                }
                Some(CredentialState::Removed(_)) => {
                    Err(RegistrationCredentialError::NotFound { owner })
                }
                Some(_) => Err(RegistrationCredentialError::Busy { owner }),
                None => Err(RegistrationCredentialError::NotFound { owner }),
            };
        };
        if Some(&registration.credential) != presented {
            return Err(RegistrationCredentialError::Unauthorized { owner });
        }
        let registration = registration.clone();
        states.insert(owner, CredentialState::Unregistering(registration.clone()));
        Ok(UnregisterPermit {
            owner,
            registration,
        })
    }

    pub(crate) fn restore_unregister(
        &self,
        permit: UnregisterPermit,
    ) -> Result<(), RegistrationCredentialError> {
        let mut states = self
            .states
            .lock()
            .map_err(|_| RegistrationCredentialError::Unavailable)?;
        if matches!(
            states.get(&permit.owner),
            Some(CredentialState::Unregistering(registration))
                if registration.incarnation == permit.registration.incarnation
        ) {
            states.insert(permit.owner, CredentialState::Active(permit.registration));
        }
        Ok(())
    }

    pub(crate) fn begin_revocation(&self, removal: RegistryRemoval) -> RevocationAction {
        let Ok(mut states) = self.states.lock() else {
            return RevocationAction::Cleanup;
        };
        let owner = removal.instance_id();
        let incarnation = removal.incarnation();
        match states.get_mut(&owner) {
            None => {
                states.insert(owner, CredentialState::Removed(incarnation));
                RevocationAction::Ignore
            }
            Some(CredentialState::Reserved(current)) if *current == incarnation => {
                states.remove(&owner);
                RevocationAction::Ignore
            }
            Some(CredentialState::Active(registration))
                if registration.incarnation == incarnation =>
            {
                states.insert(owner, CredentialState::Revoking(incarnation));
                RevocationAction::Cleanup
            }
            Some(CredentialState::Unregistering(registration))
                if registration.incarnation == incarnation =>
            {
                states.insert(owner, CredentialState::Revoking(incarnation));
                RevocationAction::Cleanup
            }
            Some(CredentialState::Registering(progress)) => {
                progress.removed_incarnations.insert(incarnation);
                RevocationAction::Ignore
            }
            _ => RevocationAction::Ignore,
        }
    }

    pub(crate) fn finish_revocation(&self, removal: RegistryRemoval) {
        let Ok(mut states) = self.states.lock() else {
            return;
        };
        if matches!(
            states.get(&removal.instance_id()),
            Some(CredentialState::Revoking(incarnation))
                if *incarnation == removal.incarnation()
        ) {
            states.remove(&removal.instance_id());
        }
    }
}

impl RegistrationPermit {
    pub(crate) fn credential(&self) -> &MutationCredential {
        &self.credential
    }

    pub(crate) fn owner(&self) -> InstanceId {
        self.owner
    }

    pub(crate) fn registration_epoch(&self) -> RegistrationEpoch {
        self.registration_epoch
    }

    pub(crate) fn previous(&self) -> Option<&CommittedRegistration> {
        self.previous.as_ref()
    }
}

impl CommittedRegistration {
    pub(crate) fn credential(&self) -> &MutationCredential {
        &self.credential
    }

    pub(crate) fn registration_epoch(&self) -> RegistrationEpoch {
        self.registration_epoch
    }

    pub(crate) fn peer(&self) -> &PeerInfo {
        &self.peer
    }

    pub(crate) fn features(&self) -> &[Feature] {
        &self.features
    }

    pub(crate) fn incarnation(&self) -> RegistryIncarnation {
        self.incarnation
    }
}

impl UnregisterPermit {
    pub(crate) fn incarnation(&self) -> RegistryIncarnation {
        self.registration.incarnation
    }
}

pub(crate) fn mutation_credential_from_headers(
    headers: &HeaderMap,
) -> Result<Option<MutationCredential>, InvalidMutationCredentialHeader> {
    let Some(value) = headers.get(crate::protocol::MUTATION_CREDENTIAL_HEADER) else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .map_err(|_| InvalidMutationCredentialHeader)?;
    MutationCredential::from_header_value(value)
        .map(Some)
        .map_err(|_| InvalidMutationCredentialHeader)
}

#[derive(Debug, thiserror::Error)]
#[error("invalid mutation credential header")]
pub(crate) struct InvalidMutationCredentialHeader;

#[derive(Debug, thiserror::Error)]
pub(crate) enum RegistrationCredentialError {
    #[error("registration mutation credential does not authorize instance {owner}")]
    Unauthorized { owner: InstanceId },
    #[error("registration mutation for instance {owner} is already in progress")]
    Busy { owner: InstanceId },
    #[error("instance {owner} has no active registration credential")]
    NotFound { owner: InstanceId },
    #[error("registration mutation state changed for instance {owner}")]
    StateChanged { owner: InstanceId },
    #[error("registration credential state is unavailable")]
    Unavailable,
}
