//! Hub-owned registration revocation shared by every registry removal path.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use velo_ext::InstanceId;

use super::{
    AbortRegistrationOutcome, RegistrationCredentials, RegistrationPermit, RevocationAction,
};
use crate::features::FeatureManager;
use crate::protocol::FeatureKey;
use crate::registry::{EvictionCallback, RegisteredPeer, RegistryIncarnation, RegistryRemoval};

/// Owns registration credentials and revokes every manager through one hook.
pub(crate) struct RegistrationLifecycle {
    credentials: Arc<RegistrationCredentials>,
    managers: Weak<HashMap<FeatureKey, Arc<dyn FeatureManager>>>,
}

impl RegistrationLifecycle {
    pub(crate) fn new(managers: &Arc<HashMap<FeatureKey, Arc<dyn FeatureManager>>>) -> Arc<Self> {
        Arc::new(Self {
            credentials: Arc::new(RegistrationCredentials::default()),
            managers: Arc::downgrade(managers),
        })
    }

    pub(crate) fn credentials(&self) -> &RegistrationCredentials {
        &self.credentials
    }

    pub(crate) fn synchronize_reservations(
        &self,
        registrations: Vec<RegisteredPeer>,
    ) -> Result<(), super::RegistrationCredentialError> {
        self.credentials.synchronize_reservations(registrations)
    }

    pub(crate) fn abort_registration(
        &self,
        permit: RegistrationPermit,
        restored_incarnation: Option<RegistryIncarnation>,
    ) -> Result<AbortRegistrationOutcome, super::RegistrationCredentialError> {
        let owner = permit.owner();
        let outcome = match self
            .credentials
            .abort_registration(permit, restored_incarnation)
        {
            Ok(outcome) => outcome,
            Err(error) => {
                self.cleanup_managers(owner);
                return Err(error);
            }
        };
        if matches!(outcome, AbortRegistrationOutcome::Revoked) {
            self.cleanup_managers(owner);
        }
        Ok(outcome)
    }

    pub(crate) fn fail_closed_registration(
        &self,
        permit: RegistrationPermit,
        reservation: Option<RegistryIncarnation>,
    ) -> Result<(), super::RegistrationCredentialError> {
        let owner = permit.owner();
        let result = self
            .credentials
            .fail_closed_registration(permit, reservation);
        self.cleanup_managers(owner);
        result
    }

    pub(crate) fn revoke(&self, removal: RegistryRemoval) {
        if !matches!(
            self.credentials.begin_revocation(removal),
            RevocationAction::Cleanup
        ) {
            return;
        }
        self.cleanup_managers(removal.instance_id());
        self.credentials.finish_revocation(removal);
    }

    fn cleanup_managers(&self, owner: InstanceId) {
        if let Some(managers) = self.managers.upgrade() {
            for manager in managers.values() {
                manager.on_unregister(owner);
            }
        }
    }

    pub(crate) fn removal_callback(self: &Arc<Self>) -> EvictionCallback {
        let lifecycle = Arc::downgrade(self);
        Arc::new(move |removal| {
            if let Some(lifecycle) = lifecycle.upgrade() {
                lifecycle.revoke(removal);
            }
        })
    }
}
