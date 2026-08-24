use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};

use axum::Router;
use futures::future::BoxFuture;
use velo_ext::{InstanceId, PeerInfo, WorkerAddress};

use super::{RegistrationCredentialError, RegistrationLifecycle};
use crate::features::{FeatureError, FeatureManager, HubContext};
use crate::protocol::{Feature, FeatureKey};
use crate::registry::{RegisteredPeer, RegistryIncarnation, RegistryRemoval};

#[test]
fn reserved_registry_occupant_cannot_begin_client_registration() {
    let managers = Arc::new(HashMap::new());
    let lifecycle = RegistrationLifecycle::new(&managers);
    let peer = make_peer();
    lifecycle
        .synchronize_reservations(vec![RegisteredPeer::new(peer.clone(), incarnation(1))])
        .unwrap();

    assert!(matches!(
        lifecycle
            .credentials()
            .begin_registration(peer.instance_id(), None),
        Err(RegistrationCredentialError::Unauthorized { .. })
    ));
}

#[test]
fn removal_before_startup_snapshot_sync_does_not_leave_phantom_reservation() {
    let managers = Arc::new(HashMap::new());
    let lifecycle = RegistrationLifecycle::new(&managers);
    let peer = make_peer();
    let removed = RegistryRemoval::new(peer.instance_id(), incarnation(1));

    lifecycle.revoke(removed);
    lifecycle
        .synchronize_reservations(vec![RegisteredPeer::new(peer.clone(), incarnation(1))])
        .unwrap();

    lifecycle
        .credentials()
        .begin_registration(peer.instance_id(), None)
        .expect("a removal delivered before snapshot sync must suppress that stale reservation");
}

#[test]
fn stale_removal_does_not_revoke_replacement_registration() {
    let manager = Arc::new(CountingManager::default());
    let managers: Arc<HashMap<_, Arc<dyn FeatureManager>>> = Arc::new(HashMap::from([(
        manager.key(),
        Arc::clone(&manager) as Arc<dyn FeatureManager>,
    )]));
    let lifecycle = RegistrationLifecycle::new(&managers);
    let peer = make_peer();

    let first = lifecycle
        .credentials()
        .begin_registration(peer.instance_id(), None)
        .unwrap();
    lifecycle
        .credentials()
        .record_registry_write(&first, incarnation(1))
        .unwrap();
    let first_credential = lifecycle
        .credentials()
        .commit_registration(&first, peer.clone(), Vec::new(), incarnation(1))
        .unwrap();

    let replacement = lifecycle
        .credentials()
        .begin_registration(peer.instance_id(), Some(&first_credential))
        .unwrap();
    lifecycle
        .credentials()
        .record_registry_write(&replacement, incarnation(2))
        .unwrap();
    let replacement_credential = lifecycle
        .credentials()
        .commit_registration(&replacement, peer.clone(), Vec::new(), incarnation(2))
        .unwrap();

    lifecycle.revoke(RegistryRemoval::new(peer.instance_id(), incarnation(1)));

    assert_eq!(
        lifecycle
            .credentials()
            .authorize(peer.instance_id(), Some(&replacement_credential))
            .unwrap(),
        incarnation(2)
    );
    assert_eq!(manager.unregister_count(), 0);
}

#[test]
fn removal_of_attempted_incarnation_prevents_registration_commit() {
    let manager = Arc::new(CountingManager::default());
    let managers: Arc<HashMap<_, Arc<dyn FeatureManager>>> = Arc::new(HashMap::from([(
        manager.key(),
        Arc::clone(&manager) as Arc<dyn FeatureManager>,
    )]));
    let lifecycle = RegistrationLifecycle::new(&managers);
    let peer = make_peer();
    let permit = lifecycle
        .credentials()
        .begin_registration(peer.instance_id(), None)
        .unwrap();
    lifecycle
        .credentials()
        .record_registry_write(&permit, incarnation(1))
        .unwrap();

    lifecycle.revoke(RegistryRemoval::new(peer.instance_id(), incarnation(1)));

    assert!(matches!(
        lifecycle
            .credentials()
            .commit_registration(&permit, peer, Vec::new(), incarnation(1),),
        Err(RegistrationCredentialError::StateChanged { .. })
    ));
    assert!(matches!(
        lifecycle.abort_registration(permit, None).unwrap(),
        super::AbortRegistrationOutcome::Revoked
    ));
    assert_eq!(manager.unregister_count(), 1);
}

#[test]
fn fail_closed_rollback_reserves_a_registry_entry_that_could_not_be_removed() {
    let manager = Arc::new(CountingManager::default());
    let managers: Arc<HashMap<_, Arc<dyn FeatureManager>>> = Arc::new(HashMap::from([(
        manager.key(),
        Arc::clone(&manager) as Arc<dyn FeatureManager>,
    )]));
    let lifecycle = RegistrationLifecycle::new(&managers);
    let peer = make_peer();
    let permit = lifecycle
        .credentials()
        .begin_registration(peer.instance_id(), None)
        .unwrap();
    lifecycle
        .credentials()
        .record_registry_write(&permit, incarnation(7))
        .unwrap();

    lifecycle
        .fail_closed_registration(permit, Some(incarnation(7)))
        .unwrap();

    assert!(matches!(
        lifecycle
            .credentials()
            .begin_registration(peer.instance_id(), None),
        Err(RegistrationCredentialError::Unauthorized { .. })
    ));
    assert_eq!(manager.unregister_count(), 1);
}

#[test]
fn replacement_is_busy_until_matching_removal_cleanup_finishes() {
    let cleanup = Arc::new(CleanupGate::default());
    let manager = Arc::new(BlockingManager {
        cleanup: Arc::clone(&cleanup),
    });
    let managers: Arc<HashMap<_, Arc<dyn FeatureManager>>> = Arc::new(HashMap::from([(
        manager.key(),
        manager as Arc<dyn FeatureManager>,
    )]));
    let lifecycle = RegistrationLifecycle::new(&managers);
    let peer = make_peer();
    let permit = lifecycle
        .credentials()
        .begin_registration(peer.instance_id(), None)
        .unwrap();
    lifecycle
        .credentials()
        .record_registry_write(&permit, incarnation(1))
        .unwrap();
    lifecycle
        .credentials()
        .commit_registration(&permit, peer.clone(), Vec::new(), incarnation(1))
        .unwrap();

    let revoke_lifecycle = Arc::clone(&lifecycle);
    let owner = peer.instance_id();
    let revoke = std::thread::spawn(move || {
        revoke_lifecycle.revoke(RegistryRemoval::new(owner, incarnation(1)));
    });
    cleanup.wait_until_started();

    assert!(matches!(
        lifecycle
            .credentials()
            .begin_registration(peer.instance_id(), None),
        Err(RegistrationCredentialError::Busy { .. })
    ));

    cleanup.release();
    revoke.join().unwrap();
    lifecycle
        .credentials()
        .begin_registration(peer.instance_id(), None)
        .expect("a fresh lifecycle may begin only after cleanup completes");
}

fn incarnation(value: u64) -> RegistryIncarnation {
    RegistryIncarnation::from_u64(value)
}

fn make_peer() -> PeerInfo {
    PeerInfo::new(
        InstanceId::new_v4(),
        WorkerAddress::from_encoded(b"registration-lifecycle-test".to_vec()),
    )
}

#[derive(Default)]
struct CountingManager {
    unregisters: Mutex<usize>,
}

impl CountingManager {
    fn unregister_count(&self) -> usize {
        *self.unregisters.lock().unwrap()
    }
}

impl FeatureManager for CountingManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::ConnectorControl
    }

    fn attach<'a>(&'a self, _ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async { Ok(()) })
    }

    fn on_register<'a>(
        &'a self,
        _instance_id: InstanceId,
        _feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async { Ok(()) })
    }

    fn on_unregister(&self, _instance_id: InstanceId) {
        *self.unregisters.lock().unwrap() += 1;
    }

    fn control_router(self: Arc<Self>) -> Router {
        Router::new()
    }

    fn public_router(self: Arc<Self>) -> Router {
        Router::new()
    }
}

#[derive(Default)]
struct CleanupGate {
    state: Mutex<CleanupState>,
    changed: Condvar,
}

#[derive(Default)]
struct CleanupState {
    started: bool,
    released: bool,
}

impl CleanupGate {
    fn wait_until_started(&self) {
        let mut state = self.state.lock().unwrap();
        while !state.started {
            state = self.changed.wait(state).unwrap();
        }
    }

    fn block_until_released(&self) {
        let mut state = self.state.lock().unwrap();
        state.started = true;
        self.changed.notify_all();
        while !state.released {
            state = self.changed.wait(state).unwrap();
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap();
        state.released = true;
        self.changed.notify_all();
    }
}

struct BlockingManager {
    cleanup: Arc<CleanupGate>,
}

impl FeatureManager for BlockingManager {
    fn key(&self) -> FeatureKey {
        FeatureKey::ConnectorControl
    }

    fn attach<'a>(&'a self, _ctx: HubContext) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async { Ok(()) })
    }

    fn on_register<'a>(
        &'a self,
        _instance_id: InstanceId,
        _feature: &'a Feature,
    ) -> BoxFuture<'a, Result<(), FeatureError>> {
        Box::pin(async { Ok(()) })
    }

    fn on_unregister(&self, _instance_id: InstanceId) {
        self.cleanup.block_until_released();
    }

    fn control_router(self: Arc<Self>) -> Router {
        Router::new()
    }

    fn public_router(self: Arc<Self>) -> Router {
        Router::new()
    }
}
