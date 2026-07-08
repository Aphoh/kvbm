//! HTTP registration transactions and rollback.

mod validation;

use anyhow::Context;
use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
};
use velo_ext::InstanceId;

use self::validation::validate_register;
use super::{CommittedRegistration, RegistrationPermit, mutation_credential_from_headers};
use crate::protocol::{RegisterRequest, RegisterResponse};
use crate::registry::{RegistryError, RegistryIncarnation, RegistryRemoval};
use crate::server::{HubError, HubServerState};

pub(crate) async fn register_instance(
    State(state): State<HubServerState>,
    headers: HeaderMap,
    Json(request): Json<RegisterRequest>,
) -> std::result::Result<Json<RegisterResponse>, HubError> {
    let peer = request.peer_info;
    let instance_id = peer.instance_id();
    let features = request.features;

    validate_register(&features, request.runtime.as_ref(), &state)?;

    let presented_credential = mutation_credential_from_headers(&headers)
        .map_err(|error| HubError::bad_request(error.to_string()))?;
    let registration_permit = state
        .registration_lifecycle()
        .credentials()
        .begin_registration(instance_id, presented_credential.as_ref())
        .map_err(HubError::from_registration_credential)?;
    if let Err(error) = stage_manager_registrations(&state, &registration_permit, &features) {
        rollback_failed_registry_write(&state, registration_permit);
        return Err(HubError::from_feature(error));
    }

    let incarnation = match state.registry().register(peer.clone()).await {
        Ok(incarnation) => incarnation,
        Err(error) => {
            rollback_failed_registry_write(&state, registration_permit);
            return Err(HubError::from_registry(error));
        }
    };
    if let Err(error) = state
        .registration_lifecycle()
        .credentials()
        .record_registry_write(&registration_permit, incarnation)
    {
        rollback_registration(&state, registration_permit, &features, incarnation).await;
        return Err(HubError::from_registration_credential(error));
    }

    for feature in &features {
        let dispatch = match state.managers().get(&feature.key()) {
            None => Err(HubError::bad_request(format!(
                "no feature manager registered for {:?}",
                feature.key()
            ))),
            Some(manager) => manager
                .on_register(instance_id, feature)
                .await
                .map_err(HubError::from_feature),
        };
        if let Err(error) = dispatch {
            rollback_registration(&state, registration_permit, &features, incarnation).await;
            return Err(error);
        }
    }

    for (key, manager) in state.managers().iter() {
        let participates = features.iter().any(|feature| feature.key() == *key);
        if let Err(error) = manager.commit_registration(
            instance_id,
            registration_permit.credential(),
            registration_permit.registration_epoch(),
            incarnation,
            participates,
        ) {
            rollback_registration(&state, registration_permit, &features, incarnation).await;
            return Err(HubError::from_feature(error));
        }
    }

    if let Some(velo) = state.velo()
        && let Err(error) = velo.register_peer(peer.clone())
    {
        rollback_registration(&state, registration_permit, &features, incarnation).await;
        return Err(HubError::internal(format!("velo register_peer: {error}")));
    }

    let prior_features = registration_permit
        .previous()
        .map(|registration| registration.features().to_vec())
        .unwrap_or_default();
    let mutation_credential = match state
        .registration_lifecycle()
        .credentials()
        .commit_registration(
            &registration_permit,
            peer.clone(),
            features.clone(),
            incarnation,
        ) {
        Ok(credential) => credential,
        Err(error) => {
            rollback_registration(&state, registration_permit, &features, incarnation).await;
            return Err(HubError::from_registration_credential(error));
        }
    };

    for manager in state.managers().values() {
        manager
            .on_register_any(instance_id, &peer, incarnation)
            .await;
    }
    for prior in &prior_features {
        if !features.iter().any(|current| current.key() == prior.key())
            && let Some(manager) = state.managers().get(&prior.key())
        {
            manager.on_unregister(instance_id);
        }
    }

    Ok(Json(RegisterResponse {
        instance_id,
        hub_instance_id: state.velo().map(|velo| velo.instance_id()),
        mutation_credential: Some(mutation_credential),
        registration_epoch: Some(registration_permit.registration_epoch()),
    }))
}

pub(crate) async fn unregister_instance(
    State(state): State<HubServerState>,
    Path(instance_id): Path<InstanceId>,
    headers: HeaderMap,
) -> std::result::Result<StatusCode, HubError> {
    let presented_credential = mutation_credential_from_headers(&headers)
        .map_err(|error| HubError::bad_request(error.to_string()))?;
    let permit = state
        .registration_lifecycle()
        .credentials()
        .begin_unregister(instance_id, presented_credential.as_ref())
        .map_err(HubError::from_registration_credential)?;
    if let Err(error) = state
        .registry()
        .unregister(instance_id, permit.incarnation())
        .await
    {
        match &error {
            RegistryError::NotFound(_) | RegistryError::StaleIncarnation { .. } => {
                state
                    .registration_lifecycle()
                    .revoke(RegistryRemoval::new(instance_id, permit.incarnation()));
            }
            _ => {
                let _ = state
                    .registration_lifecycle()
                    .credentials()
                    .restore_unregister(permit);
            }
        }
        return Err(HubError::from_registry(error));
    }
    Ok(StatusCode::NO_CONTENT)
}

fn rollback_failed_registry_write(state: &HubServerState, permit: RegistrationPermit) {
    let owner = permit.owner();
    let unchanged_incarnation = permit.previous().map(CommittedRegistration::incarnation);
    if let Some(previous) = permit.previous()
        && let Err(error) = stage_manager_authority(
            state,
            owner,
            previous.credential(),
            previous.registration_epoch(),
            previous.features(),
        )
    {
        tracing::warn!(
            instance = %owner,
            error = %error,
            "failed to restore manager authority after rejected registry write"
        );
        let _ = state
            .registration_lifecycle()
            .fail_closed_registration(permit, unchanged_incarnation);
        return;
    }
    if let Err(error) = state
        .registration_lifecycle()
        .abort_registration(permit, unchanged_incarnation)
    {
        tracing::warn!(
            instance = %owner,
            error = %error,
            "failed to restore credential after rejected registry write"
        );
    }
}

async fn rollback_registration(
    state: &HubServerState,
    permit: RegistrationPermit,
    attempted_features: &[crate::protocol::Feature],
    attempted_incarnation: RegistryIncarnation,
) {
    let owner = permit.owner();
    let Some(previous) = permit.previous().cloned() else {
        remove_registry_incarnation(state, owner, attempted_incarnation).await;
        let reservation = state.registry().current_incarnation(owner);
        let _ = state
            .registration_lifecycle()
            .fail_closed_registration(permit, reservation);
        return;
    };

    match restore_registration(state, &previous, attempted_features).await {
        Ok(restored_incarnation) => {
            match state
                .registration_lifecycle()
                .abort_registration(permit, Some(restored_incarnation))
            {
                Ok(super::AbortRegistrationOutcome::Restored) => {
                    for manager in state.managers().values() {
                        manager
                            .on_register_any(owner, previous.peer(), restored_incarnation)
                            .await;
                    }
                    return;
                }
                Ok(super::AbortRegistrationOutcome::Revoked) => {}
                Err(error) => tracing::warn!(
                    instance = %owner,
                    error = %error,
                    "failed to restore credential after rejected re-registration"
                ),
            }
            remove_registry_incarnation(state, owner, restored_incarnation).await;
            return;
        }
        Err(error) => tracing::warn!(
            instance = %owner,
            error = %error,
            "failed to restore prior registration; revoking it fail-closed"
        ),
    }
    remove_registry_incarnation(state, owner, attempted_incarnation).await;
    let reservation = state.registry().current_incarnation(owner);
    let _ = state
        .registration_lifecycle()
        .fail_closed_registration(permit, reservation);
}

async fn restore_registration(
    state: &HubServerState,
    previous: &CommittedRegistration,
    attempted_features: &[crate::protocol::Feature],
) -> anyhow::Result<RegistryIncarnation> {
    let owner = previous.peer().instance_id();
    let restored_incarnation = state
        .registry()
        .register(previous.peer().clone())
        .await
        .with_context(|| format!("restoring registry entry for {owner}"))?;

    if let Err(error) = restore_registration_after_registry(
        state,
        previous,
        attempted_features,
        restored_incarnation,
    )
    .await
    {
        remove_registry_incarnation(state, owner, restored_incarnation).await;
        return Err(error);
    }
    Ok(restored_incarnation)
}

async fn restore_registration_after_registry(
    state: &HubServerState,
    previous: &CommittedRegistration,
    attempted_features: &[crate::protocol::Feature],
    restored_incarnation: RegistryIncarnation,
) -> anyhow::Result<()> {
    let owner = previous.peer().instance_id();
    stage_manager_authority(
        state,
        owner,
        previous.credential(),
        previous.registration_epoch(),
        previous.features(),
    )
    .context("restoring staged feature authority")?;
    for attempted in attempted_features {
        let Some(manager) = state.managers().get(&attempted.key()) else {
            continue;
        };
        match previous
            .features()
            .iter()
            .find(|prior| prior.key() == attempted.key())
        {
            Some(prior) => manager
                .on_register(owner, prior)
                .await
                .with_context(|| format!("restoring {:?} registration for {owner}", prior.key()))?,
            None => manager.on_unregister(owner),
        }
    }

    for (key, manager) in state.managers() {
        let participates = previous.features().iter().any(|prior| prior.key() == *key);
        manager
            .commit_registration(
                owner,
                previous.credential(),
                previous.registration_epoch(),
                restored_incarnation,
                participates,
            )
            .with_context(|| format!("restoring {key:?} authority for {owner}"))?;
    }

    if let Some(velo) = state.velo() {
        velo.register_peer(previous.peer().clone())
            .with_context(|| format!("restoring Velo peer for {owner}"))?;
    }
    Ok(())
}

fn stage_manager_registrations(
    state: &HubServerState,
    permit: &RegistrationPermit,
    features: &[crate::protocol::Feature],
) -> Result<(), crate::features::FeatureError> {
    stage_manager_authority(
        state,
        permit.owner(),
        permit.credential(),
        permit.registration_epoch(),
        features,
    )
}

fn stage_manager_authority(
    state: &HubServerState,
    owner: InstanceId,
    credential: &crate::protocol::MutationCredential,
    registration_epoch: kvbm_protocols::cache_manifest::RegistrationEpoch,
    features: &[crate::protocol::Feature],
) -> Result<(), crate::features::FeatureError> {
    for (key, manager) in state.managers() {
        manager.stage_registration(
            owner,
            credential,
            registration_epoch,
            features.iter().any(|feature| feature.key() == *key),
        )?;
    }
    Ok(())
}

async fn remove_registry_incarnation(
    state: &HubServerState,
    owner: InstanceId,
    incarnation: RegistryIncarnation,
) {
    if let Err(error) = state.registry().unregister(owner, incarnation).await {
        if matches!(
            error,
            RegistryError::NotFound(_) | RegistryError::StaleIncarnation { .. }
        ) {
            state
                .registration_lifecycle()
                .revoke(RegistryRemoval::new(owner, incarnation));
        } else {
            tracing::warn!(
                instance = %owner,
                %incarnation,
                error = %error,
                "failed to remove rejected registry incarnation"
            );
        }
    }
}
