//! Incarnation-safe hub-to-peer heartbeat fan-out.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use velo_ext::InstanceId;

use crate::handlers::{HEARTBEAT_HANDLER, HeartbeatAck, HeartbeatRequest};
use crate::registry::{PeerRegistry, RegisteredPeer, RegistryIncarnation};

type FailureKey = (InstanceId, RegistryIncarnation);

/// Outcome of one probe against the exact registry incarnation captured when
/// the probe was launched.
#[derive(Debug)]
enum ProbeOutcome {
    Ok {
        id: InstanceId,
        incarnation: RegistryIncarnation,
        ack_seq: u64,
    },
    Failed {
        id: InstanceId,
        incarnation: RegistryIncarnation,
        reason: String,
    },
}

/// Spawn the hub-driven heartbeat manager and detached per-peer probes.
///
/// Every tick snapshots `registry.registrations()`, so each detached probe and
/// failure counter is tied to one `(InstanceId, RegistryIncarnation)`. Probe
/// completion uses conditional `touch`/`unregister`; an outcome from an older
/// incarnation therefore cannot refresh or evict its replacement.
pub(super) fn spawn_heartbeat_task(
    velo: Arc<velo::Velo>,
    registry: Arc<dyn PeerRegistry>,
    self_id: InstanceId,
    interval: Duration,
    max_failures: u32,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ProbeOutcome>();

    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.tick().await;
        let mut failures: HashMap<FailureKey, u32> = HashMap::new();
        let mut seq: u64 = 0;

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    tracing::debug!("heartbeat task: shutdown requested");
                    return;
                }
                _ = tick.tick() => {
                    seq = seq.wrapping_add(1);
                    let registrations = registry.registrations();
                    retain_current_failures(&mut failures, &registrations);
                    fan_out_probes(
                        &velo,
                        registrations,
                        self_id,
                        seq,
                        interval,
                        tx.clone(),
                    );
                }
                Some(outcome) = rx.recv() => {
                    handle_outcome(outcome, &registry, &mut failures, max_failures).await;
                }
            }
        }
    })
}

fn retain_current_failures(
    failures: &mut HashMap<FailureKey, u32>,
    registrations: &[RegisteredPeer],
) {
    failures.retain(|key, _| {
        registrations.iter().any(|registered| {
            key.0 == registered.peer().instance_id() && key.1 == registered.incarnation()
        })
    });
}

fn fan_out_probes(
    velo: &Arc<velo::Velo>,
    registrations: Vec<RegisteredPeer>,
    self_id: InstanceId,
    seq: u64,
    interval: Duration,
    tx: tokio::sync::mpsc::UnboundedSender<ProbeOutcome>,
) {
    for registered in registrations {
        let id = registered.peer().instance_id();
        let incarnation = registered.incarnation();
        if id == self_id {
            continue;
        }
        let velo = Arc::clone(velo);
        let tx = tx.clone();
        tokio::spawn(async move {
            let req = HeartbeatRequest { seq };
            let probe = async {
                let unary = velo.typed_unary(HEARTBEAT_HANDLER)?;
                let ack: HeartbeatAck = unary.payload(&req)?.instance(id).send().await?;
                Ok::<HeartbeatAck, anyhow::Error>(ack)
            };
            let outcome = match tokio::time::timeout(interval, probe).await {
                Ok(Ok(ack)) => ProbeOutcome::Ok {
                    id,
                    incarnation,
                    ack_seq: ack.seq,
                },
                Ok(Err(error)) => ProbeOutcome::Failed {
                    id,
                    incarnation,
                    reason: format!("{error:#}"),
                },
                Err(_) => ProbeOutcome::Failed {
                    id,
                    incarnation,
                    reason: format!("heartbeat probe timed out after {interval:?}"),
                },
            };
            let _ = tx.send(outcome);
        });
    }
}

async fn handle_outcome(
    outcome: ProbeOutcome,
    registry: &Arc<dyn PeerRegistry>,
    failures: &mut HashMap<FailureKey, u32>,
    max_failures: u32,
) {
    match outcome {
        ProbeOutcome::Ok {
            id,
            incarnation,
            ack_seq,
        } => {
            failures.remove(&(id, incarnation));
            if let Err(error) = registry.touch(id, incarnation).await {
                tracing::trace!(
                    instance = %id,
                    %incarnation,
                    error = %error,
                    "heartbeat: stale or unavailable touch ignored"
                );
            } else {
                tracing::trace!(instance = %id, %incarnation, ack_seq, "heartbeat: refreshed TTL");
            }
        }
        ProbeOutcome::Failed {
            id,
            incarnation,
            reason,
        } => {
            let key = (id, incarnation);
            let failures_for_incarnation = failures
                .entry(key)
                .and_modify(|count| *count += 1)
                .or_insert(1);
            tracing::warn!(
                instance = %id,
                %incarnation,
                failures = *failures_for_incarnation,
                error = %reason,
                "heartbeat: probe failed"
            );
            if *failures_for_incarnation >= max_failures {
                failures.remove(&key);
                if let Err(error) = registry.unregister(id, incarnation).await {
                    tracing::warn!(
                        instance = %id,
                        %incarnation,
                        error = %error,
                        "heartbeat: conditional unregister ignored"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::InMemoryRegistry;
    use velo_ext::{PeerInfo, WorkerAddress};

    fn peer(label: &[u8]) -> PeerInfo {
        PeerInfo::new(
            InstanceId::new_v4(),
            WorkerAddress::from_encoded(label.to_vec()),
        )
    }

    #[tokio::test]
    async fn stale_probe_failure_cannot_unregister_replacement_incarnation() {
        let concrete = Arc::new(InMemoryRegistry::builder().build());
        let registry: Arc<dyn PeerRegistry> = concrete;
        let peer = peer(b"stale-probe");
        let stale = registry.register(peer.clone()).await.unwrap();
        let current = registry.register(peer.clone()).await.unwrap();
        let mut failures = HashMap::new();

        handle_outcome(
            ProbeOutcome::Failed {
                id: peer.instance_id(),
                incarnation: stale,
                reason: "old probe timed out".to_string(),
            },
            &registry,
            &mut failures,
            1,
        )
        .await;

        assert!(registry.contains(peer.instance_id()));
        registry
            .touch(peer.instance_id(), current)
            .await
            .expect("stale probe outcome removed the replacement");
    }

    #[tokio::test]
    async fn probe_failure_counters_are_scoped_to_incarnation() {
        let concrete = Arc::new(InMemoryRegistry::builder().build());
        let registry: Arc<dyn PeerRegistry> = concrete;
        let peer = peer(b"probe-counter");
        let stale = registry.register(peer.clone()).await.unwrap();
        let current = registry.register(peer.clone()).await.unwrap();
        let mut failures = HashMap::new();

        for incarnation in [stale, current] {
            handle_outcome(
                ProbeOutcome::Failed {
                    id: peer.instance_id(),
                    incarnation,
                    reason: "independent failure".to_string(),
                },
                &registry,
                &mut failures,
                2,
            )
            .await;
        }

        assert_eq!(failures.get(&(peer.instance_id(), stale)), Some(&1));
        assert_eq!(failures.get(&(peer.instance_id(), current)), Some(&1));
        assert!(registry.contains(peer.instance_id()));
    }
}
