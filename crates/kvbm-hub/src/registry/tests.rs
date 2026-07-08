// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::Mutex;
use velo_ext::WorkerAddress;

fn make_peer() -> PeerInfo {
    PeerInfo::new(
        InstanceId::new_v4(),
        WorkerAddress::from_encoded(b"test".to_vec()),
    )
}

fn peer_with_address(id: InstanceId, address: &[u8]) -> PeerInfo {
    PeerInfo::new(id, WorkerAddress::from_encoded(address.to_vec()))
}

#[tokio::test]
async fn register_and_list() {
    let reg = InMemoryRegistry::builder().build();
    let peer = make_peer();
    reg.register(peer.clone()).await.unwrap();
    assert!(reg.contains(peer.instance_id()));
    assert_eq!(reg.list().len(), 1);
}

#[tokio::test]
async fn reregister_same_instance_replaces_single_entry() {
    let reg = InMemoryRegistry::builder().build();
    let peer = make_peer();
    reg.register(peer.clone()).await.unwrap();
    reg.register(peer.clone()).await.unwrap();
    assert_eq!(reg.list().len(), 1);
}

#[tokio::test]
async fn reregister_mints_incarnation_and_rejects_stale_mutations() {
    let reg = InMemoryRegistry::builder().build();
    let peer = make_peer();

    let first = reg.register(peer.clone()).await.unwrap();
    let second = reg.register(peer.clone()).await.unwrap();

    assert_ne!(first, second);
    assert_eq!(reg.registrations()[0].incarnation(), second);
    assert_eq!(reg.current_incarnation(peer.instance_id()), Some(second));
    assert!(!reg.is_current(peer.instance_id(), first));
    assert!(reg.is_current(peer.instance_id(), second));
    assert!(matches!(
        reg.touch(peer.instance_id(), first).await,
        Err(RegistryError::StaleIncarnation { .. })
    ));
    assert!(matches!(
        reg.unregister(peer.instance_id(), first).await,
        Err(RegistryError::StaleIncarnation { .. })
    ));
    assert!(reg.contains(peer.instance_id()));
    reg.touch(peer.instance_id(), second).await.unwrap();
    reg.unregister(peer.instance_id(), second).await.unwrap();
    assert!(!reg.contains(peer.instance_id()));
    assert_eq!(reg.current_incarnation(peer.instance_id()), None);
}

#[tokio::test]
async fn reregister_purges_every_worker_mapping_owned_by_instance() {
    let reg = InMemoryRegistry::builder().build();
    let id = InstanceId::new_v4();
    let peer_a = peer_with_address(id, b"address-a");
    let peer_b = peer_with_address(id, b"address-b");
    let stale_a = WorkerId::from_u64(peer_a.worker_id().as_u64().wrapping_add(1));
    let stale_b = WorkerId::from_u64(peer_a.worker_id().as_u64().wrapping_add(2));

    reg.register(peer_a.clone()).await.unwrap();
    reg.inner.write().by_worker.insert(stale_a, id);
    reg.register(peer_b).await.unwrap();
    assert!(!reg.inner.read().by_worker.contains_key(&stale_a));

    // Model a failed B registration restoring A: the restored write must
    // not retain any worker claim left by the attempted incarnation.
    reg.inner.write().by_worker.insert(stale_b, id);
    reg.register(peer_a.clone()).await.unwrap();

    let inner = reg.inner.read();
    let owned_workers: Vec<_> = inner
        .by_worker
        .iter()
        .filter_map(|(worker, owner)| (*owner == id).then_some(*worker))
        .collect();
    assert_eq!(owned_workers, vec![peer_a.worker_id()]);
    assert_eq!(inner.by_instance.get(&id).unwrap().peer(), &peer_a);
}

#[tokio::test]
async fn removal_callback_identifies_removed_incarnation() {
    let reg = InMemoryRegistry::builder().build();
    let removals = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&removals);
    reg.install_removal_hook(Arc::new(move |removal| {
        captured.lock().unwrap().push(removal);
    }))
    .unwrap();
    let peer = make_peer();

    let stale = reg.register(peer.clone()).await.unwrap();
    let current = reg.register(peer.clone()).await.unwrap();
    assert!(matches!(
        reg.unregister(peer.instance_id(), stale).await,
        Err(RegistryError::StaleIncarnation { .. })
    ));
    assert!(removals.lock().unwrap().is_empty());

    reg.unregister(peer.instance_id(), current).await.unwrap();
    assert_eq!(
        removals.lock().unwrap().as_slice(),
        &[RegistryRemoval::new(peer.instance_id(), current)]
    );
}

#[tokio::test]
async fn hook_installation_returns_existing_registrations() {
    let reg = InMemoryRegistry::builder().build();
    let peer = make_peer();
    let incarnation = reg.register(peer.clone()).await.unwrap();
    let removals = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&removals);

    let snapshot = reg
        .install_removal_hook(Arc::new(move |removal| {
            captured.lock().unwrap().push(removal);
        }))
        .unwrap();

    assert_eq!(
        snapshot,
        vec![RegisteredPeer::new(peer.clone(), incarnation)]
    );
    reg.unregister(peer.instance_id(), incarnation)
        .await
        .unwrap();
    assert_eq!(
        removals.lock().unwrap().as_slice(),
        &[RegistryRemoval::new(peer.instance_id(), incarnation)]
    );
}

#[test]
fn removal_hook_cannot_be_replaced() {
    let reg = InMemoryRegistry::builder().build();
    reg.install_removal_hook(Arc::new(|_| {})).unwrap();

    assert!(matches!(
        reg.install_removal_hook(Arc::new(|_| {})),
        Err(RegistryError::RemovalHookAlreadyInstalled)
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hook_snapshot_and_removal_share_a_linearization_point() {
    for _ in 0..64 {
        let reg = Arc::new(InMemoryRegistry::builder().build());
        let peer = make_peer();
        let incarnation = reg.register(peer.clone()).await.unwrap();
        let removals = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(tokio::sync::Barrier::new(3));

        let install_reg = Arc::clone(&reg);
        let install_barrier = Arc::clone(&barrier);
        let captured = Arc::clone(&removals);
        let install = tokio::spawn(async move {
            install_barrier.wait().await;
            install_reg
                .install_removal_hook(Arc::new(move |removal| {
                    captured.lock().unwrap().push(removal);
                }))
                .unwrap()
        });

        let remove_reg = Arc::clone(&reg);
        let remove_barrier = Arc::clone(&barrier);
        let remove = tokio::spawn(async move {
            remove_barrier.wait().await;
            remove_reg
                .unregister(peer.instance_id(), incarnation)
                .await
                .unwrap();
        });

        barrier.wait().await;
        let snapshot = install.await.unwrap();
        remove.await.unwrap();
        assert_eq!(snapshot.len(), removals.lock().unwrap().len());
    }
}

#[tokio::test]
async fn registrations_snapshot_pairs_peer_with_incarnation() {
    let reg = InMemoryRegistry::builder().build();
    let peer = make_peer();
    let incarnation = reg.register(peer.clone()).await.unwrap();

    assert_eq!(
        reg.registrations(),
        vec![RegisteredPeer::new(peer, incarnation)]
    );
}

#[tokio::test]
async fn conflict_on_different_instance_same_worker() {
    let reg = InMemoryRegistry::builder().build();
    let a = make_peer();
    // b shares the worker_id of a (worker_id is derived from instance_id,
    // so we craft a PeerInfo that reuses a's instance_id → same worker_id).
    // Use a different instance_id but fake-shared worker: easiest is to
    // construct two peers whose worker ids happen to match — in practice
    // we exercise the path by having two distinct ids that share a worker.
    // Since WorkerId is derived from InstanceId, we test the conflict path
    // by first registering a, then constructing b with the SAME worker_id
    // via a different instance. This requires accessing internals, so we
    // do it by directly poking the by_worker map with a known id and then
    // calling register() with a *different* instance that reuses the worker.
    reg.register(a.clone()).await.unwrap();
    // Simulate a second instance claiming the same worker by writing the
    // conflict directly — this exercises the conflict-detection branch.
    let a_wid = a.worker_id();
    let different_instance = InstanceId::new_v4();
    {
        let mut w = reg.inner.write();
        w.by_worker.insert(a_wid, different_instance);
    }
    // Now a's register should see the conflict.
    let err = reg.register(a.clone()).await.unwrap_err();
    assert!(matches!(err, RegistryError::Conflict { .. }));
}

#[tokio::test]
async fn unregister_removes_from_all_maps() {
    let reg = InMemoryRegistry::builder().build();
    let peer = make_peer();
    let incarnation = reg.register(peer.clone()).await.unwrap();
    reg.unregister(peer.instance_id(), incarnation)
        .await
        .unwrap();
    assert!(!reg.contains(peer.instance_id()));
    assert!(reg.discover_by_worker_id(peer.worker_id()).await.is_err());
}

#[tokio::test]
async fn unregister_not_found() {
    let reg = InMemoryRegistry::builder().build();
    let err = reg
        .unregister(InstanceId::new_v4(), RegistryIncarnation::from_u64(1))
        .await
        .unwrap_err();
    assert!(matches!(err, RegistryError::NotFound(_)));
}

#[tokio::test]
async fn touch_not_found() {
    let reg = InMemoryRegistry::builder().build();
    let err = reg
        .touch(InstanceId::new_v4(), RegistryIncarnation::from_u64(1))
        .await
        .unwrap_err();
    assert!(matches!(err, RegistryError::NotFound(_)));
}

#[tokio::test(start_paused = true)]
async fn prune_stale_removes_old_entries() {
    let reg = InMemoryRegistry::builder()
        .ttl(Duration::from_millis(100))
        .prune_interval(Duration::from_millis(50))
        .build();
    let peer = make_peer();
    reg.register(peer.clone()).await.unwrap();
    tokio::time::advance(Duration::from_millis(200)).await;
    reg.prune_stale();
    assert!(!reg.contains(peer.instance_id()));
}

#[tokio::test(start_paused = true)]
async fn prune_reports_the_expired_incarnation() {
    let reg = InMemoryRegistry::builder()
        .ttl(Duration::from_millis(100))
        .build();
    let removals = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&removals);
    reg.install_removal_hook(Arc::new(move |removal| {
        captured.lock().unwrap().push(removal);
    }))
    .unwrap();
    let peer = make_peer();
    let incarnation = reg.register(peer.clone()).await.unwrap();

    tokio::time::advance(Duration::from_millis(200)).await;
    reg.prune_stale();

    assert_eq!(
        removals.lock().unwrap().as_slice(),
        &[RegistryRemoval::new(peer.instance_id(), incarnation)]
    );
}

#[tokio::test(start_paused = true)]
async fn touch_refreshes_ttl() {
    let reg = InMemoryRegistry::builder()
        .ttl(Duration::from_millis(100))
        .build();
    let peer = make_peer();
    let incarnation = reg.register(peer.clone()).await.unwrap();
    tokio::time::advance(Duration::from_millis(80)).await;
    reg.touch(peer.instance_id(), incarnation).await.unwrap();
    tokio::time::advance(Duration::from_millis(80)).await;
    reg.prune_stale();
    assert!(reg.contains(peer.instance_id()));
}

#[tokio::test(start_paused = true)]
async fn protect_exempts_from_prune() {
    let reg = InMemoryRegistry::builder()
        .ttl(Duration::from_millis(100))
        .build();
    let peer = make_peer();
    reg.register(peer.clone()).await.unwrap();
    reg.protect(peer.instance_id());
    tokio::time::advance(Duration::from_millis(500)).await;
    reg.prune_stale();
    assert!(reg.contains(peer.instance_id()));
}
