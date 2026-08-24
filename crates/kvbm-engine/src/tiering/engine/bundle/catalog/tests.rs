#![allow(clippy::disallowed_macros)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex, Weak, mpsc};

use kvbm_common::{LogicalResourceId, SequenceHash};
use kvbm_protocols::cache_manifest::{
    BundleKey, CacheManifest, ModelIdentity, ResourceRequirement, ResourceRole,
};

use super::{BundleCatalog, BundleCatalogError};
use crate::tiering::engine::bundle::{BundleResourcePin, BundleResourceReference};
use crate::tiering::policy::ResourceLineage;

const HISTORY: LogicalResourceId = LogicalResourceId(40);
const CAPSULE: LogicalResourceId = LogicalResourceId(41);

struct PausedPin {
    value: Arc<str>,
    pause: Arc<ReferencePause>,
}

struct WeakPin(Weak<str>);

struct ReferencePause {
    first: AtomicBool,
    entered: Barrier,
    release: Barrier,
}

impl BundleResourcePin for PausedPin {
    type Reference = WeakPin;

    fn make_reference(&self) -> Self::Reference {
        if self.pause.first.swap(false, Ordering::AcqRel) {
            self.pause.entered.wait();
            self.pause.release.wait();
        }
        WeakPin(Arc::downgrade(&self.value))
    }
}

impl BundleResourceReference for WeakPin {
    type Pin = PausedPin;

    fn reacquire(&self) -> Option<Self::Pin> {
        self.0.upgrade().map(|value| PausedPin {
            value,
            pause: Arc::new(ReferencePause {
                first: AtomicBool::new(false),
                entered: Barrier::new(1),
                release: Barrier::new(1),
            }),
        })
    }
}

#[test]
fn publication_and_eviction_are_one_catalog_transaction() {
    let (identity, key) = identity_and_key();
    let pause = Arc::new(ReferencePause {
        first: AtomicBool::new(true),
        entered: Barrier::new(2),
        release: Barrier::new(2),
    });
    let resources = resources(Arc::clone(&pause));
    let catalog = Arc::new(Mutex::new(BundleCatalog::new()));
    let publication_catalog = Arc::clone(&catalog);
    let publication = std::thread::spawn(move || {
        publication_catalog
            .lock()
            .unwrap()
            .commit(&identity, key, 1, &resources, lineages(key))
            .unwrap();
    });

    pause.entered.wait();
    let (attempted_tx, attempted_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let eviction_catalog = Arc::clone(&catalog);
    let eviction = std::thread::spawn(move || {
        attempted_tx.send(()).unwrap();
        let invalidated = eviction_catalog
            .lock()
            .unwrap()
            .invalidate_resource(HISTORY, &[key.boundary_hash()]);
        done_tx.send(invalidated).unwrap();
    });
    attempted_rx.recv().unwrap();
    assert!(
        done_rx.try_recv().is_err(),
        "eviction cannot observe the catalog while publication is paused"
    );

    pause.release.wait();
    publication.join().unwrap();
    let invalidated = done_rx.recv().unwrap();
    eviction.join().unwrap();

    assert_eq!(invalidated.len(), 1);
    let catalog = catalog.lock().unwrap();
    assert!(catalog.lease_exact(&identity_and_key().0, &key).is_none());
    assert!(catalog.dependents(HISTORY, key.boundary_hash()).is_empty());
}

#[test]
fn invalidation_tombstone_rejects_same_generation_republication() {
    let (identity, key) = identity_and_key();
    let pause = Arc::new(ReferencePause {
        first: AtomicBool::new(false),
        entered: Barrier::new(1),
        release: Barrier::new(1),
    });
    let resources = resources(pause);
    let mut catalog = BundleCatalog::new();
    catalog
        .commit(&identity, key, 9, &resources, lineages(key))
        .unwrap();
    assert_eq!(
        catalog
            .invalidate_resource(HISTORY, &[key.boundary_hash()])
            .len(),
        1
    );

    assert_eq!(
        catalog.commit(&identity, key, 9, &resources, lineages(key)),
        Err(BundleCatalogError::RetiredGeneration {
            retired: 9,
            attempted: 9,
        })
    );
    catalog
        .commit(&identity, key, 10, &resources, lineages(key))
        .expect("a genuinely newer generation may replace the tombstone");
}

fn identity_and_key() -> (kvbm_protocols::cache_manifest::CacheIdentity, BundleKey) {
    let manifest = CacheManifest::new(
        ModelIdentity::new("catalog-test", "v1", [8; 32]).unwrap(),
        "catalog-test-v1",
        vec![
            ResourceRequirement::new(HISTORY, ResourceRole::PrefixHistory, 4).unwrap(),
            ResourceRequirement::new(CAPSULE, ResourceRole::BoundaryCapsule, 4).unwrap(),
        ],
        Default::default(),
    )
    .unwrap();
    let identity = manifest.identity();
    let key = BundleKey::new(&identity, hash(), 4).unwrap();
    (identity, key)
}

fn resources(pause: Arc<ReferencePause>) -> BTreeMap<LogicalResourceId, PausedPin> {
    [
        (
            HISTORY,
            PausedPin {
                value: Arc::from("history"),
                pause: Arc::clone(&pause),
            },
        ),
        (
            CAPSULE,
            PausedPin {
                value: Arc::from("capsule"),
                pause,
            },
        ),
    ]
    .into_iter()
    .collect()
}

fn lineages(key: BundleKey) -> Vec<ResourceLineage> {
    vec![
        ResourceLineage::new(
            HISTORY,
            ResourceRole::PrefixHistory,
            vec![key.boundary_hash()],
        ),
        ResourceLineage::new(
            CAPSULE,
            ResourceRole::BoundaryCapsule,
            vec![key.boundary_hash()],
        ),
    ]
}

fn hash() -> SequenceHash {
    SequenceHash::new(1, None, 1)
}
