// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Aggregate object-store responses from every SPMD worker.

use std::collections::HashMap;

use crate::SequenceHash;

/// Require one matching presence response from every worker for each key.
pub(super) fn aggregate_object_presence(
    requested: &[SequenceHash],
    worker_results: Vec<Vec<(SequenceHash, Option<usize>)>>,
) -> Vec<(SequenceHash, Option<usize>)> {
    let fail_all = || requested.iter().copied().map(|hash| (hash, None)).collect();
    if worker_results.is_empty() {
        return fail_all();
    }

    let positions: HashMap<SequenceHash, usize> = requested
        .iter()
        .copied()
        .enumerate()
        .map(|(position, hash)| (hash, position))
        .collect();
    if positions.len() != requested.len() {
        return fail_all();
    }

    let mut consensus = vec![None; requested.len()];
    for (worker_index, results) in worker_results.into_iter().enumerate() {
        let mut seen = vec![false; requested.len()];
        let mut valid = true;

        for (hash, size) in results {
            let Some(&position) = positions.get(&hash) else {
                valid = false;
                continue;
            };
            if seen[position] {
                valid = false;
                continue;
            }
            seen[position] = true;
            if worker_index == 0 {
                consensus[position] = size;
            } else if consensus[position] != size {
                consensus[position] = None;
            }
        }

        if !valid || seen.iter().any(|was_seen| !was_seen) {
            return fail_all();
        }
    }

    requested.iter().copied().zip(consensus).collect()
}

/// Require one successful object response from every worker for each key.
pub(super) fn aggregate_object_results(
    requested: &[SequenceHash],
    worker_results: Vec<Vec<Result<SequenceHash, SequenceHash>>>,
) -> Vec<Result<SequenceHash, SequenceHash>> {
    let fail_all = || requested.iter().copied().map(Err).collect();
    if worker_results.is_empty() {
        return fail_all();
    }

    let positions: HashMap<SequenceHash, usize> = requested
        .iter()
        .copied()
        .enumerate()
        .map(|(position, hash)| (hash, position))
        .collect();
    if positions.len() != requested.len() {
        return fail_all();
    }

    let mut succeeded = vec![true; requested.len()];
    for results in worker_results {
        let mut seen = vec![false; requested.len()];
        let mut valid = true;

        for result in results {
            let (hash, worker_succeeded) = match result {
                Ok(hash) => (hash, true),
                Err(hash) => (hash, false),
            };
            let Some(&position) = positions.get(&hash) else {
                valid = false;
                continue;
            };
            if seen[position] {
                valid = false;
                continue;
            }
            seen[position] = true;
            succeeded[position] &= worker_succeeded;
        }

        if !valid || seen.iter().any(|was_seen| !was_seen) {
            return fail_all();
        }
    }

    requested
        .iter()
        .copied()
        .zip(succeeded)
        .map(|(hash, succeeded)| if succeeded { Ok(hash) } else { Err(hash) })
        .collect()
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    fn sequence_hash(value: u64) -> SequenceHash {
        SequenceHash::new(value, None, value)
    }

    #[test]
    fn results_use_returned_hashes_across_reordered_partial_failures() {
        let a = sequence_hash(1);
        let b = sequence_hash(2);
        let c = sequence_hash(3);

        let results = aggregate_object_results(
            &[a, b, c],
            vec![vec![Ok(c), Ok(a), Err(b)], vec![Ok(b), Ok(c), Ok(a)]],
        );

        assert_eq!(results, vec![Ok(a), Err(b), Ok(c)]);
    }

    #[test]
    fn results_reject_missing_duplicate_and_unknown_hashes() {
        let a = sequence_hash(1);
        let b = sequence_hash(2);
        let unknown = sequence_hash(9);

        for malformed in [
            vec![Ok(a)],
            vec![Ok(a), Err(a)],
            vec![Ok(a), Ok(b), Ok(unknown)],
        ] {
            assert_eq!(
                aggregate_object_results(&[a, b], vec![malformed]),
                vec![Err(a), Err(b)]
            );
        }
    }

    #[test]
    fn presence_uses_returned_hashes_and_requires_rank_agreement() {
        let a = sequence_hash(1);
        let b = sequence_hash(2);
        let c = sequence_hash(3);

        let results = aggregate_object_presence(
            &[a, b, c],
            vec![
                vec![(c, Some(30)), (a, Some(10)), (b, Some(20))],
                vec![(b, None), (c, Some(30)), (a, Some(10))],
            ],
        );

        assert_eq!(results, vec![(a, Some(10)), (b, None), (c, Some(30))]);
    }

    #[test]
    fn presence_rejects_missing_duplicate_and_unknown_hashes() {
        let a = sequence_hash(1);
        let b = sequence_hash(2);
        let unknown = sequence_hash(9);

        for malformed in [
            vec![(a, Some(10))],
            vec![(a, Some(10)), (a, None)],
            vec![(a, Some(10)), (b, Some(20)), (unknown, Some(90))],
        ] {
            assert_eq!(
                aggregate_object_presence(&[a, b], vec![malformed]),
                vec![(a, None), (b, None)]
            );
        }
    }
}
