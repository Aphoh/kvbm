// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use kvbm_common::LogicalResourceId;

/// Reusable all-resource completion barrier for bundle transactions.
pub(super) struct ResourceBarrier {
    expected: BTreeSet<LogicalResourceId>,
    pending: BTreeSet<LogicalResourceId>,
    state: BarrierState,
}

impl ResourceBarrier {
    pub(super) fn new(
        resources: impl IntoIterator<Item = LogicalResourceId>,
    ) -> Result<Self, BarrierError> {
        let resources = resources.into_iter().collect::<Vec<_>>();
        if resources.is_empty() {
            return Err(BarrierError::EmptyResources);
        }
        let expected = resources.iter().copied().collect::<BTreeSet<_>>();
        if expected.len() != resources.len() {
            return Err(BarrierError::DuplicateResource);
        }
        Ok(Self {
            pending: expected.clone(),
            expected,
            state: BarrierState::Prepared,
        })
    }

    pub(super) fn start(&mut self) -> BarrierState {
        if self.state == BarrierState::Prepared {
            self.state = BarrierState::Transferring;
        }
        self.state
    }

    pub(super) fn is_pending(&self, resource: LogicalResourceId) -> bool {
        self.pending.contains(&resource)
    }

    pub(super) fn succeed(
        &mut self,
        resource: LogicalResourceId,
    ) -> Result<BarrierUpdate, BarrierError> {
        self.require_expected(resource)?;
        match self.state {
            BarrierState::Prepared => Err(BarrierError::NotStarted),
            BarrierState::Transferring => {
                if !self.pending.remove(&resource) {
                    return Ok(BarrierUpdate::Pending);
                }
                if self.pending.is_empty() {
                    self.state = BarrierState::Complete;
                    Ok(BarrierUpdate::Complete)
                } else {
                    Ok(BarrierUpdate::Pending)
                }
            }
            settled => Ok(BarrierUpdate::Settled(settled)),
        }
    }

    pub(super) fn fail(
        &mut self,
        resource: LogicalResourceId,
    ) -> Result<BarrierUpdate, BarrierError> {
        self.require_expected(resource)?;
        match self.state {
            BarrierState::Prepared => Err(BarrierError::NotStarted),
            BarrierState::Transferring if self.pending.contains(&resource) => {
                self.state = BarrierState::Aborted;
                self.pending.clear();
                Ok(BarrierUpdate::Aborted)
            }
            BarrierState::Transferring => Ok(BarrierUpdate::Pending),
            settled => Ok(BarrierUpdate::Settled(settled)),
        }
    }

    fn require_expected(&self, resource: LogicalResourceId) -> Result<(), BarrierError> {
        if self.expected.contains(&resource) {
            Ok(())
        } else {
            Err(BarrierError::UnexpectedResource { resource })
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BarrierState {
    Prepared,
    Transferring,
    Complete,
    Aborted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BarrierUpdate {
    Pending,
    Complete,
    Aborted,
    Settled(BarrierState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(in crate::tiering::engine) enum BarrierError {
    #[error("bundle transaction requires at least one resource")]
    EmptyResources,
    #[error("bundle transaction contains a duplicate resource")]
    DuplicateResource,
    #[error("bundle transaction has not started")]
    NotStarted,
    #[error("bundle transaction does not contain resource {resource:?}")]
    UnexpectedResource { resource: LogicalResourceId },
}
