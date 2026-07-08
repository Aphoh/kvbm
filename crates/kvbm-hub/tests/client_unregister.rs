// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{delete, post};
use axum::{Json, Router};
use kvbm_hub::HubClient;
use kvbm_hub::protocol::{ErrorBody, ErrorCode, RegisterRequest, RegisterResponse, paths};
use tokio::net::TcpListener;
use velo_ext::{InstanceId, PeerInfo, WorkerAddress};

struct UnregisterServer {
    address: SocketAddr,
    delete_attempts: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for UnregisterServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test]
async fn explicit_unregister_retries_after_server_failure() {
    let server = start_unregister_server().await;
    let client = register_client(&server).await;

    assert!(client.unregister().await.is_err());
    assert!(client.unregister().await.is_ok());
    assert_eq!(server.delete_attempts.load(Ordering::Acquire), 2);
}

#[tokio::test]
async fn drop_retries_after_explicit_unregister_failure() {
    let server = start_unregister_server().await;
    let client = register_client(&server).await;

    assert!(client.unregister().await.is_err());
    drop(client);

    tokio::time::timeout(Duration::from_secs(1), async {
        while server.delete_attempts.load(Ordering::Acquire) < 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("registration guard drop should retry the DELETE");
}

async fn register_client(server: &UnregisterServer) -> Arc<HubClient> {
    let client = kvbm_hub::create_client_builder()
        .host("127.0.0.1")
        .discovery_port(server.address.port())
        .control_port(server.address.port())
        .build()
        .unwrap();
    client.register_instance(make_peer()).await.unwrap();
    client
}

async fn start_unregister_server() -> UnregisterServer {
    let delete_attempts = Arc::new(AtomicUsize::new(0));
    let router = Router::new()
        .route(paths::INSTANCES, post(register))
        .route(paths::INSTANCE_BY_ID, delete(unregister))
        .with_state(Arc::clone(&delete_attempts));
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    UnregisterServer {
        address,
        delete_attempts,
        task,
    }
}

async fn register(Json(request): Json<RegisterRequest>) -> Json<RegisterResponse> {
    Json(RegisterResponse {
        instance_id: request.peer_info.instance_id(),
        hub_instance_id: None,
        mutation_credential: None,
        registration_epoch: None,
    })
}

async fn unregister(
    State(delete_attempts): State<Arc<AtomicUsize>>,
) -> Result<StatusCode, (StatusCode, Json<ErrorBody>)> {
    if delete_attempts.fetch_add(1, Ordering::AcqRel) == 0 {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                code: ErrorCode::Internal,
                message: "transient failure".to_string(),
            }),
        ));
    }
    Ok(StatusCode::NO_CONTENT)
}

fn make_peer() -> PeerInfo {
    PeerInfo::new(
        InstanceId::new_v4(),
        WorkerAddress::from_encoded(b"client-unregister-test".to_vec()),
    )
}
