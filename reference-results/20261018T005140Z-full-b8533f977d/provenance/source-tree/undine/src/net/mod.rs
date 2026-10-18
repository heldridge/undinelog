//! Minimal networking module with length-prefixed binary RPCs.
//!
//! This module provides:
//! - Registry file helpers for server discovery.
//! - Wire-friendly types for encoding field and value data.
//! - A small request/response protocol with TCP framing.

pub mod auth;
pub mod convert;
pub mod registry;
pub mod server;
pub mod system;
pub mod unauth;
pub mod wire;

use serde::{Deserialize, Serialize};
use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::Duration;
use wire::{MAX_FRAME_LEN, decode_binary, encode_binary, read_frame, write_frame};

/// RPC request message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Message {
    UnauthRead {
        idx: u64,
    },
    UnauthReadBlock {
        block: u64,
    },
    UnauthReadSegmentRaw {
        seg: u64,
    },
    UnauthReadRowsAtZ {
        seg: u64,
        z_eval: wire::NetFp,
    },
    UnauthReadRowsBatchAtZ {
        seg: u64,
        z_evals: wire::NetFpBlob,
    },
    UnauthReadRowsMatrix {
        requests: Vec<wire::NetRowReadRequest>,
    },
    UnauthOptimizedRead {
        req: wire::UnauthOptimizedReadRequest,
    },
    UnauthMultiRead {
        idxs: Vec<u64>,
    },
    UnauthRepairShard {
        seg: u64,
        registry_index: u32,
    },
    UnauthRepairShardBatch {
        seg: u64,
        registry_indices: Vec<u32>,
    },
    UnauthEvalShardsAtRowLabels {
        seg: u64,
        col_labels: wire::NetFpBlob,
        row_labels: wire::NetFpBlob,
    },
    UnauthEvalShardsAtRowLabelsMatrix {
        segs: Vec<u64>,
        col_labels: wire::NetFpBlob,
        row_labels: wire::NetFpBlob,
    },
    UnauthApplyDeltas {
        deltas: Vec<(u64, wire::NetFp)>,
        mode: wire::NetDeltaMode,
    },
    UnauthApplyBlockDeltas {
        block: u64,
        deltas: Vec<(u32, wire::NetFp)>,
        mode: wire::NetDeltaMode,
    },
    UnauthExtend {
        new_len: u64,
    },
    UnauthKappaDouble,
    UnauthKappaHalve {
        source_halves: Vec<wire::NetSourceHalfState>,
    },
    UnauthEvalPartialStitches {
        segs: Vec<u64>,
        col_labels: wire::NetFpBlob,
        row_labels: wire::NetFpBlob,
    },
    UnauthRetainClaims {
        registry_indices: Vec<u32>,
    },
    UnauthGetLen,
    AuthRead {
        req: wire::AuthReadRequest,
    },
    AuthReadSubtree {
        req: wire::AuthSubtreeReadRequest,
    },
    AuthReadSubtreeUpperPath {
        req: wire::AuthSubtreeUpperPathReadRequest,
    },
    AuthWrite {
        req: wire::AuthWriteRequest,
    },
    AuthWriteBatch {
        req: wire::AuthBatchWriteRequest,
    },
    AuthWriteFreshSubtree {
        req: wire::AuthFreshSubtreeWriteRequest,
    },
    AuthExtend {
        new_leaves: u64,
    },
    AuthGetLen,
    /// Private opening of one position.
    AuthReadPrivate {
        req: wire::AuthPrivateReadRequest,
    },
}

/// RPC response message.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Response {
    UnauthRead {
        values: wire::NetMaybeFpBlob,
    },
    UnauthReadBlock {
        rows: wire::NetRowBatchReply,
    },
    UnauthReadSegmentRaw {
        shards: Vec<wire::NetSegmentPrimaryShardReply>,
    },
    UnauthReadRowsAtZ {
        rows: wire::NetRowBatchReply,
    },
    UnauthReadRowsBatchAtZ {
        rows: wire::NetRowBatchReply,
    },
    UnauthReadRowsMatrix {
        rows: wire::NetRowMatrixReply,
    },
    UnauthOptimizedRead {
        resp: wire::UnauthOptimizedReadResponse,
    },
    UnauthMultiRead {
        values: Vec<wire::NetMaybeFpBlob>,
    },
    UnauthRepairShard {
        values: Vec<wire::NetFp>,
    },
    UnauthRepairShardBatch {
        values: Vec<Vec<wire::NetFp>>,
    },
    UnauthEvalShardsAtRowLabels {
        values: Vec<wire::NetFpBlob>,
    },
    UnauthEvalShardsAtRowLabelsMatrix {
        values: wire::NetFpBlob,
    },
    UnauthApplyDeltas {
        ok: bool,
    },
    UnauthApplyBlockDeltas {
        ok: bool,
    },
    UnauthExtend {
        ok: bool,
    },
    UnauthKappaDouble {
        ok: bool,
    },
    UnauthKappaHalve {
        ok: bool,
    },
    UnauthEvalPartialStitches {
        evals: Vec<Vec<wire::NetPartialStitchEvals>>,
    },
    UnauthRetainClaims {
        ok: bool,
    },
    UnauthLen {
        len: u64,
    },
    AuthRead {
        resp: wire::AuthReadResponse,
    },
    AuthReadSubtree {
        resp: wire::AuthSubtreeReadResponse,
    },
    AuthReadSubtreeUpperPath {
        resp: wire::AuthSubtreeUpperPathReadResponse,
    },
    AuthWrite {
        resp: wire::AuthWriteResponse,
    },
    AuthWriteBatch {
        resp: wire::AuthBatchWriteResponse,
    },
    AuthWriteFreshSubtree {
        resp: wire::AuthBatchWriteResponse,
    },
    AuthExtend {
        ok: bool,
    },
    AuthLen {
        len: u64,
    },
    Error {
        message: String,
    },
    AuthReadPrivate {
        resp: wire::AuthPrivateReadResponse,
    },
}

/// RPC client/server errors.
#[derive(Debug)]
pub enum NetError {
    Io(String),
    Codec(String),
    Remote(String),
    InvalidResponse(String),
}

impl From<io::Error> for NetError {
    fn from(err: io::Error) -> Self {
        NetError::Io(err.to_string())
    }
}

fn trace_enabled() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| {
        std::env::var("NET_DEMO_TRACE")
            .map(|val| val != "0")
            .unwrap_or(false)
    })
}

fn rpc_timeout() -> Duration {
    static RPC_TIMEOUT: OnceLock<Duration> = OnceLock::new();
    *RPC_TIMEOUT.get_or_init(|| {
        let timeout_ms = std::env::var("NET_RPC_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(3_000);
        Duration::from_millis(timeout_ms)
    })
}

/// Read timeout of a private opening. The server passes over its whole
/// database to answer, and prepares the database on the first request, so
/// the reply takes longer than that of other requests.
fn private_rpc_timeout() -> Duration {
    static PRIVATE_RPC_TIMEOUT: OnceLock<Duration> = OnceLock::new();
    *PRIVATE_RPC_TIMEOUT.get_or_init(|| {
        let timeout_ms = std::env::var("NET_PRIVATE_RPC_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(120_000);
        Duration::from_millis(timeout_ms).max(rpc_timeout())
    })
}

fn connect_with_timeout(addr: &str) -> Result<TcpStream, NetError> {
    let timeout = rpc_timeout();
    let mut last_err = None;
    let addrs = addr
        .to_socket_addrs()
        .map_err(|err| NetError::Io(err.to_string()))?;
    for sock_addr in addrs {
        match TcpStream::connect_timeout(&sock_addr, timeout) {
            Ok(stream) => {
                stream
                    .set_nodelay(true)
                    .map_err(|err| NetError::Io(err.to_string()))?;
                stream
                    .set_read_timeout(Some(timeout))
                    .map_err(|err| NetError::Io(err.to_string()))?;
                stream
                    .set_write_timeout(Some(timeout))
                    .map_err(|err| NetError::Io(err.to_string()))?;
                return Ok(stream);
            }
            Err(err) => last_err = Some(err),
        }
    }
    Err(NetError::Io(
        last_err
            .map(|err| err.to_string())
            .unwrap_or_else(|| format!("failed to resolve {addr}")),
    ))
}

fn message_kind(msg: &Message) -> &'static str {
    match msg {
        Message::UnauthRead { .. } => "UnauthRead",
        Message::UnauthReadBlock { .. } => "UnauthReadBlock",
        Message::UnauthReadSegmentRaw { .. } => "UnauthReadSegmentRaw",
        Message::UnauthReadRowsAtZ { .. } => "UnauthReadRowsAtZ",
        Message::UnauthReadRowsBatchAtZ { .. } => "UnauthReadRowsBatchAtZ",
        Message::UnauthReadRowsMatrix { .. } => "UnauthReadRowsMatrix",
        Message::UnauthOptimizedRead { .. } => "UnauthOptimizedRead",
        Message::UnauthMultiRead { .. } => "UnauthMultiRead",
        Message::UnauthRepairShard { .. } => "UnauthRepairShard",
        Message::UnauthRepairShardBatch { .. } => "UnauthRepairShardBatch",
        Message::UnauthEvalShardsAtRowLabels { .. } => "UnauthEvalShardsAtRowLabels",
        Message::UnauthEvalShardsAtRowLabelsMatrix { .. } => "UnauthEvalShardsAtRowLabelsMatrix",
        Message::UnauthApplyDeltas { .. } => "UnauthApplyDeltas",
        Message::UnauthApplyBlockDeltas { .. } => "UnauthApplyBlockDeltas",
        Message::UnauthExtend { .. } => "UnauthExtend",
        Message::UnauthKappaDouble => "UnauthKappaDouble",
        Message::UnauthKappaHalve { .. } => "UnauthKappaHalve",
        Message::UnauthEvalPartialStitches { .. } => "UnauthEvalPartialStitches",
        Message::UnauthRetainClaims { .. } => "UnauthRetainClaims",
        Message::UnauthGetLen => "UnauthGetLen",
        Message::AuthRead { .. } => "AuthRead",
        Message::AuthReadSubtree { .. } => "AuthReadSubtree",
        Message::AuthReadSubtreeUpperPath { .. } => "AuthReadSubtreeUpperPath",
        Message::AuthWrite { .. } => "AuthWrite",
        Message::AuthWriteBatch { .. } => "AuthWriteBatch",
        Message::AuthWriteFreshSubtree { .. } => "AuthWriteFreshSubtree",
        Message::AuthExtend { .. } => "AuthExtend",
        Message::AuthGetLen => "AuthGetLen",
        Message::AuthReadPrivate { .. } => "AuthReadPrivate",
    }
}

fn response_kind(resp: &Response) -> &'static str {
    match resp {
        Response::UnauthRead { .. } => "UnauthRead",
        Response::UnauthReadBlock { .. } => "UnauthReadBlock",
        Response::UnauthReadSegmentRaw { .. } => "UnauthReadSegmentRaw",
        Response::UnauthReadRowsAtZ { .. } => "UnauthReadRowsAtZ",
        Response::UnauthReadRowsBatchAtZ { .. } => "UnauthReadRowsBatchAtZ",
        Response::UnauthReadRowsMatrix { .. } => "UnauthReadRowsMatrix",
        Response::UnauthOptimizedRead { .. } => "UnauthOptimizedRead",
        Response::UnauthMultiRead { .. } => "UnauthMultiRead",
        Response::UnauthRepairShard { .. } => "UnauthRepairShard",
        Response::UnauthRepairShardBatch { .. } => "UnauthRepairShardBatch",
        Response::UnauthEvalShardsAtRowLabels { .. } => "UnauthEvalShardsAtRowLabels",
        Response::UnauthEvalShardsAtRowLabelsMatrix { .. } => "UnauthEvalShardsAtRowLabelsMatrix",
        Response::UnauthApplyDeltas { .. } => "UnauthApplyDeltas",
        Response::UnauthApplyBlockDeltas { .. } => "UnauthApplyBlockDeltas",
        Response::UnauthExtend { .. } => "UnauthExtend",
        Response::UnauthKappaDouble { .. } => "UnauthKappaDouble",
        Response::UnauthKappaHalve { .. } => "UnauthKappaHalve",
        Response::UnauthEvalPartialStitches { .. } => "UnauthEvalPartialStitches",
        Response::UnauthRetainClaims { .. } => "UnauthRetainClaims",
        Response::UnauthLen { .. } => "UnauthLen",
        Response::AuthRead { .. } => "AuthRead",
        Response::AuthReadSubtree { .. } => "AuthReadSubtree",
        Response::AuthReadSubtreeUpperPath { .. } => "AuthReadSubtreeUpperPath",
        Response::AuthWrite { .. } => "AuthWrite",
        Response::AuthWriteBatch { .. } => "AuthWriteBatch",
        Response::AuthWriteFreshSubtree { .. } => "AuthWriteFreshSubtree",
        Response::AuthExtend { .. } => "AuthExtend",
        Response::AuthLen { .. } => "AuthLen",
        Response::Error { .. } => "Error",
        Response::AuthReadPrivate { .. } => "AuthReadPrivate",
    }
}

/// Serve requests on the provided address, dispatching to `handler`.
pub fn serve<H>(addr: &str, handler: H) -> io::Result<()>
where
    H: Fn(Message) -> Response + Send + Sync + 'static,
{
    let listener = TcpListener::bind(addr)?;
    serve_with_listener(listener, handler)
}

/// Serve requests using a pre-bound listener (useful for tests).
pub fn serve_with_listener<H>(listener: TcpListener, handler: H) -> io::Result<()>
where
    H: Fn(Message) -> Response + Send + Sync + 'static,
{
    let handler = Arc::new(handler);
    for stream in listener.incoming() {
        let stream = stream?;
        let _ = stream.set_nodelay(true);
        let handler = handler.clone();
        thread::spawn(move || {
            let _ = handle_connection(stream, handler);
        });
    }
    Ok(())
}

fn serve_with_listener_until_stop<H>(
    listener: TcpListener,
    handler: H,
    stop: Arc<AtomicBool>,
) -> io::Result<()>
where
    H: Fn(Message) -> Response + Send + Sync + 'static,
{
    let handler = Arc::new(handler);
    listener.set_nonblocking(true)?;
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _addr)) => {
                let _ = stream.set_nodelay(true);
                let handler = handler.clone();
                thread::spawn(move || {
                    let _ = handle_connection(stream, handler);
                });
            }
            Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

/// Handle for a background message-level server spawned with [`spawn_with_listener`].
pub struct ServerHandle {
    stop: Arc<AtomicBool>,
    thread: thread::JoinHandle<()>,
    addr: Option<SocketAddr>,
}

impl ServerHandle {
    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.join();
    }

    pub fn addr(&self) -> Option<SocketAddr> {
        self.addr
    }
}

/// Spawn a background message-level server using a pre-bound listener.
pub fn spawn_with_listener<H>(listener: TcpListener, handler: H) -> ServerHandle
where
    H: Fn(Message) -> Response + Send + Sync + 'static,
{
    let stop = Arc::new(AtomicBool::new(false));
    let addr = listener.local_addr().ok();
    let stop_clone = stop.clone();
    let thread = thread::spawn(move || {
        let _ = serve_with_listener_until_stop(listener, handler, stop_clone);
    });
    ServerHandle { stop, thread, addr }
}

/// Send a single RPC request and return the response.
pub fn rpc_call(addr: &str, msg: &Message) -> Result<Response, NetError> {
    let trace = trace_enabled();
    if trace {
        eprintln!("rpc -> {} {}", addr, message_kind(msg));
    }
    let mut stream = connect_with_timeout(addr)?;
    if matches!(msg, Message::AuthReadPrivate { .. }) {
        stream
            .set_read_timeout(Some(private_rpc_timeout()))
            .map_err(|err| NetError::Io(err.to_string()))?;
    }
    let payload = encode_binary(msg).map_err(NetError::Codec)?;
    write_frame(&mut stream, &payload).map_err(|err| NetError::Io(err.to_string()))?;
    let frame = read_frame(&mut stream, MAX_FRAME_LEN)
        .map_err(|err| NetError::Io(err.to_string()))?
        .ok_or_else(|| NetError::InvalidResponse("eof".to_string()))?;
    let resp: Response = decode_binary(&frame).map_err(NetError::Codec)?;
    if trace {
        eprintln!("rpc <- {} {}", addr, response_kind(&resp));
    }
    Ok(resp)
}

fn handle_connection(
    mut stream: TcpStream,
    handler: Arc<dyn Fn(Message) -> Response + Send + Sync>,
) -> io::Result<()> {
    loop {
        let Some(frame) = read_frame(&mut stream, MAX_FRAME_LEN)? else {
            return Ok(());
        };
        let req: Message = match decode_binary(&frame) {
            Ok(req) => req,
            Err(err) => {
                let err_msg = Response::Error { message: err };
                let payload = encode_binary(&err_msg)
                    .map_err(|codec_err| io::Error::new(io::ErrorKind::InvalidData, codec_err))?;
                write_frame(&mut stream, &payload)?;
                return Ok(());
            }
        };
        let response = match panic::catch_unwind(AssertUnwindSafe(|| handler(req))) {
            Ok(response) => response,
            Err(payload) => Response::Error {
                message: format!("handler panic: {}", panic_payload_message(&payload)),
            },
        };
        let payload = encode_binary(&response)
            .map_err(|codec_err| io::Error::new(io::ErrorKind::InvalidData, codec_err))?;
        write_frame(&mut stream, &payload)?;
    }
}

fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}
