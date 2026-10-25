#![cfg(feature = "live-risk")]
use serde_json::Value;
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use volt_compute::live_risk::demo_frame;
const BINARY: &str = env!("CARGO_BIN_EXE_live_option_risk");
struct Server {
    child: Child,
    address: SocketAddr,
}
impl Server {
    fn start() -> Self {
        Self::start_with_output(None)
    }
    fn start_with_output(output: Option<&Path>) -> Self {
        let mut command = Command::new(BINARY);
        command.args([
            "--address",
            "127.0.0.1:0",
            "--generations",
            "1",
            "--interval-ms",
            "20",
        ]);
        if let Some(path) = output {
            command.arg("--output").arg(path);
        }
        let mut child = command
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stderr = child.stderr.take().unwrap();
        let (send, receive) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Some(url) = line.strip_prefix("Volt option risk: http://") {
                    let _ = send.send(url.parse::<SocketAddr>());
                }
            }
        });
        // Guard ownership exists before waiting, including startup failures.
        let mut server = Self {
            child,
            address: "127.0.0.1:0".parse().unwrap(),
        };
        server.address = receive
            .recv_timeout(Duration::from_secs(20))
            .expect("server did not announce bounded startup")
            .unwrap();
        assert!(server.address.ip().is_loopback());
        assert_ne!(server.address.port(), 0);
        server
    }
    fn stream(&self) -> TcpStream {
        let s = TcpStream::connect_timeout(&self.address, Duration::from_secs(3)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        s.set_write_timeout(Some(Duration::from_secs(3))).unwrap();
        s
    }
    fn request(&self, method: &str, path: &str, body: &[u8]) -> Response {
        let mut bytes=format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).into_bytes();
        bytes.extend_from_slice(body);
        self.raw(&bytes)
    }
    fn raw(&self, bytes: &[u8]) -> Response {
        let mut stream = self.stream();
        stream.write_all(bytes).unwrap();
        stream.shutdown(Shutdown::Write).unwrap();
        let mut result = Vec::new();
        stream
            .take(8 * 1024 * 1024)
            .read_to_end(&mut result)
            .unwrap();
        let end = result
            .windows(4)
            .position(|s| s == b"\r\n\r\n")
            .expect("HTTP response headers")
            + 4;
        let header = String::from_utf8(result[..end].to_vec()).unwrap();
        let status = header.split_whitespace().nth(1).unwrap().parse().unwrap();
        Response {
            status,
            header,
            body: result[end..].to_vec(),
        }
    }
    fn snapshot(&self) -> Value {
        let response = self.request("GET", "/api/snapshot", b"");
        assert_eq!(response.status, 200);
        response.json()
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
struct Response {
    status: u16,
    header: String,
    body: Vec<u8>,
}
impl Response {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }
}
fn event(reader: &mut BufReader<TcpStream>) -> (u64, Value) {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut id = None;
    let mut data = None;
    loop {
        assert!(Instant::now() < deadline, "SSE event deadline");
        let mut line = String::new();
        assert!(reader.read_line(&mut line).unwrap() > 0);
        assert!(line.len() < 4 * 1024 * 1024);
        if let Some(x) = line.strip_prefix("id: ") {
            id = Some(x.trim().parse().unwrap());
        }
        if let Some(x) = line.strip_prefix("data: ") {
            data = Some(serde_json::from_str(x.trim()).unwrap());
        }
        if line == "\n"
            && let (Some(id), Some(data)) = (id.take(), data.take())
        {
            return (id, data);
        }
    }
}
#[test]
fn actual_http_snapshot_health_dashboard_sse_and_quote_post_share_one_generation() {
    let server = Server::start();
    let initial = server.snapshot();
    assert_eq!(initial["generation"], 1);
    assert_eq!(initial["cube"]["generation"], 1);
    let health = server.request("GET", "/api/health", b"");
    assert_eq!(health.status, 200);
    assert_eq!(health.json()["generation"], 1);
    let index = server.request("GET", "/", b"");
    assert_eq!(index.status, 200);
    assert!(index.header.contains("text/html"));
    let html = String::from_utf8(index.body).unwrap();
    assert!(html.contains("/api/stream"));
    assert!(html.contains("/api/snapshot"));
    assert_eq!(server.request("GET", "/api/not-a-route", b"").status, 404);
    let mut stream = server.stream();
    stream
        .write_all(b"GET /api/stream HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut header = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        header.push_str(&line);
        if line == "\r\n" {
            break;
        }
    }
    assert!(header.contains("200 OK"));
    assert!(header.contains("text/event-stream"));
    let (id, first) = event(&mut reader);
    assert_eq!(id, 1);
    assert_eq!(first["input_sha256"], initial["input_sha256"]);
    let mut frame = demo_frame(2, initial["cutoff_ns"].as_i64().unwrap() + 1_000_000).unwrap();
    frame.position_revision = 2;
    frame.positions[0].quantity += 1;
    let posted = server.request("POST", "/api/quotes", &serde_json::to_vec(&frame).unwrap());
    assert_eq!(
        posted.status,
        200,
        "{}",
        String::from_utf8_lossy(&posted.body)
    );
    let published = posted.json();
    assert_eq!(published["generation"], 2);
    assert_eq!(published["cube"]["generation"], 2);
    assert_eq!(published["position_revision"], 2);
    assert_ne!(published["input_sha256"], initial["input_sha256"]);
    assert_ne!(published["positions_sha256"], initial["positions_sha256"]);
    assert_eq!(server.snapshot(), published);
    let (id, second) = event(&mut reader);
    assert_eq!(id, 2);
    assert_eq!(second, published);
    let health = server.request("GET", "/api/health", b"").json();
    assert_eq!(health["generation"], 2);
    assert_eq!(health["synthetic_generator"], false);
}
#[test]
fn malformed_stale_duplicate_length_and_oversized_http_requests_preserve_published_snapshot() {
    let server = Server::start();
    let before = server.snapshot();
    let stale = demo_frame(1, before["cutoff_ns"].as_i64().unwrap()).unwrap();
    assert_eq!(
        server
            .request("POST", "/api/quotes", &serde_json::to_vec(&stale).unwrap())
            .status,
        422
    );
    assert_eq!(server.snapshot(), before);
    assert_eq!(server.request("POST", "/api/quotes", b"{").status, 400);
    assert_eq!(server.snapshot(), before);
    for request in [
        "POST /api/quotes HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
        "POST /api/quotes HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2097153\r\n\r\n",
        "POST /api/quotes HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n",
        "POST /api/quotes HTTP/1.1\r\nHost: localhost\r\nContent-Length: 5\r\n\r\n{}",
    ] {
        let response = server.raw(request.as_bytes());
        assert_eq!(response.status, 400);
        assert!(response.json()["error"].is_string());
        assert_eq!(server.snapshot(), before);
    }
    // Rejections cannot poison the generation guard: the next valid full frame
    // still publishes2 and its cube mapping, with no intermediate publication.
    let next = demo_frame(2, before["cutoff_ns"].as_i64().unwrap() + 1_000_000).unwrap();
    assert_eq!(
        server
            .request("POST", "/api/quotes", &serde_json::to_vec(&next).unwrap())
            .status,
        200
    );
    assert_eq!(server.snapshot()["cube"]["generation"], 2);
}
struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "volt-live-http-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn once(args: &[&Path]) -> std::process::ExitStatus {
    let mut command = Command::new(BINARY);
    command.arg("--once");
    for arg in args {
        command.arg(arg);
    }
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("CLI --once deadline");
        }
        thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn once_mode_preserves_aliased_input_and_config_and_writes_separate_snapshot() {
    let temp = Temporary::new();
    let input = temp.0.join("frame.json");
    let output = temp.0.join("snapshot.json");
    let config = temp.0.join("config.json");
    let frame = demo_frame(1, 1_000_000_000).unwrap();
    let bytes = serde_json::to_vec(&frame).unwrap();
    fs::write(&input, &bytes).unwrap();
    assert!(!once(&[Path::new("--input"), &input, Path::new("--output"), &input]).success());
    assert_eq!(fs::read(&input).unwrap(), bytes);
    fs::write(
        &config,
        serde_json::to_vec(&volt_compute::live_risk::LiveRiskConfig::default()).unwrap(),
    )
    .unwrap();
    let config_bytes = fs::read(&config).unwrap();
    assert!(
        !once(&[
            Path::new("--config"),
            &config,
            Path::new("--output"),
            &config
        ])
        .success()
    );
    assert_eq!(fs::read(&config).unwrap(), config_bytes);
    #[cfg(unix)]
    {
        let alias = temp.0.join("alias.json");
        std::os::unix::fs::symlink(&input, &alias).unwrap();
        assert!(!once(&[Path::new("--input"), &input, Path::new("--output"), &alias]).success());
        assert_eq!(fs::read(&input).unwrap(), bytes);
    }
    assert!(once(&[Path::new("--input"), &input, Path::new("--output"), &output]).success());
    let result: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(result["generation"], 1);
    assert_eq!(result["cube"]["generation"], 1);
    assert_eq!(fs::read(&input).unwrap(), bytes);
}

#[test]
fn output_publication_failure_preserves_disk_and_http_then_same_generation_retry_succeeds() {
    let temp = Temporary::new();
    let output = temp.0.join("snapshot.json");
    let server = Server::start_with_output(Some(&output));
    let before = server.snapshot();
    let initial_bytes = fs::read(&output).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&initial_bytes).unwrap(),
        before
    );
    let blocker = output.with_extension(format!("tmp-{}", server.child.id()));
    fs::create_dir(&blocker).unwrap();
    let mut frame = demo_frame(2, before["cutoff_ns"].as_i64().unwrap() + 1_000_000).unwrap();
    frame.position_revision = 2;
    frame.positions[0].quantity += 1;
    let bytes = serde_json::to_vec(&frame).unwrap();
    let rejected = server.request("POST", "/api/quotes", &bytes);
    assert_eq!(rejected.status, 422);
    assert_eq!(server.snapshot(), before);
    assert_eq!(fs::read(&output).unwrap(), initial_bytes);
    fs::remove_dir(blocker).unwrap();
    let accepted = server.request("POST", "/api/quotes", &bytes);
    assert_eq!(
        accepted.status,
        200,
        "{}",
        String::from_utf8_lossy(&accepted.body)
    );
    let published = accepted.json();
    assert_eq!(published["generation"], 2);
    assert_eq!(published["cube"]["generation"], 2);
    assert_eq!(published["position_revision"], 2);
    assert_eq!(server.snapshot(), published);
    assert_eq!(
        serde_json::from_slice::<Value>(&fs::read(&output).unwrap()).unwrap(),
        published
    );
}
