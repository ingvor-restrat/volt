//! Local browser application for complete European quote/book frames.
use clap::Parser;
use serde::Serialize;
use std::{
    error::Error,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};
use volt_compute::{
    live_risk::{LiveRiskConfig, LiveRiskEngine, LiveRiskSnapshot, QuoteFrame, demo_frame, now_ns},
    volatility_surface::GreekConvention,
};
const HTML: &str = include_str!("../live_risk_dashboard.html");
#[derive(Parser)]
#[command(
    about = "European option surfaces, portfolio Greeks and scenarios over coherent Hypercube generations"
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8088")]
    address: String,
    /// Compute a complete JSON quote/book frame instead of starting synthetic quotes.
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    /// Atomic latest snapshot. Must differ from input/config paths.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Produce one snapshot and exit.
    #[arg(long)]
    once: bool,
    /// Serve a supplied frame without generating synthetic updates.
    #[arg(long)]
    serve: bool,
    #[arg(long)]
    example_frame: bool,
    #[arg(long, default_value_t = 250)]
    interval_ms: u64,
    /// Stop synthetic updates after this many generations; HTTP remains available.
    #[arg(long, default_value_t = 0)]
    generations: u64,
    #[arg(long)]
    sticky_log_moneyness: bool,
}
struct App {
    engine: Mutex<LiveRiskEngine>,
    snapshot: RwLock<Arc<LiveRiskSnapshot>>,
    demo: AtomicBool,
    output: Option<PathBuf>,
    limit: usize,
}
impl App {
    fn snapshot(&self) -> Result<Arc<LiveRiskSnapshot>, String> {
        self.snapshot
            .read()
            .map(|s| Arc::clone(&s))
            .map_err(|_| "snapshot lock failed".into())
    }
    fn apply(&self, frame: QuoteFrame) -> Result<Arc<LiveRiskSnapshot>, String> {
        let mut engine = self.engine.lock().map_err(|_| "calculation lock failed")?;
        let next = Arc::new(
            engine
                .update_and_publish(frame, |next| {
                    if let Some(path) = &self.output {
                        atomic_json(path, next)
                            .map_err(|e| volt_compute::ComputeError::Execution(e.to_string()))?;
                    }
                    Ok(())
                })
                .map_err(|e| e.to_string())?,
        );
        *self
            .snapshot
            .write()
            .map_err(|_| "snapshot publication lock failed")? = Arc::clone(&next);
        Ok(next)
    }
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if let Some(p) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p)?;
    }
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let result = (|| -> Result<(), Box<dyn Error>> {
        let bytes = serde_json::to_vec(value)?;
        fs::write(&temp, bytes)?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}
fn read_json<T: serde::de::DeserializeOwned>(
    path: &Path,
    limit: usize,
) -> Result<T, Box<dyn Error>> {
    let f = fs::File::open(path)?;
    if f.metadata()?.len() > limit as u64 {
        return Err("JSON input exceeds byte cap".into());
    }
    let mut bytes = Vec::new();
    f.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("JSON input exceeds byte cap".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.interval_ms < 20 || args.interval_ms > 60_000 {
        return Err("interval must be20..60000milliseconds".into());
    }
    if args.example_frame {
        println!(
            "{}",
            serde_json::to_string_pretty(&demo_frame(1, now_ns())?)?
        );
        return Ok(());
    }
    if let Some(out) = &args.output {
        for input in [&args.input, &args.config].into_iter().flatten() {
            if fs::canonicalize(out)
                .ok()
                .zip(fs::canonicalize(input).ok())
                .is_some_and(|(a, b)| a == b)
                || out == input
            {
                return Err("output must not overwrite input/config".into());
            }
        }
    }
    let mut config = args
        .config
        .as_ref()
        .map(|p| read_json::<LiveRiskConfig>(p, 65536))
        .transpose()?
        .unwrap_or_default();
    if args.sticky_log_moneyness {
        config.convention = GreekConvention::StickyLogMoneynessFiniteBump {
            relative_spot_bump: 0.001,
        };
    }
    let limit = config.max_request_bytes;
    let mut engine = LiveRiskEngine::new(config)?;
    let frame = if let Some(p) = &args.input {
        read_json::<QuoteFrame>(p, limit)?
    } else {
        demo_frame(1, now_ns())?
    };
    let snapshot = Arc::new(engine.update(frame)?);
    if let Some(p) = &args.output {
        atomic_json(p, snapshot.as_ref())?;
    }
    if args.once || (args.input.is_some() && !args.serve) {
        if args.output.is_none() {
            println!("{}", serde_json::to_string_pretty(snapshot.as_ref())?);
        }
        return Ok(());
    }
    let demo = args.input.is_none();
    let app = Arc::new(App {
        engine: Mutex::new(engine),
        snapshot: RwLock::new(snapshot),
        demo: AtomicBool::new(demo),
        output: args.output,
        limit,
    });
    if demo {
        let a = Arc::clone(&app);
        let interval = Duration::from_millis(args.interval_ms);
        let max = args.generations;
        thread::spawn(move || {
            loop {
                thread::sleep(interval);
                if !a.demo.load(Ordering::Acquire) {
                    break;
                }
                let Ok(current) = a.snapshot() else {
                    break;
                };
                if max > 0 && current.generation >= max {
                    a.demo.store(false, Ordering::Release);
                    break;
                }
                let Some(generation) = current.generation.checked_add(1) else {
                    break;
                };
                let result = demo_frame(generation, now_ns().max(current.cutoff_ns))
                    .map_err(|e| e.to_string())
                    .and_then(|f| a.apply(f));
                if let Err(e) = result {
                    eprintln!("synthetic update rejected: {e}");
                }
            }
        });
    }
    let listener = TcpListener::bind(&args.address)?;
    eprintln!("Volt option risk: http://{}", listener.local_addr()?);
    eprintln!(
        "{}",
        if demo {
            "Synthetic quotes; POST /api/quotes accepts an external complete frame and stops the generator."
        } else {
            "Frame/replay mode; POST /api/quotes publishes newer complete quote/book frames."
        }
    );
    let active = Arc::new(AtomicUsize::new(0));
    for connection in listener.incoming() {
        let mut stream = match connection {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept: {e}");
                continue;
            }
        };
        if active.fetch_add(1, Ordering::AcqRel) >= 16 {
            active.fetch_sub(1, Ordering::AcqRel);
            let _ = reply(
                &mut stream,
                "503 Service Unavailable",
                "application/json",
                br#"{"error":"connection limit"}"#,
            );
            continue;
        }
        let a = Arc::clone(&app);
        let count = Arc::clone(&active);
        thread::spawn(move || {
            struct Guard(Arc<AtomicUsize>);
            impl Drop for Guard {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::AcqRel);
                }
            }
            let _guard = Guard(count);
            if let Err(e) = handle(&mut stream, &a)
                && e.kind() != std::io::ErrorKind::BrokenPipe
            {
                eprintln!("request: {e}");
            }
        });
    }
    Ok(())
}
struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}
fn read_request(stream: &mut TcpStream, limit: usize) -> Result<Request, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    let end = loop {
        if let Some(i) = bytes.windows(4).position(|x| x == b"\r\n\r\n") {
            break i + 4;
        }
        if bytes.len() >= 8192 {
            return Err("request headers too large".into());
        }
        let mut chunk = [0u8; 2048];
        let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("incomplete request".into());
        }
        bytes.extend_from_slice(&chunk[..n]);
    };
    if end > 8192 {
        return Err("request headers too large".into());
    }
    let header = std::str::from_utf8(&bytes[..end]).map_err(|_| "invalid header")?;
    let mut lines = header.split("\r\n");
    let first: Vec<_> = lines.next().unwrap_or("").split_whitespace().collect();
    if first.len() != 3 || !matches!(first[2], "HTTP/1.0" | "HTTP/1.1") {
        return Err("invalid request line".into());
    }
    let method = first[0].to_string();
    let path = first[1].split('?').next().unwrap_or("/").to_string();
    let mut length = None;
    for line in lines.filter(|l| !l.is_empty()) {
        let (key, value) = line.split_once(':').ok_or("invalid header")?;
        if key.eq_ignore_ascii_case("transfer-encoding") {
            return Err("chunked requests are not supported; send Content-Length".into());
        }
        if key.eq_ignore_ascii_case("content-length") {
            if length.is_some() {
                return Err("duplicate Content-Length".into());
            }
            length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| "invalid Content-Length")?,
            );
        }
    }
    let n = length.unwrap_or(0);
    if n > limit {
        return Err("request body exceeds byte cap".into());
    }
    let mut body = bytes[end..].to_vec();
    if body.len() > n {
        return Err("unexpected trailing request bytes".into());
    }
    while body.len() < n {
        let mut chunk = [0u8; 8192];
        let wanted = (n - body.len()).min(chunk.len());
        let read = stream
            .read(&mut chunk[..wanted])
            .map_err(|e| e.to_string())?;
        if read == 0 {
            return Err("incomplete request body".into());
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Ok(Request { method, path, body })
}
fn reply(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}
fn json_reply(stream: &mut TcpStream, status: &str, value: &impl Serialize) -> std::io::Result<()> {
    let body = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    reply(stream, status, "application/json", &body)
}
fn handle(stream: &mut TcpStream, app: &App) -> std::io::Result<()> {
    let req = match read_request(stream, app.limit) {
        Ok(r) => r,
        Err(e) => return json_reply(stream, "400 Bad Request", &serde_json::json!({"error":e})),
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => reply(
            stream,
            "200 OK",
            "text/html; charset=utf-8",
            HTML.as_bytes(),
        ),
        ("GET", "/api/snapshot") => match app.snapshot() {
            Ok(s) => json_reply(stream, "200 OK", s.as_ref()),
            Err(e) => json_reply(
                stream,
                "503 Service Unavailable",
                &serde_json::json!({"error":e}),
            ),
        },
        ("GET", "/api/health") => json_reply(
            stream,
            "200 OK",
            &serde_json::json!({"generation":app.snapshot().ok().map(|s|s.generation),"synthetic_generator":app.demo.load(Ordering::Acquire)}),
        ),
        ("POST", "/api/quotes") => {
            let frame = serde_json::from_slice::<QuoteFrame>(&req.body);
            match frame {
                Err(e) => json_reply(
                    stream,
                    "400 Bad Request",
                    &serde_json::json!({"error":e.to_string()}),
                ),
                Ok(f) => match app.apply(f) {
                    Ok(s) => {
                        app.demo.store(false, Ordering::Release);
                        json_reply(stream, "200 OK", s.as_ref())
                    }
                    Err(e) => json_reply(
                        stream,
                        "422 Unprocessable Entity",
                        &serde_json::json!({"error":e}),
                    ),
                },
            }
        }
        ("GET", "/api/stream") => {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n"
            )?;
            stream.flush()?;
            let mut last = 0;
            loop {
                let s = app.snapshot().map_err(std::io::Error::other)?;
                if s.generation != last {
                    last = s.generation;
                    write!(
                        stream,
                        "id: {last}\nevent: snapshot\ndata: {}\n\n",
                        serde_json::to_string(s.as_ref()).map_err(std::io::Error::other)?
                    )?;
                    stream.flush()?;
                } else {
                    stream.write_all(b": keepalive\n\n")?;
                    stream.flush()?;
                }
                thread::sleep(Duration::from_millis(250));
            }
        }
        ("GET", "/favicon.ico") => reply(stream, "204 No Content", "image/x-icon", b""),
        _ => json_reply(
            stream,
            "404 Not Found",
            &serde_json::json!({"error":"route not found"}),
        ),
    }
}
