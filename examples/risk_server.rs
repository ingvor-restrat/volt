//! Local HTTP dashboard for the deterministic Volt risk simulation.

mod support;

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use support::risk_model::{RiskSimulation, RiskSnapshot};

const DASHBOARD: &str = include_str!("../demo/index.html");

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
struct Config {
    address: String,
    interval: Duration,
    seed: u64,
}

fn main() -> Result<()> {
    let config = parse_args()?;
    let mut simulation = RiskSimulation::new(config.seed)?;
    let initial = simulation.next_snapshot()?;
    let state = Arc::new(RwLock::new(initial));
    let generator_state = Arc::clone(&state);
    let interval = config.interval;

    thread::spawn(move || loop {
        thread::sleep(interval);
        match simulation.next_snapshot() {
            Ok(snapshot) => {
                if let Ok(mut current) = generator_state.write() {
                    *current = snapshot;
                }
            }
            Err(error) => eprintln!("risk simulation update failed: {error}"),
        }
    });

    let listener = TcpListener::bind(&config.address)?;
    println!("Volt risk dashboard: http://{}", config.address);
    println!(
        "Deterministic seed {} · refresh {} ms",
        config.seed,
        config.interval.as_millis()
    );
    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, state, interval) {
                        if error.kind() != std::io::ErrorKind::BrokenPipe {
                            eprintln!("request failed: {error}");
                        }
                    }
                });
            }
            Err(error) => eprintln!("accept failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    state: Arc<RwLock<RiskSnapshot>>,
    interval: Duration,
) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut request = [0_u8; 8_192];
    let read = stream.read(&mut request)?;
    let first_line = String::from_utf8_lossy(&request[..read])
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    let path = first_line.split_whitespace().nth(1).unwrap_or("/");
    let path = path.split('?').next().unwrap_or(path);

    match path {
        "/" | "/index.html" => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            DASHBOARD.as_bytes(),
        ),
        "/api/snapshot" => {
            let body = state
                .read()
                .ok()
                .and_then(|snapshot| serde_json::to_vec(&*snapshot).ok())
                .unwrap_or_else(|| br#"{"error":"snapshot unavailable"}"#.to_vec());
            respond(&mut stream, "200 OK", "application/json", &body)
        }
        "/api/stream" => stream_snapshots(&mut stream, state, interval),
        "/favicon.ico" => respond(&mut stream, "204 No Content", "image/x-icon", &[]),
        _ => respond(
            &mut stream,
            "404 Not Found",
            "application/json",
            br#"{"error":"not found"}"#,
        ),
    }
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn stream_snapshots(
    stream: &mut TcpStream,
    state: Arc<RwLock<RiskSnapshot>>,
    interval: Duration,
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n"
    )?;
    stream.flush()?;
    let mut last_step = 0;
    loop {
        let payload = state.read().ok().and_then(|snapshot| {
            if snapshot.step == last_step {
                None
            } else {
                last_step = snapshot.step;
                serde_json::to_string(&*snapshot).ok()
            }
        });
        if let Some(payload) = payload {
            write!(stream, "event: snapshot\ndata: {payload}\n\n")?;
            stream.flush()?;
        }
        thread::sleep(interval.min(Duration::from_millis(250)));
    }
}

fn parse_args() -> Result<Config> {
    let mut config = Config {
        address: "127.0.0.1:8081".to_owned(),
        interval: Duration::from_millis(180),
        seed: 0x7017,
    };
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--address" => config.address = value(&mut arguments, "--address")?,
            "--interval-ms" => {
                config.interval = Duration::from_millis(value(&mut arguments, "--interval-ms")?)
            }
            "--seed" => config.seed = value(&mut arguments, "--seed")?,
            "-h" | "--help" => {
                println!(
                    "risk_server — local Volt browser dashboard\n\n\
                     cargo run --example risk_server -- [options]\n\n\
                     --address HOST:PORT  bind address (default 127.0.0.1:8081)\n\
                     --interval-ms N     refresh delay (default 180)\n\
                     --seed N            deterministic random seed"
                );
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument {unknown}; use --help").into()),
        }
    }
    if config.interval < Duration::from_millis(25) {
        return Err("--interval-ms must be at least 25".into());
    }
    Ok(config)
}

fn value<T>(arguments: &mut impl Iterator<Item = String>, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    arguments
        .next()
        .ok_or_else(|| format!("{name} requires a value").into())
        .and_then(|raw| {
            raw.parse()
                .map_err(|error| format!("invalid value for {name}: {error}").into())
        })
}
