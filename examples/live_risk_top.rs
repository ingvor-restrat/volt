//! Compact terminal view of native `live_option_risk` snapshots. No analytics live here.
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use serde_json::Value;
use std::{
    error::Error,
    fs::File,
    io::{self, IsTerminal, Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    thread,
    time::Duration,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const CAP: usize = 64 * 1024 * 1024;
const RULE: &str =
    "────────────────────────────────────────────────────────────────────────────────────────────";
#[derive(Debug)]
struct Config {
    address: SocketAddr,
    input: Option<PathBuf>,
    ticks: Option<usize>,
    interval: Duration,
    record: bool,
    no_color: bool,
    rows: usize,
}
fn parse() -> Result<Config> {
    let mut c = Config {
        address: "127.0.0.1:8088".parse()?,
        input: None,
        ticks: None,
        interval: Duration::from_millis(500),
        record: false,
        no_color: false,
        rows: 8,
    };
    let mut a = std::env::args().skip(1);
    while let Some(k) = a.next() {
        match k.as_str() {
            "--address" => c.address = a.next().ok_or("missing address")?.parse()?,
            "--input" => c.input = Some(a.next().ok_or("missing input")?.into()),
            "--ticks" => c.ticks = Some(a.next().ok_or("missing ticks")?.parse()?),
            "--interval-ms" => {
                c.interval = Duration::from_millis(a.next().ok_or("missing interval")?.parse()?)
            }
            "--rows" => c.rows = a.next().ok_or("missing rows")?.parse()?,
            "--record" => c.record = true,
            "--no-color" => c.no_color = true,
            "--help" | "-h" => {
                println!("Native European Greeks monitor\n\ncargo run --example live_risk_top -- [options]\n\nStart native analytics first: cargo run -p volt-compute --features live-risk --bin live_option_risk\n\n--address IP:PORT  HTTP snapshot service (default 127.0.0.1:8088)\n--input FILE       Saved native snapshot or whitespace/JSONL snapshot sequence\n--ticks N          Bounded refreshes (default 1 when piped, 32 when recording)\n--interval-ms N    Refresh delay 0..60000 ms (default 500)\n--rows N           Visible chain rows 1..20 (default 8; held positions first)\n--record          ANSI frames separated by form feed; deterministic with --input\n--no-color        Plain text, no styling escapes\nq / Esc           Quit interactive alternate screen");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {k}").into()),
        }
    }
    if c.ticks == Some(0)
        || c.ticks.is_some_and(|n| n > 100_000)
        || c.interval.as_millis() > 60_000
        || !(1..=20).contains(&c.rows)
    {
        return Err("ticks must be 1..100000, interval 0..60000 ms, rows 1..20".into());
    }
    Ok(c)
}
fn bounded(mut r: impl Read) -> Result<Vec<u8>> {
    let mut b = Vec::new();
    r.by_ref().take(CAP as u64 + 1).read_to_end(&mut b)?;
    if b.len() > CAP {
        return Err("snapshot exceeds 64 MiB cap".into());
    }
    Ok(b)
}
fn validate(v: &Value) -> Result<()> {
    if v["schema"] != "volt.compute.live_risk.v1"
        || v["generation"].as_u64().is_none()
        || v["spot"].as_f64().is_none()
        || !v["portfolio"].is_object()
        || !v["rows"].is_array()
        || !v["scenarios"].is_array()
    {
        return Err("expected native volt.compute.live_risk.v1 snapshot".into());
    }
    Ok(())
}
fn replay(path: &PathBuf) -> Result<Vec<Value>> {
    let b = bounded(File::open(path)?)?;
    let mut out = Vec::new();
    for v in serde_json::Deserializer::from_slice(&b).into_iter::<Value>() {
        let v = v?;
        validate(&v)?;
        out.push(v);
        if out.len() > 1024 {
            return Err("replay exceeds 1024 snapshots".into());
        }
    }
    if out.is_empty() {
        return Err("empty snapshot replay".into());
    }
    Ok(out)
}
fn decode_http(b: &[u8]) -> Result<Value> {
    let end = b
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("incomplete HTTP headers")?;
    if end > 16384 {
        return Err("HTTP headers exceed 16 KiB".into());
    }
    let h = std::str::from_utf8(&b[..end])?;
    let mut lines = h.split("\r\n");
    if !matches!(lines.next(), Some("HTTP/1.1 200 OK" | "HTTP/1.0 200 OK")) {
        return Err("snapshot service returned non-200 response".into());
    }
    let mut length = None;
    for l in lines {
        let (k, v) = l.split_once(':').ok_or("malformed HTTP header")?;
        if k.eq_ignore_ascii_case("transfer-encoding") {
            return Err("chunked HTTP unsupported; native service uses Content-Length".into());
        }
        if k.eq_ignore_ascii_case("content-length") {
            if length.is_some() {
                return Err("duplicate Content-Length".into());
            }
            length = Some(v.trim().parse::<usize>()?);
        }
    }
    let body = &b[end + 4..];
    if length != Some(body.len()) {
        return Err("snapshot Content-Length mismatch".into());
    }
    let v = serde_json::from_slice(body)?;
    validate(&v)?;
    Ok(v)
}
fn fetch(address: SocketAddr) -> Result<Value> {
    let mut s = TcpStream::connect_timeout(&address, Duration::from_secs(3))?;
    s.set_read_timeout(Some(Duration::from_secs(3)))?;
    s.set_write_timeout(Some(Duration::from_secs(3)))?;
    write!(
        s,
        "GET /api/snapshot HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
    )?;
    decode_http(&bounded(s)?)
}
fn text(v: &Value, n: usize) -> String {
    v.as_str()
        .unwrap_or("--")
        .chars()
        .filter(|c| !c.is_control())
        .take(n)
        .collect()
}
fn number(v: &Value, width: usize, decimals: usize) -> String {
    match v.as_f64().filter(|v| v.is_finite()) {
        Some(x) => format!("{x:>width$.decimals$}"),
        None => format!("{:>width$}", "--"),
    }
}
fn style(s: &str, code: &str, color: bool) -> String {
    if color {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.into()
    }
}
fn render(v: &Value, rows: usize, color: bool) -> String {
    let p = &v["portfolio"];
    let synthetic = v["synthetic"].as_bool().unwrap_or(false);
    let mut o = style(
        " VOLT / RISK   EUROPEAN GREEKS MONITOR\n",
        "1;38;5;45",
        color,
    );
    o += &style(
        &format!(
            " gen {:>6}  spot {} {}  underlying {}  book revision {}\n",
            v["generation"],
            number(&v["spot"], 9, 3),
            text(&v["currency"], 6),
            text(&v["underlying_id"], 18),
            v["position_revision"]
        ),
        "38;5;244",
        color,
    );
    o += &style(
        if synthetic {
            " source SYNTHETIC QUOTES / native CPU IV, surface, Greeks + Hypercube\n"
        } else {
            " source SUPPLIED FRAME / native CPU IV, surface, Greeks + Hypercube\n"
        },
        "38;5;75",
        color,
    );
    let convention = if v["convention"].to_string().contains("sticky") {
        "sticky log-forward-moneyness / finite spot bumps"
    } else {
        "fixed volatility / analytical spot partials"
    };
    let eligible = v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["quote"]["eligible"] == true)
        .count();
    o += &format!(" Greeks {convention}\n eligible quotes {eligible}/{}  CCY={}  vol pt=1 percentage point  px=one underlying price unit\n", v["rows"].as_array().unwrap().len(), text(&v["currency"], 6));
    o += &style(
        &format!(
            "{RULE}\n {:<10} {:>14} {:>12} {:>15} {:>17}\n",
            "LIVE BOOK", "MARK/CCY", "DELTA/sh", "GAMMA/sh/px", "VEGA/CCY/vol-pt"
        ),
        "1;38;5;221",
        color,
    );
    o += &format!(
        " {} {:>14} {:>12} {:>15} {:>17}\n",
        if p["complete"] == true {
            "COMPLETE  "
        } else {
            "INCOMPLETE"
        },
        number(&p["value"], 14, 2),
        number(&p["delta"], 12, 3),
        number(&p["gamma"], 15, 4),
        number(&p["vega_per_vol_point"], 17, 2)
    );
    o += &format!(
        " theta {}/day   rho {}/bp   positions {}   missing {}\n",
        number(&p["theta_per_day"], 10, 2),
        number(&p["rho_per_bp"], 10, 3),
        p["position_count"],
        p["unpriced_contracts"].as_array().map_or(0, Vec::len)
    );
    o += &style(
        &format!(
            "{RULE}\n {:<20} {:>6} {:>6} {:<5} {:>7} {:>7} {:>6} {:>7} {:>9} {:>9}\n",
            "CHAIN (held first)",
            "K",
            "DTE",
            "TYPE",
            "BID",
            "ASK",
            "MID IV",
            "UNITS",
            "DELTA",
            "VEGA/PT"
        ),
        "1;38;5;221",
        color,
    );
    let mut chain: Vec<_> = v["rows"].as_array().unwrap().iter().collect();
    chain.sort_by_key(|r| r["position_units"].as_f64().unwrap_or(0.) == 0.);
    for r in chain.iter().take(rows) {
        let q = &r["quote"]["quote"];
        let i = &q["input"];
        let dte = i["maturity_years"]
            .as_f64()
            .map(|t| Value::from(t * 365.))
            .unwrap_or(Value::Null);
        let iv = r["quote"]["mid_iv"]["volatility"]
            .as_f64()
            .map(|x| Value::from(x * 100.))
            .unwrap_or(Value::Null);
        o += &format!(
            " {:<20} {} {} {:<5} {} {} {} {} {} {}{}\n",
            text(&q["contract_id"], 20),
            number(&i["strike"], 6, 1),
            number(&dte, 6, 1),
            text(&i["side"], 5),
            number(&q["bid"], 7, 2),
            number(&q["ask"], 7, 2),
            number(&iv, 6, 2),
            number(&r["position_units"], 7, 1),
            number(&r["greeks"]["delta"], 9, 4),
            number(&r["greeks"]["vega_per_vol_point"], 9, 3),
            if r["quote"]["eligible"] == true {
                ""
            } else {
                " !"
            }
        );
    }
    o += &format!(
        " showing {}/{} quotes; chain Greeks per unit; IV percent; ! ineligible\n",
        rows.min(chain.len()),
        chain.len()
    );
    o+=&style(&format!("{RULE}\n SCENARIO                    SPOT %    VOL PT      REPRICE/CCY       LOCAL/CCY\n"),"1;38;5;221",color);
    for s in v["scenarios"].as_array().unwrap().iter().take(6) {
        let spot = s["spot_change"]
            .as_f64()
            .map(|x| Value::from(x * 100.))
            .unwrap_or(Value::Null);
        let vol = s["volatility_change"]
            .as_f64()
            .map(|x| Value::from(x * 100.))
            .unwrap_or(Value::Null);
        o += &format!(
            " {:<26} {} {} {} {}\n",
            text(&s["name"], 26),
            number(&spot, 7, 2),
            number(&vol, 9, 2),
            number(&s["pnl"], 16, 2),
            number(&s["local_pnl"], 15, 2)
        );
    }
    o += &style(
        &format!(
            "{RULE}\n calc {} ms  surface knots {} / issues {}  grid {}  cube gen {}\n",
            number(&v["timings"]["total_ms"], 8, 3),
            v["surface"]["diagnostics"]["knots"],
            v["surface"]["diagnostics"]["issues"]
                .as_array()
                .map_or(0, Vec::len),
            v["grid"].as_array().map_or(0, Vec::len),
            v["cube"]["generation"]
        ),
        "38;5;244",
        color,
    );
    o += &style(
        " q / Esc quit   -- = unavailable   full-frame generations / native snapshots\n",
        "2;38;5;75",
        color,
    );
    o
}
// ANSI styling consumes no terminal columns; keep narrow terminals from wrapping rows.
fn clip_line(s: &str, width: usize) -> String {
    let mut out = String::new();
    let mut visible = 0;
    let mut escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            escape = true;
        }
        if escape {
            out.push(c);
            if c == 'm' {
                escape = false;
            }
        } else {
            if visible >= width {
                break;
            }
            out.push(c);
            visible += 1;
        }
    }
    if s.contains('\x1b') {
        out.push_str("\x1b[0m");
    }
    out
}
struct Terminal {
    active: bool,
}
impl Terminal {
    fn enter(active: bool) -> Result<Self> {
        let t = Self { active };
        if active {
            terminal::enable_raw_mode()?;
            execute!(io::stdout(), EnterAlternateScreen, Hide)?;
        }
        Ok(t)
    }
    fn draw(&self, s: &str) -> Result<()> {
        execute!(io::stdout(), MoveTo(0, 0), Clear(ClearType::All))?;
        let (width, height) = terminal::size()?;
        let mut out = io::stdout().lock();
        for l in s.lines().take(height.saturating_sub(1) as usize) {
            write!(
                out,
                "{}\r\n",
                clip_line(l, width.saturating_sub(1) as usize)
            )?;
        }
        out.flush()?;
        Ok(())
    }
}
impl Drop for Terminal {
    fn drop(&mut self) {
        if self.active {
            let _ = execute!(io::stdout(), Show, LeaveAlternateScreen);
            let _ = terminal::disable_raw_mode();
        }
    }
}
fn main() -> Result<()> {
    let c = parse()?;
    let saved = c.input.as_ref().map(replay).transpose()?;
    let interactive = io::stdout().is_terminal() && !c.record;
    let color = !c.no_color && (interactive || c.record);
    let limit = c
        .ticks
        .or(c.record.then_some(saved.as_ref().map_or(32, Vec::len)))
        .or((!interactive).then_some(1));
    let terminal = Terminal::enter(interactive)?;
    let mut tick = 0;
    loop {
        let v = if let Some(s) = &saved {
            s[tick % s.len()].clone()
        } else {
            fetch(c.address)?
        };
        let frame = render(&v, c.rows, color);
        tick += 1;
        if interactive {
            terminal.draw(&frame)?;
        } else {
            print!("{frame}");
            if c.record {
                print!("\n\x0c\n");
            }
            io::stdout().flush()?;
        }
        if limit.is_some_and(|n| tick >= n) {
            break;
        }
        if interactive {
            if event::poll(c.interval)?
                && matches!(event::read()?,Event::Key(k) if k.kind==KeyEventKind::Press && matches!(k.code,KeyCode::Char('q')|KeyCode::Esc))
            {
                break;
            }
        } else {
            thread::sleep(c.interval);
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn snapshot() -> Value {
        json!({"schema":"volt.compute.live_risk.v1","generation":1,"spot":100.,"portfolio":{"complete":false,"unpriced_contracts":["missing"]},"rows":[],"scenarios":[]})
    }
    #[test]
    fn narrow_terminal_preserves_style_and_row_width() {
        assert_eq!(clip_line("\x1b[31mabcdef\x1b[0m", 3), "\x1b[31mabc\x1b[0m");
    }
    #[test]
    fn nulls_and_incomplete_book_remain_visible() {
        let s = render(&snapshot(), 8, false);
        assert!(s.contains("INCOMPLETE"));
        assert!(s.contains("--"));
        assert!(!s.contains('\x1b'));
    }
    #[test]
    fn untrusted_labels_cannot_inject_terminal_controls() {
        assert_eq!(text(&json!("abc\u{1b}[2J\nxyz"), 6), "abc[2J");
    }
    #[test]
    fn http_checks_length_status_and_schema() {
        let b = serde_json::to_vec(&snapshot()).unwrap();
        let h = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", b.len());
        let mut raw = h.into_bytes();
        raw.extend(b);
        assert!(decode_http(&raw).is_ok());
        raw.pop();
        assert!(decode_http(&raw).is_err());
        assert!(decode_http(b"HTTP/1.1 500 ERROR\r\nContent-Length: 0\r\n\r\n").is_err());
        assert!(validate(&json!({})).is_err());
    }
}
