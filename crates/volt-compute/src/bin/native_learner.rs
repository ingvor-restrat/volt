//! Bounded JSON-lines worker; stdout contains protocol responses only.
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use volt_compute::native_learner::{Command, MAX_MESSAGE_BYTES, NativeLearner, SCHEMA};

#[derive(Deserialize)]
struct Request {
    id: String,
    #[serde(flatten)]
    command: Command,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut learner = None;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    loop {
        let mut line = Vec::new();
        loop {
            let available = input.fill_buf()?;
            if available.is_empty() {
                break;
            }
            let count = available
                .iter()
                .position(|x| *x == b'\n')
                .map(|n| n + 1)
                .unwrap_or(available.len());
            if line.len() + count > MAX_MESSAGE_BYTES {
                return Err("request exceeds native worker limit".into());
            }
            let finished = available[count - 1] == b'\n';
            line.extend_from_slice(&available[..count]);
            input.consume(count);
            if finished {
                break;
            }
        }
        let count = line.len();
        if count == 0 {
            break;
        }
        if count > MAX_MESSAGE_BYTES {
            return Err("request exceeds native worker limit".into());
        }
        let raw: Value = match serde_json::from_slice(&line) {
            Ok(x) => x,
            Err(e) => {
                writeln!(
                    output,
                    "{}",
                    json!({"schema":SCHEMA,"id":null,"ok":false,"error":e.to_string()})
                )?;
                output.flush()?;
                continue;
            }
        };
        let id = raw.get("id").cloned().unwrap_or(Value::Null);
        let request = serde_json::from_value::<Request>(raw);
        let mut shutdown = false;
        let result = request.map_err(|e| e.to_string()).and_then(|request| {
            if request.id.is_empty() || request.id.len() > 128 {
                return Err("invalid request id".into());
            }
            match request.command {
                Command::Shutdown => {
                    shutdown = true;
                    Ok(json!({"shutdown":true}))
                }
                Command::Configure { config } => {
                    if learner.is_some() {
                        return Err("worker is already configured".into());
                    }
                    learner = Some(NativeLearner::new(*config).map_err(|e| e.to_string())?);
                    learner
                        .as_mut()
                        .unwrap()
                        .handle(Command::Status)
                        .map_err(|e| e.to_string())
                }
                Command::Restore { checkpoint } => {
                    if learner.is_some() {
                        return Err("restore requires a fresh worker".into());
                    }
                    learner = Some(NativeLearner::restore(*checkpoint).map_err(|e| e.to_string())?);
                    learner
                        .as_mut()
                        .unwrap()
                        .handle(Command::Status)
                        .map_err(|e| e.to_string())
                }
                command => learner
                    .as_mut()
                    .ok_or("configure or restore first".to_string())?
                    .handle(command)
                    .map_err(|e| e.to_string()),
            }
        });
        let response = match result {
            Ok(result) => json!({"schema":SCHEMA,"id":id,"ok":true,"result":result}),
            Err(error) => json!({"schema":SCHEMA,"id":id,"ok":false,"error":error}),
        };
        let mut encoded = serde_json::to_vec(&response)?;
        if encoded.len() > MAX_MESSAGE_BYTES {
            encoded = serde_json::to_vec(
                &json!({"schema":SCHEMA,"id":response["id"],"ok":false,"error":"response exceeds native message limit; reduce checkpoint/model size"}),
            )?;
        }
        output.write_all(&encoded)?;
        output.write_all(b"\n")?;
        output.flush()?;
        if shutdown {
            break;
        }
    }
    Ok(())
}
