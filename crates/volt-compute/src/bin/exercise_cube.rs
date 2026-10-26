//! Persistent bounded JSONL bridge to native Hypercube exercise-risk publication.
use std::io::{self, BufRead, Read, Write};
use volt_compute::exercise_publication::{ExercisePublisher, PublicationInput};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut publisher = ExercisePublisher::new();
    let mut input = io::stdin().lock();
    let mut output = io::BufWriter::new(io::stdout().lock());
    loop {
        let mut bytes = Vec::new();
        // read_until alone could allocate an unbounded hostile line.
        let n = (&mut input)
            .take(8 * 1024 * 1024 + 1)
            .read_until(b'\n', &mut bytes)?;
        if n == 0 {
            break;
        }
        if bytes.len() > 8 * 1024 * 1024 {
            return Err("publication line exceeds 8MiB".into());
        }
        let result = serde_json::from_slice::<PublicationInput>(&bytes)
            .map_err(|e| e.to_string())
            .and_then(|v| publisher.publish(&v).map_err(|e| e.to_string()));
        let response = match result {
            Ok(v) => serde_json::json!({"ok":true,"publication":v}),
            Err(e) => serde_json::json!({"ok":false,"error":e}),
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
    }
    Ok(())
}
