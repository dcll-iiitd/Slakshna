# Repository Guidelines

## Project Structure & Module Organization

SLAKSHNA combines a Rust peer-to-peer node with a Python federated-learning pipeline. Rust code lives in `src/`; `src/main.rs` drives epochs, `src/history.rs` defines wire records, and `src/network/mesh.rs` handles discovery and gossip. `ml_engine.py` connects the node to the vendored Bhaskera framework in `Bhaskera/`. Experiment drivers are under `scripts/`, plotting utilities under `plots_script/`, and documentation under `docs/`. Runtime output belongs in `data/`, `logs/`, `ml_models/`, `ml_states/`, or `results/` and is not source.

## Build, Test, and Development Commands

- `cd Bhaskera && bash setup.sh && source bhaskera-activate.sh && cd ..`: prepare the Python/Bhaskera environment.
- `bash setup.sh`: install Python dependencies, create runtime directories, and build the release binary.
- `cargo check`: quickly validate Rust changes.
- `cargo build --release`: build `target/release/iiitd`.
- `cargo test`: run the available Rust unit tests.
- `python ml_engine.py <my_id> <peer_id...>`: exercise one ML epoch without networking.
- `./target/release/iiitd --config node1.toml`: run a node with an explicit configuration.

Run experiment and plotting scripts from the repository root because they use relative paths.

## Coding Style & Naming Conventions

Format Rust with `cargo fmt` and follow standard Rust naming: `snake_case` functions/modules, `CamelCase` types, and `SCREAMING_SNAKE_CASE` constants. Use four-space indentation and `snake_case` in Python; keep scripts compatible with the existing environment. Send Python diagnostics to `stderr`: Rust parses the final `stdout` line from `ml_engine.py` as JSON. Avoid renaming `RecordKind` variants or fields without coordinating a federation-wide wire-format migration.

## Testing Guidelines

Add focused Rust unit tests near the module under test using `#[cfg(test)]`; use descriptive names such as `stable_identity_survives_restart`. Run `cargo fmt --check`, `cargo check`, and `cargo test` before submission. For ML changes, run the standalone engine and verify that its last output line matches the Rust–Python JSON contract. There is no enforced coverage threshold.

## Commit & Pull Request Guidelines

History generally uses concise Conventional Commit prefixes such as `feat:`, `docs:`, and `refactor:`. Keep each commit scoped and use an imperative summary. Pull requests should explain the behavior change, list verification commands, identify configuration or wire-format effects, and link relevant issues. Include plots or log excerpts when changing experiments or training behavior; never commit secrets, node key material, generated checkpoints, or large runtime logs.

## Configuration & Operational Notes

All nodes in a federation must share `[federation].id`. Preserve each node's `data_dir`: deleting it changes node identity. Review GPU selection, delta-size limits, clipping thresholds, and discovery settings before multi-node runs.
