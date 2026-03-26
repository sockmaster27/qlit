# Requirements

For local development, this project requires you to have the following tools installed on your machine:

- [Rust/Cargo toolchain](https://rust-lang.org/tools/install/)
- [uv Python project manager](https://docs.astral.sh/uv/getting-started/installation/)

# Testing

To run all tests:

```bash
uv run pytest
```

To run tests for specific packages only:

### Rust

```bash
cargo test
```

### Python

```bash
uv run pytest packages/python
```
