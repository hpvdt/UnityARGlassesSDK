# AGENTS.md - ar-drivers-rs Project Guide

## Project Information

Refer to [README.md](README.md) for the project overview, supported devices, Linux dependencies and udev setup,
build/run quickstart, protocol blog posts, contribution information, licensing, and legal notes.

## Agent Role

You are an autonomous agent. Never stop until the task is completely finished. If you reach your output limit or finish
one logical step, immediately continue in the next response without saying "continued" or asking permission. Do not
output any TODO lists or "next steps" unless the user explicitly asks for a plan — just do the work.

## Architecture

The crate is a single library built as both `rlib` and `cdylib`. Every supported glasses model implements one common
device trait defined in the crate root. The root also owns the shared event, error, and display-mode types, runs device
discovery across all enabled drivers, and can fall back to a simulated SimMotion device when no hardware is found. A
singleton connection layer runs sensor fusion on a background thread, and a C ABI layer exposes the library to the
Unity integration.

Display configuration covers mirrored 1080p, full side-by-side stereo, half-resolution side-by-side upscaled by the
device, and high-refresh-rate (120 Hz) variants of both mirrored and side-by-side modes. Not every device supports
every mode.

### Feature Flags

The library uses Cargo feature flags for conditional compilation:

- `xreal`: Enables XREAL device support (requires: hidapi, tinyjson, bytemuck)
- `rokid`: Enables Rokid device support (requires: rusb)
- `grawoow`: Enables Grawoow device support (requires: rusb, tinyjson, bytemuck)
- `mad_gaze`: Enables Mad Gaze device support (requires: serialport)

All features are enabled by default.

## Rust Guardrails

### Formatting and Imports

- Let `rustfmt` define layout. Follow the repository's `rustfmt.toml` when one is
  present; do not align fields, arguments, or comments by hand.
- Use the line-ending style configured by the repository.
- Group imports consistently: standard library, external crates, then
  `crate`/`super`/`self`. Let rustfmt sort names within each group.
- Import the concrete types and traits used by the module. Avoid glob imports.
- Prefer one module-level import over repeated fully qualified paths when that
  makes the code easier to read.
- only exported definitions (by `pub use`) should be imported directly, everything
  else should be invoked under it's preceding qualifier, including mod, enum, and error

### Naming and API Shape

- Use standard Rust naming: `snake_case` for functions, methods, modules, fields,
  and locals; `UpperCamelCase` for types and traits; `SCREAMING_SNAKE_CASE` for
  constants.
- Preserve established public or ABI names when changing them would break a
  caller, but do not copy legacy naming inconsistencies into new APIs.
- Use `Self` in constructors and inherent implementations. Implement `Default`
  when there is one unsurprising baseline configuration, and make `new()`
  delegate to it where appropriate.
- Builder-style configuration methods take and return `self`; state-changing
  operations take `&mut self`; read-only operations take `&self`.
- Re-exporting a definition under a different name is strictly forbidden.
- Definitions that are not defined or exported in a crate root (`lib.rs`) or module
  root (`mod.rs`) are supporting data structure,
  they have to be referenced through their preceding module names.

### Documentation and Comments

- Give public items useful `///` doc comments. Use `//!` for module-level behavior
  or constraints.
- Document observable behavior: units, blocking behavior, feature or platform
  availability, error conditions, panics, and safety requirements when relevant.
- Explain non-obvious constraints, magic values, numerical thresholds, and
  invariants. Do not add comments that merely restate the code.
- Keep existing copyright, attribution, and source notes intact when editing a
  file.

### Errors and Control Flow

- Return `Result` from fallible APIs. Use a shared error type when callers need a
  stable error surface, add `From` conversions for reusable lower-level errors,
  and use `?` to preserve the original cause.
- Use specific typed error variants when callers need to distinguish recovery
  paths. Preserve error meaning and message wording during refactors when callers
  may depend on them.
- Do not use `unwrap()` or `expect()` for ordinary runtime failures in library
  code. They are acceptable in tests.
- Prefer early returns for invalid inputs and guard conditions. Use `match` when
  every enum case matters or when it expresses branching more clearly.
- Do not silently discard errors. If best-effort processing intentionally
  continues, keep enough context to diagnose the failure.

### Modules and Conditional Compilation

- Keep modules cohesive and give each one a clear responsibility. Split a module
  when its responsibilities or private implementation details stop being related.
- Keep helpers private by default. Expose `pub(crate)` for genuine cross-module
  internals and `pub` only for public API needed by downstream users.
- Keep `#[cfg(...)]` gates next to the module, import, implementation, or function
  they control. Optional functionality and its dependencies should be guarded by
  the same feature.
- Keep behavior consistent across platform-specific implementations when the
  public API is shared.
- Treat `Cargo.toml` and the crate root as the source of truth for supported
  devices, optional dependencies, feature gates, and device discovery. When
  adding device support, update those surfaces together with the cohesive driver
  module and its shared-trait implementation.

### Types and Data Handling

- Prefer domain types and newtypes when they prevent invalid combinations of
  primitive values. Use exact-width integers for binary formats and external
  interfaces whose widths are fixed.
- Give repeated constants and thresholds descriptive names. Include units in a
  name or doc comment when the type alone cannot express them.
- Make byte order explicit when reading or writing binary data. Validate lengths,
  tags, ranges, and conversions before indexing, slicing, or casting bytes.
- Preserve numerical precision deliberately. Reject non-finite or degenerate data
  before normalization, division, decomposition, or other sensitive operations.
- Prefer iterators when they make the transformation clearer; use loops when
  control flow, mutation, or early exit is easier to understand that way.

### Concurrency, FFI, and Unsafe Code

- Keep lock acquisition and thread lifecycle logic centralized. Propagate poison
  and join failures rather than introducing new panics.
- Use atomics with an explicitly chosen ordering; keep the ordering decision in
  one named constant when multiple operations share it.
- Avoid `unsafe` when a safe abstraction is practical. Keep unavoidable unsafe
  blocks and unsafe impls as small as possible, and add a `SAFETY:` comment that
  states the invariant being upheld.
- Treat exported symbol names, signatures, layouts, ownership, and lifetimes as
  ABI. Do not change them without coordinating and testing all callers.
- Never allow a panic to unwind across an `extern "C"` boundary. Validate raw
  pointers and lengths before dereferencing, and document caller obligations.
- Use an explicit representation such as `#[repr(C)]` when a type's layout is
  shared across an FFI or binary boundary.

## Testing

Hardware paths require physical devices; discovery reports a not-found error when no supported glasses are connected.
The deterministic SimMotion fixture in `src/sim/` is the fallback for development and testing without hardware, and most
integration tests run against it.

### Test Layout

- Unit test suites live in a sibling file next to the implementation, wired in behind `#[cfg(test)]`. Use the
  `_tests` filename suffix for new suites (some older files use `_test`).
- Test-only code belongs in the test suite file, never in the production source. The only `#[cfg(test)]` gate
  allowed in a production file is the `mod foo_test;` wiring; gating individual methods, helpers, constants, or
  imports with `#[cfg(test)]` is forbidden. Suites that need private state are wired with
  `#[cfg(test)] #[path = "foo_tests.rs"] mod foo_tests;` inside the implementation file (see `sim_motion.rs`), so
  the suite itself can hold the test-only `impl` block or free functions.
- Tests covering success, malformed input, boundary values, and error variants of the same behavior belong in the
  same suite.
- Use integration tests under `tests/` for behavior exercised through the public API.
- Prefer deterministic tests and local fixtures. Keep tests that require external resources, timing, or environment
  state clearly separate and document their prerequisites.
- Compare floating-point results with a tolerance derived from the algorithm; use exact equality only for values
  that are constructed exactly.

### Validation

For Rust changes, run the narrowest relevant checks first, then broaden them:

```bash
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

Adapt feature flags and targets when a project does not support building every
combination together. Run narrower package, module, or test checks first for fast
feedback, but complete the broad checks applicable to the repository before
submitting a change.

## Git (Version Control)

- Commit message always have the following format:

```
[{{LLM MODEL}}] {{Task Info}} {{Optional Subtask Info}}
```

- If a task contains multiple subtasks. Each subtask should have it's own commit
- If HEAD is DETACHED, create a temporary branch and commit into it

## Documentation/Markdown Files

- Indentation is 4 spaces, continuation indentation is 6 spaces
- Hard wrap is 120 characters. The only exceptions are Table and markup sections
  which can be longer.
- Duplicated or contradicting information should be actively deleted (including information in code comment)

### Acronyms

Every acronym used in the documentation (e.g. this guide, `TODO.md`) must
appear in this list. Add a new acronym here in the same change that introduces it; otherwise spell the term out.

- **AHRS:** Attitude and Heading Reference System.
- **FRD:** Forward-Right-Down aerospace coordinate frame.
- **RMS:** Root Mean Square.
- **RUB:** Right-Up-Back Android sensor coordinate frame.
- **SGD:** Stochastic Gradient Descent.
- **SPD:** Symmetric Positive-Definite.

### Formulas

Every math formula (e.g. equation, pseudo-algorithm) in the documentation (e.g. this guide, `TODO.md`) should be
in LaTeX math block (enclosed a pair of `$` or `$$`).

### Symbols

Every symbol used in the documentation (e.g. this guide, `TODO.md`) and symbolic variable name in the code must
appear in this list. Add a new symbol here in the same change that introduces it; otherwise use the full name reference.

You should avoid abusing 1 symbol to refer to different concepts. This include symbols written in different alphabets
(e.g. `\mu` in LaTex math and `mu` in code should always refer to the same concept)

### TODO.md Format

- Contain only a flat checklist of issues grouped by severity.
- All issues are grouped by severity heading (e.g. `## High severity`).

#### Issue Format

- Each issue is a `- [ ]` or `- [x]` checkbox followed by a short name & indented fields:
  - **Summary:** Short description.
  - **Position:** Path of the block code comment that explains the issue, e.g. `src/path/to/file.rs (issue_summary)`. The block comment should have the following sections:
    - always start with `TODO: issue_summary`.
    - detailed explanation.
    - recommended fix (if applicable).
  - **Unit test:** Path of the failing unit test(s) that reveals the issue, e.g. `src/path/to/file.rs (issue_summary)`.
    - issue should always come with one or more unit tests
- Keep items that are checked (`[x]`) only when the fix has already been merged; remove them on cleanup passes.
