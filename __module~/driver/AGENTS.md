# AGENTS.md - Project Guide

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
- Only definitions re-exported with `pub use` may be imported directly; reference
  everything else through its preceding qualifier (module, enum, or error type).

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
- Definitions that are neither defined in nor re-exported from a crate root
  (`lib.rs`) or module root (`mod.rs`) are supporting data structures and must be
  referenced through their preceding module names.

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

- Commit messages always have the following format:

```
[{{LLM MODEL}}] {{Task Info}} {{Optional Subtask Info}}
```

- If a task contains multiple subtasks, each subtask should have its own commit
- If HEAD is DETACHED, create a temporary branch and commit into it

## Planning

- Any inconsistency or contradiction discovered during the planning stage must be immediately raised and highlighted in the plan
- no plan shall be executed until the inconsistency or contradiction is full addressed

## Documentation (including Markdown & Comments)

Before starting to work on code, actively enforce the following guardrails on every document you read; apply
corrections in one or more preceding git commits if necessary:

- Indentation is 4 spaces, continuation indentation is 6 spaces.
- Hard wrap is 120 characters. The only exceptions are table and markup sections
  which can be longer.
- Duplicated or contradicting information should be merged or deleted.
- Inconsistent or dangling references should be fixed.
- Spelling and syntax errors should be fixed.
- All references must point to existing code or artefacts; references to historical
  objects must be deleted.

### Acronyms

Every acronym used in the documentation (e.g. this guide, a `TODO.md`) must
appear in this list. Add a new acronym here in the same change that introduces it; otherwise spell the term out.

- **ABI:** Application Binary Interface.
- **AHRS:** Attitude and Heading Reference System.
- **API:** Application Programming Interface.
- **FFI:** Foreign Function Interface.
- **FRD:** Forward-Right-Down aerospace coordinate frame.
- **LLM:** Large Language Model.
- **RMS:** Root Mean Square.
- **RUB:** Right-Up-Back Android sensor coordinate frame.
- **SGD:** Stochastic Gradient Descent.
- **SPD:** Symmetric Positive-Definite.

### Formulas

Every math formula (e.g. equation, pseudo-algorithm) in the documentation (e.g. this guide, a `TODO.md`) should
be in a LaTeX math block (enclosed in a pair of `$` or `$$`).

### Symbols

Every symbol used in the documentation (e.g. this guide, a `TODO.md`) and every symbolic variable name in the code
must appear in the following list, with each entry containing the following information:

- the meaning of the symbol.
- (optional) the definitive equation that relates it to other symbols.
- (if it is a vector, matrix or tensor) its dimensions.

Add a new symbol here in the same change that introduces it; otherwise use the full name.

- **$A$:** Soft-iron correction matrix, $A = D^{-1} = M^{1/2} / r$; $3 \times 3$.
- **$A_w$:** Current working soft-iron correction used as the gravity preconditioner; the code field
  `gravity_frame` stores $A_w^{-1}$ with eigenvalues clamped to $[0.25, 4]$; $3 \times 3$.
- **$B$:** Online-optimizer minibatch; $|B|$ is its observation count.
- **$B_r$:** Replay minibatch size (`replay_minibatch_size`).
- **$b$:** Hard-iron offset vector, $b = \mu + r d$; $3 \times 1$.
- **$c$:** Shape-prior scale of the regularization target $c I$.
- **$D$:** Symmetric positive-definite soft-iron distortion matrix; $3 \times 3$.
- **$d$:** Normalized-offset candidate in cache-normalization units, $d = -\tfrac{1}{2} Q^{-1} q$; $3 \times 1$.
- **$e_{r,i}$:** Radial algebraic residual of observation $i$, $e_{r,i} = \phi_i^T \theta - 1$.
- **$e_{g,i}$:** Gravity-projection residual of observation $i$, $e_{g,i} = \psi_i^T \theta - \kappa$.
- **$G$:** Gravity-carrying subset of a minibatch; $|G|$ is its observation count.
- **$g_i$:** Normalized gravity direction of observation $i$; $3 \times 1$.
- **$H$:** Mean Gram matrix of the retained mean-centered unit directions,
  $H = \frac{1}{n} \sum_i \varphi(\hat{u}_i) \varphi(\hat{u}_i)^T$; $9 \times 9$.
- **$I$:** Identity matrix in the regularization target $c I$ and in $\|Q - c I\|_F^2$; $3 \times 3$.
- **$J_r$:** Radial online objective, $J_r = \frac{1}{2 n} \sum_i e_{r,i}^2 + \frac{\lambda}{2} \|Q - c I\|_F^2$.
- **$J_g$:** Gravity-surrogate objective, $J_g = \frac{w_g}{2 n_g} \sum_i e_{g,i}^2$.
- **$k$:** Diversity neighbor count (`num_neighbors`).
- **$M$:** Normalized shape matrix, $M = Q / \gamma$; $3 \times 3$.
- **$m_i$:** Ideal calibrated unit magnetic vector, $m_i = A\, (x_i - b)$ with $\|m_i\| = 1$; $3 \times 1$.
- **$N$:** `MagCalibrator` cache capacity in rows.
- **$n$:** Number of terms in an objective or Gram average.
- **$n_g$:** Gravity-carrying term count in $J_g$.
- **$n_i$:** Ellipsoid normal at $u_i$, $n_i = Q u_i + q / 2$; $3 \times 1$.
- **$p$:** Ramped count of cold-start replay updates per sample.
- **$Q$:** Symmetric ellipsoid shape matrix in $u_i^T Q\, u_i + q^T u_i = 1$; $3 \times 3$.
- **$q$:** Ellipsoid linear coefficient vector in $u_i^T Q\, u_i + q^T u_i = 1$; $3 \times 1$.
- **$R$:** Diagonal feature-space regularization weights $\operatorname{diag}(1, 1, 1, 2, 2, 2, 0, 0, 0)$; $9 \times 9$.
- **$r$:** RMS radius of the retained magnetometer samples (code field `sample_rms_radius` in
  `mag_model::MagModel`).
- **$s_i$:** Gravity normal projection, $s_i = \tilde{g}_i^T n_i$.
- **$s_{\theta,j}$, $s_\kappa$:** Diagonal feature-energy scales normalizing the optimizer descent step,
  $s_{\theta,j} = \frac{1}{|B|} \sum_{i \in B} \phi_{i,j}^2 + \frac{w_g}{|G|} \sum_{i \in G} \psi_{i,j}^2
  + \lambda R_{jj} + \epsilon$, and $s_\kappa = w_g + \epsilon$.
- **$u_i$:** Normalized magnetometer sample, $u_i = (x_i - \mu) / r$; $3 \times 1$.
- **$\hat{u}_i$:** Mean-centered unit direction of retained sample $i$; $3 \times 1$.
- **$w_g$:** Gravity term weight (`gravity_weight`).
- **$x_i$:** Raw retained magnetometer sample vector; $3 \times 1$.
- **$\gamma$:** Ellipsoid normalization scale, $\gamma = 1 + d^T Q d$.
- **$\epsilon$:** Numerical floor of the optimizer feature-energy scales.
- **$\theta$:** Packed online coefficients $[Q_{00}, Q_{11}, Q_{22}, Q_{01}, Q_{02}, Q_{12}, q_0, q_1, q_2]$ (code
  field `parameters`); $9 \times 1$.
- **$\theta_{\mathrm{prior}}$:** Prior coefficient vector $[c, c, c, 0, 0, 0, 0, 0, 0]$; $9 \times 1$.
- **$\tilde{g}_i$:** Preconditioned gravity direction of observation $i$,
  $\tilde{g}_i = A_w^{-1} g_i$; $3 \times 1$.
- **$\kappa$:** Learned gravity projection scalar.
- **$\lambda$:** Shape regularization weight.
- **$\mu$:** Sample mean of the retained magnetometer samples (code field `sample_mean` in
  `mag_model::MagModel`); $3 \times 1$.
- **$\nabla_\theta$, $\nabla_\kappa$:** Gradients of $J_r + J_g$ with respect to
  $\theta$ ($\nabla_\theta$; $9 \times 1$) and $\kappa$ ($\nabla_\kappa$; scalar).
- **$\phi(u)$:** Ellipsoid-fit feature vector with cross-term weight $2$; $9 \times 1$.
- **$\varphi(u)$:** Direction-feature vector with cross-term weight $\sqrt{2}$; $9 \times 1$.
- **$\psi(u, g)$:** Gravity-surrogate feature vector, $\psi(u, g)^T \theta = g^T (Q u + q / 2)$; $9 \times 1$.

You should avoid abusing one symbol to refer to different concepts. This includes symbols written in different
alphabets (e.g. `\mu` in LaTeX math and `mu` in code should always refer to the same concept).

### TODO.md Format

- Contains only a flat checklist of issues, grouped under severity headings (e.g. `## High severity`).

#### Issue Format

- Each issue is a `- [ ]` or `- [x]` checkbox followed by a short name and indented fields:
  - **Summary:** Short description.
  - **Position:** Path of the block comment in code that explains the issue, e.g. `src/path/to/file.rs (issue_summary)`.
    The block comment must be consistent with both code and documentation; every symbol should be annotated with a
    variable name in the code. The block comment should have the following sections:
    - always start with `TODO: issue_summary`.
    - detailed explanation.
    - recommended fix (if applicable).
  - **Unit test:** Path of the failing unit test(s) that reveals the issue, e.g. `src/path/to/file.rs (issue_summary)`.
    - issue should always come with one or more unit tests
- Keep items that are checked (`[x]`) only when the fix has already been merged; remove them on cleanup passes.
