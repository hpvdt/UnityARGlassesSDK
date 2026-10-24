# AGENTS.md - AR Glasses SDK

This repository is a Unity package (`ca.hpvdt.arglassessdk`) holding two cooperating modules:

- **C# module:** the Unity-facing runtime in `Runtime/`, with its `Tests/`, `Scenes/`, and `Prefabs/`, plus the
  prebuilt native libraries in `Plugins/`.
- **Rust driver:** the device, protocol, and sensor-fusion library in `__module~/driver/`; its architecture, Rust
  guardrails, and Cargo checks are in the [driver guide](__module~/driver/AGENTS.md).

The conventions on this page are shared by both modules. Module guides only add what is specific to their language and
subsystem, and must not restate these rules.

## Documentation (including Markdown & Comments)

Update documentation relevant to the current change, using the following guardrails:

- Markdown format is defined by the [package configuration](.markdownlint.jsonc).
  [dev/make-all.sh](dev/make-all.sh) rewrites Markdown and Rust formatting, builds a release library,
  and copies it into the SDK's `Plugins/` directory; use the validation commands below for checks without rewriting files.
- Duplicated or contradicting information should be merged or deleted.
- Inconsistent or dangling references should be fixed.
- Spelling and syntax errors should be fixed.
- References to current behavior must point to existing code or artefacts. Retain research citations, attribution,
  and versioned evidence relevant to the current design.

### Validation

For documentation-only changes, run these checks from the SDK root:

```bash
git ls-files -z --cached --others --exclude-standard -- '*.md' | xargs -0 markdownlint-cli2
git diff --check
```

### Formulas

Every math formula (e.g. equation, pseudo-algorithm) in the documentation (e.g. this guide, a `TODO.md`) should
be in a LaTeX math block (enclosed in a pair of `$` or `$$`).

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
- Check (`[x]`) the item only when committing the fixed code; don't remove any of them.

## Git (Version Control)

- Commit messages always have the following format:

```
[{{LLM MODEL}}] {{Task Info}} {{Optional Subtask Info}}
```

- If a task contains multiple subtasks:
    - each subtask should have its own commit.
    - after each commit:
        - launch a subagent to review it for guardrail compliance
- If HEAD is DETACHED, create a temporary branch and commit into it
- If the commit can't compile cleanly, it should have [WIP] in its commit message

## Planning

- Any inconsistency or contradiction discovered during the planning stage must be immediately raised and
  highlighted in the plan
- no plan shall be executed until the inconsistency or contradiction is fully addressed
