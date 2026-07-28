# reimpl/deterministic-stems Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Program spec: `docs/superpowers/specs/2026-07-28-reimpl-off-main-design.md`.
> Single-task feature; the task review doubles as the final review.

**Goal:** Feature 3: deterministic stem counts. `connect_stems` joins stems in
the set-iteration order of string-hashed `Part` objects; Python salts string
hashing per process, so the same orthomosaic yielded different stem counts on
every run (247/255/258 across three interpreters in the original testing).
Reference fix: old-stack commit `2ae179c` (PR #4) — re-exec `winmol_run.py`
once with `PYTHONHASHSEED=0` before anything hashes into a set. ~50 LOC target.

**Why re-exec (not sorted iteration):** the parity requirement — pinned-seed
runs reproduce the validated legacy stem counts; changing iteration semantics
would change counts. The plugin path already pins the seed via `child_env()`;
this guard covers direct CLI + `winmol_batch.py` + any caller.

## Global Constraints
- Branch `reimpl/deterministic-stems` off `reimpl/plugin-onnx` (b45e4bf). NO PR.
- Anti-slop ~50 LOC. Fast tests only (no model/fixture download).

## Task 1: guard + observable seed + test

**Files:** Modify `winmol_run.py` (top-of-file guard + one seed print at the
start of `__main__`); Create `tests/test_determinism.py` (~45 LOC).

- Guard (port from `git show 2ae179c -- winmol_run.py`, concise comment):
  ```python
  if os.environ.get("PYTHONHASHSEED") != "0":
      os.environ["PYTHONHASHSEED"] = "0"
      os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)
  ```
  placed BEFORE all project imports (only `os`/`sys` above it). `-u` re-added
  so the plugin's log stream stays unbuffered.
- Observable: first line of the `__main__` block prints
  `Determinism: PYTHONHASHSEED=<value>` (diagnostic contract the test pins).
- Test (fast, self-contained): (1) static — the guard block appears before any
  project import in the source; (2) behavioral — run
  `[sys.executable, "winmol_run.py", bogus args…]` with incoming
  PYTHONHASHSEED unset/"1"/"2"; each child re-execs; assert stdout contains
  `Determinism: PYTHONHASHSEED=0` for all three (exit code is nonzero later —
  bogus model — that's fine, capture output; assert the seed line preceded the
  failure).
- TDD; flake8; suite stays green (46 + new).
- Commit `fix(determinism): pin PYTHONHASHSEED=0 via re-exec guard in winmol_run`.
- Push branch; NO PR.

## Verification
Suite green; flake8; the three-seed behavioral test proves normalization.
Full-pipeline determinism (3× identical stems on a real ortho) is deferred to
the user's end-to-end testing round — the guard + inheritance (spawn workers
get the seed via os.environ) is the mechanism, pinned here.
