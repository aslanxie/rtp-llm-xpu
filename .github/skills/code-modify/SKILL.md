---
name: code-modify
description: 'Discipline and workflow for all code changes in rtp-llm: bug fix, refactor, rebase, upstream sync, or feature work. Use when: planning or reviewing any code change, writing a commit, deciding whether to touch shared vs device-specific code, or handling a deferred item.'
---

# Code Modification Discipline for rtp-llm

Applies to all change types: bug fix, refactor, rebase, upstream-sync, feature addition.

## Core Principles

| # | Principle | Short Rule |
|---|-----------|------------|
| 1 | Peer-first | Compare CUDA/ROCm implementations before touching anything XPU-specific |
| 2 | Scope discipline | Never modify shared code to solve a device-specific problem |
| 3 | Trace both directions | Read who calls this, and what this calls, before changing it |
| 4 | File-level context | Evaluate a change in its full file context, not just the diff hunk |
| 5 | One fix, one commit | Small focused commits; deferred items get a TODO, not silence |
| 6 | TODO + scope explanation, not a half-solution | When a full solution is premature, not possible now, or not worth the effort at this moment, write a precise TODO with scope explanation instead of shipping an incomplete or incorrect implementation |

---

## Procedure

### Step 1 — Orient with Peer Implementations

Before reading the XPU path in detail, pull up the CUDA and ROCm equivalents side by side.

**Why:** Three implementations make the intended pattern obvious. What looks like a bug in the XPU path is often just missing a line that both peers have, or the peers have the same "flaw" and it is intentional.

**How:**
1. Identify the file being changed (e.g. `xpu_impl/foo.py`, `XpuFoo.cc`).
2. Find the CUDA peer: same module name under `cuda_impl/` or `CudaFoo.cc`.
3. Find the ROCm peer: same module name under `rocm_impl/` or check `#elif USING_ROCM` blocks.
4. Read all three implementations before forming an opinion.
5. Only proceed if the XPU path diverges from both peers in a way that is clearly wrong or clearly intentional.

**Red flag:** You are about to "fix" something that CUDA and ROCm do identically. Stop and re-evaluate — the behaviour is almost certainly correct.

---

### Step 2 — Trace the Call Chain in Both Directions

Before touching any function, method, or class:

1. **Who calls this?** — Find all callers (use `grep_search` or `vscode_listCodeUsages`). Understand the contract each caller expects.
2. **What does this call?** — Identify every downstream dependency: other functions, registry lookups, kernel launches, IPC, config keys.
3. **What is the invariant?** — State in one sentence what this code is supposed to guarantee. If you can't state it, you don't understand it well enough to change it.
4. **What breaks if the invariant shifts?** — Identify exactly which callers or callees would need to change if the interface changes.

Do not begin writing code until you can answer all four points.

---

### Step 3 — Evaluate the Change at File Level

A change that looks clean in a 10-line diff hunk can be wrong or inconsistent when read in the context of the full file.

Before finalising:
1. Re-read the entire function (not just the changed lines).
2. Scan the surrounding functions in the same file for patterns the change should be consistent with (naming, error handling, logging, device guard placement).
3. Check that the change does not introduce an inconsistency that a reviewer would flag (e.g. adding a device check in one place but not a symmetric place 20 lines below).
4. If the file has a header comment or a class docstring describing its purpose, verify the change is consistent with that description.

---

### Step 4 — Apply Scope Discipline

**Rule:** Shared code is changed for shared reasons. Device-specific code is changed for device-specific reasons. Never cross the boundary.

#### Decision tree

```
Is the bug/feature only observable on one device?
├── Yes → modify device-specific code only (xpu_impl/, #if USING_XPU block, XPU factory branch)
│          Do NOT touch shared code, shared base class, or shared config
└── No  → the fix belongs in shared code; validate that it does not regress any device
```

#### Signals that you are about to cross the boundary wrongly

- You are adding an `if device == "xpu":` block inside a shared utility.
- You are adding a parameter to a shared function signature to handle an XPU edge case.
- You are changing a default value in shared config because XPU needs a different default.

**Correct alternative in each case:**
- Override the utility in the XPU-specific subclass or wrapper.
- Add the parameter to the XPU subclass constructor and pass it locally.
- Set the XPU override in the XPU-specific config layer, not the shared layer.

---

### Step 5 — Write Small, Focused Commits

**Rule: one logical fix = one commit.**

#### What counts as "one fix"

- A single root cause addressed (e.g. wrong tensor layout).
- A single missing dispatch entry added.
- A single type or naming correction.
- A single deferred item resolved.

A commit that fixes two independent bugs is two commits. A commit that fixes a bug and adds a test for it is one commit (the test is part of the fix).

#### Commit message format

```
[XPU] <area>: <what changed> (<why, if not obvious>)

Examples:
[XPU] attention: fix KV cache layout — BHSD not BSHD on decode path
[XPU] tp: add broadcast after sampling — peers (CUDA/ROCm) both do this
[XPU] build: exclude flash_attention from XPU target — no SYCL port yet
```

#### When to defer instead of bundle

If a fix requires touching N files but only M < N are clearly correct right now:
- Land the M-file fix as its own commit.
- Add a `TODO(xpu)` comment at each remaining site (see Step 6).
- Do **not** bundle a partial, untested change into the same commit.

---

### Step 6 — TODO + Scope Explanation, Not a Half-Solution

**Rule:** A known gap that is not fixed in this PR must be documented with a `TODO(xpu)` comment at the exact site. Silence is not acceptable.

This principle applies whenever:
- The current change **does not cover all cases** (edge conditions, tensor shapes, quantisation formats, etc.) and handling them fully requires disproportionate effort.
- The change is **not the best-performing solution** and a proper optimisation needs a dedicated kernel, a different data layout, or a new API that isn't available yet.
- The full solution would require **touching many files or subsystems** that are out of scope for this PR.
- It is **not the right moment** to address the gap (blocked on upstream, on a dependency, or on a decision that hasn't been made yet).

In all these cases: land the partial fix that *is* correct and safe, and write a precise TODO at every remaining site. A half-solution with a clear TODO is always preferable to a half-solution that looks complete.

#### TODO comment format

```python
# TODO(xpu): <what needs to happen> — <why deferred>
# Peers (CUDA/ROCm) do X here. XPU needs Y but Y is not yet available.
# Track: <issue URL or "no tracking issue yet">
```

```cpp
// TODO(xpu): <what needs to happen> — <why deferred>
// CUDA path calls foo(); XPU equivalent bar() is not yet implemented.
// Track: <issue URL or "no tracking issue yet">
```

#### What to include in the TODO

| Field | Purpose |
|-------|---------|
| What needs to happen | Concrete action (not vague "fix later") |
| Why deferred | Missing kernel / unsupported op / not the right moment / scope too large |
| Peer reference | Which device does it right and how |
| Tracking | Issue URL; if none exists, create one before merging |

**Never** leave a gap undocumented. A reviewer reading the code should always be able to tell: "this is knowingly deferred, here is why, here is what is needed."

---

## Anti-Patterns

| Anti-Pattern | Why Wrong | Correct Approach |
|---|---|---|
| Fixing something that CUDA and ROCm do identically | The behaviour is almost certainly intentional | Check peer impl first; if all three are wrong, note it but don't silently "fix" XPU only |
| Touching shared code for a device-specific problem | Risks regressing other devices; violates boundary | Solve in device-specific layer; override, don't pollute |
| Changing a function without tracing its callers | Silent contract breakage; hard to catch in review | Always grep callers before changing a signature or semantics |
| Evaluating a diff hunk without reading the surrounding file | Inconsistencies not caught; reviewers flag them | Re-read full function + nearby functions before finalising |
| Bundling multiple independent fixes in one commit | Hard to revert, cherry-pick, or bisect | One fix = one commit |
| Leaving a parity gap undocumented | Future maintainers don't know gap is known | `TODO(xpu)` + tracking issue at the exact site |
| Writing a half-solution without a TODO | Misleads reviewers; harder to land the real fix later | Either land the full solution or defer with a clear TODO + scope explanation |
| Using a generic default branch to cover XPU | Silent breakage on new upstream changes | Explicit `xpu` branch in every conditional |

---

## Quick Checklist (use before every commit)

- [ ] Did I read the CUDA and ROCm peers before making this change?
- [ ] Is my change limited to the appropriate scope (device-specific or shared, not both)?
- [ ] Can I trace all callers of anything I modified?
- [ ] Does the change look consistent when I re-read the full file?
- [ ] Is this commit one logical fix only?
- [ ] Have I added `TODO(xpu)` with a tracking reference for every deferred item?
- [ ] Does the commit message name the area, what changed, and why?

---

## Output

After applying this skill to a change or review, report:
- Peers consulted (CUDA / ROCm path compared)
- Scope boundary verdict (device-specific / shared / correctly crossing)
- Call-chain summary (callers found, downstream deps identified)
- File-level consistency issues found (if any)
- Commits proposed or written (one per logical fix)
- Deferred items and their TODO placements
