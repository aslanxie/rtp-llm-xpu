---
name: code-review
description: 'Perform AI code review on local changes or a GitHub PR, duplicating the LLLLKKKK/rtp-llm review bot format (P0-P3 ratings, checklists, strengths). Use when: pre-reviewing before PR submission, reviewing a PR diff, or self-checking staged changes.'
---

# AI Code Review Bot

Perform structured code review that matches the rtp-llm upstream review system (LLLLKKKK's AI-assisted review bot).

## Inputs

- **Target**: one of:
  - `local` — review unstaged/staged git diff in the workspace (default)
  - `branch <name>` — review diff of branch vs main
  - `pr <URL>` — fetch and review a GitHub PR diff
  - `files <path1> <path2> ...` — review specific files
- **Scope** (optional): `full` (default) or `quick` (skip checklists, issues only)
- **Language** (optional): `zh` (Chinese, default — matches upstream) or `en` (English)

## When to Use

- Before submitting a PR to alibaba/rtp-llm — catch issues the review bot will flag
- After pushing fixes to check if review round is clean
- Self-review of any code change

## Procedure

### 1. Gather the Diff

Based on the target:

- **local**: Run `git diff HEAD` (or `git diff --cached` for staged only)
- **branch**: Run `git diff main...<branch> -- . ':!internal_source'`
- **pr**: Fetch diff via `curl -sH "Accept: application/vnd.github.v3.diff" https://api.github.com/repos/{owner}/{repo}/pulls/{number}`
- **files**: Read the specified files and compare against main

Filter out:
- Files under `internal_source/` (internal overlay, not reviewed)
- Binary files, generated files, lock files (unless lock file is the point of the PR)
- Files with only whitespace/comment changes (note them as P3 at most)

### 2. Analyze Each Changed File

For each file in the diff, analyze against the three checklists below. Track:
- **Issues**: problems that should be raised as P0/P1/P2/P3
- **Checklist violations**: checklist items that fail (linked to issues or checklist-only)
- **Strengths**: positive patterns worth calling out

### 3. Classify Issues by Priority

| Level | Name | Criteria | Effect |
|-------|------|----------|--------|
| **P0** | Critical | Security vulnerability, credential/URL leak, data loss, crash in production path | BLOCKING — must fix before merge |
| **P1** | Blocking | Correctness bug, breaking API change, missing referenced target, test infrastructure that silently skips, state/resource leak | BLOCKING — must fix before merge |
| **P2** | Non-blocking Suggestion | DRY violation, missing tests for new logic, hardcoded paths, loose test tolerance, type discipline issues, missing error handling | Non-blocking — should fix but won't block merge |
| **P3** | Minor Nit | Trailing whitespace, import order, naming conventions, docstring style, dead code comments | Non-blocking — nice to fix |

**Status decision:**
- If **any P0 or P1** exists → Status: **BLOCKING**
- Otherwise → Status: **LGTM** + include "lgtm ready to ci"

### 4. Output the Review

Use this exact format (matching upstream bot output):

```
## AI Code Review - PR #<number or "local">

Status: <LGTM | BLOCKING>

Summary: P0/<count> · P1/<count> · P2/<count> · P3/<count>

[if LGTM] lgtm ready to ci

### Blocking Issues

#### P0
• <issue title> @ `<file:line>`
  ◦ 建议：<specific actionable suggestion>

#### P1
• <issue title> @ `<file:line>`
  ◦ 建议：<specific actionable suggestion>

### Non-blocking Suggestions

#### P2
• <issue title> @ `<file:line>`
  ◦ 建议：<specific actionable suggestion>

#### P3
• <issue title> @ `<file:line>`
  ◦ 建议：<specific actionable suggestion>

### Checklist Violations (<fail count> fail / <total checked> total)

General Principles Checklist
• [6.1] <category> — <rule> → issue `<linked issue title>` <explanation>
  OR
• [6.1] <category> — <rule> → checklist-only <explanation why not promoted to issue>

RTP-LLM Checklist
• [<letter>] <category> — <rule> → issue `<linked issue title>` <explanation>

Python Static-First Checklist
• [P.<letter>] <category> — <rule> → issue `<linked issue title>` <explanation>

### Strengths
• <positive observation about the code>
• <another positive observation>
```

**Formatting rules:**
- If a P-level section has 0 items, omit that section entirely
- Each issue must reference a specific file:line
- Each suggestion (建议) must be concrete and actionable — no vague "consider improving"
- Checklist items link to issues with `→ issue '<title>'` or note `→ checklist-only` with rationale
- Strengths should acknowledge genuinely good patterns (not filler)
- Default language is Chinese for issue titles and suggestions (matching upstream)

---

## Checklists

### General Principles Checklist [6.1]

Run ALL items against every PR. Mark pass/fail for each. Only report failures in output.

#### Software Engineering
- [ ] **SRP**: Each class/function has a single responsibility
- [ ] **OCP**: Extension via new code, not modifying existing stable interfaces
- [ ] **LSP**: Subclass/override preserves base class contract (return types, exceptions, invariants)
- [ ] **ISP**: Interfaces are cohesive; callers don't depend on methods they don't use
- [ ] **DIP**: High-level policy doesn't depend on unnecessary concrete details
- [ ] **DRY**: Repeated non-trivial logic is extracted or explicitly reused (>5 lines duplicated = violation)
- [ ] **YAGNI**: No speculative features or premature abstractions added

#### Architecture
- [ ] **Error semantics**: fail-fast / retry / fallback / silent behavior is explicit, not accidental
- [ ] **Compatibility**: Public API / persisted data / config / env migration is safe
  - New enum values append-only (no renumber)
  - Default value changes evaluated for existing user impact
  - Return type changes have compatibility handling
  - New config fields propagated to all model variants
- [ ] **Observability**: Logs / metrics / timeouts are actionable, not noise
  - New enum/status values have corresponding string mappings
  - Error logs include enough context to diagnose
  - No swallowed exceptions in production paths
- [ ] **Rollback path**: Risky behavior has an operational rollback mechanism (feature flag, config switch)
- [ ] **Dependency direction**: No circular deps or cross-layer surprises
- [ ] **Concurrency**: Shared mutable state protected; async cleanup via try/finally
  - ContextVar.set() has matching .reset() in finally block
  - No race conditions in shared data structures
- [ ] **State invariants**: Create/update/fail/retry/rollback paths are valid

#### Tests
- [ ] **Unit tests**: New logic has focused unit tests + related integration/smoke tests
- [ ] **Boundary cases**: Empty, single element, max value, off-by-one covered
- [ ] **Cross-platform**: Distributed / cross-platform changes have corresponding coverage
- [ ] **Test infrastructure**: Tests actually run (not commented out in BUILD, not silently skipped)
  - unittest.main() at end of file, not mid-file
  - BUILD targets enabled (not commented out) — use tags=["manual"] if hardware-specific
  - Test tolerance meaningful (atol << output magnitude)
  - Self-checks that trivial inputs don't produce trivially-passing results

#### Quality
- [ ] **No unrelated formatting**: Logic changes don't include unrelated whitespace/import changes
- [ ] **PR description**: Motivation and design explained
- [ ] **Commit structure**: Logical commits that aid review and bisection
- [ ] **Mega-PR split**: If >500 lines changed, has the PR been split into independent changes?

---

### RTP-LLM Domain Checklist

Run for PRs touching rtp-llm-specific code. Skip for pure infra/docs changes.

#### [A] Compatibility & Configuration
- [ ] Default value changes evaluated for existing user impact
- [ ] New model config fields propagated to all model variants
- [ ] Bazel `environ` list includes all env vars read by repo rules
- [ ] requirements and lock files are in sync (version, hash, Python version match)
- [ ] No internal URLs (alibaba-inc.com, artlab, /mnt/nas) in open-source files

#### [B] Correctness & Logic
- [ ] No logic errors, off-by-one, null/zero check gaps
- [ ] No dead code / unreachable branches (commented-out code, impossible conditions)
- [ ] Shared object in-place modification uses shallow copy
- [ ] Interface return type changes have compatible handling for callers
- [ ] Boundary cases: empty input, single element, max value handled
- [ ] `x or default` vs `x if x is not None else default` — correct falsy semantics
- [ ] select() branches all resolve to existing targets

#### [C] API & Protocol
- [ ] OpenAI API response format matches spec (streaming chunks, finish_reason placement)
- [ ] SSE format: `data: {...}\n\n` with proper termination `data: [DONE]\n\n`
- [ ] Content-Type headers correct for response type

#### [D] Performance
- [ ] No per-forward memory allocation on hot path (can move to __init__)
- [ ] No unnecessary tensor copies (use view/reshape when possible)
- [ ] Lazy imports for heavy optional dependencies

#### [E] Device Integration (CUDA / ROCm / XPU)
- [ ] Device-specific code behind proper config guards (--config=xpu, using_xpu select)
- [ ] New device paths follow existing registration patterns (same as CUDA/ROCm siblings)
- [ ] Dummy/stub targets for anything not yet implemented
- [ ] Device capability checks at strategy/executor/router registration time
- [ ] No cross-device imports at module level (lazy import within guarded blocks)

#### [H] Testing & CI
- [ ] Test coverage sufficient: major refactors have equivalence coverage, new features have end-to-end
- [ ] CPU-runnable tests exist for platform-agnostic logic (config parsing, weight loading, factory selection)
- [ ] Hardware-specific tests use tags=["manual", "<device>"] not commented-out BUILD entries
- [ ] Test BUILD targets are registered and enabled (not commented out)

#### [I] Code Quality
- [ ] Same functionality uses unified helper functions (no copy-paste between sibling classes)
- [ ] Pre-commit checks pass (trailing whitespace, isort, black)
- [ ] Import order follows project convention

---

### Python Static-First Checklist

Run for PRs with Python changes. Skip for pure C++/Bazel/config changes.

#### [P.A] Static Structure & Type Discipline
- [ ] No getattr/setattr with literal string keys — declare explicit class attributes
- [ ] No hasattr for control flow branching (use isinstance or explicit attribute)
- [ ] Public APIs don't use **kwargs pass-through without validation
- [ ] String dispatch uses Enum or Literal, not raw string comparison (note: existing code may use strings — only flag if introducing NEW string dispatch)

#### [P.B] Function Design
- [ ] Functions have explicit return types (not implicit None)
- [ ] No mutable default arguments (list, dict, set)
- [ ] Generator/iterator functions are clearly marked

#### [P.F] Language Pitfalls
- [ ] No module-level import side effects (unless matching existing pattern in same file)
- [ ] No bare `except:` — always catch specific exceptions
- [ ] f-string expressions don't have side effects
- [ ] `isinstance()` preferred over `type() ==` for type checks

#### [P.G] Test Conventions
- [ ] Data-driven tests use pytest.mark.parametrize (or note if unittest style matches directory convention)
- [ ] Test method names describe the scenario, not just "test_1", "test_2"
- [ ] No `print()` debugging left in test code

#### [P.H] Type Annotations
- [ ] `Any` usage has a comment explaining why
- [ ] New public functions/methods have type annotations
- [ ] Complex return types are aliased for readability

---

## Special Rules

### Checklist-only vs Issue

When a checklist item fails but doesn't warrant its own P-level issue:
- Mark as `→ checklist-only` with brief rationale (e.g., "matches existing pattern in same file", "impact too small to justify separate fix")
- This happens when: the violation is inherited from existing code, the fix would be a separate refactor PR, or the risk is purely theoretical

### Comparing with Peer Implementations

Before flagging a pattern as a violation, check:
1. Do CUDA/ROCm/CPU implementations of the same component use the same pattern?
2. Is this an established convention in the rtp-llm codebase?
3. If yes → note as `checklist-only` ("沿用既有模式，不单独立 issue")

### Scaling Checklist Depth

- **Small PRs** (1-3 files, <100 lines): Run all checklists but with ~23-30 total items
- **Medium PRs** (4-10 files, 100-500 lines): Full checklists, ~56 total items
- **Large PRs** (10+ files, 500+ lines): Full checklists with sub-items expanded, ~93-104 total items
- Report total as `(<fail> fail / <total checked> total)`

### Internal URL Detection (always P0)

Always scan for these patterns — they are automatic P0 blockers:
```
alibaba-inc.com
artlab.alibaba
/mnt/nas1/
aliyuncs.com.*key
--access-key-id
--access-key-secret
```

### Version Number Checks

If PR updates dependency versions:
- Verify version is monotonically increasing (not regressing)
- Verify lock file hash matches the new wheel/package
- Verify Python version in lock file matches target runtime

## Output

The complete review in the format specified in Step 4 above. Must include:
1. Status line (LGTM or BLOCKING)
2. Summary counts
3. All P-level issue sections (only non-empty ones)
4. Checklist violations section
5. Strengths section (minimum 2 items)
