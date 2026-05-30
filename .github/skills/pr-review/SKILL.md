---
name: pr-review
description: 'Review and address GitHub PR comments. Use when: user shares a PR link and asks to review comments, fix review feedback, or summarize review status. Fetches comments via GitHub API and guides the fix process.'
---

# PR Review Comment Workflow

## Inputs

- **PR URL** — e.g. `https://github.com/owner/repo/pull/123`
- **Reviewer name** (optional) — filter to a specific reviewer's comments

## When to Use
- User shares a PR link and asks to review/address comments
- User wants to fetch and triage review feedback
- User wants to fix issues raised in PR reviews

## Procedure

### 1. Fetch PR Review Comments via GitHub API

Two separate endpoints are needed — common mistake is only checking one:

- **Issue-level comments** (AI review summaries with P0/P1/P2/P3, general discussion):
  `GET /repos/{owner}/{repo}/issues/{pr}/comments?per_page=100`
- **Inline review comments** (file-level, line-specific):
  `GET /repos/{owner}/{repo}/pulls/{pr}/comments?per_page=100`

Key rules for the fetch script:
- Write as a standalone `.py` file, run via `python3 /tmp/script.py` — avoid inline python in bash (quoting issues with f-strings, brackets, backslashes)
- Use `urllib.request` (no external deps), set `timeout=15`
- Set header `Accept: application/vnd.github.v3+json`
- Filter by `c["user"]["login"]` to isolate specific reviewer's comments
- Print full `c["body"]` for issue comments (AI summaries can be long), truncate only if needed
- Sort inline comments by `c["created_at"]` to see latest round

### 2. Triage

Separate AI (Copilot) vs human reviewer comments; prioritize human P0/P1.

### 3. Compare with existing code

Before fixing, check how similar files (CUDA/ROCm equivalents) handle the same pattern — don't fix what isn't broken in peer implementations.

### 4. Don't blindly accept suggestions

Especially toolchain changes — always build-verify before committing (e.g. `enabled=True` on compile features can break unrelated targets).

### 5. Fix locally

Do NOT push without user's explicit instruction.

### 6. Reply with summary

Group fixes by reviewer's P-level tags, include what was fixed and what was intentionally not changed (with rationale).

## Output
- Report: list of comments fetched, triage result (P0/P1/P2/P3 counts), fixes applied, build verification status
