---
name: xpu-sync
description: 'Sync rtp-llm-xpu with upstream alibaba/rtp-llm:main. Use when: syncing upstream, syncing fork, pulling latest upstream changes. Handles merge conflicts automatically.'
---

# Merge Upstream into rtp-llm-xpu

## Inputs (from .env)

- **WORK_DIR** — absolute path to the local rtp-llm-xpu workspace
- **REPO_URL** — the rtp-llm-xpu fork Git URL
- **BRANCH** — the local branch to merge into (default: `main`)

## When to Use
- Syncing the fork with the latest alibaba/rtp-llm:main
- Pulling upstream changes into the local branch

## Procedure

Follow these steps EXACTLY in order. Do NOT run any commands outside this procedure.

### 1. Configure Remotes

```
cd $WORK_DIR
git remote set-url origin $REPO_URL 2>/dev/null || git remote add origin $REPO_URL
git remote set-url upstream https://github.com/alibaba/rtp-llm.git 2>/dev/null || git remote add upstream https://github.com/alibaba/rtp-llm.git
```

### 2. Switch to Branch and Pull Latest

```
cd $WORK_DIR
git checkout $BRANCH
git pull origin $BRANCH
```

### 3. Fetch Upstream and Merge

```
git fetch upstream
git merge upstream/main
```

### 4. Handle Merge Result

- **Clean merge**: Record the new merge commit hash. Done.
- **Conflicts**: Follow the conflict resolution procedure below.

## Conflict Resolution

If `git merge upstream/main` reports conflicts:

1. List conflicted files:
   ```
   git diff --name-only --diff-filter=U
   ```

2. For each conflicted file:
   - Read the file and examine the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
   - **XPU-specific files** (paths containing `xpu`): prefer the local (ours) version, incorporate any non-conflicting upstream changes
   - **Upstream-owned files** (BUILD, configs, deps, non-XPU code): prefer the upstream (theirs) version, preserve any local XPU-related additions (e.g., `xpu` config blocks, XPU device selections)
   - **Bazel files** (.bzl, WORKSPACE, BUILD): merge carefully — keep XPU config entries, accept upstream dependency updates
   - Edit the file to resolve all conflict markers

3. After resolving all files:
   ```
   git add -A
   git commit --no-edit
   ```

4. Verify no conflicts remain:
   ```
   git diff --name-only --diff-filter=U
   ```

## Output
- Report: merge status (clean/resolved), commit hash, number of conflicts resolved (if any)
- Do NOT push to origin

## See Also
- **code-modify** — apply this discipline when resolving merge conflicts: compare with CUDA/ROCm peers, keep changes device-scoped, one fix per commit, add TODO(xpu) for any gap left unresolved
