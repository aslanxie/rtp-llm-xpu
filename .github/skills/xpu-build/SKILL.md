---
name: xpu-build
description: 'Build rtp-llm-xpu with Bazel for Intel XPU target. Use when: compiling after sync, rebuilding after code changes, fixing build errors. Automatically fixes common build failures.'
---

# Build rtp-llm-xpu (XPU)

## Inputs (from .env)

- **WORK_DIR** — absolute path to the local rtp-llm-xpu workspace

## When to Use
- After merging upstream changes
- After modifying any source code
- When a clean rebuild is needed

## Procedure

### 1. Auto-detect clean strategy

Check what changed in the last commit to decide the clean strategy:

```
cd $WORK_DIR
git diff --name-only HEAD~1 HEAD
```

Classify the changed files:

- If **any** file matches `WORKSPACE`, `deps/git.bzl`, `deps/http.bzl`, `deps/pip.bzl`: **full clean** (external deps changed)
  ```
  cd $WORK_DIR
  bazel clean --expunge
  ```

- Else if **any** file matches `.bazelrc`, `arch_config/*`, `bazel/*`: **light clean** (build config changed)
  ```
  cd $WORK_DIR
  bazel clean
  ```

- Else (only source code like `.py`, `.cc`, `.h`, `BUILD`): **skip clean** — Bazel rebuilds incrementally

When unsure, use `bazel clean` (without `--expunge`) to avoid re-downloading all dependencies.

### 2. Build the XPU target

Save all output to `./logs/build.log`. Do NOT read the raw log; use the targeted
checks in step 3 instead.

```
cd $WORK_DIR
mkdir -p ./logs
bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=xpu \
  --test_output=errors --test_env="LOG_LEVEL=INFO" --jobs=32 \
  2>&1 | tee ./logs/build.log
BUILD_EXIT=${PIPESTATUS[0]}
echo "Build exit code: $BUILD_EXIT"
```

### 3. Check result from log

```
# Final status lines (last 5 lines contain INFO/FAIL summary)
tail -5 ./logs/build.log
```

- If `Build completed successfully` appears → build passed. Proceed to step 4.
- If `FAILED` or non-zero exit code → build failed. Go to **Build Error Resolution**.

### 4. Create protobuf symlinks

If build succeeded:
```
cd $WORK_DIR
ln -sf $(pwd)/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py $(pwd)/rtp_llm/cpp/model_rpc/proto/
ln -sf $(pwd)/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py $(pwd)/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py
```

## Build Error Resolution

If the build fails, diagnose from the saved log — do NOT re-read the full log.

### 1. Identify the failing target and error category

```
# Failing targets
grep -E "^ERROR:|^Target.*FAILED" ./logs/build.log | tail -20

# Error lines (C++ / Python / config)
grep -E "error:|Error:" ./logs/build.log | grep -v "^INFO" | head -30
```

Common categories and what to grep for:

| Category | Grep pattern | Fix |
|---|---|---|
| Missing XPU config | `grep "no such configuration\|--config=xpu" ./logs/build.log` | Check `arch_config/`, `.bazelrc` |
| Missing dependency | `grep "no such target\|not declared" ./logs/build.log` | Check `BUILD`, `deps/`, `WORKSPACE` |
| C++ compile error | `grep "error:.*\.cc\|error:.*\.h" ./logs/build.log \| head -20` | Add XPU guards or stubs |
| Python import error | `grep "ImportError\|ModuleNotFoundError" ./logs/build.log` | Add conditional import |
| Proto/gRPC mismatch | `grep "protobuf\|grpc" ./logs/build.log \| head -10` | Regenerate protobuf symlinks |

### 2. Apply fix and retry

After applying the fix, retry (skip clean on retry):

```
bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=xpu \
  --test_output=errors --test_env="LOG_LEVEL=INFO" --jobs=32 \
  2>&1 | tee ./logs/build.log
BUILD_EXIT=${PIPESTATUS[0]}
tail -5 ./logs/build.log
```

Repeat until build succeeds or the error is beyond auto-fix (then STOP and report).

## Output
- Report: build status (pass/fail), clean strategy used (none/light/full), duration, number of build actions, any fixes applied
- Full log available at `./logs/build.log` for manual inspection if needed
