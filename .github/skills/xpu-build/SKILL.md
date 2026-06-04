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

```
cd $WORK_DIR
bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=xpu --test_output=errors --test_env="LOG_LEVEL=INFO" --jobs=32
```

### 3. Create protobuf symlinks

If build succeeds:
```
cd $WORK_DIR
ln -sf $(pwd)/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py $(pwd)/rtp_llm/cpp/model_rpc/proto/
ln -sf $(pwd)/bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py $(pwd)/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py
```

## Build Error Resolution

If the build fails, diagnose and fix:

1. Read the error output carefully. Common categories:
   - **Missing XPU config**: Check `arch_config/`, `.bazelrc` for missing `--config=xpu` definitions
   - **Missing dependency**: Check `BUILD` files, `deps/`, `WORKSPACE` for new upstream deps not wired for XPU
   - **C++ compile error**: Check if new upstream code uses CUDA-only APIs — add XPU guards or stubs
   - **Python import error**: Check if new upstream modules import unavailable packages — add conditional imports
   - **Proto/gRPC mismatch**: Regenerate protobuf symlinks

2. Apply the fix, then retry the build (skip any clean on retry):
   ```
   bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=xpu --test_output=errors --test_env="LOG_LEVEL=INFO" --jobs=32
   ```

3. Repeat until build succeeds or the error is beyond auto-fix (then STOP and report).

## Output
- Report: build status (pass/fail), clean strategy used (none/light/full), duration, number of build actions, any fixes applied
