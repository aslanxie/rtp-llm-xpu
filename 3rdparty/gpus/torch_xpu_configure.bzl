"""Repository rule for torch XPU autoconfiguration.

`torch_xpu_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: The python binary path. Used to detect site-packages.

This rule creates a repository pointing at the system-installed PyTorch XPU
site-packages directory, detected automatically from the Python interpreter
on PATH or via `PYTHON_BIN_PATH`.
"""

def _resolve_venv_python(repository_ctx, python_bin):
    """Resolve a possibly-symlinked python to the actual venv python.

    When PYTHON_BIN_PATH points to a symlink (e.g. /opt/conda310/bin/python3 ->
    /opt/venv/bin/python3), Python doesn't activate the venv because pyvenv.cfg
    isn't found relative to the invoked path. This walks the symlink chain to
    find the first python whose parent directory contains pyvenv.cfg.
    Falls back to the original path if no venv is found.
    """
    result = repository_ctx.execute([
        python_bin, "-c",
        "import os, sys\npath = sys.executable\nfor _ in range(10):\n    if os.path.islink(path):\n        t = os.readlink(path)\n        if not os.path.isabs(t): t = os.path.join(os.path.dirname(path), t)\n        path = os.path.normpath(t)\n        p = os.path.dirname(os.path.dirname(path))\n        if os.path.isfile(os.path.join(p, 'pyvenv.cfg')):\n            print(path); raise SystemExit\n    else: break\nprint(sys.executable)",
    ])
    if result.return_code == 0 and result.stdout.strip():
        return result.stdout.strip()
    return python_bin

def _torch_xpu_configure_impl(repository_ctx):
    # Check if XPU torch is actually available; if not, create a dummy repo
    # so CUDA/ROCm builds don't fail.
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "")
    if not python_bin:
        python_bin = repository_ctx.which("python3")
        if python_bin == None:
            # No python3 — create dummy and return
            repository_ctx.file("BUILD.bazel", "# dummy torch_xpu repo (no python3 found)\n")
            return
        python_bin = str(python_bin)

    # Resolve symlinked python to venv python so import torch works
    python_bin = _resolve_venv_python(repository_ctx, python_bin)

    # Check if torch.xpu is available in this Python
    check = repository_ctx.execute([
        python_bin, "-c", "import torch; assert hasattr(torch, 'xpu')",
    ])
    if check.return_code != 0:
        # torch.xpu not available — create dummy BUILD and return
        repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD.bazel")
        # Create minimal torch directory stubs so select() targets resolve
        repository_ctx.file("torch/lib/.empty", "")
        return

    # Auto-detect site-packages from the Python interpreter
    result = repository_ctx.execute([
        python_bin,
        "-c",
        "import site; print(site.getsitepackages()[0])",
    ])
    if result.return_code != 0:
        fail("Failed to detect site-packages: " + result.stderr)

    site_packages = result.stdout.strip()

    # List site-packages entries and symlink each one into the repo root,
    # reproducing the same layout as new_local_repository(path = site_packages).
    ls_result = repository_ctx.execute(["ls", "-1", site_packages])
    if ls_result.return_code != 0:
        fail("Failed to list site-packages: " + ls_result.stderr)
    for entry in ls_result.stdout.strip().split("\n"):
        if entry:
            repository_ctx.symlink(site_packages + "/" + entry, entry)

    # Generate BUILD file from the provided build_file (overwrites any
    # symlinked BUILD that may exist in site-packages).
    build_file = repository_ctx.attr.build_file
    repository_ctx.symlink(build_file, "BUILD.bazel")

torch_xpu_configure = repository_rule(
    implementation = _torch_xpu_configure_impl,
    attrs = {
        "build_file": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "BUILD file for the torch XPU repository.",
        ),
    },
    environ = [
        "PYTHON_BIN_PATH",
    ],
    doc = "Auto-detects PyTorch XPU site-packages and creates a repository.",
)
