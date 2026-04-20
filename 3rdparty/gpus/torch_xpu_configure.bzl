"""Repository rule for torch XPU autoconfiguration.

`torch_xpu_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: The python binary path. Used to detect site-packages.

This rule creates a repository pointing at the system-installed PyTorch XPU
site-packages directory, detected automatically from the Python interpreter
on PATH or via `PYTHON_BIN_PATH`.
"""

def _torch_xpu_configure_impl(repository_ctx):
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH", "")
    if not python_bin:
        python_bin = repository_ctx.which("python3")
        if not python_bin:
            fail("Could not find python3 on PATH. Set PYTHON_BIN_PATH.")

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
