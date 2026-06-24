load("@rules_python//python:pip.bzl", "pip_parse")

PIP_EXTRA_ARGS = [
    "--cache-dir=~/.cache/pip",
    "--extra-index-url=https://mirrors.aliyun.com/pypi/simple/",
    "--verbose",
]

def pip_deps():
    pip_parse(
        name = "pip_cpu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_cpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_ppu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
    )

    pip_parse(
        name = "pip_gpu_cuda12_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_cuda12_9_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_torch_gpu_cuda12_9.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_cuda12_arm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_cuda12_arm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 3600,
        quiet = False,
    )

    pip_parse(
        name = "pip_gpu_rocm_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_rocm.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS,
        timeout = 12000,
    )

    # XPU lockfile was generated with Python 3.12 (PyTorch XPU requires ==3.12).
    # pip_parse only declares the hub repo and parses the hashed lockfile; it
    # does not download wheels, so declaring it in every container is cheap.
    # The actual whl_library fetches DO run the interpreter and would fail on a
    # Python 3.10 container (e.g. scikit-learn==1.8.0 is an XPU-only transitive
    # pin that Requires-Python>=3.11). Those fetches are gated by xpu_pip_gate
    # below on TF_NEED_XPU, so `bazel sync` / non-XPU builds never resolve the
    # XPU wheels.
    pip_parse(
        name = "pip_xpu_torch",
        requirements_lock = "@rtp_deps//:requirements_lock_xpu.txt",
        python_interpreter = "/opt/conda310/bin/python3",
        extra_pip_args = PIP_EXTRA_ARGS + ["--extra-index-url=https://download.pytorch.org/whl/xpu"],
        timeout = 3600,
    )

    # Gate XPU wheel installation on TF_NEED_XPU (set by --config=xpu, see
    # .bazelrc). Without this, `bazel sync` on a non-XPU Python 3.10 container
    # eagerly fetches every pip_xpu_torch_* whl_library and fails to resolve
    # XPU-only pins (e.g. scikit-learn==1.8.0, Requires-Python>=3.11).
    _xpu_pip_gate(name = "xpu_pip_gate")


def _xpu_pip_gate_impl(repository_ctx):
    # When building for XPU (--config=xpu sets TF_NEED_XPU=1) re-export the real
    # install_deps from @pip_xpu_torch so the XPU wheels are declared/fetched.
    # Otherwise expose a no-op so non-XPU builds and `bazel sync` never declare
    # the pip_xpu_torch_* whl_library repos (which would resolve XPU-only pins
    # under an incompatible interpreter, e.g. Python 3.10).
    enabled = repository_ctx.os.environ.get("TF_NEED_XPU", "0") == "1"
    repository_ctx.file("BUILD.bazel", "")
    if enabled:
        repository_ctx.file(
            "requirements.bzl",
            "load(\"@pip_xpu_torch//:requirements.bzl\", _install_deps = \"install_deps\")\n" +
            "def install_deps(**kwargs):\n" +
            "    _install_deps(**kwargs)\n",
        )
    else:
        repository_ctx.file(
            "requirements.bzl",
            "def install_deps(**kwargs):\n" +
            "    pass\n",
        )

_xpu_pip_gate = repository_rule(
    implementation = _xpu_pip_gate_impl,
    environ = ["TF_NEED_XPU"],
    configure = True,
)
