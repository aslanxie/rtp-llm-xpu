"""Repository rule for Intel XPU autoconfiguration.

`xpu_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: The python binary path. Used to detect site-packages
    for the torch_xpu repository.
  * `ONEAPI_ROOT`: Path to Intel oneAPI installation. Default: /opt/intel/oneapi
"""

_ONEAPI_ROOT = "ONEAPI_ROOT"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        Label("//3rdparty/gpus/%s.tpl" % tpl),
        substitutions,
    )

def to_list_of_strings(elements):
    result = ""
    for element in elements:
        result += ("\"" + element + "\",")
    return result

def verify_build_defines(params):
    """Verify all variables substituted into crosstool/BUILD are present."""
    missing = []
    pattern = [
        "%{cxx_builtin_include_directories}",
        "%{extra_no_canonical_prefixes_flags}",
        "%{host_compiler_path}",
        "%{host_compiler_prefix}",
        "%{host_compiler_warnings}",
        "%{unfiltered_compile_flags}",
        "%{linker_bin_path}",
        "%{compiler_deps}",
        "%{linker_files}",
        "%{win_linker_files}",
        "%{msvc_cl_path}",
        "%{msvc_env_include}",
        "%{msvc_env_lib}",
        "%{msvc_env_path}",
        "%{msvc_env_tmp}",
        "%{msvc_lib_path}",
        "%{msvc_link_path}",
        "%{msvc_ml_path}",
    ]
    for p in pattern:
        if p not in params:
            missing.append(p)
    if missing:
        auto_configure_fail(
            "crosstool/BUILD.tpl template is missing these variables: " +
            str(missing) +
            ". Are you using a modified BUILD.tpl? Please update it.")

def auto_configure_fail(msg):
    """Output failure message for auto configuration."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def _cxx_inc_convert(path):
    """Convert path returned by the compiler to its true path."""
    path = path.strip()
    if path.startswith("("):
        path = path.strip("()")
    return path

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    """Get built-in include directories from compiler."""
    lang = "c++" if lang_is_cpp else "c"
    result = repository_ctx.execute([cc, "-E", "-x" + lang, "-", "-v"])
    stderr = result.stderr
    index1 = stderr.find("#include <...>")
    if index1 == -1:
        return []
    index1 = stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = stderr.find("\n ", index1 + 1)
    if index2 == -1:
        return []
    index3 = stderr.find("End of search list", index2)
    if index3 == -1:
        return []
    inc_dirs = stderr[index2:index3]
    return [
        repository_ctx.path(_cxx_inc_convert(p))
        for p in inc_dirs.split("\n")
        if len(p.strip()) > 0
    ]

def get_cxx_inc_directories(repository_ctx, cc):
    """Compute the list of default C and C++ include directories."""
    includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
    includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)
    includes_cpp_set = {str(d): None for d in includes_cpp}
    return includes_cpp + [
        inc for inc in includes_c if str(inc) not in includes_cpp_set
    ]

def _oneapi_root(repository_ctx):
    """Return the oneAPI root path."""
    oneapi_root = repository_ctx.os.environ.get(_ONEAPI_ROOT, "")
    if not oneapi_root:
        # Try common install locations
        for path in ["/opt/intel/oneapi", "/opt/oneapi"]:
            if repository_ctx.path(path).exists:
                return path
        auto_configure_fail(
            "Cannot find Intel oneAPI. Set ONEAPI_ROOT environment variable " +
            "or install to /opt/intel/oneapi.")
    return oneapi_root

def _find_icx(repository_ctx, oneapi_root):
    """Find icx compiler path."""
    candidate = oneapi_root + "/compiler/latest/bin/icx"
    if repository_ctx.path(candidate).exists:
        return candidate
    # Try without 'latest' symlink
    result = repository_ctx.execute(["which", "icx"])
    if result.return_code == 0:
        return result.stdout.strip()
    auto_configure_fail("Cannot find icx compiler. Ensure Intel oneAPI is installed.")

def _find_icpx(repository_ctx, oneapi_root):
    """Find icpx compiler path."""
    candidate = oneapi_root + "/compiler/latest/bin/icpx"
    if repository_ctx.path(candidate).exists:
        return candidate
    result = repository_ctx.execute(["which", "icpx"])
    if result.return_code == 0:
        return result.stdout.strip()
    auto_configure_fail("Cannot find icpx compiler. Ensure Intel oneAPI is installed.")

def _xpu_configure_impl(repository_ctx):
    """Implementation of the xpu_configure repository rule."""
    oneapi_root = _oneapi_root(repository_ctx)
    icx_path = _find_icx(repository_ctx, oneapi_root)
    icpx_path = _find_icpx(repository_ctx, oneapi_root)

    # Resolve symlinks so paths match what the compiler reports to Bazel
    icx_path = str(repository_ctx.path(icx_path).realpath)
    icpx_path = str(repository_ctx.path(icpx_path).realpath)
    oneapi_compiler_dir = str(repository_ctx.path(oneapi_root + "/compiler/latest").realpath)
    oneapi_include = oneapi_compiler_dir + "/include"

    # Use icpx as the host compiler for getting include directories
    host_compiler_includes = get_cxx_inc_directories(repository_ctx, icpx_path)

    host_compiler_prefix = "/usr/bin"

    # --- Generate the crosstool wrapper script ---
    _tpl(
        repository_ctx,
        "crosstool:clang/bin/crosstool_wrapper_driver_xpu",
        {
            "%{icx_path}": icx_path,
            "%{icpx_path}": icpx_path,
            "%{oneapi_include_path}": oneapi_include,
        },
        out = "crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc",
    )

    # --- Generate crosstool BUILD and cc_toolchain_config ---
    xpu_defines = {}
    xpu_defines["%{host_compiler_path}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
    xpu_defines["%{host_compiler_prefix}"] = host_compiler_prefix
    xpu_defines["%{linker_bin_path}"] = "/usr/bin"
    xpu_defines["%{extra_no_canonical_prefixes_flags}"] = ""
    xpu_defines["%{unfiltered_compile_flags}"] = to_list_of_strings([
        "-DUSING_XPU=1",
    ])
    xpu_defines["%{host_compiler_warnings}"] = to_list_of_strings([
        "-Wno-error",
    ])
    xpu_defines["%{cxx_builtin_include_directories}"] = to_list_of_strings(
        [str(d) for d in host_compiler_includes] + [oneapi_include],
    )
    xpu_defines["%{compiler_deps}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
    xpu_defines["%{linker_files}"] = "clang/bin/crosstool_wrapper_driver_is_not_gcc"
    xpu_defines["%{win_linker_files}"] = ":empty"

    # Dummy Windows defines (required by verify_build_defines)
    xpu_defines["%{msvc_cl_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_include}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_lib}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_env_tmp}"] = "msvc_not_used"
    xpu_defines["%{msvc_lib_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_link_path}"] = "msvc_not_used"
    xpu_defines["%{msvc_ml_path}"] = "msvc_not_used"

    verify_build_defines(xpu_defines)

    _tpl(repository_ctx, "crosstool:BUILD", xpu_defines)
    _tpl(
        repository_ctx,
        "crosstool:xpu_cc_toolchain_config.bzl",
        out = "crosstool/cc_toolchain_config.bzl",
    )

    # --- Generate xpu/BUILD with SYCL runtime libraries ---
    sycl_lib = oneapi_compiler_dir + "/lib/libsycl.so"
    ze_loader_lib = ""
    for path in ["/usr/lib/x86_64-linux-gnu/libze_loader.so",
                 "/usr/lib64/libze_loader.so",
                 oneapi_compiler_dir + "/lib/libze_loader.so"]:
        if repository_ctx.path(path).exists:
            ze_loader_lib = path
            break

    xpu_build_substitutions = {}
    if repository_ctx.path(sycl_lib).exists:
        repository_ctx.symlink(sycl_lib, "xpu/lib/libsycl.so")
        xpu_build_substitutions["%{sycl_runtime_lib}"] = "xpu/lib/libsycl.so"
    else:
        xpu_build_substitutions["%{sycl_runtime_lib}"] = ""

    if ze_loader_lib:
        repository_ctx.symlink(ze_loader_lib, "xpu/lib/libze_loader.so")
        xpu_build_substitutions["%{ze_loader_lib}"] = "xpu/lib/libze_loader.so"
    else:
        xpu_build_substitutions["%{ze_loader_lib}"] = ""

    xpu_build_substitutions["%{copy_rules}"] = ""

    # Symlink SYCL headers
    sycl_include = oneapi_compiler_dir + "/include"
    if repository_ctx.path(sycl_include + "/sycl").exists:
        repository_ctx.symlink(sycl_include, "xpu/include")

    _tpl(repository_ctx, "xpu:BUILD", xpu_build_substitutions)

    # --- Generate py_runtime so .bazelrc can set --python_top=@local_config_xpu//:python_runtime ---
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH, "")
    if not python_bin:
        python_bin = str(repository_ctx.which("python3"))
    else:
        python_bin = str(python_bin)
    repository_ctx.file("BUILD.bazel", content = """
py_runtime(
    name = "python_runtime",
    interpreter_path = "%s",
    python_version = "PY3",
    stub_shebang = "#!%s",
    visibility = ["//visibility:public"],
)
""" % (python_bin, python_bin))

    # --- Torch XPU site-packages setup ---
    if python_bin:
        result = repository_ctx.execute([
            str(python_bin), "-c",
            "import site; print(site.getsitepackages()[0])",
        ])
        if result.return_code == 0:
            site_packages = result.stdout.strip()
            repository_ctx.file(
                "xpu/site_packages.bzl",
                'XPU_SITE_PACKAGES = "%s"\n' % site_packages,
            )

xpu_configure = repository_rule(
    implementation = _xpu_configure_impl,
    environ = [
        _ONEAPI_ROOT,
        _PYTHON_BIN_PATH,
    ],
    doc = """Configures the Intel XPU (icx/icpx) C/C++ toolchain.

Add the following to your WORKSPACE:

```python
xpu_configure(name = "local_config_xpu")
```
""",
)
