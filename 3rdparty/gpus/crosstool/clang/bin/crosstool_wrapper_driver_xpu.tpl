#!/usr/bin/env python3
"""Crosstool wrapper for compiling with Intel oneAPI compilers (icx/icpx).

SYNOPSIS:
  crosstool_wrapper_driver_xpu [options passed in by cc_library()
                                or cc_binary() rule]

DESCRIPTION:
  This script is expected to be called by the cc_library() or cc_binary() bazel
  rules. It routes compilation to Intel icx (C) or icpx (C++) compilers,
  filtering out GCC-specific flags that are unsupported by the Intel toolchain.
"""

from __future__ import print_function

import os
import subprocess
import sys

# Template values set by xpu_configure.bzl
ICX_PATH = '%{icx_path}'
ICPX_PATH = '%{icpx_path}'
ONEAPI_INCLUDE = '%{oneapi_include_path}'

# GCC flags that icx/icpx do not support — silently dropped.
_UNSUPPORTED_FLAGS = frozenset([
    '-Wno-stringop-truncation',
    '-Wno-stringop-overflow',
    '-Wno-maybe-uninitialized',
    '-Wno-format-overflow',
    '-Wno-class-memaccess',
    '-pass-exit-codes',
])

# Prefixes of GCC flags to drop (matched via startswith).
_UNSUPPORTED_PREFIXES = (
    '-Wformat-truncation=',
    '-Wformat-overflow=',
    '-Wstringop-truncation',
    '-Wstringop-overflow=',
)


def _is_cpp(argv):
    """Heuristic: if we see -x c++ or a .cpp/.cc/.cxx source, use icpx."""
    for i, arg in enumerate(argv):
        if arg == '-x' and i + 1 < len(argv) and argv[i + 1] in ('c++', 'cu'):
            return True
        if arg.endswith(('.cpp', '.cc', '.cxx', '.C')):
            return True
    return False


def _is_assembler(argv):
    """Check if this is an assembler invocation."""
    for i, arg in enumerate(argv):
        if arg == '-x' and i + 1 < len(argv) and argv[i + 1] in ('assembler', 'assembler-with-cpp'):
            return True
        if arg.endswith('.S'):
            return True
    return False


def _filter_flags(argv):
    """Remove GCC-only flags that icx/icpx would reject."""
    filtered = []
    for arg in argv:
        if arg in _UNSUPPORTED_FLAGS:
            continue
        if any(arg.startswith(p) for p in _UNSUPPORTED_PREFIXES):
            continue
        if arg.startswith('-mcpu='):
            filtered.append('-march=native')
            continue
        filtered.append(arg)
    return filtered


def _expand_params_files(argv):
    """Expand @params file arguments inline."""
    expanded = []
    for arg in argv:
        if arg.startswith('@') and not arg.startswith('@rpath'):
            params_file = arg[1:]
            try:
                with open(params_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            expanded.append(line)
            except IOError:
                expanded.append(arg)
        else:
            expanded.append(arg)
    return expanded


def main():
    argv = sys.argv[1:]
    argv = _expand_params_files(argv)
    argv = _filter_flags(argv)

    is_asm = _is_assembler(argv)
    use_cxx = _is_cpp(argv)

    if use_cxx:
        compiler = ICPX_PATH
        extra = ['-isystem', ONEAPI_INCLUDE, '-include', 'cstdint']
    elif is_asm:
        compiler = ICX_PATH
        extra = []
    else:
        compiler = ICX_PATH
        extra = ['-isystem', ONEAPI_INCLUDE, '-D_GNU_SOURCE', '-include', 'stdint.h', '-include', 'unistd.h']

    cmd = [compiler] + extra + argv
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
