# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from glob import glob
from setuptools import setup, Extension


with open('README.md') as f:
    long_description = f.read()


def get_pkg_config(package, option):
    """Get pkg-config values for a package."""
    try:
        result = subprocess.run(
            ['pkg-config', option, package],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


# Base configuration - include bundled speex headers first
# Headers are at src/include/speex/*.h, so include src/include to allow #include "speex/..."
include_dirs = ['src', 'src/include']
library_dirs = []
libraries = ['speexdsp']
define_macros = []
extra_compile_args = ['-std=c++11']
extra_link_args = []

# Try to get paths from pkg-config
pkg_cflags = get_pkg_config('speexdsp', '--cflags')
pkg_libs = get_pkg_config('speexdsp', '--libs')

print(f"[setup.py] pkg-config --cflags speexdsp: {pkg_cflags}")
print(f"[setup.py] pkg-config --libs speexdsp: {pkg_libs}")

for flag in pkg_cflags:
    if flag.startswith('-I'):
        include_dirs.append(flag[2:])

for flag in pkg_libs:
    if flag.startswith('-L'):
        library_dirs.append(flag[2:])

# Add paths from environment variables
env_cppflags = os.environ.get('CPPFLAGS', '')
env_ldflags = os.environ.get('LDFLAGS', '')
env_c_include = os.environ.get('C_INCLUDE_PATH', '')
env_library_path = os.environ.get('LIBRARY_PATH', '')

print(f"[setup.py] CPPFLAGS: {env_cppflags}")
print(f"[setup.py] LDFLAGS: {env_ldflags}")
print(f"[setup.py] C_INCLUDE_PATH: {env_c_include}")
print(f"[setup.py] LIBRARY_PATH: {env_library_path}")

for flag in env_cppflags.split():
    if flag.startswith('-I'):
        path = flag[2:]
        if path not in include_dirs:
            include_dirs.append(path)

for flag in env_ldflags.split():
    if flag.startswith('-L'):
        path = flag[2:]
        if path not in library_dirs:
            library_dirs.append(path)

for path in env_c_include.split(':'):
    if path and path not in include_dirs:
        include_dirs.append(path)

for path in env_library_path.split(':'):
    if path and path not in library_dirs:
        library_dirs.append(path)

# Add common paths for fallback (check /usr first for manylinux)
common_include_paths = [
    '/usr/include',
    '/usr/include/speex',
    '/usr/local/include',
    '/usr/local/include/speex',
]
common_lib_paths = [
    '/usr/lib',
    '/usr/lib64',
    '/usr/local/lib',
    '/usr/local/lib64',
]

for path in common_include_paths:
    if os.path.exists(path) and path not in include_dirs:
        include_dirs.append(path)

for path in common_lib_paths:
    if os.path.exists(path) and path not in library_dirs:
        library_dirs.append(path)

print(f"[setup.py] Final include_dirs: {include_dirs}")
print(f"[setup.py] Final library_dirs: {library_dirs}")

# Check if speex header exists
for inc_dir in include_dirs:
    speex_header = os.path.join(inc_dir, 'speex', 'speex_preprocess.h')
    if os.path.exists(speex_header):
        print(f"[setup.py] Found speex header at: {speex_header}")
        break
else:
    print("[setup.py] WARNING: speex/speex_preprocess.h not found in any include directory!")

# Use pre-generated SWIG wrapper (no SWIG dependency at build time)
sources = [
    'src/noise_suppression.cpp',
    'src/speexdsp_ns_wrap.cpp',
]

setup(
    name='speexdsp-ns-vulcanlabs',
    version='0.1.3',
    description='Python bindings of speexdsp noise suppression library (Vulcanlabs fork with multi-platform wheels)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Lucky Wong, Vulcanlabs',
    author_email='dev@vulcanlabs.co',
    url='https://github.com/hiendang7613vulcan/speexdsp-ns-vulcanlabs',
    packages=['speexdsp_ns'],
    ext_modules=[
        Extension(
            name='speexdsp_ns._speexdsp_ns',
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++'
    ],
    license='BSD',
    keywords=['speexdsp_ns', 'noise suppression', 'audio processing', 'speech enhancement'],
    platforms=['Linux', 'MacOS'],
    package_dir={
        'speexdsp_ns': 'src'
    }
)
