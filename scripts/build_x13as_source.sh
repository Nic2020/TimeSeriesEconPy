#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Build X-13ARIMA-SEATS from the Census Bureau Fortran source pinned in
# src/tsecon/_mirror.py and install it at src/tsecon/x13/_binary/x13as.
#
# Used by the docs jobs (checks.yml, docs.yml) so the executed X-13 examples
# in docs/reference/x13.md run against the same pinned, SHA-256-verified
# source the wheels build from. Linux hosts only; the flags match the
# manylinux recipe in wheels.yml (CIBW_BEFORE_ALL_LINUX).
#
# Usage (from the repository root):
#     bash scripts/build_x13as_source.sh

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${repo_root}/src/tsecon/x13/_binary"

# Read the pins without importing the full package (see wheels.yml).
eval "$(python3 - "${repo_root}/src/tsecon/_mirror.py" <<'PY'
import importlib.util, shlex, sys
spec = importlib.util.spec_from_file_location("_mirror_meta", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for name in ("X13AS_VERSION", "X13AS_SOURCE_URL", "X13AS_SOURCE_SHA256"):
    print(f"{name}={shlex.quote(getattr(mod, name))}")
PY
)"

if ! command -v gfortran >/dev/null 2>&1; then
  sudo apt-get update -q
  sudo apt-get install -y -q gfortran
fi

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT
cd "${workdir}"
curl -sSfL -o x13as_src.tar.gz "${X13AS_SOURCE_URL}"
echo "${X13AS_SOURCE_SHA256}  x13as_src.tar.gz" | sha256sum -c -
tar xzf x13as_src.tar.gz
cd "x13as_asciisrc-${X13AS_VERSION}"
# Same flags as the wheels: -fallow-argument-mismatch for the legacy arity
# site in adpdrg.f, and static Fortran/GCC runtimes with a dynamic libc.
sed -i.bak 's/-static /-static-libgfortran -static-libgcc -static-libquadmath /' makefile.gf
make -f makefile.gf FFLAGS='-O2 -fallow-argument-mismatch' >build.log 2>&1 || {
  tail -50 build.log
  exit 1
}
mkdir -p "${dest}"
cp x13as_ascii "${dest}/x13as"
chmod +x "${dest}/x13as"
echo "Installed ${dest}/x13as (${X13AS_VERSION})"
