#!/usr/bin/env bash
#
# setup_swann_env.sh
#
# Sets up the environment for this repo (csu_swann_noaa_hot):
#   - conda env (python 3.12) from environment.yml (pip packages under `pip:`)
#   - Julia install + project instantiation
#   - lrose-core RPM unpack, merged into the conda env
#
# Assumes this script and environment.yml live at the top level of the
# already-cloned repo, and is run from there.
#
# All "already done?" checks live in the Preflight section below, so the
# Install Steps section reads as a clean, linear sequence you can copy/paste
# into a terminal by hand if you don't want to run the whole script.
#
# Usage:
#   ./setup_swann_env.sh              # run everything
#   ./setup_swann_env.sh --skip-julia
#   ./setup_swann_env.sh --lrose-version lrose-core-20260425 --lrose-os rockylinux_9
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via flags or env vars)
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="swann_py312"
ENV_YAML="${ENV_YAML:-${REPO_DIR}/swann_py312.yaml}"

LROSE_VERSION="lrose-core-20260425"
LROSE_OS="rockylinux_9"
LROSE_URL="https://github.com/NCAR/lrose-core/releases/download/${LROSE_VERSION}/${LROSE_VERSION}-${LROSE_OS}.x86_64.rpm"

# Set this to wherever your conda installs live; avoid hardcoding another
# user's home directory (the original history used /nhc_home/abrammer/...).
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV_PATH="${CONDA_BASE}/envs/${CONDA_ENV}"

SKIP_CONDA=false
SKIP_JULIA=false
SKIP_LROSE=false

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-conda) SKIP_CONDA=true; shift ;;
    --skip-julia) SKIP_JULIA=true; shift ;;
    --skip-lrose) SKIP_LROSE=true; shift ;;
    --lrose-version) LROSE_VERSION="$2"; shift 2 ;;
    --lrose-os) LROSE_OS="$2"; shift 2 ;;
    --conda-base) CONDA_BASE="$2"; CONDA_ENV_PATH="${CONDA_BASE}/envs/${CONDA_ENV}"; shift 2 ;;
    --env-yaml) ENV_YAML="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

# ===========================================================================
# Preflight — decide what already exists, flip SKIP_* accordingly.
# Nothing here mutates the system, it only inspects it.
# ===========================================================================
log "Preflight checks"

set +u
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set -u

if conda env list | grep -qE "^\s*${CONDA_ENV}\s"; then
  echo "  conda env '${CONDA_ENV}' already exists, will update (not recreate) from yaml"
fi
if [[ ! -f "$ENV_YAML" ]]; then
  echo "  ERROR: environment yaml not found: ${ENV_YAML}" >&2
  echo "  (pass --env-yaml /path/to/environment.yml)" >&2
  exit 1
fi

if command -v julia &>/dev/null; then
  echo "  julia already on PATH ($(command -v julia)), will skip installer"
  SKIP_JULIA=true
fi

if ! $SKIP_LROSE && [[ ! -d "$CONDA_ENV_PATH" ]] && $SKIP_CONDA; then
  echo "  ERROR: conda env path not found: ${CONDA_ENV_PATH}, and --skip-conda was given" >&2
  echo "  (create the env first, or drop --skip-conda)" >&2
  exit 1
fi

# ===========================================================================
# Install Steps — plain sequential commands, safe to copy/paste by hand.
# ===========================================================================

# --- 1. Conda env from environment.yml (python + pip packages) -------------
if ! $SKIP_CONDA; then
  log "Setting up conda env '${CONDA_ENV}' from ${ENV_YAML}"
  conda env create -f "$ENV_YAML" 2>/dev/null || conda env update -n "$CONDA_ENV" -f "$ENV_YAML" --prune
  conda activate "$CONDA_ENV"
fi

# --- 2. Julia ----------------------------------------------------------------
if ! $SKIP_JULIA; then
  log "Installing Julia"
  mkdir -p ./tmp && curl -fsSL https://install.julialang.org | TMPDIR=./tmp sh -s -- --yes
  rm -rf ./tmp

  # Make julia available on PATH whenever this conda env is activated
  CONDA_ENV_PREFIX="${CONDA_PREFIX:-$CONDA_ENV_PATH}"
  if [[ ! -d "$CONDA_ENV_PREFIX" ]]; then
    echo "  ERROR: conda env not found at ${CONDA_ENV_PREFIX}, can't register Julia PATH hook" >&2
    exit 1
  fi

  log "Registering Julia PATH with conda activate/deactivate hooks"
  mkdir -p "${CONDA_ENV_PREFIX}/etc/conda/activate.d" "${CONDA_ENV_PREFIX}/etc/conda/deactivate.d"
  cat > "${CONDA_ENV_PREFIX}/etc/conda/activate.d/julia_path.sh" << EOF
export PATH="${JULIA_BIN_DIR}:\${PATH}"
EOF

  cat > "${CONDA_ENV_PREFIX}/etc/conda/deactivate.d/julia_path.sh" << EOF
export PATH="\${PATH#${JULIA_BIN_DIR}:}"
EOF

  # Pick up the PATH change in this shell too, without a full activate/deactivate cycle.
  export PATH="${JULIA_BIN_DIR}:${PATH}"
fi

log "Instantiating Julia project"
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# --- 3. lrose-core -> merge into conda env -----------------------------------
if ! $SKIP_LROSE; then
  log "Installing lrose-core (${LROSE_VERSION}, ${LROSE_OS})"

  mkdir lrose
  cd lrose
  curl -OL "$LROSE_URL"
  rpm2cpio "${LROSE_VERSION}-${LROSE_OS}.x86_64.rpm" | cpio -idmv

  for sub in lib include bin share docs; do
    mkdir -p "${CONDA_ENV_PATH}/${sub}"
    mv "usr/local/lrose/${sub}"/* "${CONDA_ENV_PATH}/${sub}/"
  done

  cd ../
  rm -rf lrose

  echo "  lrose binaries merged into ${CONDA_ENV_PATH}"
  echo "  add this to your env activation (or a conda env script):"
  echo "    export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:\${LD_LIBRARY_PATH:-}"
fi

log "Done. Verify with: conda activate ${CONDA_ENV} && samurai -h"