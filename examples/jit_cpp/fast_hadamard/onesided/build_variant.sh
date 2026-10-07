# Build one compile-time variant of the one-sided kernel into
# outputs/so_var/<name>.so, for A/B-ing the PTO_OS_* switches documented at the
# top of the kernel. Run from the fast_hadamard/ directory.
#
#   bash onesided/build_variant.sh base
#   bash onesided/build_variant.sh big -DPTO_OS_BIGTILE=1
#   python3 onesided/validate_variant.py onesided/outputs/so_var/base.so  # ALWAYS
#
# Validate before timing: a variant that computes the wrong answer is a broken
# switch, not a slow technique, and its timing means nothing.
set -e
mkdir -p onesided/outputs/so_var
bisheng -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 -Wno-ignored-attributes \
  --cce-aicore-arch=dav-c220-cube -isystem ${ASCEND_TOOLKIT_HOME}/include "${@:2}" \
  onesided/fast_hadamard_onesided.cpp -o onesided/outputs/so_var/$1.so
echo "built $1"
