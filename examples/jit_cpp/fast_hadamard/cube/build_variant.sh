# Build one compile-time variant of the cube kernel into outputs/so_var/<name>.so,
# for A/B-ing the PTO_HAD_* switches documented at the top of the kernel. Run
# from the fast_hadamard/ directory.
#
#   bash cube/build_variant.sh base -DPTO_HAD_DBUF=0
#   bash cube/build_variant.sh deep -DPTO_HAD_DBUF=1
#   python3 cube/validate_variant.py cube/outputs/so_var/base.so ...   # ALWAYS
#   python3 cube/devtime_bench.py <cfg.json>                           # then time
#
# Validate before timing: a variant that computes the wrong answer is a broken
# switch, not a slow technique, and its timing means nothing.
set -e
mkdir -p cube/outputs/so_var
bisheng -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 -Wno-ignored-attributes \
  --cce-aicore-arch=dav-c220-cube -isystem ${ASCEND_TOOLKIT_HOME}/include "${@:2}" \
  cube/fast_hadamard_cube.cpp -o cube/outputs/so_var/$1.so
echo "built $1"
