#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

// Radix-sqrt(N) Hadamard transform on the cube unit.
//
// H_N is separable: with N = R * C and the Sylvester (natural) ordering,
// H_N = H_R (x) H_C, so for a sample viewed as an R x C row-major matrix X,
//
//     Y = H_R * X * H_C
//
// and reading Y back row-major gives the length-N transform in natural order.
// R and C are the balanced split of log2(N), R = 2^floor(m/2) and
// C = 2^ceil(m/2), so both factors are about sqrt(N) on a side and the whole
// transform is two matrix-matrix products -- one applied from the right, one
// from the left -- instead of log2(N) butterfly stages on the vector unit.
// Accumulation is fp32 and the result is rounded twice rather than log2(N)
// times, so this is also the more accurate of the two kernels.
//
// A tile is S consecutive samples, which in GM is one contiguous (S*R x C)
// matrix. Both multiplies are then whole-tile: a right multiply does not mix
// rows, so X_stack * H_C transforms all S samples at once; a left multiply
// does not mix columns, so H_R * T_cat does the same with the samples laid
// side by side. Two TMATMULs per tile, whatever S is.
//
// The two paths differ only in how the per-sample blocks get from a vertical
// stack to a horizontal one between the multiplies:
//
//   C == 16  A 16-column tile is a single NZ block-column, so (S*R x 16) and
//            (16 x S*R) are the SAME BYTES -- fractal s is sample s either
//            way. The flip costs nothing, both GM transfers stay one
//            contiguous descriptor, and the tile is 7 descriptors for any S.
//
//   C > 16   Block-columns are still contiguous, so the flip is S TEXTRACTs
//            into one (R x S*C) L1 tile, and the result comes back out as S
//            per-sample Acc sub-tiles. The fixpipe L0C->GM path is 2D-to-2D
//            only (TStoreAccNz2nd asserts the three outer GlobalTensor dims
//            are 1), so those stores cannot be folded into one strided
//            descriptor -- a 5D scatter faults the aicore.
//
// Measured on 910B2 (msprof device time, bd=24, batch 8192), against the
// vector butterfly kernel in standard/: N=1024 1.64x, N=4096 2.09x. N=256 is
// 0.63x there and 1.51x at batch 65536; see the README for the block_dim
// sweep, which is worth up to 1.71x on the small-N shapes.
//
// (N, R, C, samples per tile)

// Samples per tile at N=256. S*N <= 16384 keeps both L0C accumulators inside
// 128 KB; the rest of the table is fixed by the same bound.
#ifndef PTO_HAD_S256
#define PTO_HAD_S256 32
#endif
// Load and store a 16-column tile through a Layout::NZ view, i.e. as one
// contiguous burst instead of the default ND2NZ gather, which reads only
// C * sizeof(half) = 32 bytes per descriptor row. Worth 4.5x at N=256 and is
// the single largest win in this kernel; only valid when C == 16.
#ifndef PTO_HAD_NZLOAD
#define PTO_HAD_NZLOAD 1
#endif
#ifndef PTO_HAD_NZSTORE
#define PTO_HAD_NZSTORE 1
#endif
// Split the per-core work by whole tiles rather than by samples, so at most
// one core walks the power-of-two fallback down to single-sample tiles.
#ifndef PTO_HAD_TILE_SPLIT
#define PTO_HAD_TILE_SPLIT 1
#endif
// Diagnostic only: drop both TMATMULs, keeping every transfer and handshake,
// to measure the data-movement floor of this tile geometry. Output is wrong.
// It is what showed the matmuls were nearly free and the N=256 cost was all
// in the load descriptor shape.
#ifndef PTO_HAD_NOMM
#define PTO_HAD_NOMM 0
#endif

#define FAST_HADAMARD_CUBE_CASES(X) \
  X(256, 16, 16, PTO_HAD_S256)      \
  X(512, 16, 32, 16)                \
  X(1024, 32, 32, 8)                \
  X(2048, 32, 64, 4)                \
  X(4096, 64, 64, 2)                \
  X(8192, 64, 128, 1)               \
  X(16384, 128, 128, 1)

#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

namespace {

constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L0AB_BYTES = 64 * 1024;
constexpr uint32_t L0C_BYTES = 128 * 1024;

// Distinct ids for the cross-tile edges; the intra-tile Sync pairs use id 0.
// Macros, not constexpr: bisheng rejects anything but the literal EVENT_ID*
// token as the third argument of set_flag / wait_flag.
#define EV_X0 EVENT_ID1   // L1 X slot 0
#define EV_X1 EVENT_ID2   // L1 X slot 1
#define EV_ACC EVENT_ID3  // output accumulator

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void Sync(uint32_t id = 0) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

// The only Mat layout TLOAD / TMOV / TEXTRACT accept on A2/A3.
template <uint32_t Rows, uint32_t Cols>
using MatTile = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor, Rows,
                     Cols, SLayout::RowMajor, 512>;

// NZ view of a 16-column tile: Shape<1, 1, Rows/16, 16, C0>, contiguous.
template <uint32_t Rows, uint32_t Cols>
using GmNz = GlobalTensor<half, Shape<1, 1, Rows / 16, 16, 16>,
                          Stride<Rows * Cols, Rows * Cols, 256, 16, 1>,
                          Layout::NZ>;

template <uint32_t Rows, uint32_t Cols>
using GmTile =
    GlobalTensor<half, TileShape2D<half, Rows, Cols, Layout::ND>,
                 BaseShape2D<half, Rows, Cols, Layout::ND>, Layout::ND>;

// Buffer map. Addresses that hold the two Hadamard factors must not depend on
// S: they are loaded once and reused by every tile size the driver picks.
template <uint32_t R, uint32_t C, uint32_t S>
struct Layout {
  static constexpr uint32_t kN = R * C;
  static constexpr uint32_t kStackRows = S * R;
  static constexpr uint32_t kTileElems = S * R * C;

  // L1
  static constexpr unsigned kL1Hl = 0;
  static constexpr unsigned kL1Hc = kL1Hl + R * R * sizeof(half);
  // Two X slots so the next tile's TLOAD can be issued before this tile's
  // compute: MTE2 then runs underneath MTE1/M/FIX instead of in front of them.
  static constexpr unsigned kL1X = kL1Hc + C * C * sizeof(half);
  static constexpr unsigned kL1XSlot = kTileElems * sizeof(half);
  static constexpr unsigned kL1T = kL1X + 2 * kL1XSlot;
  static_assert(kL1T + kTileElems * sizeof(half) <= L1_BYTES, "L1 overflow");

  static constexpr unsigned xSlot(bool pong) {
    return kL1X + (pong ? kL1XSlot : 0);
  }

  // L0A: H_R, then the stacked X tile
  static constexpr unsigned kL0aHl = 0;
  static constexpr unsigned kL0aX = kL0aHl + R * R * sizeof(half);
  static_assert(kL0aX + kTileElems * sizeof(half) <= L0AB_BYTES,
                "L0A overflow");

  // L0B: H_C, then the concatenated T tile
  static constexpr unsigned kL0bHc = 0;
  static constexpr unsigned kL0bT = kL0bHc + C * C * sizeof(half);
  static_assert(kL0bT + kTileElems * sizeof(half) <= L0AB_BYTES,
                "L0B overflow");

  // L0C: the stacked accumulator, then the concatenated one
  static constexpr unsigned kL0cStack = 0;
  static constexpr unsigned kL0cCat =
      kL0cStack + kTileElems * sizeof(float);
  static_assert(kL0cCat + kTileElems * sizeof(float) <= L0C_BYTES,
                "L0C overflow");
};

// GM -> L1 -> L0A/L0B for the two Hadamard factors. Done once per core; both
// stay resident for the whole kernel.
template <uint32_t R, uint32_t C>
AICORE void loadHadamardFactors(__gm__ half *h_left, __gm__ half *h_right) {
  using L = Layout<R, C, 1>;

  MatTile<R, R> hl_l1;
  MatTile<C, C> hc_l1;
  TASSIGN(hl_l1, L::kL1Hl);
  TASSIGN(hc_l1, L::kL1Hc);

  GmTile<R, R> hl_gm(h_left);
  GmTile<C, C> hc_gm(h_right);
  TASSIGN(hl_gm, h_left);
  TASSIGN(hc_gm, h_right);

  TLOAD(hl_l1, hl_gm);
  TLOAD(hc_l1, hc_gm);
  Sync<PIPE_MTE2, PIPE_MTE1>();

  TileLeft<half, R, R> hl_l0;
  TileRight<half, C, C> hc_l0;
  TASSIGN(hl_l0, L::kL0aHl);
  TASSIGN(hc_l0, L::kL0bHc);

  TMOV(hl_l0, hl_l1);
  TMOV(hc_l0, hc_l1);
  Sync<PIPE_MTE1, PIPE_M>();
}

// Issue the GM->L1 load for one tile into slot `pong` and signal MTE2->MTE1.
// The caller must already hold the slot (MTE1->MTE2 on the same event).
template <uint32_t R, uint32_t C, uint32_t S, bool Pong, event_t EvX>
AICORE void issueTileLoad(__gm__ half *x, uint32_t sample) {
  using L = Layout<R, C, S>;

  wait_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // slot free

  MatTile<L::kStackRows, C> x_l1;
  TASSIGN(x_l1, L::xSlot(Pong));
  __gm__ half *x_tile = x + sample * L::kN;

  if constexpr (C == 16 && PTO_HAD_NZLOAD) {
    // A 16-column ND tile is a single block-column, so its NZ image is
    // byte-identical to the GM bytes. Declaring the source Layout::NZ turns
    // the default ND2NZ gather -- which reads C*sizeof(half) = 32 BYTES per
    // descriptor row -- into one contiguous burst over the whole tile.
    GmNz<L::kStackRows, C> x_gm(x_tile);
    TASSIGN(x_gm, x_tile);
    TLOAD(x_l1, x_gm);
  } else {
    GmTile<L::kStackRows, C> x_gm(x_tile);
    TASSIGN(x_gm, x_tile);
    TLOAD(x_l1, x_gm);
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, EvX);
}

// Y[sample .. sample+S) = H_R * X * H_C, for a tile whose load has already
// been issued into slot Pong.
template <uint32_t R, uint32_t C, uint32_t S, bool Pong, event_t EvX>
AICORE void hadamardTile(__gm__ half *y, uint32_t sample) {
  using L = Layout<R, C, S>;
  constexpr uint32_t kStackRows = L::kStackRows;

  MatTile<kStackRows, C> x_l1;
  TASSIGN(x_l1, L::xSlot(Pong));
  wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);  // this tile's load has landed

  if constexpr (C == 16) {
    // With R == C == 16 a (S*R x 16) NZ tile and a (16 x S*R) NZ tile are the
    // SAME BYTES: 16 columns is one block-column, so the tile is just S
    // 16x16 fractals in sample order, and fractal s is sample s whichever way
    // the tile is declared. The vertical stack therefore reads back as the
    // horizontal concatenation [X_0 | X_1 | ... ] for free, and the
    // per-sample layout flip the general path pays for disappears.
    //
    // So apply H_R first, on the horizontal view (one left multiply covers
    // every sample), then H_C on the vertical view (one right multiply), and
    // both GM transfers stay a single contiguous descriptor. The whole tile
    // costs 7 descriptors regardless of S, which is what makes N=256 -- the
    // size where the general path is descriptor-bound -- run at memory speed.
    static_assert(R == 16, "the free reinterpret needs square 16x16 fractals");
    constexpr uint32_t kCat = S * R;

    MatTile<R, kCat> x_cat;
    TileRight<half, R, kCat> x_l0b;
    TASSIGN(x_cat, L::xSlot(Pong));  // same bytes as x_l1, read the other way
    TASSIGN(x_l0b, L::kL0bT);
    TMOV(x_l0b, x_cat);
    set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // slot free for the load after next
    Sync<PIPE_MTE1, PIPE_M>();

    TileLeft<half, R, R> hl_l0;
    TileAcc<float, R, kCat> acc_cat;
    TASSIGN(hl_l0, L::kL0aHl);
    TASSIGN(acc_cat, L::kL0cStack);
#if !PTO_HAD_NOMM
    TMATMUL(acc_cat, hl_l0, x_l0b);
#endif
    Sync<PIPE_M, PIPE_FIX>();

    // Same reinterpret on the accumulator: an Acc fractal is 16x16 fp32, so
    // (R x S*R) and (S*R x R) are again one block-column of S fractals.
    TileAcc<float, kStackRows, C> acc_vert;
    MatTile<kStackRows, C> t_l1;
    TASSIGN(acc_vert, L::kL0cStack);
    TASSIGN(t_l1, L::kL1T);
    TMOV(t_l1, acc_vert);
    Sync<PIPE_FIX, PIPE_MTE1>();

    TileLeft<half, kStackRows, C> t_l0a;
    TASSIGN(t_l0a, L::kL0aX);
    TMOV(t_l0a, t_l1);
    Sync<PIPE_MTE1, PIPE_M>();

    TileRight<half, C, C> hc_l0;
    TileAcc<float, kStackRows, C> acc_out;
    TASSIGN(hc_l0, L::kL0bHc);
    TASSIGN(acc_out, L::kL0cCat);
    wait_flag(PIPE_FIX, PIPE_M, EV_ACC);  // previous tile's TSTORE has drained
#if !PTO_HAD_NOMM
    TMATMUL(acc_out, t_l0a, hc_l0);
#endif
    Sync<PIPE_M, PIPE_FIX>();

    __gm__ half *y_tile = y + sample * L::kN;
    if constexpr (PTO_HAD_NZSTORE) {
      GmNz<kStackRows, C> y_gm(y_tile);
      TASSIGN(y_gm, y_tile);
      TSTORE(y_gm, acc_out);
    } else {
      GmTile<kStackRows, C> y_gm(y_tile);
      TASSIGN(y_gm, y_tile);
      TSTORE(y_gm, acc_out);
    }
    set_flag(PIPE_FIX, PIPE_M, EV_ACC);
  } else {
    // --- right multiply: X * H_C, S samples stacked vertically ---
    TileLeft<half, kStackRows, C> x_l0;
    TileRight<half, C, C> hc_l0;
    TileAcc<float, kStackRows, C> acc_stack;
    TASSIGN(x_l0, L::kL0aX);
    TASSIGN(hc_l0, L::kL0bHc);
    TASSIGN(acc_stack, L::kL0cStack);

    TMOV(x_l0, x_l1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // slot free for the load after next
    Sync<PIPE_MTE1, PIPE_M>();
#if !PTO_HAD_NOMM
    TMATMUL(acc_stack, x_l0, hc_l0);
#endif
    Sync<PIPE_M, PIPE_FIX>();

    // --- vertical stack -> horizontal concatenation, in L1 ---
    // Block-columns are contiguous in NZ, so sample s's (R x C) result lands
    // at s * R * C and the S writes read back as one (R x S*C) tile.
    for (uint32_t s = 0; s < S; ++s) {
      MatTile<R, C> t_block;
      TASSIGN(t_block, L::kL1T + s * R * C * sizeof(half));
      TEXTRACT(t_block, acc_stack, s * R, 0);
    }
    Sync<PIPE_FIX, PIPE_MTE1>();

    // --- left multiply: H_R * T, S samples side by side ---
    MatTile<R, S * C> t_l1;
    TileRight<half, R, S * C> t_l0;
    TASSIGN(t_l1, L::kL1T);
    TASSIGN(t_l0, L::kL0bT);

    TMOV(t_l0, t_l1);
    Sync<PIPE_MTE1, PIPE_M>();

    TileLeft<half, R, R> hl_l0;
    TileAcc<float, R, S * C> acc_cat;
    TASSIGN(hl_l0, L::kL0aHl);
    TASSIGN(acc_cat, L::kL0cCat);

    wait_flag(PIPE_FIX, PIPE_M, EV_ACC);  // previous tile's TSTORE has drained
#if !PTO_HAD_NOMM
    TMATMUL(acc_cat, hl_l0, t_l0);
#endif
    Sync<PIPE_M, PIPE_FIX>();

    // Each sample is a contiguous (R x C) block-column range of acc_cat.
    // The fixpipe L0C->GM path is 2D-to-2D only (TStoreAccNz2nd asserts the
    // three outer GlobalTensor dims are 1), so these cannot be folded into one
    // strided descriptor -- a 5D scatter faults the aicore.
    for (uint32_t s = 0; s < S; ++s) {
      TileAcc<float, R, C> acc_block;
      TASSIGN(acc_block, L::kL0cCat + s * R * C * sizeof(float));

      __gm__ half *y_row = y + (sample + s) * L::kN;
      GmTile<R, C> y_gm(y_row);
      TASSIGN(y_gm, y_row);
      TSTORE(y_gm, acc_block);
    }
    set_flag(PIPE_FIX, PIPE_M, EV_ACC);
  }
}

// Consume [sample, end) with the largest power-of-two tile that still fits,
// halving on the way down. A core with 43 samples left runs 32 + 8 + 2 + 1
// rather than 43 single-sample tiles.
//
// Each size runs its own two-slot pipeline: the next tile's TLOAD is issued
// before the current tile's compute, so MTE2 overlaps MTE1/M/FIX instead of
// sitting in front of them. The slot flags are primed on entry and drained on
// exit, so every tile size leaves the flag state balanced for the next one.
template <uint32_t R, uint32_t C, uint32_t S>
AICORE void runTiles(__gm__ half *x, __gm__ half *y, uint32_t &sample,
                     uint32_t end) {
  if (end - sample >= S) {
    set_flag(PIPE_MTE1, PIPE_MTE2, EV_X0);  // both slots start free
    set_flag(PIPE_MTE1, PIPE_MTE2, EV_X1);

    // bisheng needs the flag id to be a compile-time literal, so the slot
    // parity is a template argument and the loop dispatches to one of two
    // instantiations rather than carrying it in a variable.
    uint32_t cur = sample;
    bool pong = false;
    issueTileLoad<R, C, S, false, EV_X0>(x, cur);

    while (true) {
      const uint32_t next = cur + S;
      const bool has_next = (end - next >= S);
      if (pong) {
        if (has_next) {
          issueTileLoad<R, C, S, false, EV_X0>(x, next);
        }
        hadamardTile<R, C, S, true, EV_X1>(y, cur);
      } else {
        if (has_next) {
          issueTileLoad<R, C, S, true, EV_X1>(x, next);
        }
        hadamardTile<R, C, S, false, EV_X0>(y, cur);
      }
      if (!has_next) {
        break;
      }
      cur = next;
      pong = !pong;
    }
    sample = cur + S;

    wait_flag(PIPE_MTE1, PIPE_MTE2, EV_X0);  // drain the two primes
    wait_flag(PIPE_MTE1, PIPE_MTE2, EV_X1);
  }
  if constexpr (S > 1) {
    runTiles<R, C, S / 2>(x, y, sample, end);
  }
}

template <uint32_t N, uint32_t R, uint32_t C, uint32_t S>
AICORE void runCubeHadamard(__gm__ half *x, __gm__ half *y,
                            __gm__ half *h_left, __gm__ half *h_right,
                            uint32_t batch) {
  static_assert(R * C == N, "R * C must equal N");
  static_assert(R % 16 == 0 && C % 16 == 0, "cube tiles need 16-aligned dims");

  const uint32_t num_cores = block_num;
  const uint32_t core_id = get_block_idx();

  // Split by TILES, not by samples. An even split of samples leaves most cores
  // with a count that is not a multiple of S, and each one then walks the
  // power-of-two fallback down to single-sample tiles -- which cost the same
  // handshakes as a full tile for a fraction of the work. Rounding the split
  // up to whole tiles instead leaves at most one ragged core and is worth 2x
  // at shapes where batch/cores lands just above a tile boundary, even though
  // it can idle the last few cores.
#if PTO_HAD_TILE_SPLIT
  const uint32_t tiles_per_core = DIV_ROUNDUP(DIV_ROUNDUP(batch, S), num_cores);
  const uint32_t begin = tiles_per_core * S * core_id;
#else
  const uint32_t begin = DIV_ROUNDUP(batch, num_cores) * core_id;
#endif
  if (begin >= batch) {
    return;
  }
#if PTO_HAD_TILE_SPLIT
  uint32_t end = begin + tiles_per_core * S;
#else
  uint32_t end = begin + DIV_ROUNDUP(batch, num_cores);
#endif
  if (end > batch) {
    end = batch;
  }

  loadHadamardFactors<R, C>(h_left, h_right);

  set_flag(PIPE_FIX, PIPE_M, EV_ACC);  // output accumulator starts free

  uint32_t sample = begin;
  runTiles<R, C, S>(x, y, sample, end);

  wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
}

AICORE void dispatchCubeHadamard(__gm__ half *x, __gm__ half *y,
                                 __gm__ half *h_left, __gm__ half *h_right,
                                 uint32_t batch, uint32_t n) {
  switch (n) {
#define FAST_HADAMARD_CUBE_DISPATCH(N, R, C, S)                      \
  case N:                                                            \
    runCubeHadamard<N, R, C, S>(x, y, h_left, h_right, batch);       \
    return;
    FAST_HADAMARD_CUBE_CASES(FAST_HADAMARD_CUBE_DISPATCH)
#undef FAST_HADAMARD_CUBE_DISPATCH
    default:
      return;
  }
}

}  // namespace

#endif  // cube core

__global__ AICORE void fast_hadamard_cube_fp16(__gm__ void *x, __gm__ void *y,
                                               __gm__ void *h_left,
                                               __gm__ void *h_right,
                                               uint32_t batch, uint32_t n) {
#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  dispatchCubeHadamard((__gm__ half *)x, (__gm__ half *)y,
                       (__gm__ half *)h_left, (__gm__ half *)h_right, batch, n);
#endif
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint8_t *y, uint8_t *h_left, uint8_t *h_right,
                            uint32_t batch, uint32_t n) {
  fast_hadamard_cube_fp16<<<blockDim, nullptr, stream>>>(x, y, h_left, h_right,
                                                         batch, n);
}
