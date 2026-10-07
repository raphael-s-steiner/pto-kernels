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
// Measured on 910B2 (msprof device time, batch 8192, with the wrapper's
// block_dim schedule), against the vector butterfly kernel in standard/:
// N=1024 1.64x, N=4096 2.09x, N=256 0.84x -- and N=256 is 1.26x at batch 64 and
// 1.95x at batch 65536. Geometric mean 1.42x over the benchmark grid.
//
// N=256 is the weak case because a tile is the same 16 KB at every size
// (S * N = 8192 throughout the table), so N=256 yields the fewest tiles for a
// given batch, and a core needs a RUN of tiles to fill its pipes: measured, a
// tile costs 340 ns when it is one of two on a core and 91 ns on a long run.
// That is what PTO_HAD_DBUF below and the host-side block_dim schedule in
// jit_util_hadamard_cube.py are for; the README derives both.
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
// Run the C == 16 tile as two interleaved phases instead of one straight-line
// body. The straight-line body visits MTE1 -> M -> FIX -> MTE1 -> M -> FIX, and
// because each pipe is in-order its SECOND MTE1 use cannot start until its own
// first matmul and L0C->L1 move have retired -- which also pins the next tile's
// first MTE1 behind it. So only the GM->L1 load ever overlapped, and a core
// moved 94 GB/s with a short tile run against 359 GB/s with a long one.
//
// Splitting the body at the L1 turnaround and issuing A(i+1) BEFORE B(i) makes
// the MTE1 order A0 A1 B0 A2 B1 ..., so MTE1 always has the next tile's work
// queued while the current tile is in M/FIX. That needs two of every
// intermediate (L0B X, L0C stack and cat, L1 T, L0A T) and a 4-deep X ring to
// keep the prefetch two tiles ahead; Layout::kDbuf gates it on all of that
// fitting. Set to 0 for the single-buffered control.
#ifndef PTO_HAD_DBUF
#define PTO_HAD_DBUF 1
#endif
// Diagnostic only: drop both TMATMULs, keeping every transfer and handshake,
// to measure the data-movement floor of this tile geometry. Output is wrong.
// It is what showed the matmuls were nearly free and the N=256 cost was all
// in the load descriptor shape.
#ifndef PTO_HAD_NOMM
#define PTO_HAD_NOMM 0
#endif
// Diagnostic only: truncate the kernel to measure what the fixed cost is made
// of. 1 = return at once (block dispatch floor), 2 = also load the Hadamard
// factors and return (adds the per-core prologue). Output is wrong.
#ifndef PTO_HAD_STOP
#define PTO_HAD_STOP 0
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
// token as the third argument of set_flag / wait_flag. They reach the flag
// instructions as template non-type arguments, which is the one other form it
// accepts.
//
// The two drivers never coexist in one instantiation, so they get separate
// names over a shared id pool. Keep the shallow ids exactly as they are: moving
// them renumbers flags in a path that did not otherwise change, and that alone
// measured a reproducible 0.92x at N=1024 batch 65536.
#define EV_X0 EVENT_ID1   // shallow: L1 X slot 0
#define EV_X1 EVENT_ID2   // shallow: L1 X slot 1
#define EV_ACC EVENT_ID3  // shallow: output accumulator

#define EV_D0 EVENT_ID0   // deep: L1 X ring slot 0
#define EV_D1 EVENT_ID1   // deep: L1 X ring slot 1
#define EV_D2 EVENT_ID2   // deep: L1 X ring slot 2
#define EV_D3 EVENT_ID3   // deep: L1 X ring slot 3
#define EV_PA0 EVENT_ID4  // deep: phase A, even tile
#define EV_PA1 EVENT_ID5  // deep: phase A, odd tile
#define EV_PB0 EVENT_ID6  // deep: phase B, even tile
#define EV_PB1 EVENT_ID7  // deep: phase B, odd tile

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
  static constexpr unsigned kTileBytes = kTileElems * sizeof(half);
  static constexpr unsigned kAccBytes = kTileElems * sizeof(float);

  // Two of every intermediate plus a 4-deep X ring. Every case in the table
  // holds S * N <= 16384, so all of them fit except the widest, whose two
  // accumulators already fill L0C; that one keeps the shallow path.
  static constexpr bool kDbuf =
      PTO_HAD_DBUF && C == 16 &&
      R * R * sizeof(half) + 2 * kTileBytes <= L0AB_BYTES &&
      C * C * sizeof(half) + 2 * kTileBytes <= L0AB_BYTES &&
      4 * kAccBytes <= L0C_BYTES &&
      R * R * sizeof(half) + C * C * sizeof(half) + 6 * kTileBytes <= L1_BYTES;

  static constexpr unsigned kXSlots = kDbuf ? 4 : 2;
  static constexpr unsigned kBuf = kDbuf ? 2 : 1;

  // L1: the two factors, the X ring, then the T turnaround slots.
  static constexpr unsigned kL1Hl = 0;
  static constexpr unsigned kL1Hc = kL1Hl + R * R * sizeof(half);
  static constexpr unsigned kL1X = kL1Hc + C * C * sizeof(half);
  static constexpr unsigned kL1XSlot = kTileBytes;
  static constexpr unsigned kL1T = kL1X + kXSlots * kL1XSlot;
  static_assert(kL1T + kBuf * kTileBytes <= L1_BYTES, "L1 overflow");

  static constexpr unsigned xSlot(uint32_t slot) {
    return kL1X + slot * kL1XSlot;
  }
  static constexpr unsigned l1T(bool par) {
    return kL1T + (par ? kTileBytes : 0);
  }

  // L0A: H_R, then the stacked X tile
  static constexpr unsigned kL0aHl = 0;
  static constexpr unsigned kL0aX = kL0aHl + R * R * sizeof(half);
  static_assert(kL0aX + kBuf * kTileBytes <= L0AB_BYTES, "L0A overflow");
  static constexpr unsigned l0aX(bool par) {
    return kL0aX + (par ? kTileBytes : 0);
  }

  // L0B: H_C, then the concatenated T tile
  static constexpr unsigned kL0bHc = 0;
  static constexpr unsigned kL0bT = kL0bHc + C * C * sizeof(half);
  static_assert(kL0bT + kBuf * kTileBytes <= L0AB_BYTES, "L0B overflow");
  static constexpr unsigned l0bT(bool par) {
    return kL0bT + (par ? kTileBytes : 0);
  }

  // L0C: the stacked accumulator, then the concatenated one -- as a pair, so
  // the deep path's two tiles in flight get [stack|cat][stack|cat].
  static constexpr unsigned kL0cStack = 0;
  static constexpr unsigned kL0cCat = kL0cStack + kAccBytes;
  static_assert(kBuf * 2 * kAccBytes <= L0C_BYTES, "L0C overflow");
  static constexpr unsigned l0cStack(bool par) {
    return (par ? 2 * kAccBytes : 0);
  }
  static constexpr unsigned l0cCat(bool par) {
    return (par ? 2 * kAccBytes : 0) + kAccBytes;
  }
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
template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, event_t EvX>
AICORE void issueTileLoad(__gm__ half *x, uint32_t sample) {
  using L = Layout<R, C, S>;

  wait_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // slot free

  MatTile<L::kStackRows, C> x_l1;
  TASSIGN(x_l1, L::xSlot(Slot));
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
  TASSIGN(x_l1, L::xSlot(Pong ? 1u : 0u));
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
    TASSIGN(x_cat, L::xSlot(Pong ? 1u : 0u));  // same bytes as x_l1, read the other way
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

// --- deep pipeline (C == 16 only) ----------------------------------------
//
// The straight-line tile above is split here at the L1 turnaround. Phase A
// takes the loaded X slot to L0B, left-multiplies by H_R and parks the result
// back in L1; phase B takes it to L0A, right-multiplies by H_C and stores. The
// driver issues A(i+1) before B(i), so each pipe always has the next tile's
// work queued behind the current tile's.
//
// Every WAR edge between tiles is covered transitively by the in-order pipes
// plus the RAW flags below, given two slots per intermediate: with issue order
// A0 A1 B0 A2 B1 A3 B2 ..., the reuse of a slot by A(i+2) sits behind B(i)'s
// MTE1, which waits on A(i)'s FIX. So no extra WAR flags are needed, and the
// shallow path's EV_ACC edge disappears entirely.

// X slot -> L0B, H_R * X -> L0C, L0C -> L1 T slot. Releases the X slot and
// leaves t_l1(Par) ready, signalled FIX->MTE1 on EvPA.
template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, bool Par,
          event_t EvX, event_t EvPA>
AICORE void tilePhaseA() {
  using L = Layout<R, C, S>;
  constexpr uint32_t kCat = S * R;

  MatTile<R, kCat> x_cat;
  TileRight<half, R, kCat> x_l0b;
  TASSIGN(x_cat, L::xSlot(Slot));  // same bytes as the loaded tile, read wide
  TASSIGN(x_l0b, L::l0bT(Par));
  wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);  // this tile's load has landed
  TMOV(x_l0b, x_cat);
  set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // X slot free for the load 4 tiles on
  set_flag(PIPE_MTE1, PIPE_M, EvPA);

  TileLeft<half, R, R> hl_l0;
  TileAcc<float, R, kCat> acc_cat;
  TASSIGN(hl_l0, L::kL0aHl);
  TASSIGN(acc_cat, L::l0cStack(Par));
  wait_flag(PIPE_MTE1, PIPE_M, EvPA);
#if !PTO_HAD_NOMM
  TMATMUL(acc_cat, hl_l0, x_l0b);
#endif
  set_flag(PIPE_M, PIPE_FIX, EvPA);

  // Same free reinterpret on the accumulator: an Acc fractal is 16x16 fp32, so
  // (R x S*R) and (S*R x R) are again one block-column of S fractals.
  TileAcc<float, L::kStackRows, C> acc_vert;
  MatTile<L::kStackRows, C> t_l1;
  TASSIGN(acc_vert, L::l0cStack(Par));
  TASSIGN(t_l1, L::l1T(Par));
  wait_flag(PIPE_M, PIPE_FIX, EvPA);
  TMOV(t_l1, acc_vert);
  set_flag(PIPE_FIX, PIPE_MTE1, EvPA);
}

// L1 T slot -> L0A, T * H_C -> L0C, L0C -> GM.
template <uint32_t R, uint32_t C, uint32_t S, bool Par, event_t EvPA,
          event_t EvPB>
AICORE void tilePhaseB(__gm__ half *y, uint32_t sample) {
  using L = Layout<R, C, S>;
  constexpr uint32_t kStackRows = L::kStackRows;

  MatTile<kStackRows, C> t_l1;
  TileLeft<half, kStackRows, C> t_l0a;
  TASSIGN(t_l1, L::l1T(Par));
  TASSIGN(t_l0a, L::l0aX(Par));
  wait_flag(PIPE_FIX, PIPE_MTE1, EvPA);  // phase A parked T in L1
  TMOV(t_l0a, t_l1);
  set_flag(PIPE_MTE1, PIPE_M, EvPB);

  TileRight<half, C, C> hc_l0;
  TileAcc<float, kStackRows, C> acc_out;
  TASSIGN(hc_l0, L::kL0bHc);
  TASSIGN(acc_out, L::l0cCat(Par));
  wait_flag(PIPE_MTE1, PIPE_M, EvPB);
#if !PTO_HAD_NOMM
  TMATMUL(acc_out, t_l0a, hc_l0);
#endif
  set_flag(PIPE_M, PIPE_FIX, EvPB);

  __gm__ half *y_tile = y + sample * L::kN;
  wait_flag(PIPE_M, PIPE_FIX, EvPB);
  if constexpr (PTO_HAD_NZSTORE) {
    GmNz<kStackRows, C> y_gm(y_tile);
    TASSIGN(y_gm, y_tile);
    TSTORE(y_gm, acc_out);
  } else {
    GmTile<kStackRows, C> y_gm(y_tile);
    TASSIGN(y_gm, y_tile);
    TSTORE(y_gm, acc_out);
  }
}

// One loop iteration: prefetch tile i+3 into the ring slot it is about to
// vacate, start phase A of tile i+1, finish phase B of tile i. Slot and parity
// are template arguments because the flag ids must be literal, and all of them
// follow from i mod 4, so the loop is a four-way switch over that.
template <uint32_t R, uint32_t C, uint32_t S, uint32_t SlotLoad,
          uint32_t SlotA, bool ParA, bool ParB, event_t EvXLoad, event_t EvXA,
          event_t EvPA_A, event_t EvPA_B, event_t EvPB_B>
AICORE void deepStep(__gm__ half *x, __gm__ half *y, uint32_t base, uint32_t i,
                     uint32_t n) {
  if (i + 3 < n) {
    issueTileLoad<R, C, S, SlotLoad, EvXLoad>(x, base + (i + 3) * S);
  }
  if (i + 1 < n) {
    tilePhaseA<R, C, S, SlotA, ParA, EvXA, EvPA_A>();
  }
  tilePhaseB<R, C, S, ParB, EvPA_B, EvPB_B>(y, base + i * S);
}

template <uint32_t R, uint32_t C, uint32_t S>
AICORE void runTilesDeep(__gm__ half *x, __gm__ half *y, uint32_t &sample,
                         uint32_t end) {
  const uint32_t base = sample;
  const uint32_t n = (end - base) / S;
  if (n == 0) {
    return;
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, EV_D0);  // all four slots start free
  set_flag(PIPE_MTE1, PIPE_MTE2, EV_D1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EV_D2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EV_D3);

  // Fill the ring three deep and open phase A of tile 0, so the first loop
  // iteration already has MTE2 two tiles ahead of MTE1.
  issueTileLoad<R, C, S, 0, EV_D0>(x, base);
  if (n > 1) {
    issueTileLoad<R, C, S, 1, EV_D1>(x, base + S);
  }
  if (n > 2) {
    issueTileLoad<R, C, S, 2, EV_D2>(x, base + 2 * S);
  }
  tilePhaseA<R, C, S, 0, false, EV_D0, EV_PA0>();

  for (uint32_t i = 0; i < n; ++i) {
    switch (i & 3u) {
      case 0:  // load slot 3, A slot 1 (odd), B even
        deepStep<R, C, S, 3, 1, true, false, EV_D3, EV_D1, EV_PA1, EV_PA0,
                 EV_PB0>(x, y, base, i, n);
        break;
      case 1:  // load slot 0, A slot 2 (even), B odd
        deepStep<R, C, S, 0, 2, false, true, EV_D0, EV_D2, EV_PA0, EV_PA1,
                 EV_PB1>(x, y, base, i, n);
        break;
      case 2:  // load slot 1, A slot 3 (odd), B even
        deepStep<R, C, S, 1, 3, true, false, EV_D1, EV_D3, EV_PA1, EV_PA0,
                 EV_PB0>(x, y, base, i, n);
        break;
      default:  // load slot 2, A slot 0 (even), B odd
        deepStep<R, C, S, 2, 0, false, true, EV_D2, EV_D0, EV_PA0, EV_PA1,
                 EV_PB1>(x, y, base, i, n);
        break;
    }
  }
  sample = base + n * S;

  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_D0);  // drain the four primes
  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_D1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_D2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_D3);
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
  if constexpr (Layout<R, C, S>::kDbuf) {
    runTilesDeep<R, C, S>(x, y, sample, end);
  } else if (end - sample >= S) {
    set_flag(PIPE_MTE1, PIPE_MTE2, EV_X0);  // both slots start free
    set_flag(PIPE_MTE1, PIPE_MTE2, EV_X1);

    // bisheng needs the flag id to be a compile-time literal, so the slot
    // parity is a template argument and the loop dispatches to one of two
    // instantiations rather than carrying it in a variable.
    uint32_t cur = sample;
    bool pong = false;
    issueTileLoad<R, C, S, 0, EV_X0>(x, cur);

    while (true) {
      const uint32_t next = cur + S;
      const bool has_next = (end - next >= S);
      if (pong) {
        if (has_next) {
          issueTileLoad<R, C, S, 0, EV_X0>(x, next);
        }
        hadamardTile<R, C, S, true, EV_X1>(y, cur);
      } else {
        if (has_next) {
          issueTileLoad<R, C, S, 1, EV_X1>(x, next);
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
#if PTO_HAD_STOP >= 2
  return;
#endif

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
#if PTO_HAD_STOP >= 1
  return;
#endif
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
