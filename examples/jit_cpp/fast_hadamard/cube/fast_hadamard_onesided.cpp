#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

// One-sided Hadamard transform on the cube unit: Y = X * H_N as a single right
// multiplication. H is symmetric, so the right operand alone gives the natural
// (Sylvester) ordered transform, there is no transpose, and nothing is turned
// around through L1 -- a tile touches each pipe ONCE:
//
//     GM --MTE2--> L1 --MTE1--> L0A --M--> L0C --FIX--> GM
//
// The two-sided kernel beside it cannot: its tile visits MTE1 and M twice, so
// consecutive tiles serialize behind the second visit. This one spends
// sqrt(N)/2 times as many MACs -- N*N per sample against 2*N*sqrt(N) -- to buy
// that straight pipe, which is the right side of the trade while N is small,
// and it rounds to fp16 exactly ONCE, the most accurate of the three kernels
// here. Past 512 the MACs grow 4x per doubling while the bytes only double, so
// the two-sided kernel takes over: the table below stops there at a compute
// crossover, not a capacity limit.
//
// Above N=128, H_N is not applied as one dense (N x N) operand. It is factored
// H_N = H_B (x) H_M and applied as B*B matmuls of a FIXED size M against a
// resident +-H_M, summed in L0C -- see BLOCKED ACCUMULATION below, which is
// what lets N=512 work at all (1.33-1.56x at N=256 and 1.64-2.55x at N=512 over
// the dense form, whose H streamed from GM every tile; N <= 128 is untouched,
// B == 1 there and that IS the dense matmul).
//
// Dropping the matmuls measures this kernel at 51-91% MOVEMENT, and its GM legs
// cannot be made conversion-free the way the two-sided kernel's can -- see
// issueTileLoad, where that is a counting argument and not a tuning gap.
//
// See the README for the measurements behind every constant here.

// Samples per tile, as S * N, for N <= 128 -- where B == 1 and all of H sits in
// L0B. 16384 keeps two L0C accumulators (S*N*4 each) inside 128 KB and two L0A
// slots inside 64 KB.
#ifndef PTO_OS_TILE_ELEMS
#define PTO_OS_TILE_ELEMS 16384
#endif
#ifndef PTO_OS_S64
#define PTO_OS_S64 (PTO_OS_TILE_ELEMS / 64)
#endif
#ifndef PTO_OS_S128
#define PTO_OS_S128 (PTO_OS_TILE_ELEMS / 128)
#endif
// N >= 256 runs the blocked form, whose L0A slice is only (S x M), and it wants
// a SMALLER tile than TILE_ELEMS / N would give it: a tall tile was justified
// only by halving H's per-tile L1->L0B traffic, which blocking makes zero, and
// with that gone a short tile's deeper buffering wins (1.24x at N=512). 16 at
// N=256 is a cliff rather than a continuation -- below 32 the tile stops
// covering the fan-out over B output blocks.
#ifndef PTO_OS_S256
#define PTO_OS_S256 32
#endif
#ifndef PTO_OS_S512
#define PTO_OS_S512 16
#endif

// BLOCKED ACCUMULATION -- the vector butterfly's staging, done on the cube.
//
// With N = B*M and H_N = H_B (x) H_M, blocking the product by M columns gives
//
//     Y[:, jM:(j+1)M] = sum_i  X[:, iM:(i+1)M] * (H_B[i,j] * H_M)
//
// and H_B[i,j] is +-1, so the ONLY right operands that ever occur are +H_M and
// -H_M. The matmul is restricted to a fixed efficient size, the sum over i runs
// in the L0C accumulator exactly as the vector core accumulates its log2(N)
// butterfly stages, and H near the matmul shrinks from N*N to 2*M*M -- resident
// in L0B for every N, so per-tile H traffic is zero, not merely smaller. Same
// MAC count as the dense form, same single fp32 rounding, still one pass over
// each pipe per block. H_B is never materialized: H_B[i,j] = (-1)^popcount(i&j).
//
// M MUST match _BLOCK in the host wrapper, which builds +-H_M; a mismatch hands
// the kernel a wrong-sized operand and it computes garbage. 64, not the 128
// that would still fit L0B: what matters is that the (S x M) L0A slice stays
// small enough to double-buffer deeply, not that the matmuls are few and large
// (3.01x over M=32 and 1.24x over M=128 at N=512). Resweep it if the tile
// geometry changes -- it did not survive a geometry change once already.
#ifndef PTO_OS_BLOCK
#define PTO_OS_BLOCK 64
#endif

#define FAST_HADAMARD_ONESIDED_CASES(X) \
  X(64, PTO_OS_S64)                     \
  X(128, PTO_OS_S128)                   \
  X(256, PTO_OS_S256)                   \
  X(512, PTO_OS_S512)

#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

namespace {

constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L0AB_BYTES = 64 * 1024;
constexpr uint32_t L0C_BYTES = 128 * 1024;

// Ids are unique only per (src pipe, dst pipe) pair. Macros, not constexpr:
// bisheng accepts only a literal EVENT_ID* token or a template non-type
// argument as the third set_flag / wait_flag argument.
#define EV_X0 EVENT_ID0  // L1 X ring slot 0
#define EV_X1 EVENT_ID1  // L1 X ring slot 1
#define EV_X2 EVENT_ID2  // L1 X ring slot 2
#define EV_X3 EVENT_ID3  // L1 X ring slot 3
// L0A slot free again: the consumer telling the producer it may overwrite. NOT
// optional -- the two-sided kernel gets this for free from its L1 turnaround,
// but here each pipe is touched once, so nothing holds MTE1 back and it runs
// ahead by slots, letting tile i+2's TMOV land on the slot tile i's TMATMUL is
// still reading. Its absence is a race that moves with the shape.
#define EV_A0 EVENT_ID4  // (M->MTE1)    L0A/L0B slot 0 free
#define EV_A1 EVENT_ID5  // (M->MTE1)    L0A/L0B slot 1 free
#define EV_ACC EVENT_ID6  // (FIX->M)     L0C accumulator free

// Sync<>'s default id 0 collides with EV_X0 on (MTE2, MTE1), so it must not be
// called on that pair anywhere inside the tile loop.

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void Sync(uint32_t id = 0) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

// The only Mat layout TLOAD / TMOV accept on A2/A3.
template <uint32_t Rows, uint32_t Cols>
using MatTile = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor, Rows,
                     Cols, SLayout::RowMajor, 512>;

template <uint32_t Rows, uint32_t Cols>
using GmTile =
    GlobalTensor<half, TileShape2D<half, Rows, Cols, Layout::ND>,
                 BaseShape2D<half, Rows, Cols, Layout::ND>, Layout::ND>;

// Same as MatTile but with a settable valid row count. The short-batch load's
// destination must agree with a GM source whose row count is a runtime value.
template <uint32_t Rows, uint32_t Cols, int RowValid>
using MatTileV = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor,
                      RowValid, Cols, SLayout::RowMajor, 512>;

// Same as GmTile but with a DYNAMIC row count. The base shape keeps the static
// row stride (Cols); the DIM_0..2 strides it also carries are unused, because
// the ND shape's leading dims are 1.
template <uint32_t Rows, uint32_t Cols>
using GmTileDyn =
    GlobalTensor<half, TileShape2D<half, DYNAMIC, Cols, Layout::ND>,
                 BaseShape2D<half, Rows, Cols, Layout::ND>, Layout::ND>;

template <uint32_t N, uint32_t S>
struct Layout {
  static constexpr uint32_t kTileElems = S * N;
  static constexpr unsigned kTileBytes = kTileElems * sizeof(half);
  static constexpr unsigned kAccBytes = kTileElems * sizeof(float);
  static constexpr unsigned kHBytes = N * N * sizeof(half);

  // All of H fits half of L0B, so the single resident copy is also
  // double-bufferable: one dense matmul per tile and no H traffic at all.
  // Otherwise: blocked accumulation against a resident +-H_M pair, which is no
  // more traffic. Those are the only two forms, so every size's H is loaded
  // once per core and no tile ever touches MTE2 for H.
  static constexpr bool kResident = (kHBytes * 2 <= L0AB_BYTES);  // N <= 128
  static constexpr uint32_t kM = (N < PTO_OS_BLOCK) ? N : (uint32_t)PTO_OS_BLOCK;
  static constexpr uint32_t kBlk = N / kM;  // B, the number of M-column blocks
  static constexpr bool kBlocked = !kResident && (kBlk > 1);
  static_assert(kResident || kBlocked, "every size must reach one of the bodies");
  static_assert(!kBlocked || (kBlk * kM == N && kM % 16 == 0),
                "M must divide N in whole fractal columns");

  static constexpr unsigned kHPair = kM * kM * sizeof(half);  // one of +-H_M
  static constexpr unsigned kXSlots = 4;

  // L1: H, either all of it or the +-H_M pair, then the X ring.
  static constexpr unsigned kL1H = 0;
  static constexpr unsigned kL1X =
      kL1H + (kResident ? kHBytes : 2 * kHPair);
  static_assert(kL1X + kXSlots * kTileBytes <= L1_BYTES, "L1 overflow");

  static constexpr unsigned xSlot(uint32_t slot) {
    return kL1X + slot * kTileBytes;
  }
  // The +-H_M pair in L1, indexed by SIGN: 0 is +H_M and 1 is -H_M.
  static constexpr unsigned l1HSign(uint32_t sign) {
    return kL1H + (sign & 1u) * kHPair;
  }

  // L0A: the whole (S x N) tile when resident, else one (S x M) block of X.
  // Two slots wherever they fit -- both counts are derived, so a tile size that
  // does not fit gets shallower buffers instead of failing to compile.
  static constexpr unsigned kL0aX = 0;
  static constexpr unsigned kL0aSlot =
      kResident ? kTileBytes : S * kM * sizeof(half);
  static constexpr unsigned kAccSlots = (2 * kAccBytes <= L0C_BYTES) ? 2u : 1u;
  static constexpr unsigned kL0aSlots = (2 * kL0aSlot <= L0AB_BYTES) ? 2u : 1u;
  static_assert(kL0aSlot <= L0AB_BYTES, "L0A overflow");
  static constexpr unsigned l0aX(uint32_t i) {
    return kL0aX + (kL0aSlots == 2u ? (i & 1u) : 0u) * kL0aSlot;
  }
  // Columns [i*M, (i+1)*M) of the L1 X tile. An L1 Mat tile is
  // block-column-major, so whole 16-column groups are one contiguous range --
  // the slice an L0A Left tile canNOT do, being fractal-ROW-major.
  static constexpr unsigned xBlock(uint32_t slot, uint32_t i) {
    return xSlot(slot) + i * kM * S * sizeof(half);
  }

  // L0B: H resident in full, or the resident +-H_M pair indexed by SIGN.
  static constexpr unsigned kL0bH = 0;
  static_assert(kResident ? (kHBytes <= L0AB_BYTES) : (2 * kHPair <= L0AB_BYTES),
                "L0B overflow");
  static constexpr unsigned l0bH(uint32_t sign) {
    return kL0bH + (kResident ? 0u : (sign & 1u) * kHPair);
  }

  // L0C: the accumulator, two slots where they fit.
  static_assert(kAccBytes <= L0C_BYTES, "L0C overflow");
  static constexpr unsigned l0cAcc(bool par) {
    return (kAccSlots == 2u && par) ? kAccBytes : 0;
  }
  // Output block j: columns [j*M, (j+1)*M) of the accumulator, contiguous for
  // the same reason -- an L0C tile is block-column-major too.
  static constexpr unsigned l0cBlock(unsigned base, uint32_t j) {
    return base + j * kM * S * sizeof(float);
  }
};

// H_B[i,j] = (-1)^popcount(i & j) for the Sylvester matrix, so the sign of
// block (i,j) is a parity, and H_B itself is never built or read. Returns 1 for
// -H_M and 0 for +H_M, which are also their L0B slot indices.
AICORE inline uint32_t blockSign(uint32_t i, uint32_t j) {
  uint32_t v = i & j;
  v ^= v >> 8;
  v ^= v >> 4;
  v ^= v >> 2;
  v ^= v >> 1;
  return v & 1u;
}

// GM -> L1 -> L0B for H, once per core. The host buffer is the row-major
// (N x N) matrix followed by +H_M and -H_M at offset N*N, so one buffer feeds
// either form and only one of the two is ever read.
template <uint32_t N, uint32_t S>
AICORE void loadHadamard(__gm__ half *h) {
  using L = Layout<N, S>;
  if constexpr (L::kBlocked) {
    for (uint32_t s = 0; s < 2; ++s) {  // s = 0 is +H_M, s = 1 is -H_M
      MatTile<L::kM, L::kM> h_l1;
      TASSIGN(h_l1, L::l1HSign(s));
      __gm__ half *src = h + N * N + s * L::kM * L::kM;
      GmTile<L::kM, L::kM> h_gm(src);
      TASSIGN(h_gm, src);
      TLOAD(h_l1, h_gm);
    }
    Sync<PIPE_MTE2, PIPE_MTE1>();
    for (uint32_t s = 0; s < 2; ++s) {
      MatTile<L::kM, L::kM> h_l1;
      TileRight<half, L::kM, L::kM> h_l0b;
      TASSIGN(h_l1, L::l1HSign(s));
      TASSIGN(h_l0b, L::l0bH(s));
      TMOV(h_l0b, h_l1);
    }
    Sync<PIPE_MTE1, PIPE_M>();
  } else if constexpr (L::kResident) {
    MatTile<N, N> h_l1;
    TASSIGN(h_l1, L::kL1H);
    GmTile<N, N> h_gm(h);
    TASSIGN(h_gm, h);
    TLOAD(h_l1, h_gm);
    Sync<PIPE_MTE2, PIPE_MTE1>();

    TileRight<half, N, N> h_l0b;
    TASSIGN(h_l0b, L::kL0bH);
    TMOV(h_l0b, h_l1);
    Sync<PIPE_MTE1, PIPE_M>();
  }
}

// GM -> L1 X slot, signalling MTE2->MTE1. The caller already holds the slot.
template <uint32_t N, uint32_t S, uint32_t Slot, event_t EvX>
AICORE void issueTileLoad(__gm__ half *x, uint32_t sample, uint32_t rows) {
  using L = Layout<N, S>;

  wait_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // slot free

  MatTileV<S, N, DYNAMIC> x_l1;
  TASSIGN(x_l1, L::xSlot(Slot));
  x_l1.SetValidRow(rows);
  __gm__ half *x_tile = x + sample * N;

  // A CONVERTING ND->NZ load, and it cannot be made conversion-free. H_N is a
  // tensor power of H_2, so a compile-time operand can apply anything to one
  // FIELD of the GM byte index p, and the layout fixes which fields exist: NZ
  // gives the right operand p[0,4) u p[a+4,t) and the left operand p[4,a+4);
  // ND is the mirror image, since TMOV from an ND L1 tile reads the bytes as
  // fractals AND TRANSPOSES (measured) rather than converting.
  //
  // Every operand here, dense or blocked, is a RIGHT operand, so exactly one
  // field is covered and the other keeps live transform bits -- p[4,a+4) under
  // NZ, p[0,4) under ND, both non-empty for every N in the table. Covering both
  // means a left multiply as well, which is the two-sided kernel: "one-sided"
  // and "conversion-free GM transfers" are mutually exclusive by construction.
  GmTileDyn<S, N> x_gm(x_tile);
  TASSIGN(x_gm, x_tile);
  x_gm.template SetShape<GlobalTensorDim::DIM_3>(rows);
  TLOAD(x_l1, x_gm);
  set_flag(PIPE_MTE2, PIPE_MTE1, EvX);
}

// One input block of the blocked form: an (S x M) column slice of X to L0A,
// then B matmuls against +-H_M, one into each output block's accumulator. Slot
// parity is a template argument because the flag ids must be literal.
template <uint32_t N, uint32_t S, uint32_t XSlot, bool APar, event_t EvA>
AICORE void blockStep(unsigned accBase, uint32_t i) {
  using L = Layout<N, S>;

  MatTile<S, L::kM> x_l1;
  TileLeft<half, S, L::kM> x_l0a;
  TASSIGN(x_l1, L::xBlock(XSlot, i));
  TASSIGN(x_l0a, L::l0aX(APar ? 1u : 0u));

  wait_flag(PIPE_M, PIPE_MTE1, EvA);  // block i-2 has finished with the slot
  TMOV(x_l0a, x_l1);
  Sync<PIPE_MTE1, PIPE_M>();

  for (uint32_t j = 0; j < L::kBlk; ++j) {
    TileAcc<float, S, L::kM> acc_j;
    TASSIGN(acc_j, L::l0cBlock(accBase, j));
    TileRight<half, L::kM, L::kM> h_j;
    TASSIGN(h_j, L::l0bH(blockSign(i, j)));
    // i == 0 initializes every output block; every later block accumulates.
    // Both run on PIPE_M in order, so the read-after-write between them needs
    // no flag.
    if (i == 0) {
      TMATMUL(acc_j, x_l0a, h_j);
    } else {
      TMATMUL_ACC(acc_j, x_l0a, h_j);
    }
  }
  set_flag(PIPE_M, PIPE_MTE1, EvA);  // slot free for block i+2
}

// Y[sample .. sample+S) = X * H, for a tile already loaded into X slot Slot.
// One pass over the contraction with no L1 turnaround, so the next tile's MTE1
// is never stuck behind this tile's matmul.
template <uint32_t N, uint32_t S, uint32_t Slot, bool Par, event_t EvX,
          event_t EvA>
AICORE void onesidedTile(__gm__ half *h, __gm__ half *y, uint32_t sample,
                         uint32_t rows) {
  using L = Layout<N, S>;

  wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);  // this tile's load has landed

  TileAcc<float, S, N> acc;
  TASSIGN(acc, L::l0cAcc(Par));

  if constexpr (L::kResident) {
    // H is already in L0B and stays there: one matmul, nothing to move.
    MatTile<S, N> x_l1;
    TileLeft<half, S, N> x_l0a;
    TASSIGN(x_l1, L::xSlot(Slot));
    TASSIGN(x_l0a, L::l0aX(Par ? 1u : 0u));
    wait_flag(PIPE_M, PIPE_MTE1, EvA);  // tile i-2 has finished reading the slot
    TMOV(x_l0a, x_l1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // X slot free again

    TileRight<half, N, N> h_l0b;
    TASSIGN(h_l0b, L::kL0bH);
    wait_flag(PIPE_FIX, PIPE_M, EV_ACC);  // the slot's previous store drained
    Sync<PIPE_MTE1, PIPE_M>();
    TMATMUL(acc, x_l0a, h_l0b);
    set_flag(PIPE_M, PIPE_MTE1, EvA);  // slot free for tile i+2
  } else if constexpr (L::kBlocked) {
    // +-H_M likewise, so a block costs one TMOV and B matmuls.
    wait_flag(PIPE_FIX, PIPE_M, EV_ACC);  // the slot's previous store drained
    const unsigned accBase = L::l0cAcc(Par);
    for (uint32_t i = 0; i < L::kBlk; ++i) {
      if (i & 1u) {
        blockStep<N, S, Slot, true, EV_A1>(accBase, i);
      } else {
        blockStep<N, S, Slot, false, EV_A0>(accBase, i);
      }
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // last block has read the X slot
  }
  Sync<PIPE_M, PIPE_FIX>();

  // The cube ran all S rows whatever `rows` says -- the transform is per-sample,
  // so the rows past `rows` only reach output rows that are never stored.
  TileAcc<float, S, N, DYNAMIC> acc_out;
  TASSIGN(acc_out, L::l0cAcc(Par));
  acc_out.SetValidRow(rows);
  __gm__ half *y_tile = y + sample * N;
  GmTileDyn<S, N> y_gm(y_tile);
  TASSIGN(y_gm, y_tile);
  y_gm.template SetShape<GlobalTensorDim::DIM_3>(rows);
  TSTORE(y_gm, acc_out);
  set_flag(PIPE_FIX, PIPE_M, EV_ACC);
}

// The rows a core is left with when its span is shorter than one tile, 1..S-1.
// The tile geometry is the compile-time S, but the TRANSFERS are not: an
// ND->NZ load moves the GM source's row count (A2/A3) and TSTORE moves
// `src.GetValidRow()`, so MTE2 reads exactly `rows` rows and FIX writes exactly
// `rows` rows. The cube still runs a full S-row tile over whatever else L1
// holds -- the transform is per-sample, so those rows only reach output rows
// that are never stored.
//
// The blocked body is the ring's own `blockStep`, so the B blocks of one tile
// double-buffer L0A against each other exactly as they do in the ring. It is
// NOT enough to sync M->MTE1 once per block instead: a single tile is the whole
// of a short-span core's work, so serializing its B blocks is the critical
// path, and doing that cost 1.35x at N=512 and 1.14x at N=256. That is a cost
// of this body alone and has nothing to do with how the span was divided.
template <uint32_t N, uint32_t S>
AICORE void partialTile(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                        uint32_t sample, uint32_t rows) {
  using L = Layout<N, S>;

  {
    MatTileV<S, N, DYNAMIC> x_l1;
    TASSIGN(x_l1, L::xSlot(0));
    x_l1.SetValidRow(rows);
    __gm__ half *x_tile = x + sample * N;
    GmTileDyn<S, N> x_gm(x_tile);
    TASSIGN(x_gm, x_tile);
    x_gm.template SetShape<GlobalTensorDim::DIM_3>(rows);
    TLOAD(x_l1, x_gm);
  }
  Sync<PIPE_MTE2, PIPE_MTE1>(7);

  TileAcc<float, S, N> acc;
  TASSIGN(acc, L::l0cAcc(false));
  if constexpr (L::kResident) {
    MatTile<S, N> x_l1;
    TileLeft<half, S, N> x_l0a;
    TASSIGN(x_l1, L::xSlot(0));
    TASSIGN(x_l0a, L::l0aX(0));
    TMOV(x_l0a, x_l1);
    TileRight<half, N, N> h_l0b;
    TASSIGN(h_l0b, L::kL0bH);
    Sync<PIPE_MTE1, PIPE_M>(7);
    TMATMUL(acc, x_l0a, h_l0b);
  } else {
    set_flag(PIPE_M, PIPE_MTE1, EV_A0);
    set_flag(PIPE_M, PIPE_MTE1, EV_A1);
    for (uint32_t i = 0; i < L::kBlk; ++i) {
      if (i & 1u) {
        blockStep<N, S, 0, true, EV_A1>(0, i);
      } else {
        blockStep<N, S, 0, false, EV_A0>(0, i);
      }
    }
    wait_flag(PIPE_M, PIPE_MTE1, EV_A0);
    wait_flag(PIPE_M, PIPE_MTE1, EV_A1);
  }
  Sync<PIPE_M, PIPE_FIX>();

  TileAcc<float, S, N, DYNAMIC> acc_out;
  TASSIGN(acc_out, L::l0cAcc(false));
  acc_out.SetValidRow(rows);
  __gm__ half *y_tile = y + sample * N;
  GmTileDyn<S, N> y_gm(y_tile);
  TASSIGN(y_gm, y_tile);
  y_gm.template SetShape<GlobalTensorDim::DIM_3>(rows);
  TSTORE(y_gm, acc_out);
  Sync<PIPE_FIX, PIPE_M>(7);
}

// A core's span as ONE ring of `m` tiles, every one at its natural position
// and every one MASKED to `rowsAt` rows.
//
// A full tile is simply the case where that min picks S, so there is no special
// last tile: no back-up, no duplicated rows, and no separate remainder unit
// behind its own prime and drain. The remainder is an ordinary ring tile, which
// means it overlaps its predecessor like any other, and it is a FULL-SIZE S
// tile with a short transfer rather than a small tile of its own.
//
// The masking is free on the transfers -- an ND->NZ load moves the GM source's
// row count and TSTORE moves `src.GetValidRow()` -- but not on the tile: the
// cube runs all S rows regardless. That costs nothing here because the only
// tile ever short is the last one, and it is overlapped.
template <uint32_t N, uint32_t S>
AICORE void runRing(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                    uint32_t base, uint32_t end) {
  // The span fixes the tile count, so the caller has nothing to pass but the
  // range. Exactly ONE tile is ever short -- tile `last`, with `rest` rows --
  // so every row count below is a pick between `rest` and the constant S, and
  // there is no arithmetic per tile.
  const uint32_t span = end - base;
  const uint32_t last = (span - 1) / S;        // span >= S, so this is m - 1
  const uint32_t rest = span - last * S;       // 1..S

  set_flag(PIPE_MTE1, PIPE_MTE2, EV_X0);  // all four slots start free
  set_flag(PIPE_MTE1, PIPE_MTE2, EV_X1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EV_X2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EV_X3);
  set_flag(PIPE_FIX, PIPE_M, EV_ACC);  // one credit per accumulator slot
  if constexpr (Layout<N, S>::kAccSlots == 2u) {
    set_flag(PIPE_FIX, PIPE_M, EV_ACC);
  }
  set_flag(PIPE_M, PIPE_MTE1, EV_A0);  // both L0A/L0B slots start free
  set_flag(PIPE_M, PIPE_MTE1, EV_A1);

  issueTileLoad<N, S, 0, EV_X0>(x, base, last == 0 ? rest : S);
  if (last > 0) issueTileLoad<N, S, 1, EV_X1>(x, base + S, last == 1 ? rest : S);
  if (last > 1) issueTileLoad<N, S, 2, EV_X2>(x, base + 2 * S, last == 2 ? rest : S);

  for (uint32_t i = 0; i <= last; ++i) {
    // Slot and accumulator parity follow i, and the flag ids must be literal,
    // so the loop is a four-way switch over i mod 4. Each arm prefetches the
    // tile three ahead, if there is one, and runs tile i.
    const uint32_t cur = base + i * S;
    const uint32_t crow = (i == last) ? rest : S;
    // `next` and `nrow` are only read under `prefetch`, so neither needs a
    // guard of its own -- an address three tiles past the end is never used.
    const bool prefetch = (i + 3 <= last);
    const uint32_t next = cur + 3 * S;
    const uint32_t nrow = (i + 3 == last) ? rest : S;
    switch (i & 3u) {
      case 0:
        if (prefetch) issueTileLoad<N, S, 3, EV_X3>(x, next, nrow);
        onesidedTile<N, S, 0, false, EV_X0, EV_A0>(h, y, cur, crow);
        break;
      case 1:
        if (prefetch) issueTileLoad<N, S, 0, EV_X0>(x, next, nrow);
        onesidedTile<N, S, 1, true, EV_X1, EV_A1>(h, y, cur, crow);
        break;
      case 2:
        if (prefetch) issueTileLoad<N, S, 1, EV_X1>(x, next, nrow);
        onesidedTile<N, S, 2, false, EV_X2, EV_A0>(h, y, cur, crow);
        break;
      default:
        if (prefetch) issueTileLoad<N, S, 2, EV_X2>(x, next, nrow);
        onesidedTile<N, S, 3, true, EV_X3, EV_A1>(h, y, cur, crow);
        break;
    }
  }

  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_X0);  // drain the four primes
  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_X1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_X2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EV_X3);
  wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
  if constexpr (Layout<N, S>::kAccSlots == 2u) {
    wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
  }
  wait_flag(PIPE_M, PIPE_MTE1, EV_A0);
  wait_flag(PIPE_M, PIPE_MTE1, EV_A1);
}

// One masked tile at the SMALLEST instantiated tile size that covers `rows`.
//
// Masking is free on the TRANSFERS but not on the tile: the cube still runs all
// S rows whatever the row count says, so a 1..15 row span in an S = 256 tile
// wastes 240 rows of TMOV and MACs -- 0.19 us at N=64 and 0.32 us at N=128,
// which is 15-20% of a short-span core, enough to lose to the chain this
// replaced. Choosing the level by `rows` bounds the waste at under half a tile
// and costs a scalar compare chain plus one partialTile instantiation per
// level.
//
// The recursion is a compile-time CHOICE, not a chain: every level below the
// chosen one returns before doing any work, so exactly one tile ever runs.
template <uint32_t N, uint32_t S>
AICORE void maskedTile(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                       uint32_t sample, uint32_t rows) {
  if constexpr (S > 16) {
    if (rows <= S / 2) {
      maskedTile<N, S / 2>(x, h, y, sample, rows);
      return;
    }
  }
  partialTile<N, S>(x, h, y, sample, rows);
}

// Consume [sample, end) as exactly ONE unit of work: a ring whose every tile
// carries its own row count, or -- when the span is shorter than a single tile
// -- one standalone masked tile. Either way nothing is left over, so there is
// no second unit and no chain of them.
//
// The two arms exist because a LONE tile has nothing to overlap. Inside the
// ring the short tile is free, since it pipelines against its predecessor; on
// its own it would pay the ring's prime and drain for no overlap at all, which
// measured 1.05-1.11x. Hence: ring when there is something to overlap with,
// standalone when there is not.
//
// This replaced a recursive power-of-two HALVING chain: run span/S whole tiles,
// then tail-recurse on S/2, down to 16, and mask the last 1..15 rows. That was
// correct and it cost nothing in bulk -- work is split by TILES, so every core
// but the last-with-work has a span that is a whole multiple of S and never
// descends at all, and the one core that does descend is the truncated, UNDER-
// loaded one, whose serialization hides in its slack (N=64 batch 65535 against
// 65536: 5.46 vs 5.48 us, no difference).
//
// What it cost was the regime where that slack is gone -- tiles_per_core of 1
// or 2, where the descending core IS the critical path. There each level is a
// full prime/ring/drain for one tile, with no predecessor to overlap, and a
// 255-sample span at N=64 walked five of them back to back. Collapsing that to
// one unit is 1.28-1.56x over the four sizes at short and unaligned spans, and
// parity everywhere else, including every aligned shape and the whole bulk
// range. Note that N=512 (S=16) never had a chain to walk and still gains: two
// thirds of the win is the ring fold and the blocked partialTile body, not the
// chain. The README has the full grid.
//
// Halving back down was NOT the cheap alternative it looks like: a 16-row ring
// tile costs ~0.11 us at the margin, so the 15 small tiles that cover a
// 255-sample span lose to four serialized big ones, and a two-level form (S
// then 16) measured 0.78-0.86x -- slower than the chain it was meant to fix.
// The short span wants ONE big masked tile, which is what maskedTile gives it.
//
// Two further shapes were built and measured before this one was settled on,
// and both are worse. Running the ring over WHOLE tiles only and leaving the
// remainder to a standalone maskedTile is simpler still, but the remainder
// then overlaps nothing: 1.045x and 1.071x at N=64, where S = 256 makes it a
// large un-overlapped tile. Putting EVERY span through the ring, the short
// ones included, fixes that but loses the other end, 1.05-1.11x on spans below
// one tile. The split above takes the better arm of each.
template <uint32_t N, uint32_t S>
AICORE void runTiles(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                     uint32_t &sample, uint32_t end) {
  const uint32_t base = sample;
  const uint32_t span = end - base;
  if (span >= S) {
    // The mask rides the RING: every tile is full-size S and carries its own
    // row count, so the remainder is an ordinary ring tile rather than a
    // separate drained unit -- and nothing is recomputed.
    runRing<N, S>(x, h, y, base, end);
  } else if (span > 0) {
    // A span shorter than one tile is a single tile either way, and a lone
    // tile has nothing to overlap, so it skips the ring's prime/drain and runs
    // standalone. Putting it through the ring instead measured 1.05-1.11x.
    maskedTile<N, S>(x, h, y, base, span);
  }
  sample = end;
}

template <uint32_t N, uint32_t S>
AICORE void runOnesided(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                        uint32_t batch) {
  static_assert(N % 16 == 0, "cube tiles need 16-aligned N");
  static_assert(S >= 1, "at least one sample per tile");

  const uint32_t num_cores = block_num;
  const uint32_t core_id = get_block_idx();

  // Split by TILES, not samples: an even sample split leaves most cores walking
  // the power-of-two fallback down to its smallest tile, which costs a full
  // tile's handshakes for a fraction of the work.
  const uint32_t tiles_per_core = DIV_ROUNDUP(DIV_ROUNDUP(batch, S), num_cores);
  const uint32_t begin = tiles_per_core * S * core_id;
  if (begin >= batch) {
    return;
  }
  uint32_t end = begin + tiles_per_core * S;
  if (end > batch) {
    end = batch;
  }

  loadHadamard<N, S>(h);

  uint32_t sample = begin;
  runTiles<N, S>(x, h, y, sample, end);
}

#define FAST_HADAMARD_ONESIDED_DISPATCH_CASE(NVal, SVal) \
  case NVal:                                             \
    runOnesided<NVal, SVal>(x, h, y, batch);             \
    return;

AICORE void dispatchOnesided(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                             uint32_t batch, uint32_t n) {
  // Every transfer below needs a row count > 0, and an empty batch has no work.
  if (batch == 0) {
    return;
  }
  switch (n) {
    FAST_HADAMARD_ONESIDED_CASES(FAST_HADAMARD_ONESIDED_DISPATCH_CASE)
    default:
      return;
  }
}

}  // namespace

#endif  // cube core

__global__ AICORE void fast_hadamard_onesided_fp16(__gm__ void *x,
                                                   __gm__ void *y,
                                                   __gm__ void *h,
                                                   uint32_t batch,
                                                   uint32_t n) {
#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  dispatchOnesided((__gm__ half *)x, (__gm__ half *)h, (__gm__ half *)y, batch,
                   n);
#endif
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint8_t *y, uint8_t *h, uint32_t batch,
                            uint32_t n) {
  // blockDim counts cube cores, so there is no doubling here.
  fast_hadamard_onesided_fp16<<<blockDim, nullptr, stream>>>(x, y, h, batch, n);
}
