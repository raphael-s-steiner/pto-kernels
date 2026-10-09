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
AICORE void issueTileLoad(__gm__ half *x, uint32_t sample) {
  using L = Layout<N, S>;

  wait_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // slot free

  MatTile<S, N> x_l1;
  TASSIGN(x_l1, L::xSlot(Slot));
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
  GmTile<S, N> x_gm(x_tile);
  TASSIGN(x_gm, x_tile);
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
AICORE void onesidedTile(__gm__ half *h, __gm__ half *y, uint32_t sample) {
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

  __gm__ half *y_tile = y + sample * N;
  GmTile<S, N> y_gm(y_tile);
  TASSIGN(y_gm, y_tile);
  TSTORE(y_gm, acc);
  set_flag(PIPE_FIX, PIPE_M, EV_ACC);
}

// Tile i starts at base + i*stride, except a tail tile, which ends at `end`.
AICORE inline uint32_t tileStart(uint32_t base, uint32_t stride,
                                 uint32_t n_full, uint32_t end, uint32_t i) {
  return (i < n_full) ? (base + i * stride) : (end - stride);
}

// Consume [sample, end) with the largest power-of-two tile that still fits,
// halving on the way down. 16 is the smallest: a tile's Rows IS S here, and PTO
// wants Rows a multiple of 16 while TMOV's TExtract rejects even Rows == 1. So
// the last 1..15 samples are covered by backing one more 16-tile up to END,
// which recomputes up to 15 samples -- harmless, because the transform is
// per-sample and the overlapping store writes byte-identical values. It does
// mean batch >= 16; the host wrapper pads below that.
//
// The X ring is four deep so MTE2 stays two tiles ahead of MTE1; the tile body
// touches each pipe once, so nothing further is needed to keep them overlapped.
template <uint32_t N, uint32_t S>
AICORE void runTiles(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                     uint32_t &sample, uint32_t end) {
  constexpr bool kTail = (S == 16);  // the level that mops up a short remainder
  const uint32_t base = sample;
  const uint32_t span = end - base;
  const uint32_t n_full = span / S;
  const uint32_t n = n_full + ((kTail && (span % S) != 0) ? 1u : 0u);

  if (n > 0) {
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

    issueTileLoad<N, S, 0, EV_X0>(x, tileStart(base, S, n_full, end, 0));
    if (n > 1) issueTileLoad<N, S, 1, EV_X1>(x, tileStart(base, S, n_full, end, 1));
    if (n > 2) issueTileLoad<N, S, 2, EV_X2>(x, tileStart(base, S, n_full, end, 2));

    for (uint32_t i = 0; i < n; ++i) {
      // Slot and accumulator parity follow i, and the flag ids must be literal,
      // so the loop is a four-way switch over i mod 4.
      switch (i & 3u) {
        case 0:
          if (i + 3 < n) issueTileLoad<N, S, 3, EV_X3>(x, tileStart(base, S, n_full, end, i + 3));
          onesidedTile<N, S, 0, false, EV_X0, EV_A0>(h, y, tileStart(base, S, n_full, end, i));
          break;
        case 1:
          if (i + 3 < n) issueTileLoad<N, S, 0, EV_X0>(x, tileStart(base, S, n_full, end, i + 3));
          onesidedTile<N, S, 1, true, EV_X1, EV_A1>(h, y, tileStart(base, S, n_full, end, i));
          break;
        case 2:
          if (i + 3 < n) issueTileLoad<N, S, 1, EV_X1>(x, tileStart(base, S, n_full, end, i + 3));
          onesidedTile<N, S, 2, false, EV_X2, EV_A0>(h, y, tileStart(base, S, n_full, end, i));
          break;
        default:
          if (i + 3 < n) issueTileLoad<N, S, 2, EV_X2>(x, tileStart(base, S, n_full, end, i + 3));
          onesidedTile<N, S, 3, true, EV_X3, EV_A1>(h, y, tileStart(base, S, n_full, end, i));
          break;
      }
    }
    sample = kTail ? end : (base + n_full * S);

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
  if constexpr (S > 16) {
    runTiles<N, S / 2>(x, h, y, sample, end);
  }
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
