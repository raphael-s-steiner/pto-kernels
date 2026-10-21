#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

#ifndef PTO_OS_TILE_ELEMS
// Samples per tile for N <= 128, where all of H is resident in L0B. Sized so
// two fp32 accumulators and two L0A slots fit their buffers.
#define PTO_OS_TILE_ELEMS 16384
#endif
#ifndef PTO_OS_S64
#define PTO_OS_S64 (PTO_OS_TILE_ELEMS / 64)
#endif
#ifndef PTO_OS_S128
#define PTO_OS_S128 (PTO_OS_TILE_ELEMS / 128)
#endif

#ifndef PTO_OS_S256
// N >= 256 runs the blocked form, whose L0A slice is only (S x M). It wants a
// shorter tile than TILE_ELEMS / N: deeper buffering beats a tall tile once
// per-tile H traffic is zero. Tuned per N -- resweep on any geometry change.
#define PTO_OS_S256 32
#endif
#ifndef PTO_OS_S512
#define PTO_OS_S512 16
#endif

#ifndef PTO_OS_BLOCK
// M of the blocked form, H_N = H_B (x) H_M. MUST equal _BLOCK in the host
// wrapper, which builds the +-H_M operand: a mismatch hands the kernel a
// wrong-sized operand and it silently computes garbage.
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

// Ids are unique per (src pipe, dst pipe) pair, and one id must not carry two
// channels out of the SAME pipe: a waiter on one channel can take the other's
// credit and run ahead. Hence the L0A write-after-read edge (M -> MTE1) keeps
// ids clear of the end-of-tile handoff (M -> FIX).
#define EV_X0 EVENT_ID0
#define EV_X1 EVENT_ID1
#define EV_X2 EVENT_ID2
#define EV_X3 EVENT_ID3

#define EV_A0 EVENT_ID4
#define EV_A1 EVENT_ID5
#define EV_ACC EVENT_ID6

// Default id 0 collides with EV_X0 on (MTE2, MTE1), so this must not be called
// on that pair inside the tile loop.
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void Sync(uint32_t id = 0) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

template <uint32_t Rows, uint32_t Cols>
using MatTile = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor, Rows,
                     Cols, SLayout::RowMajor, 512>;

template <uint32_t Rows, uint32_t Cols>
using GmTile =
    GlobalTensor<half, TileShape2D<half, Rows, Cols, Layout::ND>,
                 BaseShape2D<half, Rows, Cols, Layout::ND>, Layout::ND>;

template <uint32_t Rows, uint32_t Cols, int RowValid>
using MatTileV = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor,
                      RowValid, Cols, SLayout::RowMajor, 512>;

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

  // Two forms only, and every size reaches one: H resident in L0B whole, or
  // the +-H_M pair resident instead. Either way no tile ever loads H.
  static constexpr bool kResident = (kHBytes * 2 <= L0AB_BYTES);
  static constexpr uint32_t kM = (N < PTO_OS_BLOCK) ? N : (uint32_t)PTO_OS_BLOCK;
  static constexpr uint32_t kBlk = N / kM;
  static constexpr bool kBlocked = !kResident && (kBlk > 1);
  static_assert(kResident || kBlocked, "every size must reach one of the bodies");
  static_assert(!kBlocked || (kBlk * kM == N && kM % 16 == 0),
                "M must divide N in whole fractal columns");

  static constexpr unsigned kHPair = kM * kM * sizeof(half);

  // Four, because runRing prefetches three tiles ahead: tile i and tile i+4
  // share a slot, which is the distance its credit arithmetic assumes.
  static constexpr unsigned kXSlots = 4;

  static constexpr unsigned kL1H = 0;
  static constexpr unsigned kL1X =
      kL1H + (kResident ? kHBytes : 2 * kHPair);
  static_assert(kL1X + kXSlots * kTileBytes <= L1_BYTES, "L1 overflow");

  static constexpr unsigned xSlot(uint32_t slot) {
    return kL1X + slot * kTileBytes;
  }

  static constexpr unsigned l1HSign(uint32_t sign) {
    return kL1H + (sign & 1u) * kHPair;
  }

  static constexpr unsigned kL0aX = 0;
  static constexpr unsigned kL0aSlot =
      kResident ? kTileBytes : S * kM * sizeof(half);
  static constexpr unsigned kAccSlots = (2 * kAccBytes <= L0C_BYTES) ? 2u : 1u;
  static constexpr unsigned kL0aSlots = (2 * kL0aSlot <= L0AB_BYTES) ? 2u : 1u;
  static_assert(kL0aSlot <= L0AB_BYTES, "L0A overflow");
  static constexpr unsigned l0aX(uint32_t i) {
    return kL0aX + (kL0aSlots == 2u ? (i & 1u) : 0u) * kL0aSlot;
  }

  static constexpr unsigned xBlock(uint32_t slot, uint32_t i) {
    return xSlot(slot) + i * kM * S * sizeof(half);
  }

  static constexpr unsigned kL0bH = 0;
  static_assert(kResident ? (kHBytes <= L0AB_BYTES) : (2 * kHPair <= L0AB_BYTES),
                "L0B overflow");
  static constexpr unsigned l0bH(uint32_t sign) {
    return kL0bH + (kResident ? 0u : (sign & 1u) * kHPair);
  }

  static_assert(kAccBytes <= L0C_BYTES, "L0C overflow");
  static constexpr unsigned l0cAcc(bool par) {
    return (kAccSlots == 2u && par) ? kAccBytes : 0;
  }

  static constexpr unsigned l0cBlock(unsigned base, uint32_t j) {
    return base + j * kM * S * sizeof(float);
  }
};

// H_B[i,j] = (-1)^popcount(i & j), so H_B is a parity of the loop indices and
// is never built or read.
AICORE inline uint32_t blockSign(uint32_t i, uint32_t j) {
  uint32_t v = i & j;
  v ^= v >> 8;
  v ^= v >> 4;
  v ^= v >> 2;
  v ^= v >> 1;
  return v & 1u;
}

// GM -> L1 -> L0B once per core. The host buffer holds the (N x N) matrix
// followed by +H_M and -H_M at offset N*N; only one of the two forms is read.
template <uint32_t N, uint32_t S>
AICORE void loadHadamard(__gm__ half *h) {
  using L = Layout<N, S>;
  if constexpr (L::kBlocked) {
    for (uint32_t s = 0; s < 2; ++s) {
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

template <uint32_t N, uint32_t S, uint32_t Slot, event_t EvX>
AICORE void issueTileLoad(__gm__ half *x, uint32_t sample, uint32_t rows,
                          bool waitSlot) {
  using L = Layout<N, S>;

  if (waitSlot) wait_flag(PIPE_MTE1, PIPE_MTE2, EvX);

  MatTileV<S, N, DYNAMIC> x_l1;
  TASSIGN(x_l1, L::xSlot(Slot));
  x_l1.SetValidRow(rows);
  __gm__ half *x_tile = x + sample * N;

  // A converting ND->NZ load, and it cannot be avoided here. A compile-time
  // operand can absorb the layout change for one field of the GM byte index;
  // every operand in a one-sided product is a RIGHT operand, so it covers one
  // field and the other keeps live transform bits. Conversion-free transfers
  // would need a left multiply too, which is the two-sided kernel.
  GmTileDyn<S, N> x_gm(x_tile);
  TASSIGN(x_gm, x_tile);
  x_gm.template SetShape<GlobalTensorDim::DIM_3>(rows);
  TLOAD(x_l1, x_gm);
  set_flag(PIPE_MTE2, PIPE_MTE1, EvX);
}

template <uint32_t N, uint32_t S, uint32_t XSlot, bool APar, event_t EvA>
AICORE void blockStep(unsigned accBase, uint32_t i) {
  using L = Layout<N, S>;

  MatTile<S, L::kM> x_l1;
  TileLeft<half, S, L::kM> x_l0a;
  TASSIGN(x_l1, L::xBlock(XSlot, i));
  TASSIGN(x_l0a, L::l0aX(APar ? 1u : 0u));

  wait_flag(PIPE_M, PIPE_MTE1, EvA);
  TMOV(x_l0a, x_l1);
  Sync<PIPE_MTE1, PIPE_M>();

  for (uint32_t j = 0; j < L::kBlk; ++j) {
    TileAcc<float, S, L::kM> acc_j;
    TASSIGN(acc_j, L::l0cBlock(accBase, j));
    TileRight<half, L::kM, L::kM> h_j;
    TASSIGN(h_j, L::l0bH(blockSign(i, j)));

    if (i == 0) {
      TMATMUL(acc_j, x_l0a, h_j);
    } else {
      TMATMUL_ACC(acc_j, x_l0a, h_j);
    }
  }
  set_flag(PIPE_M, PIPE_MTE1, EvA);
}

template <uint32_t N, uint32_t S, uint32_t Slot, bool Par, event_t EvX,
          event_t EvA>
AICORE void onesidedTile(__gm__ half *h, __gm__ half *y, uint32_t sample,
                         uint32_t rows, bool freeSlot, bool waitAcc,
                         bool freeAcc, bool waitA, bool freeA) {
  using L = Layout<N, S>;

  // Nothing else holds MTE1 back -- a tile touches each pipe once -- so
  // without the EvA write-after-read edge below MTE1 runs ahead by slots and
  // lands tile i+2's TMOV on the slot tile i's TMATMUL is still reading.
  wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);

  TileAcc<float, S, N> acc;
  TASSIGN(acc, L::l0cAcc(Par));

  if constexpr (L::kResident) {

    MatTile<S, N> x_l1;
    TileLeft<half, S, N> x_l0a;
    TASSIGN(x_l1, L::xSlot(Slot));
    TASSIGN(x_l0a, L::l0aX(Par ? 1u : 0u));
    if (waitA) wait_flag(PIPE_M, PIPE_MTE1, EvA);
    TMOV(x_l0a, x_l1);
    if (freeSlot) set_flag(PIPE_MTE1, PIPE_MTE2, EvX);

    TileRight<half, N, N> h_l0b;
    TASSIGN(h_l0b, L::kL0bH);

    if (waitAcc) wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
    Sync<PIPE_MTE1, PIPE_M>();
    TMATMUL(acc, x_l0a, h_l0b);
    if (freeA) set_flag(PIPE_M, PIPE_MTE1, EvA);
  } else if constexpr (L::kBlocked) {

    if (waitAcc) wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
    const unsigned accBase = L::l0cAcc(Par);
    for (uint32_t i = 0; i < L::kBlk; ++i) {
      if (i & 1u) {
        blockStep<N, S, Slot, true, EV_A1>(accBase, i);
      } else {
        blockStep<N, S, Slot, false, EV_A0>(accBase, i);
      }
    }
    if (freeSlot) set_flag(PIPE_MTE1, PIPE_MTE2, EvX);
  }

  Sync<PIPE_M, PIPE_FIX>();

  // The cube ran all S rows whatever `rows` says. The transform is per-sample,
  // so the surplus rows reach only output rows this store never writes.
  TileAcc<float, S, N, DYNAMIC> acc_out;
  TASSIGN(acc_out, L::l0cAcc(Par));
  acc_out.SetValidRow(rows);
  __gm__ half *y_tile = y + sample * N;
  GmTileDyn<S, N> y_gm(y_tile);
  TASSIGN(y_gm, y_tile);
  y_gm.template SetShape<GlobalTensorDim::DIM_3>(rows);
  TSTORE(y_gm, acc_out);
  if (freeAcc) set_flag(PIPE_FIX, PIPE_M, EV_ACC);
}

// A core's span as one ring: every tile at its natural position, every tile
// masked to min(S, rows left). A full tile is just where that min picks S, so
// there is no special last tile.
//
// NOTE: the ring does NOT drain itself. Anything placed after it that reuses
// L1, L0A or the accumulator must first hand the buffers over on FIX -> M,
// M -> MTE1 and MTE1 -> MTE2, or the cube faults with FIXP reading L0C while
// M overwrites it.
template <uint32_t N, uint32_t S>
AICORE void runRing(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                    uint32_t base, uint32_t end) {

  // Exactly one tile is ever short, so every row count below picks between
  // `rest` and the constant S.
  const uint32_t span = end - base;
  const uint32_t last = (span - 1) / S;
  const uint32_t rest = span - last * S;

  using L = Layout<N, S>;
  constexpr uint32_t kAcc = L::kAccSlots;

  // No primed credits, and none drained. A credit says a resource is free
  // again, which is only worth saying once someone has held it: tile j reuses
  // tile j-4's slot, so it waits iff j >= 4 and signals iff tile j+4 exists.
  // The pairing is exact, and tile j's load is issued an iteration after tile
  // j-4 was consumed, so a credit is always produced before it is awaited.
  //
  // The blocked body is the exception: its L0A credits rotate between the B
  // blocks within a tile, so the run's first two blocks have no predecessor.
  if constexpr (!L::kResident) {
    set_flag(PIPE_M, PIPE_MTE1, EV_A0);
    set_flag(PIPE_M, PIPE_MTE1, EV_A1);
  }

  issueTileLoad<N, S, 0, EV_X0>(x, base, last == 0 ? rest : S, false);
  if (last > 0)
    issueTileLoad<N, S, 1, EV_X1>(x, base + S, last == 1 ? rest : S, false);
  if (last > 1)
    issueTileLoad<N, S, 2, EV_X2>(x, base + 2 * S, last == 2 ? rest : S, false);

  for (uint32_t i = 0; i <= last; ++i) {
    const uint32_t cur = base + i * S;
    const uint32_t crow = (i == last) ? rest : S;
    const bool prefetch = (i + 3 <= last);
    const uint32_t next = cur + 3 * S;
    const uint32_t nrow = (i + 3 == last) ? rest : S;

    const bool pwait = (i >= 1);
    const bool fslot = (i + 4 <= last);
    const bool wacc = (i >= kAcc);
    const bool facc = (i + kAcc <= last);
    const bool wa = (i >= 2);
    const bool fa = (i + 2 <= last);

    switch (i & 3u) {
      case 0:
        if (prefetch) issueTileLoad<N, S, 3, EV_X3>(x, next, nrow, pwait);
        onesidedTile<N, S, 0, false, EV_X0, EV_A0>(h, y, cur, crow, fslot, wacc,
                                                   facc, wa, fa);
        break;
      case 1:
        if (prefetch) issueTileLoad<N, S, 0, EV_X0>(x, next, nrow, pwait);
        onesidedTile<N, S, 1, true, EV_X1, EV_A1>(h, y, cur, crow, fslot, wacc,
                                                  facc, wa, fa);
        break;
      case 2:
        if (prefetch) issueTileLoad<N, S, 1, EV_X1>(x, next, nrow, pwait);
        onesidedTile<N, S, 2, false, EV_X2, EV_A0>(h, y, cur, crow, fslot, wacc,
                                                   facc, wa, fa);
        break;
      default:
        if (prefetch) issueTileLoad<N, S, 2, EV_X2>(x, next, nrow, pwait);
        onesidedTile<N, S, 3, true, EV_X3, EV_A1>(h, y, cur, crow, fslot, wacc,
                                                  facc, wa, fa);
        break;
    }
  }

  if constexpr (!L::kResident) {
    wait_flag(PIPE_M, PIPE_MTE1, EV_A0);
    wait_flag(PIPE_M, PIPE_MTE1, EV_A1);
  }
}

// One tile, standalone. It has no neighbour to wait for or signal to, so
// every credit is skipped. Not runRing: that would instantiate its loop and
// all four slot/parity combinations at every size runShortSpan can reach.
template <uint32_t N, uint32_t S>
AICORE void oneTile(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                    uint32_t sample, uint32_t rows) {
  using L = Layout<N, S>;
  if constexpr (!L::kResident) {
    set_flag(PIPE_M, PIPE_MTE1, EV_A0);
    set_flag(PIPE_M, PIPE_MTE1, EV_A1);
  }
  issueTileLoad<N, S, 0, EV_X0>(x, sample, rows, false);
  onesidedTile<N, S, 0, false, EV_X0, EV_A0>(h, y, sample, rows, false, false,
                                             false, false, false);
  if constexpr (!L::kResident) {
    wait_flag(PIPE_M, PIPE_MTE1, EV_A0);
    wait_flag(PIPE_M, PIPE_MTE1, EV_A1);
  }
}

// A span of 1..S samples, as one tile of the smallest size that covers it.
// Masking is free on the transfers but not on the tile -- the cube runs all S
// rows regardless -- so a short span in a top-level tile wastes most of it.
// A compile-time choice, not a chain: exactly one tile runs. 16 is the floor,
// below which TMOV's TExtract rejects the tile.
template <uint32_t N, uint32_t S>
AICORE void runShortSpan(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                         uint32_t sample, uint32_t span) {
  if constexpr (S > 16) {
    if (span <= S / 2) {
      runShortSpan<N, S / 2>(x, h, y, sample, span);
      return;
    }
  }
  oneTile<N, S>(x, h, y, sample, span);
}

// Consume [sample, end) ascending, as one unit of work: a ring of S-sample
// tiles, or a single smaller tile when the span cannot fill half of one. The
// remainder stays INSIDE the ring, where it overlaps its predecessor; split
// off it would need the handoff above and overlap nothing.
template <uint32_t N, uint32_t S>
AICORE void runTiles(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                     uint32_t &sample, uint32_t end) {
  const uint32_t span = end - sample;
  if (span == 0) {
    return;
  }

  if constexpr (S > 16) {
    if (span <= S / 2) {
      runShortSpan<N, S / 2>(x, h, y, sample, span);
      sample = end;
      return;
    }
  }
  runRing<N, S>(x, h, y, sample, end);
  sample = end;
}

template <uint32_t N, uint32_t S>
AICORE void runOnesided(__gm__ half *x, __gm__ half *h, __gm__ half *y,
                        uint32_t batch) {
  static_assert(N % 16 == 0, "cube tiles need 16-aligned N");
  static_assert(S >= 1, "at least one sample per tile");

  const uint32_t num_cores = block_num;
  const uint32_t core_id = get_block_idx();

  // Split by TILES, so every core but the last gets a whole number of them
  // and only the truncated core ends on a short tile.
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

  if (batch == 0) {
    return;
  }
  switch (n) {
    FAST_HADAMARD_ONESIDED_CASES(FAST_HADAMARD_ONESIDED_DISPATCH_CASE)
    // Unsupported n: the host wrapper validates, so reaching here is a bug
    // upstream rather than something to handle.
    default:
      return;
  }
}

}

#endif

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

  fast_hadamard_onesided_fp16<<<blockDim, nullptr, stream>>>(x, y, h, batch, n);
}
