#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

// One-sided Hadamard transform on the cube unit: Y = X * H_N, as a single
// right-multiplication -- no transpose and no intermediate turned around
// through L1, unlike the two-sided split in cube/.
//
// H_N is not applied as one dense (N x N) operand. It is factored
// H_N = H_B (x) H_M and applied as B*B matmuls of a FIXED size M against a
// resident +-H_M, with the sum over input blocks accumulated in L0C -- the
// cube's version of the vector core's butterfly stages. See "BLOCKED
// ACCUMULATION" below for the derivation; it is what lets N=512 work at all.
//
// The companion kernel in cube/ uses the balanced Kronecker split
// H_N = H_R (x) H_C and pays two matmuls of about sqrt(N) on a side. This one
// spends sqrt(N)/2 times as many MACs -- 8x at N=256 -- to buy a tile body that
// touches each pipe ONCE:
//
//     GM --MTE2--> L1 --MTE1--> L0A --M--> L0C --FIX--> GM
//
// The two-sided kernel cannot: its tile has to turn the intermediate around
// through L1 between the two multiplies, so it visits MTE1 and M twice and
// consecutive tiles serialize behind the second visit. That serialization, not
// arithmetic, is what makes the two-sided kernel lose at small N -- measured
// there, a tile costs 340 ns on a core with a short run against 91 ns on a long
// one. On the cube a fp16 matmul is nearly free next to the data movement, so
// trading MACs for a straight pipe is the right side of the trade while N is
// small.
//
// WHAT USED TO CAP IT. Written as one dense product, the whole of H has to be
// read once per tile, so H's L1->L0B traffic per sample is N*N*2 / S bytes
// against 4*N bytes of HBM traffic for X and Y. With S bounded by L0C capacity
// (S*N fixed) that ratio grows as N^2: H is 32 KB at N=128 and sits in L0B for
// the whole kernel, 128 KB at N=256 and only fits L1, and 512 KB at N=512 --
// all of L1, so it streams from GM on every tile at 4x the HBM traffic. That
// last regime ran at 0.6-0.8x of the vector kernel and set the 512 ceiling.
//
// The blocked form below removes that term outright rather than managing it:
// H near the matmul becomes +-H_M at 16 KB for every N, loaded once per core.
// Against the dense form, paired, 3 passes, null control at 1.00x -- including
// the tile geometry and block_dim retune that the change licensed:
//
//     N=256   1.33-1.56x       N=512   1.64-2.55x
//
// N<=128 is untouched (0.98-1.02x): B == 1 there, which IS the single dense
// matmul. The result is that this kernel now beats the vector butterfly at
// every size it covers -- N=512 went from 0.72x of it to 1.50x -- and beats the
// two-sided kernel everywhere the two overlap, 1.11-2.44x.
//
// WHAT CAPS IT NOW is arithmetic, not capacity. The one-sided product is N*N
// MACs per sample however H is factored, and at N=512 that is already most of
// the time -- dropping the matmuls (PTO_OS_NOMM) is 3.96x faster at batch
// 65536, against 2.34x at N=256. Past 512 the MACs grow 4x per doubling while
// the bytes only double, so the two-sided kernel in cube/ -- 2*N*sqrt(N) MACs
// -- takes over. PTO_OS_BIG compiles 1024 and 2048, which the blocked form is
// correct at and the dense form could never reach (H would be 2 MB and 8 MB);
// measured there the two-sided kernel is 1.7x faster at 1024 and 4.8x at 2048,
// so they stay off. 512 is now a compute crossover, not a capacity limit.
//
// THAT IS ONLY TRUE AT THE TOP OF THE RANGE. Measured per size with
// PTO_OS_NOMM (device time, batch 1024 / 4096, each at its own best block_dim),
// the matmuls are:
//
//     N=64   9.0% / 9.3%      N=256  22.7% / 23.2%
//     N=128 12.2% / 17.1%     N=512  29.1% / 48.8%
//
// so this kernel is 51-91% MOVEMENT, and only at N=512 with a large batch is
// arithmetic even half of it. That matters because its GM legs cannot be made
// conversion-free (see issueTileLoad), so the movement it is spending most of
// its time on is movement it cannot reduce the way cube/ can.
// H is symmetric, so there is no transpose to worry about: Y = X * H with H as
// the right operand gives the natural (Sylvester) ordered transform, and the
// host builds H once.
//
// Accumulation is fp32 and the result is rounded to fp16 exactly ONCE -- the
// single matmul is also the most accurate of the three kernels here, against
// twice for the two-sided split and log2(N) times for the vector butterfly.
//
// (N, samples per tile)

// Samples per tile, as S * N, for the sizes that do not override it below
// (N <= 128, where B == 1). 16384 keeps two L0C accumulators (S*N*4 each)
// inside 128 KB and two L0A slots inside 64 KB.
//
// Raising it halves H's per-sample traffic and lowering it buys shallower
// buffers. Measured against 8192 while the dense form was shipping: 16384 won
// by 1.31-1.94x at N=256/512, where H traffic dominated, and lost by at most
// 1.12x at N<=128. The blocked form makes H traffic zero, so that argument now
// only applies to N <= 128 -- which is why 256 and 512 override it.
#ifndef PTO_OS_TILE_ELEMS
#define PTO_OS_TILE_ELEMS 16384
#endif
// Per-N override of samples per tile. N <= 128 holds all of H in L0B and is
// capped by L0A, which must hold two (S x N) slots, so it takes
// PTO_OS_TILE_ELEMS / N. N >= 256 runs the blocked form, whose L0A slot is only
// (S x M), and it wants a SMALLER tile -- see PTO_OS_S512.
#ifndef PTO_OS_S64
#define PTO_OS_S64 (PTO_OS_TILE_ELEMS / 64)
#endif
#ifndef PTO_OS_S128
#define PTO_OS_S128 (PTO_OS_TILE_ELEMS / 128)
#endif
#ifndef PTO_OS_S256
#define PTO_OS_S256 32
#endif
// 16 and 32, not TILE_ELEMS/N = 32 and 64.
//
// These two inverted when the blocked form landed, and the reason is worth
// keeping: the tall tile used to be justified SOLELY by halving H's per-tile
// L1->L0B traffic, which the blocked form makes zero. With that gone the
// smaller tile's deeper buffering wins instead -- measured over S512 in
// {16,32,64} x S256 in {16,32,64,128} x M in {32,64,128}, paired:
//
//     N=512   S=64 -> 16   1.24x at batch 65536, 1.31x at 8192
//     N=256   S=64 -> 32   1.07x at batch 8192, 1.00x at 65536, 0.95x at 1024
//
// S256=16 is a cliff, not a continuation (40.8 us against 27.1 at batch
// 65536): below 32 the tile stops covering the fan-out over B output blocks.
#ifndef PTO_OS_S512
#define PTO_OS_S512 16
#endif

// Split the per-core work by whole tiles rather than by samples, so at most one
// core is left with a count that is not a multiple of S and has to walk the
// halving chain (and then the backed-up 16-sample tail) for a fraction of a
// tile's work at a full tile's handshake cost.
#ifndef PTO_OS_TILE_SPLIT
#define PTO_OS_TILE_SPLIT 1
#endif
// Diagnostic only: drop the matmuls, keeping every transfer and handshake, to
// measure the data-movement floor of this tile geometry. Output is wrong.
#ifndef PTO_OS_NOMM
#define PTO_OS_NOMM 0
#endif

// BLOCKED ACCUMULATION -- the vector butterfly's staging, done on the cube.
//
// Everywhere H does not already fit L0B, replace the single dense (N x N) right
// operand by a Kronecker factorization and let the L0C accumulator do the
// butterfly's add/subtract. With N = B*M and H_N = H_B (x) H_M, blocking the
// product by M columns gives
//
//     Y[:, jM:(j+1)M] = sum_i  X[:, iM:(i+1)M] * (H_B[i,j] * H_M)
//
// and H_B[i,j] is +-1, so the ONLY right operands that ever occur are +H_M and
// -H_M. That is the whole idea: the matmul is restricted to a fixed efficient
// size M, the sum over i is accumulated in L0C by TMATMUL_ACC exactly as the
// vector core accumulates its log2(N) butterfly stages, and H near the matmul
// shrinks from N*N to 2*M*M -- which is resident in L0B for every N, so the per
// tile H traffic that caps the dense form is exactly zero. (Zero, not merely
// less: the two operands are loaded once per core.)
//
// Same MAC count as the dense form, B*B matmuls instead of one, same single
// fp32 rounding, and still one pass over each pipe per block with no L1
// turnaround. H_B is never materialized: H_B[i,j] = (-1)^popcount(i & j).
#ifndef PTO_OS_BLOCKED
#define PTO_OS_BLOCKED 1
#endif
// M above, and it MUST match _BLOCK in the host wrapper, which builds +-H_M --
// a mismatch just hands the kernel a wrong-sized operand and it computes
// garbage. 128 is the largest that fits L0B's 64 KB, and the argument for it is
// that B = N/M matmuls shrink as B*B, so the largest M issues the fewest.
//
// That argument is WRONG, and measurably so. Swept paired at the shipped tile
// geometry, M=64 against M=128 and M=32:
//
//     N=512 batch 65536    M=32 300.5us   M=64 99.9us   M=128 123.8us
//     N=256 batch 65536    M=32  40.8us   M=64 27.2us   M=128  27.8us
//
// M=64 is 3.01x over M=32 and 1.24x over M=128 at N=512. Fewer, larger matmuls
// is the wrong objective here: what matters is that the (S x M) L0A slice stays
// small enough to double-buffer deeply, and 64 is where that balances. Resweep
// this if the tile geometry changes -- it did not survive a geometry change
// once already.
#ifndef PTO_OS_BLOCK
#define PTO_OS_BLOCK 64
#endif

// Sizes above 512, which only the blocked form can reach at all -- the dense
// form's H would be 2 MB. Off by default: the capacity ceiling is gone but an
// arithmetic one takes its place (the one-sided product is N*N MACs per sample
// whatever the form), so past 512 the two-sided kernel in cube/ wins. On for
// the probe that measured where that crossover is; see the README.
#ifndef PTO_OS_BIG
#define PTO_OS_BIG 0
#endif
#ifndef PTO_OS_S1024
#define PTO_OS_S1024 16
#endif
#ifndef PTO_OS_S2048
#define PTO_OS_S2048 16
#endif
#if PTO_OS_BIG
#define FAST_HADAMARD_ONESIDED_BIG_CASES(X) \
  X(1024, PTO_OS_S1024)                     \
  X(2048, PTO_OS_S2048)
#else
#define FAST_HADAMARD_ONESIDED_BIG_CASES(X)
#endif

#define FAST_HADAMARD_ONESIDED_CASES(X) \
  X(64, PTO_OS_S64)                     \
  X(128, PTO_OS_S128)                   \
  X(256, PTO_OS_S256)                   \
  X(512, PTO_OS_S512)                   \
  FAST_HADAMARD_ONESIDED_BIG_CASES(X)

#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

namespace {

constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L0AB_BYTES = 64 * 1024;
constexpr uint32_t L0C_BYTES = 128 * 1024;

// Ids are unique only per (src pipe, dst pipe) pair, so the X ring (MTE2<->MTE1)
// and the H chunk ring (MTE1->M) may reuse numbers. Macros, not constexpr:
// bisheng accepts only a literal EVENT_ID* token or a template non-type
// argument as the third argument of set_flag / wait_flag.
#define EV_X0 EVENT_ID0  // L1 X ring slot 0
#define EV_X1 EVENT_ID1  // L1 X ring slot 1
#define EV_X2 EVENT_ID2  // L1 X ring slot 2
#define EV_X3 EVENT_ID3  // L1 X ring slot 3
// L0A (and, when the contraction is split, the paired L0B) slot is free again:
// M -> MTE1, i.e. the consumer telling the producer it may overwrite.
//
// This edge is NOT optional here, and its absence is a race that moves with the
// shape. The two-sided kernel gets the same protection for free: its tile body
// turns around through L1, so it has a FIX -> MTE1 edge that stalls MTE1 inside
// every tile. This body deliberately has no such edge -- each pipe is touched
// once -- so nothing holds MTE1 back and it runs ahead by slots, letting tile
// i+2's TMOV land on the L0A slot tile i's TMATMUL is still reading.
#define EV_A0 EVENT_ID4  // (M->MTE1)    L0A/L0B slot 0 free
#define EV_A1 EVENT_ID5  // (M->MTE1)    L0A/L0B slot 1 free
#define EV_ACC EVENT_ID6  // (FIX->M)     L0C accumulator free

// H's L1 staging ring, used in BOTH directions: MTE2->MTE1 when a chunk has
// landed, MTE1->MTE2 when its slot may be overwritten. Reusing the numbers of
// EV_A0/EV_A1 is safe because those sit on (M->MTE1), a different pipe pair.
//
// What is NOT safe, and was a bug here: staging a chunk with the generic
// Sync<PIPE_MTE2, PIPE_MTE1>() helper, whose id 0 collides with EV_X0 on that
// very pair. Only the streamed-H case runs that Sync inside the tile loop, so
// it corrupted the X ring's credit counts at N=512 and nowhere else.
#define EV_H0 EVENT_ID4  // (MTE2<->MTE1) H L1 ring slot 0
#define EV_H1 EVENT_ID5  // (MTE2<->MTE1) H L1 ring slot 1

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
  static constexpr bool kResident = (kHBytes * 2 <= L0AB_BYTES);  // N <= 128

  // Otherwise: blocked accumulation against a resident +-H_M (see PTO_OS_BLOCK),
  // or, with that switched off, the dense product split into contraction chunks
  // of the largest Kc whose (Kc x N) slice fits half of L0B.
  static constexpr uint32_t kM = (N < PTO_OS_BLOCK) ? N : (uint32_t)PTO_OS_BLOCK;
  static constexpr uint32_t kBlk = N / kM;  // B, the number of M-column blocks
  static constexpr bool kBlocked =
      (PTO_OS_BLOCKED != 0) && !kResident && (kBlk > 1);
  static_assert(!kBlocked || kBlk * kM == N, "M must divide N");

  // Contraction chunk, which both non-resident forms walk: the blocked form's
  // is one M-column block of X, the chunked form's is a Kc-row slice of H. Kc
  // stays a multiple of 16 -- it indexes whole fractal columns.
  static constexpr uint32_t kKc =
      kResident ? N
                : (kBlocked ? kM : (L0AB_BYTES / 2) / (N * sizeof(half)));
  static constexpr uint32_t kChunks = N / kKc;
  static_assert(kKc * kChunks == N && kKc % 16 == 0, "Kc must tile N by 16s");

  // H placement. Resident in L0B when there is only one chunk: loaded once per
  // core and never touched again, which is the whole point at small N.
  static constexpr bool kHInL0B = (kChunks == 1);
  // The blocked form is resident too -- +-H_M, two M x M tiles, loaded once per
  // core. Otherwise H is resident in L1 if it fits beside the X ring, else
  // streamed from GM into a two-slot L1 ring on every tile.
  static constexpr unsigned kXSlots = 4;
  static constexpr bool kHInL1 =
      !kHInL0B && !kBlocked && (kHBytes + kXSlots * kTileBytes <= L1_BYTES);

  // L1: H (all of it, +-H_M, or a 2-slot chunk ring), then the X ring.
  static constexpr unsigned kL1H = 0;
  static constexpr unsigned kL1HChunk =
      kBlocked ? kM * kM * sizeof(half) : kKc * N * sizeof(half);
  static constexpr unsigned kL1HBytes =
      kHInL0B ? kHBytes : (kHInL1 ? kHBytes : 2 * kL1HChunk);
  static constexpr unsigned kL1X = kL1H + kL1HBytes;
  static_assert(kL1X + kXSlots * kTileBytes <= L1_BYTES, "L1 overflow");

  static constexpr unsigned xSlot(uint32_t slot) {
    return kL1X + slot * kTileBytes;
  }
  // Row chunk kc of H inside L1. Resident: H is (N x N) row-major in NZ, and a
  // row chunk is Kc/16 fractal ROWS -- not contiguous unless kKc == N. So the
  // resident L1 copy is staged as kChunks separate (Kc x N) tiles instead, which
  // the host hands over in exactly that order.
  static constexpr unsigned l1HChunk(uint32_t kc) {
    return kL1H + (kHInL1 ? kc : (kc & 1u)) * kL1HChunk;
  }

  // L0A: one contraction chunk of X, two slots. With a single chunk that is the
  // whole (S x N) tile and the slots alternate per TILE; when the contraction is
  // split it is (S x Kc) and they alternate per CHUNK.
  static constexpr unsigned kL0aX = 0;
  static constexpr unsigned kL0aChunk = S * kKc * sizeof(half);
  // Two slots everywhere they fit. A tall tile at a chunked size runs out of
  // L0C before L0A, so kSlots can be 1 with the L0A ring still doubled; both
  // are derived, so a tile size that does not fit simply gets shallower buffers
  // instead of failing to compile.
  static constexpr unsigned kAccSlots = (2 * kAccBytes <= L0C_BYTES) ? 2u : 1u;
  static constexpr unsigned kL0aSlots = (2 * kL0aChunk <= L0AB_BYTES) ? 2u : 1u;
  static_assert(kL0aChunk <= L0AB_BYTES, "L0A overflow");
  static constexpr unsigned l0aX(uint32_t i) {
    return kL0aX + (kL0aSlots == 2u ? (i & 1u) : 0u) * kL0aChunk;
  }
  // Columns [kc*Kc, (kc+1)*Kc) of the L1 X tile. An L1 Mat tile is
  // block-column-major, so whole 16-column groups are one contiguous range --
  // this is the slice the L0A Left tile canNOT do, because a Left tile is
  // fractal-ROW-major and the same columns are strided there.
  static constexpr unsigned xChunk(uint32_t slot, uint32_t kc) {
    return xSlot(slot) + kc * kKc * S * sizeof(half);
  }

  // L0B: H, resident in full, as the resident +-H_M pair, or a two-slot chunk
  // ring. The blocked pair and the chunk ring index the same way -- slot 0/1 --
  // so l0bH serves both; for the pair the index is the SIGN, not a chunk.
  static constexpr unsigned kL0bH = 0;
  static constexpr unsigned kL0bChunk =
      kBlocked ? kM * kM * sizeof(half) : kKc * N * sizeof(half);
  static_assert(kHInL0B ? (kHBytes <= L0AB_BYTES)
                        : (2 * kL0bChunk <= L0AB_BYTES),
                "L0B overflow");
  static constexpr unsigned l0bH(uint32_t kc) {
    return kL0bH + (kHInL0B ? 0 : (kc & 1u) * kL0bChunk);
  }

  // L0C: the accumulator, two slots where they fit.
  static_assert(kAccBytes <= L0C_BYTES, "L0C overflow");
  static constexpr unsigned l0cAcc(bool par) {
    return (kAccSlots == 2u && par) ? kAccBytes : 0;
  }
  // Output block j: columns [j*M, (j+1)*M) of the accumulator. An L0C tile is
  // block-column-major like an L1 Mat tile, so whole 16-column groups are one
  // contiguous range and each block is an (S x M) accumulator in its own right.
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

// GM -> L1, and for the resident case on to L0B. Done once per core.
//
// h holds kChunks consecutive (Kc x N) row chunks of H, which for a row-major
// (N x N) matrix is just H itself; the host lays it out that way so each chunk
// is one contiguous descriptor whatever Kc is. The blocked form's +H_M and
// -H_M follow it, two (M x M) matrices at offset N*N -- appended rather than
// substituted so that one host buffer feeds either form of the kernel and the
// A/B between them is a compile-time switch and nothing else.
template <uint32_t N, uint32_t S>
AICORE void loadHadamard(__gm__ half *h) {
  using L = Layout<N, S>;
  if constexpr (L::kBlocked) {
    for (uint32_t s = 0; s < 2; ++s) {  // s = 0 is +H_M, s = 1 is -H_M
      MatTile<L::kM, L::kM> h_l1;
      TASSIGN(h_l1, L::l1HChunk(s));
      __gm__ half *src = h + N * N + s * L::kM * L::kM;
      GmTile<L::kM, L::kM> h_gm(src);
      TASSIGN(h_gm, src);
      TLOAD(h_l1, h_gm);
    }
    Sync<PIPE_MTE2, PIPE_MTE1>();
    for (uint32_t s = 0; s < 2; ++s) {
      MatTile<L::kM, L::kM> h_l1;
      TileRight<half, L::kM, L::kM> h_l0b;
      TASSIGN(h_l1, L::l1HChunk(s));
      TASSIGN(h_l0b, L::l0bH(s));
      TMOV(h_l0b, h_l1);
    }
    Sync<PIPE_MTE1, PIPE_M>();
  } else if constexpr (L::kHInL0B) {
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
  } else if constexpr (L::kHInL1) {
    for (uint32_t kc = 0; kc < L::kChunks; ++kc) {
      MatTile<L::kKc, N> h_l1;
      TASSIGN(h_l1, L::l1HChunk(kc));
      __gm__ half *src = h + kc * L::kKc * N;
      GmTile<L::kKc, N> h_gm(src);
      TASSIGN(h_gm, src);
      TLOAD(h_l1, h_gm);
    }
    Sync<PIPE_MTE2, PIPE_MTE1>();
  }
  // kHInL1 == false and kHInL0B == false: nothing resident, every tile streams.
}

// GM -> L1 X slot, signalling MTE2->MTE1. The caller already holds the slot.
template <uint32_t N, uint32_t S, uint32_t Slot, event_t EvX>
AICORE void issueTileLoad(__gm__ half *x, uint32_t sample) {
  using L = Layout<N, S>;

  wait_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // slot free

  MatTile<S, N> x_l1;
  TASSIGN(x_l1, L::xSlot(Slot));
  __gm__ half *x_tile = x + sample * N;

  // The descriptor row here is N * sizeof(half) bytes, 128 at the smallest N in
  // the table, so this is a CONVERTING ND->NZ load and it would be worth
  // avoiding -- this kernel is 51-91% movement (see the header). It cannot be
  // avoided, and the proof is a counting argument rather than anything about
  // Hadamard matrices.
  //
  // Write the GM byte index of a tile as p = [ sample | i ]. Whatever operands
  // we precompute, each matmul applies them to one FIELD of p, and which fields
  // exist is fixed by the layout:
  //
  //   conversion-free NZ:  right operand owns p[0,4) u p[a+4,t)
  //                        left  operand owns p[4, a+4)
  //   conversion-free ND:  the mirror image -- TMOV from an ND L1 tile reads
  //                        the bytes as fractals AND TRANSPOSES (measured), so
  //                        the left operand owns p[0,4) u the top bits and the
  //                        right operand owns the middle.
  //
  // This kernel is ONE-SIDED: every operand, dense or blocked, is a RIGHT
  // operand. So exactly one of those two fields is ever covered, and the other
  // one keeps whichever transform bits it holds -- p[4, a+4) under NZ, p[0,4)
  // under ND. Both are non-empty for every N here (A >= 16 forces a >= 4, and
  // log2(N) >= 6 > 4), so some transform bit is never multiplied at all.
  //
  // Covering both fields means a left multiply AND a right multiply, which is
  // the two-sided kernel in cube/ -- and that kernel, once it had
  // conversion-free legs, duly overtook this one at N=512 from batch 256 up.
  // So "one-sided" and "conversion-free GM transfers" are mutually exclusive by
  // construction, not by accident of the layouts.
  GmTile<S, N> x_gm(x_tile);
  TASSIGN(x_gm, x_tile);
  TLOAD(x_l1, x_gm);
  set_flag(PIPE_MTE2, PIPE_MTE1, EvX);
}

// Y[sample .. sample+S) = X * H, for a tile already loaded into X slot Slot.
//
// One pass over the contraction: each pipe is visited once for X and once per H
// chunk, and there is no L1 turnaround, so the next tile's MTE1 is never stuck
// behind this tile's matmul.
// One contraction chunk: X column-slice and H row-chunk to L0A/L0B, then a
// matmul into the running accumulator. Slot parity is a template argument
// because the flag ids must be literal.
template <uint32_t N, uint32_t S, uint32_t XSlot, bool CPar, event_t EvA,
          event_t EvH, typename AccT>
AICORE void chunkStep(__gm__ half *h, AccT &acc, uint32_t kc,
                      uint32_t chunks) {
  using L = Layout<N, S>;

  // X: slice the L1 tile by columns, then move just that chunk to L0A. An L1
  // Mat tile is block-column-major, so whole 16-column groups are contiguous --
  // this is the slice an L0A Left tile canNOT do, being fractal-ROW-major.
  MatTile<S, L::kKc> x_l1;
  TileLeft<half, S, L::kKc> x_l0a;
  TASSIGN(x_l1, L::xChunk(XSlot, kc));
  TASSIGN(x_l0a, L::l0aX(CPar ? 1u : 0u));

  // H: row chunk kc, contiguous in GM and in L1.
  MatTile<L::kKc, N> h_l1;
  TileRight<half, L::kKc, N> h_l0b;
  TASSIGN(h_l1, L::l1HChunk(kc));
  TASSIGN(h_l0b, L::l0bH(CPar ? 1u : 0u));

  if constexpr (!L::kHInL1) {
    // Nothing resident: stage this chunk GM -> L1 first. H is the same every
    // tile, so these reads are served by L2, not HBM.
    __gm__ half *src = h + kc * L::kKc * N;
    GmTile<L::kKc, N> h_gm(src);
    TASSIGN(h_gm, src);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EvH);  // chunk kc-2 has read this slot
    TLOAD(h_l1, h_gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, EvH);
  }
  wait_flag(PIPE_M, PIPE_MTE1, EvA);  // chunk kc-2 has finished with the slot
  if constexpr (!L::kHInL1) {
    wait_flag(PIPE_MTE2, PIPE_MTE1, EvH);  // this chunk's H has landed
  }
  TMOV(x_l0a, x_l1);
  TMOV(h_l0b, h_l1);
  if constexpr (!L::kHInL1) {
    set_flag(PIPE_MTE1, PIPE_MTE2, EvH);  // H L1 slot free for chunk kc+2
  }
  // A full barrier, so chunk kc+1's TMOVs do not start under chunk kc's matmul:
  // the slot alternation above is allocated but NOT exploited. That is
  // deliberate, and measured rather than assumed -- a PTO_OS_NOMM probe puts the
  // matmuls at 0% of N=512's time and 7-8% of N=256's, so these sizes are purely
  // movement-bound and overlapping M under MTE1 has nothing to win. Splitting
  // this into a distance-1 pair would only pay if that ever changes.
  Sync<PIPE_MTE1, PIPE_M>();
#if !PTO_OS_NOMM
  if (kc == 0) {
    TMATMUL(acc, x_l0a, h_l0b);
  } else {
    TMATMUL_ACC(acc, x_l0a, h_l0b);
  }
#endif
  set_flag(PIPE_M, PIPE_MTE1, EvA);  // slot free for chunk kc+2
}

// One input block of the blocked form: move X's M-column block i to L0A, then
// fan it out over all B output blocks, each against +H_M or -H_M.
//
// This is the cube's version of one butterfly stage. The vector core walks
// log2(N) stages, reading and rewriting the whole sample each time; here the B
// partial products that make output block j land in the SAME L0C accumulator
// and TMATMUL_ACC sums them in place, so the combination costs no traffic at
// all -- block i is read from L1 once and never written back.
template <uint32_t N, uint32_t S, uint32_t XSlot, bool APar, event_t EvA>
AICORE void blockStep(unsigned accBase, uint32_t i) {
  using L = Layout<N, S>;

  // Columns [i*M, (i+1)*M) of the L1 X tile. An L1 Mat tile is
  // block-column-major, so whole 16-column groups are one contiguous range --
  // the slice an L0A Left tile canNOT do, being fractal-ROW-major.
  MatTile<S, L::kM> x_l1;
  TileLeft<half, S, L::kM> x_l0a;
  TASSIGN(x_l1, L::xChunk(XSlot, i));
  TASSIGN(x_l0a, L::l0aX(APar ? 1u : 0u));

  wait_flag(PIPE_M, PIPE_MTE1, EvA);  // block i-2 has finished with the slot
  TMOV(x_l0a, x_l1);
  Sync<PIPE_MTE1, PIPE_M>();

#if !PTO_OS_NOMM
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
#endif
  set_flag(PIPE_M, PIPE_MTE1, EvA);  // slot free for block i+2
}

template <uint32_t N, uint32_t S, uint32_t Slot, bool Par, event_t EvX,
          event_t EvA>
AICORE void onesidedTile(__gm__ half *h, __gm__ half *y, uint32_t sample) {
  using L = Layout<N, S>;

  wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);  // this tile's load has landed

  TileAcc<float, S, N> acc;
  TASSIGN(acc, L::l0cAcc(Par));

  if constexpr (L::kHInL0B) {
    // The whole of H is already in L0B and stays there: one matmul, nothing to
    // move, and MTE1 is free for the next tile the moment the TMOV below lands.
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
#if !PTO_OS_NOMM
    TMATMUL(acc, x_l0a, h_l0b);
#endif
    set_flag(PIPE_M, PIPE_MTE1, EvA);  // slot free for tile i+2
  } else if constexpr (L::kBlocked) {
    // +-H_M is already in L0B and stays there, so a block costs one TMOV and B
    // matmuls, and the tile never touches MTE2 for H at all.
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
  } else {
    wait_flag(PIPE_FIX, PIPE_M, EV_ACC);  // the slot's previous store drained
    for (uint32_t kc = 0; kc < L::kChunks; ++kc) {
      if (kc & 1u) {
        chunkStep<N, S, Slot, true, EV_A1, EV_H1>(h, acc, kc, L::kChunks);
      } else {
        chunkStep<N, S, Slot, false, EV_A0, EV_H0>(h, acc, kc, L::kChunks);
      }
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // last chunk has read the X slot
  }
  Sync<PIPE_M, PIPE_FIX>();

  __gm__ half *y_tile = y + sample * N;
  GmTile<S, N> y_gm(y_tile);
  TASSIGN(y_gm, y_tile);
  TSTORE(y_gm, acc);
  set_flag(PIPE_FIX, PIPE_M, EV_ACC);
}

// Consume [sample, end) with the largest power-of-two tile that still fits,
// halving on the way down.
//
// Unlike the two-sided kernel there is no single-sample fallback: a tile's Rows
// IS S here (the two-sided tile's rows are S*R, 16-aligned for free), and PTO
// requires Rows to be 1 or a multiple of 16 -- while TMOV's TExtract rejects
// even Rows == 1. So 16 is the smallest tile, and the last 1..15 samples are
// covered by backing one more 16-tile up to END instead. That recomputes up to
// 15 samples, which is free of consequence: the transform is per-sample, so the
// overlapping store writes byte-identical values, whether the overlap falls in
// this core's range or the previous core's.
//
// It does mean the kernel needs batch >= 16; the host wrapper pads below that.
//
// The X ring is four deep so MTE2 stays two tiles ahead of MTE1. The tile body
// touches each pipe once, so no further interleaving is needed to keep them
// overlapped -- which is the whole advantage over the two-sided tile.
// Tile i starts at base + i*stride, except a tail tile, which ends at `end`.
AICORE inline uint32_t tileStart(uint32_t base, uint32_t stride,
                                 uint32_t n_full, uint32_t end, uint32_t i) {
  return (i < n_full) ? (base + i * stride) : (end - stride);
}

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
    if constexpr (!Layout<N, S>::kHInL0B && !Layout<N, S>::kHInL1 &&
                  !Layout<N, S>::kBlocked) {
      set_flag(PIPE_MTE1, PIPE_MTE2, EV_H0);  // both H L1 slots start free
      set_flag(PIPE_MTE1, PIPE_MTE2, EV_H1);
    }

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
    if constexpr (!Layout<N, S>::kHInL0B && !Layout<N, S>::kHInL1 &&
                  !Layout<N, S>::kBlocked) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EV_H0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EV_H1);
    }
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

  // Split by TILES, not by samples: an even split of samples leaves most cores
  // with a count that is not a multiple of S, and each then walks the
  // power-of-two fallback down to single-sample tiles, which cost the same
  // handshakes as a full tile for a fraction of the work.
#if PTO_OS_TILE_SPLIT
  const uint32_t tiles_per_core = DIV_ROUNDUP(DIV_ROUNDUP(batch, S), num_cores);
  const uint32_t begin = tiles_per_core * S * core_id;
#else
  const uint32_t begin = DIV_ROUNDUP(batch, num_cores) * core_id;
#endif
  if (begin >= batch) {
    return;
  }
#if PTO_OS_TILE_SPLIT
  uint32_t end = begin + tiles_per_core * S;
#else
  uint32_t end = begin + DIV_ROUNDUP(batch, num_cores);
#endif
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
