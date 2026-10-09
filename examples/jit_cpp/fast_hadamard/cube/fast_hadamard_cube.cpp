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
// Take the SAME contiguous burst at C > 16, where ND and NZ are no longer the
// same bytes. It still works, for a different reason: NZ position is a
// PERMUTATION OF THE BIT FIELDS of ND position,
//
//   ND:  [ row>>4 | row&15 | col>>4 | col&15 ]
//   NZ:  [ col>>4 | row>>4 | row&15 | col&15 ]
//
// and a Sylvester Hadamard is invariant under any bit permutation of its
// index, because H[i][j] = (-1)^popcount(i & j) and popcount is blind to which
// bit positions the set bits sit in. So feeding the cube the ND bytes read as
// NZ computes the transform of a permuted input, which IS the permuted
// transform -- and reading the accumulator back out through the same NZ view
// undoes the permutation exactly. Load and store must therefore flip TOGETHER:
// either alone emits a permuted result.
//
// The one restriction is that the permutation lifts col>>4 above the row
// index, which for a tile of S stacked samples lifts it above the SAMPLE index
// too -- so it would mix samples unless the tile holds one. Hence S == 1,
// which is the top of the size table (N >= 8192) and exactly where the
// transfers are largest.
#ifndef PTO_HAD_NZGM
#define PTO_HAD_NZGM 1
#endif
// DIAGNOSTIC, COMPUTES WRONG ANSWERS. Drops the S == 1 guard above so the
// contiguous NZ descriptor is used at every size. The results are permuted
// across samples and meaningless, exactly like PTO_HAD_NOMM's; the point is to
// price the GM leg at the NARROW tiles (C = 32 and 64), which are the ones a
// restructure would have to reach and the only ones where the ND descriptor row
// is still small enough to hurt. Never ship this on.
#ifndef PTO_HAD_NZUNSAFE
#define PTO_HAD_NZUNSAFE 0
#endif
// Split the per-core work by whole tiles rather than by samples, so at most
// one core walks the power-of-two fallback down to single-sample tiles.
// Run the tile as two interleaved phases instead of one straight-line body. The straight-line body visits MTE1 -> M -> FIX -> MTE1 -> M -> FIX, and
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
//
// Worth 1.17-1.29x at N=256 and 1.15-1.70x at N=512..8192, measured paired
// against this switch off, with null controls at 1.00x.
#ifndef PTO_HAD_DBUF
#define PTO_HAD_DBUF 1
#endif
// Run the GENERAL (C != 16) tile as two phases as well.
//
// The phase split was written for C == 16 and gated on it, which read like a
// dependency on the free NZ reinterpret that case enjoys. It is not: the
// reinterpret is what makes the C == 16 TURNAROUND cheap, while the split is
// about pipe ORDER and needs nothing but a cut at the turnaround and two of
// every intermediate. The general body already pays its reshape (S TEXTRACTs
// plus S TSTOREs) in both forms.
//
// So the general body is the case that should gain MORE, not less: it visits
// each pipe twice like the C == 16 body, and has far more work queued behind
// the second visit. All four of kDbuf's capacity conditions already hold for
// every size from 512 to 8192; only N=16384 fails them, its two tile slots
// being 100% of L0A and L0B on their own.
#ifndef PTO_HAD_DBUF_GEN
#define PTO_HAD_DBUF_GEN 1
#endif
// Apply each stage's factor as H_f = H_acc (x) H_mat with H_mat of size M, so
// the matmul is M-sized and the sum over blocks runs in the L0C accumulator --
// the same trick the one-sided kernel in onesided/ uses, applied to each of the
// two one-sided stages this kernel is made of.
//
// The MAC count is unchanged: B*B matmuls of (rows x M) by (M x M) is the same
// rows*f*f as one dense product, whatever M is. What changes is footprint, and
// the term that matters is the SLOT, not the operand:
//
//     phase A's X slice in L0A    2*S*R*C  ->  2*S*R*M_b
//     phase B's T slice in L0B    2*S*R*C  ->  2*M_a*S*C
//     phase B's operand in L0A    2*R*R    ->  4*M_a*M_a
//     phase A's operand in L0B    2*C*C    ->  4*M_b*M_b
//
// At N=16384 that takes the deep path's L0A and L0B from 96 KB each -- over the
// 64 KB budget, which is why it was locked out -- to 48 KB at M=64. It does NOT
// shrink the accumulators, so PTO_HAD_ACCALIAS is needed with it.
//
// M must divide its factor and be at least 16, the cube's fractal size, so
// N=256 and N=512 cannot be blocked at all: their factors are already 16.
// 0 means dense everywhere.
#ifndef PTO_HAD_BLOCK
#define PTO_HAD_BLOCK 64
#endif
// Block every size that CAN be blocked, not just the ones that must be. Off by
// default, and the default is not a guess: blocking a single phase into two
// blocks costs 1.78-1.86x on the deep path at N=8192 (measured against the
// dense deep body at bd 16/20/24, with two null controls reading 0.996-1.008x),
// because the blocked body walks MTE1 -> M -> MTE1 -> M and so uses each pipe
// kC1 times per phase -- re-serializing consecutive tiles, which is the exact
// thing the two-phase interleave exists to prevent. On for A/B only.
#ifndef PTO_HAD_BLOCK_ALWAYS
#define PTO_HAD_BLOCK_ALWAYS 0
#endif
// Let the concatenated accumulator share the stacked one's L0C space. It is
// dead the moment phase A's TEXTRACTs have run, and the FIX -> MTE1 -> M chain
// between the phases already orders that, so WITHIN a tile the reuse is free.
// ACROSS tiles it is not: with the phases interleaved, A(i+2) writes the slot
// B(i) is still storing from, so the deep path needs one extra FIX -> M edge
// per slot -- which is exactly the EV_ACC edge the deep path had dropped.
//
// Halves L0C, which is what lets N=16384 hold two tiles in flight (256 -> 128
// KB) and what makes N=32768 representable at all.
#ifndef PTO_HAD_ACCALIAS
#define PTO_HAD_ACCALIAS 1
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
  X(16384, 128, 128, 1)             \
  X(32768, 128, 256, 1)

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
// Shallow blocked bodies. Every id below is distinct WITHIN its source pipe,
// which is the invariant that matters: two channels out of the same pipe under
// one id let the wrong waiter take a credit -- see the README section on it.
// EV_SWB shares id 3 with EV_ACC, but EV_ACC's source is FIX and this one's is
// M, so they are different channels either way the hardware keys them.
#define EV_SWB  EVENT_ID3  // (M  -> MTE1) left multiply, slice free
#define EV_SWA0 EVENT_ID4  // (M  -> MTE1) right multiply, slot 0 free
#define EV_SWA1 EVENT_ID5  // (M  -> MTE1) right multiply, slot 1 free
#define EV_SRA0 EVENT_ID6  // (MTE1 -> M)  right multiply, slot 0 staged
#define EV_SRA1 EVENT_ID7  // (MTE1 -> M)  right multiply, slot 1 staged

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

// NZ view of a tile: Shape<1, Cols/16, Rows/16, 16, C0> with the canonical
// strides, i.e. block-column major over 16x16 row-major fractals. At Cols == 16
// that is one block-column and the bytes coincide with ND; above it the two
// differ by the bit permutation described at PTO_HAD_NZGM. Either way the
// descriptor is contiguous over the whole tile.
template <uint32_t Rows, uint32_t Cols>
using GmNz = GlobalTensor<half, TileShape2D<half, Rows, Cols, Layout::NZ>,
                          BaseShape2D<half, Rows, Cols, Layout::NZ>,
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

  // Mat-factor of each stage, and the matching accumulation factor. A factor
  // at or below the block size is left dense (kMa == R, kR1 == 1).
  // Does the DENSE SHALLOW path fit? That is the fallback every size has, so it
  // -- not the deep path -- is what decides whether blocking is needed at all.
  // L0A carries H_R plus one X slice, L0B carries H_C plus one T slice, L0C the
  // two accumulators. Blocking to reach the DEEP path instead is a measured
  // 1.28x regression at N=16384 (56.86 us blocked-deep against 44.30 us
  // dense-shallow, batch 1024, and the same 1.28-1.33x at all 24 block_dims),
  // because blocking's serialization cost exceeds what the interleave buys.
  // The deep path's own fit is the same four conditions with double buffering,
  // so kDbuf below repeats these with the slot counts doubled.
  static constexpr bool kDenseShallowFits =
      R * R * sizeof(half) + kTileBytes <= L0AB_BYTES &&
      C * C * sizeof(half) + kTileBytes <= L0AB_BYTES &&
      2 * kAccBytes <= L0C_BYTES;
  static constexpr bool kBlockIt =
      (PTO_HAD_BLOCK_ALWAYS != 0) || !kDenseShallowFits;

  // Move both GM transfers onto the contiguous NZ descriptor. At C == 16 the
  // two layouts are the same bytes and each leg is independent
  // (PTO_HAD_NZLOAD / PTO_HAD_NZSTORE); above it they differ by a bit
  // permutation that only cancels if BOTH legs use the NZ view, and only
  // stays inside one sample if the tile holds one. See PTO_HAD_NZGM.
  //
  // Not on the blocked path: its stores are row SUB-BLOCKS of the output, and
  // TStoreAccNz2nz derives the destination block-column stride from the shape
  // (gShape2 * gShape3 * C0) rather than from gStride1, so it can only write a
  // whole NZ tile.
  static constexpr bool kNzGm = (PTO_HAD_NZGM != 0) && (C > 16) &&
                                (S == 1 || PTO_HAD_NZUNSAFE != 0) && !kBlockIt;

  // Which GM view each leg uses. EVERY load and store site reads these two,
  // so the pair cannot drift apart: kNzGm makes both true at once, which is
  // what the permutation needs, while at C == 16 the bytes coincide and the two
  // switches are free to differ. A store site that forgets them emits a
  // permuted result -- rel_rms sqrt(2), which validate_variant.py reports at
  // every shape of that size.
  static constexpr bool kNzLoad = (C == 16 && PTO_HAD_NZLOAD != 0) || kNzGm;
  static constexpr bool kNzStore = (C == 16 && PTO_HAD_NZSTORE != 0) || kNzGm;
  static constexpr uint32_t kMa =
      (!kBlockIt || PTO_HAD_BLOCK == 0 || PTO_HAD_BLOCK >= R)
          ? R
          : (uint32_t)PTO_HAD_BLOCK;
  static constexpr uint32_t kMb =
      (!kBlockIt || PTO_HAD_BLOCK == 0 || PTO_HAD_BLOCK >= C)
          ? C
          : (uint32_t)PTO_HAD_BLOCK;
  static constexpr uint32_t kR1 = R / kMa;   // blocks of H_R
  static constexpr uint32_t kC1 = C / kMb;   // blocks of H_C
  static_assert(kR1 * kMa == R && kC1 * kMb == C, "M must divide its factor");
  static_assert(kMa % 16 == 0 && kMb % 16 == 0, "M must be 16-aligned");
  static constexpr bool kBlkA = (kC1 > 1);   // phase A (right multiply) blocked
  static constexpr bool kBlkB = (kR1 > 1);   // phase B (left multiply) blocked

  // Only where blocking is active. Aliasing buys nothing without it (the
  // slots, not the accumulators, are what overflow), and leaving it off keeps
  // every unblocked size's L0C offsets -- and so its codegen -- untouched.
  static constexpr bool kAlias = (PTO_HAD_ACCALIAS != 0) && (kC1 > 1 || kR1 > 1);

  // Operands, resident for the whole kernel: +-H_M when blocked, else H itself.
  static constexpr unsigned kOpABytes = kBlkB ? 2 * kMa * kMa * sizeof(half)
                                              : R * R * sizeof(half);
  static constexpr unsigned kOpBBytes = kBlkA ? 2 * kMb * kMb * sizeof(half)
                                              : C * C * sizeof(half);
  // Per-matmul data slices.
  static constexpr unsigned kSliceA = S * R * kMb * sizeof(half);  // L0A, phase A
  static constexpr unsigned kSliceB = kMa * S * C * sizeof(half);  // L0B, phase B

  // Block-level double buffering for the blocked SHALLOW right multiply: with
  // two L0A slices MTE1 can stage block i+1 while M is still working on block
  // i, which is exactly the overlap blocking otherwise gives up (blocking
  // measured 1.49-1.53x on this path at N=8192/16384, bd 20 and 24, against a
  // 0.995-1.000x null control). Phase A only -- phase B's slice is kMa*S*C,
  // twice as wide here, and two of those plus the operand overflow L0B.
  static constexpr uint32_t kSlotsA =
      (kBlkA && kOpABytes + 2 * kSliceA <= L0AB_BYTES) ? 2u : 1u;

  // Two of every intermediate plus a 4-deep X ring. Every case in the table
  // holds S * N <= 16384, so all of them fit except the widest, whose two
  // accumulators already fill L0C; that one keeps the shallow path.
  //
  // `!kBlockIt` is the load-bearing term: a blocked size takes the SHALLOW
  // path, always. Blocking a single stage costs 1.49-1.86x (it makes the body
  // walk MTE1 -> M -> MTE1 -> M, so consecutive tiles stop overlapping), and no
  // size that NEEDS blocking can reach the deep path regardless -- what
  // overflows at N=32768 is the accumulator, which double buffering doubles.
  // So the deep path stays dense, and with kMa == R and kMb == C its operands
  // and slices are exactly the dense ones: H whole, and full-tile slices.
  static constexpr bool kDbuf =
      PTO_HAD_DBUF && !kBlockIt && (C == 16 || PTO_HAD_DBUF_GEN) &&
      R * R * sizeof(half) + 2 * kTileBytes <= L0AB_BYTES &&
      C * C * sizeof(half) + 2 * kTileBytes <= L0AB_BYTES &&
      4 * kAccBytes <= L0C_BYTES &&
      (R * R + C * C) * sizeof(half) + 6 * kTileBytes <= L1_BYTES;

  static constexpr unsigned kXSlots = kDbuf ? 4 : 2;
  static constexpr unsigned kBuf = kDbuf ? 2 : 1;

  // Both the deep and the shallow general bodies have blocked forms, so a
  // blocked size no longer has to reach the deep path -- which matters, because
  // the sizes that MUST block are exactly the ones too wide for it. The C == 16
  // bodies are the only dense ones left, and M is clamped to the factor there,
  // so kC1 and kR1 are 1 and this cannot fire; it is here to catch a future
  // case-table entry that breaks that. The L0C assert below is what enforces
  // the other half: a blocked shallow size needs kAlias to fit its two
  // accumulators, and without it this is a compile error, not a wrong answer.
  static_assert(!(kC1 > 1 || kR1 > 1) || C != 16,
                "the C == 16 bodies are dense and would read +-H_M as H");

  // L1: the two factors, the X ring, then the T turnaround slots.

  static constexpr unsigned kL1Hl = 0;
  static constexpr unsigned kL1Hc = kL1Hl + kOpABytes;
  static constexpr unsigned kL1X = kL1Hc + kOpBBytes;
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
  static constexpr unsigned kL0aX = kL0aHl + kOpABytes;
  static constexpr unsigned kL0aSlot = (C == 16) ? kTileBytes : kSliceA;
  static_assert(kL0aX + kBuf * kL0aSlot <= L0AB_BYTES, "L0A overflow");
  static constexpr unsigned l0aX(bool par) {
    return kL0aX + (par ? kL0aSlot : 0);
  }
  // Operand slot: 0 is +H_Ma, 1 is -H_Ma (and 0 is H_R when dense).
  static constexpr unsigned l0aOp(uint32_t neg) {
    return kL0aHl + (kBlkB ? neg * kMa * kMa * sizeof(half) : 0u);
  }

  // L0B: H_C, then the concatenated T tile
  static constexpr unsigned kL0bHc = 0;
  static constexpr unsigned kL0bT = kL0bHc + kOpBBytes;
  static constexpr unsigned kL0bSlot = (C == 16) ? kTileBytes : kSliceB;
  static_assert(kL0bT + kBuf * kL0bSlot <= L0AB_BYTES, "L0B overflow");
  static constexpr unsigned l0bT(bool par) {
    return kL0bT + (par ? kL0bSlot : 0);
  }
  static constexpr unsigned l0bOp(uint32_t neg) {
    return kL0bHc + (kBlkA ? neg * kMb * kMb * sizeof(half) : 0u);
  }

  // L0C: the stacked accumulator, then the concatenated one -- as a pair, so
  // the deep path's two tiles in flight get [stack|cat][stack|cat].
  static constexpr unsigned kL0cStack = 0;
  static constexpr unsigned kL0cCat = kL0cStack + kAccBytes;
  // With kAlias the cat accumulator shares the stack one's space, halving L0C.
  static constexpr unsigned kAccPair = (kAlias ? 1u : 2u) * kAccBytes;
  static_assert(kBuf * kAccPair <= L0C_BYTES, "L0C overflow");
  static constexpr unsigned l0cStack(bool par) {
    return (par ? kAccPair : 0);
  }
  static constexpr unsigned l0cCat(bool par) {
    return (par ? kAccPair : 0) + (kAlias ? 0u : kAccBytes);
  }
  // Output block a of the blocked phase B: rows [a*Ma, (a+1)*Ma) of the (R x
  // S*C) result, laid out as kR1 separate contiguous (Ma x S*C) accumulators --
  // an L0C tile is block-COLUMN-major, so a row slice of one tall tile would be
  // strided, while kR1 short tiles are each contiguous and together occupy the
  // same bytes.
  static constexpr unsigned l0cCatBlock(bool par, uint32_t a) {
    return l0cCat(par) + a * kMa * S * C * sizeof(float);
  }
  // Column block j of phase A's accumulator: whole 16-column groups, which ARE
  // contiguous in an L0C tile.
  static constexpr unsigned l0cStackBlock(bool par, uint32_t j) {
    return l0cStack(par) + j * kMb * kStackRows * sizeof(float);
  }
};

// GM -> L1 -> L0A/L0B for the two Hadamard factors. Done once per core; both
// stay resident for the whole kernel.
// S is a template parameter although no address here depends on it: the block
// decision does (kDenseShallowFits weighs the tile size), and loading the
// factors under a different decision from the one the tile bodies make would
// stage dense H where they expect +-H_M.
template <uint32_t R, uint32_t C, uint32_t S>
AICORE void loadHadamardFactors(__gm__ half *h_left, __gm__ half *h_right) {
  using L = Layout<R, C, S>;

  // Each buffer is H_f followed by +H_M and -H_M (see _factor_buffer on the
  // host). A blocked stage reads the pair at offset f*f and never touches the
  // dense prefix; a dense stage reads the prefix and never touches the pair.
  if constexpr (L::kBlkB) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMa, L::kMa> hl_l1;
      TASSIGN(hl_l1, L::kL1Hl + neg * L::kMa * L::kMa * sizeof(half));
      __gm__ half *src = h_left + R * R + neg * L::kMa * L::kMa;
      GmTile<L::kMa, L::kMa> hl_gm(src);
      TASSIGN(hl_gm, src);
      TLOAD(hl_l1, hl_gm);
    }
  } else {
    MatTile<R, R> hl_l1;
    TASSIGN(hl_l1, L::kL1Hl);
    GmTile<R, R> hl_gm(h_left);
    TASSIGN(hl_gm, h_left);
    TLOAD(hl_l1, hl_gm);
  }
  if constexpr (L::kBlkA) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMb, L::kMb> hc_l1;
      TASSIGN(hc_l1, L::kL1Hc + neg * L::kMb * L::kMb * sizeof(half));
      __gm__ half *src = h_right + C * C + neg * L::kMb * L::kMb;
      GmTile<L::kMb, L::kMb> hc_gm(src);
      TASSIGN(hc_gm, src);
      TLOAD(hc_l1, hc_gm);
    }
  } else {
    MatTile<C, C> hc_l1;
    TASSIGN(hc_l1, L::kL1Hc);
    GmTile<C, C> hc_gm(h_right);
    TASSIGN(hc_gm, h_right);
    TLOAD(hc_l1, hc_gm);
  }
  Sync<PIPE_MTE2, PIPE_MTE1>();

  if constexpr (L::kBlkB) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMa, L::kMa> hl_l1;
      TileLeft<half, L::kMa, L::kMa> hl_l0;
      TASSIGN(hl_l1, L::kL1Hl + neg * L::kMa * L::kMa * sizeof(half));
      TASSIGN(hl_l0, L::l0aOp(neg));
      TMOV(hl_l0, hl_l1);
    }
  } else {
    MatTile<R, R> hl_l1;
    TileLeft<half, R, R> hl_l0;
    TASSIGN(hl_l1, L::kL1Hl);
    TASSIGN(hl_l0, L::kL0aHl);
    TMOV(hl_l0, hl_l1);
  }
  if constexpr (L::kBlkA) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMb, L::kMb> hc_l1;
      TileRight<half, L::kMb, L::kMb> hc_l0;
      TASSIGN(hc_l1, L::kL1Hc + neg * L::kMb * L::kMb * sizeof(half));
      TASSIGN(hc_l0, L::l0bOp(neg));
      TMOV(hc_l0, hc_l1);
    }
  } else {
    MatTile<C, C> hc_l1;
    TileRight<half, C, C> hc_l0;
    TASSIGN(hc_l1, L::kL1Hc);
    TASSIGN(hc_l0, L::kL0bHc);
    TMOV(hc_l0, hc_l1);
  }
  Sync<PIPE_MTE1, PIPE_M>();
}

// H_f[i,j] = (-1)^popcount(i & j) for the Sylvester matrix, so the sign of
// block (i,j) is a parity and H_acc itself is never built or read. Returns 1
// for -H_M and 0 for +H_M, which are also their operand slot indices.
// One column block of the blocked shallow right multiply. Slot and flag ids
// are template arguments because bisheng requires a literal EVENT_ID*, so the
// caller branches on i & 1 rather than indexing.
template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, uint32_t P,
          event_t EvRdy, event_t EvWar>
AICORE void shallowBlockA(uint32_t i);

AICORE inline uint32_t blockSign(uint32_t i, uint32_t j) {
  uint32_t v = i & j;
  v ^= v >> 4;
  v ^= v >> 2;
  v ^= v >> 1;
  return v & 1u;
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

  if constexpr (L::kNzLoad) {
    // Declaring the source Layout::NZ turns the default ND2NZ gather -- which
    // writes L1 in C0-sized 32-BYTE pieces however wide the tile is -- into one
    // contiguous burst over the whole tile. At C == 16 that is a pure
    // reinterpret; above it the permutation it introduces is undone by the
    // matching NZ store, which L::kNzGm guarantees is in use.
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
    if constexpr (L::kNzStore) {
      GmNz<kStackRows, C> y_gm(y_tile);
      TASSIGN(y_gm, y_tile);
      TSTORE(y_gm, acc_out);
    } else {
      GmTile<kStackRows, C> y_gm(y_tile);
      TASSIGN(y_gm, y_tile);
      TSTORE(y_gm, acc_out);
    }
    set_flag(PIPE_FIX, PIPE_M, EV_ACC);
  } else if constexpr (L::kBlkA || L::kBlkB) {
    // Blocked shallow body: the same two multiplies as below, each applied as
    // blocks against a resident +-H_M so that neither factor is ever in L0A or
    // L0B whole. Blocking costs 1.8x per two-way split on the deep path, so it
    // is only ever worth paying where there is no dense form at all -- a size
    // whose H_C alone would fill L0B. Then it is not a speedup, it is the
    // difference between running and not compiling.
    constexpr uint32_t kSlot = Pong ? 1u : 0u;

    if constexpr (L::kAlias) {
      // The two accumulators share one L0C range here, so the previous tile's
      // stores must have drained before this tile's FIRST matmul -- not merely
      // before the left multiply, as in the unaliased form below.
      wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
    }

    // --- right multiply: X * H_C, as kC1 column blocks summed in L0C ---
    for (uint32_t i = 0; i < L::kC1; ++i) {
      if constexpr (L::kSlotsA == 2) {
        if (i & 1u) {
          shallowBlockA<R, C, S, kSlot, 1, EV_SRA1, EV_SWA1>(i);
        } else {
          shallowBlockA<R, C, S, kSlot, 0, EV_SRA0, EV_SWA0>(i);
        }
      } else {
        shallowBlockA<R, C, S, kSlot, 0, EV_SRA0, EV_SWA0>(i);
      }
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // X slot free (all blocks read)
    Sync<PIPE_M, PIPE_FIX>();

    // --- vertical stack -> horizontal concatenation, in L1 ---
    // Grouped by the left multiply's ROW blocks, so each of its kR1 slices is
    // one contiguous (Ma x S*C) tile rather than a strided row slice.
    TileAcc<float, kStackRows, C> acc_stack;
    TASSIGN(acc_stack, L::l0cStack(false));
    for (uint32_t a = 0; a < L::kR1; ++a) {
      for (uint32_t sm = 0; sm < S; ++sm) {
        MatTile<L::kMa, C> t_block;
        TASSIGN(t_block, L::kL1T + (a * S + sm) * L::kMa * C * sizeof(half));
        TEXTRACT(t_block, acc_stack, sm * R + a * L::kMa, 0);
      }
    }
    Sync<PIPE_FIX, PIPE_MTE1>();

    // --- left multiply: H_R * T, as kR1 row blocks summed in L0C ---
    // Under kAlias this writes over acc_stack. The extracts have drained it and
    // M is ordered behind them transitively: FIX -> MTE1 by the Sync just made,
    // then MTE1 -> M by the one inside the loop.
    TileRight<half, L::kMa, S * C> t_l0;
    TASSIGN(t_l0, L::kL0bT);
    for (uint32_t ap = 0; ap < L::kR1; ++ap) {
      MatTile<L::kMa, S * C> t_l1_ap;
      TASSIGN(t_l1_ap, L::kL1T + ap * S * L::kMa * C * sizeof(half));
      if (ap > 0) {
        wait_flag(PIPE_M, PIPE_MTE1, EV_SWB);
      }
      TMOV(t_l0, t_l1_ap);
      Sync<PIPE_MTE1, PIPE_M>();
      if constexpr (!L::kAlias) {
        if (ap == 0) {
          wait_flag(PIPE_FIX, PIPE_M, EV_ACC);  // previous tile's stores drained
        }
      }
#if !PTO_HAD_NOMM
      for (uint32_t a = 0; a < L::kR1; ++a) {
        TileAcc<float, L::kMa, S * C> acc_a;
        TASSIGN(acc_a, L::l0cCatBlock(false, a));
        TileLeft<half, L::kMa, L::kMa> h_a;
        TASSIGN(h_a, L::l0aOp(blockSign(ap, a)));
        if (ap == 0) {
          TMATMUL(acc_a, h_a, t_l0);
        } else {
          TMATMUL_ACC(acc_a, h_a, t_l0);
        }
      }
#endif
      if (ap + 1 < L::kR1) {
        set_flag(PIPE_M, PIPE_MTE1, EV_SWB);
      }
    }
    Sync<PIPE_M, PIPE_FIX>();

    for (uint32_t a = 0; a < L::kR1; ++a) {
      for (uint32_t sm = 0; sm < S; ++sm) {
        TileAcc<float, L::kMa, C> acc_block;
        TASSIGN(acc_block,
                L::l0cCatBlock(false, a) + sm * L::kMa * C * sizeof(float));
        __gm__ half *y_row = y + (sample + sm) * L::kN + a * L::kMa * C;
        // Always ND here: these are row SUB-BLOCKS of the output, and
        // TStoreAccNz2nz takes the destination block-column stride from the
        // shape, so it can only write a whole NZ tile. kNzGm is false on this
        // path for exactly that reason.
        GmTile<L::kMa, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      }
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
      if constexpr (L::kNzStore) {
        // The second half of the pair: writing the accumulator out through the
        // NZ view undoes the permutation the NZ load introduced, so the result
        // lands in natural order. Unlike the ND path (TStoreAccNz2nd, which
        // asserts the three outer GlobalTensor dims are 1) this one carries a
        // real block-column stride, so a wide tile is one descriptor.
        GmNz<R, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      } else {
        GmTile<R, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      }
    }
    set_flag(PIPE_FIX, PIPE_M, EV_ACC);
  }
}

template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, uint32_t P,
          event_t EvRdy, event_t EvWar>
AICORE void shallowBlockA(uint32_t i) {
  using L = Layout<R, C, S>;
  constexpr uint32_t kStackRows = L::kStackRows;

  TileLeft<half, kStackRows, L::kMb> x_l0;
  TASSIGN(x_l0, L::kL0aX + P * L::kSliceA);
  MatTile<kStackRows, L::kMb> x_l1_i;
  TASSIGN(x_l1_i, L::xSlot(Slot) + i * L::kMb * kStackRows * sizeof(half));

  // Only the block kSlotsA back shared this slot, so with two slots MTE1 stages
  // block i+1 while M is still on block i.
  if (i >= L::kSlotsA) {
    wait_flag(PIPE_M, PIPE_MTE1, EvWar);
  }
  TMOV(x_l0, x_l1_i);
  set_flag(PIPE_MTE1, PIPE_M, EvRdy);

  wait_flag(PIPE_MTE1, PIPE_M, EvRdy);
#if !PTO_HAD_NOMM
  for (uint32_t j = 0; j < L::kC1; ++j) {
    TileAcc<float, kStackRows, L::kMb> acc_j;
    TASSIGN(acc_j, L::l0cStackBlock(false, j));
    TileRight<half, L::kMb, L::kMb> h_j;
    TASSIGN(h_j, L::l0bOp(blockSign(i, j)));
    if (i == 0) {
      TMATMUL(acc_j, x_l0, h_j);
    } else {
      TMATMUL_ACC(acc_j, x_l0, h_j);
    }
  }
#endif
  if (i + L::kSlotsA < L::kC1) {
    set_flag(PIPE_M, PIPE_MTE1, EvWar);
  }
}

// --- deep pipeline --------------------------------------------------------
//
// Both phase bodies branch on C internally. The C == 16 pair uses that case's
// free NZ reinterpret so its turnaround is two TMOVs; the general pair pays the
// same S TEXTRACTs plus S TSTOREs the shallow body pays. The SPLIT itself is
// independent of either -- it is about pipe order, not about the reshape.
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

  if constexpr (C != 16) {
    // General body, same cut: right multiply, then park T in L1. X is the LEFT
    // operand here (and T the right one in phase B), the mirror of the C == 16
    // case, so the two share Layout's slot helpers and capacity asserts.
    MatTile<L::kStackRows, C> x_l1;
    TileLeft<half, L::kStackRows, C> x_l0a;
    TileRight<half, C, C> hc_l0;
    TileAcc<float, L::kStackRows, C> acc_stack;
    TASSIGN(x_l1, L::xSlot(Slot));
    TASSIGN(x_l0a, L::l0aX(Par));
    TASSIGN(hc_l0, L::kL0bHc);
    TASSIGN(acc_stack, L::l0cStack(Par));

    wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);  // this tile's load has landed
    TMOV(x_l0a, x_l1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EvX);  // X slot free for the load 4 tiles on
    set_flag(PIPE_MTE1, PIPE_M, EvPA);

    wait_flag(PIPE_MTE1, PIPE_M, EvPA);
#if !PTO_HAD_NOMM
    TMATMUL(acc_stack, x_l0a, hc_l0);
#endif
    set_flag(PIPE_M, PIPE_FIX, EvPA);

    // Vertical stack -> horizontal concatenation, in L1. Block-columns are
    // contiguous in NZ, so sample s's (R x C) result lands at s * R * C and the
    // S writes read back as one (R x S*C) tile in phase B.
    wait_flag(PIPE_M, PIPE_FIX, EvPA);
    for (uint32_t s = 0; s < S; ++s) {
      MatTile<R, C> t_block;
      TASSIGN(t_block, L::l1T(Par) + s * R * C * sizeof(half));
      TEXTRACT(t_block, acc_stack, s * R, 0);
    }
    set_flag(PIPE_FIX, PIPE_MTE1, EvPA);
    return;
  }

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

  if constexpr (C != 16) {
    // General body: left multiply on the concatenated view, then store each
    // sample separately. The fixpipe L0C->GM path is 2D-to-2D only, so these
    // cannot be folded into one strided descriptor -- a 5D scatter faults.
    MatTile<R, S * C> t_l1;
    TileRight<half, R, S * C> t_l0b;
    TileLeft<half, R, R> hl_l0;
    TileAcc<float, R, S * C> acc_cat;
    TASSIGN(t_l1, L::l1T(Par));
    TASSIGN(t_l0b, L::l0bT(Par));
    TASSIGN(hl_l0, L::kL0aHl);
    TASSIGN(acc_cat, L::l0cCat(Par));

    wait_flag(PIPE_FIX, PIPE_MTE1, EvPA);  // phase A parked T in L1
    TMOV(t_l0b, t_l1);
    set_flag(PIPE_MTE1, PIPE_M, EvPB);

    wait_flag(PIPE_MTE1, PIPE_M, EvPB);
#if !PTO_HAD_NOMM
    TMATMUL(acc_cat, hl_l0, t_l0b);
#endif
    set_flag(PIPE_M, PIPE_FIX, EvPB);

    // No EV_ACC edge, as in the C == 16 phase body: with two cat accumulators
    // the write-after-read on slot Par is already ordered, because B(i+2)'s
    // TMOV waits on A(i+2)'s FIX->MTE1 flag and A(i+2)'s TEXTRACTs follow
    // B(i)'s TSTOREs on the in-order FIX pipe.
    wait_flag(PIPE_M, PIPE_FIX, EvPB);
    for (uint32_t s = 0; s < S; ++s) {
      TileAcc<float, R, C> acc_block;
      TASSIGN(acc_block, L::l0cCat(Par) + s * R * C * sizeof(float));
      __gm__ half *y_row = y + (sample + s) * L::kN;
      if constexpr (L::kNzStore) {  // must match the load -- see PTO_HAD_NZGM
        GmNz<R, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      } else {
        GmTile<R, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      }
    }
    return;
  }

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
  if constexpr (L::kNzStore) {
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
  const uint32_t tiles_per_core = DIV_ROUNDUP(DIV_ROUNDUP(batch, S), num_cores);
  const uint32_t begin = tiles_per_core * S * core_id;
  if (begin >= batch) {
    return;
  }
  uint32_t end = begin + tiles_per_core * S;
  if (end > batch) {
    end = batch;
  }

  loadHadamardFactors<R, C, S>(h_left, h_right);
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
