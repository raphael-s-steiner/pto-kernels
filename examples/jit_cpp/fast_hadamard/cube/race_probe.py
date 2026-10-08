"""Repeat one fixed shape many times and report whether the answer MOVES.

validate_variant.py runs each shape once, so it cannot separate a logic bug
from a missing synchronisation edge -- and a missing edge can sit at a low
enough rate to pass a whole sweep. This runs one shape repeatedly: a stable bad
set is a logic bug, a moving one is a race. The blocked phase bodies' first
version failed 4 times in 30 here while passing all 252 cases of the sweep.

The error MAGNITUDE names the stage. Dropping one of k accumulation terms in an
orthogonal transform leaves relative RMS sqrt(1/k) -- 0.707 for k = 2, 0.500
for k = 4 -- so at a shape whose two stages have different block counts, the
value says which stage lost its term. That is what identified the defect as a
TSTORE running ahead of the last matmul rather than as corrupt data.

  python3 cube/race_probe.py <so> [n] [batch] [block_dim] [reps]
"""
import ctypes, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch_npu  # noqa
from cube.jit_util_hadamard_cube import hadamard_factors
from jit_util_common import get_current_stream_ptr, load_cdll, load_required_symbol
BASE=pathlib.Path(__file__).resolve().parents[1]
ARGS=[ctypes.c_uint32]+[ctypes.c_void_p]*5+[ctypes.c_uint32]*2
torch.npu.set_device("npu:0")
def fwht(x):
    y=x.to(torch.float64).clone(); n=y.shape[-1]; h=1
    while h<n:
        y=y.view(-1,n//(2*h),2,h); a,b=y[:,:,0,:].clone(),y[:,:,1,:].clone()
        y[:,:,0,:],y[:,:,1,:]=a+b,a-b; y=y.reshape(-1,n); h*=2
    return y
fn=load_required_symbol(load_cdll(str(BASE/sys.argv[1])),"call_kernel",ARGS)
N,batch,bd=[int(v) for v in (sys.argv[2:5] if len(sys.argv)>4 else (8192,8,3))]
print(f"Same shape N={N} batch={batch} bd={bd}, repeated. Stable set => logic")
print("bug; moving set => a missing edge (race).\n")
sets=[]; worst=[]
REPS=int(sys.argv[5]) if len(sys.argv)>5 else 10
for rep in range(REPS):
    torch.manual_seed(rep)
    x=torch.randn(batch,N,device="npu:0",dtype=torch.float16)
    y=torch.full_like(x,float("nan"))
    hl,hc=hadamard_factors(N,x.device)
    fn(bd,get_current_stream_ptr(),x.data_ptr(),y.data_ptr(),
       hl.data_ptr(),hc.data_ptr(),batch,N)
    torch.npu.synchronize()
    got=y.to(torch.float64); ref=fwht(x)
    errs=[float(((got[i]-ref[i])**2).mean().sqrt()/(ref[i]**2).mean().sqrt())
          for i in range(batch)]
    bad=tuple(i for i,e in enumerate(errs) if e>1e-3)
    worst.append(max(errs)); sets.append(bad); 
print(f"  {REPS} reps: {sum(1 for b in sets if b)} bad, "
      f"worst rel_rms {max(worst):.4f}, outcomes {sorted(set(sets))[:3]}")
print(f"distinct outcomes: {len(set(sets))}"
      f"  -> {'RACE' if len(set(sets))>1 else 'deterministic'}")
