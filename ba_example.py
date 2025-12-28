from time import perf_counter
from pathlib import Path
from datetime import datetime
import torch
import pypose as pp

from ba_helpers import Reproj, least_square_error
from datapipes.bal_loader import get_problem, read_bal_data
from bae.sparse.py_ops import *
from bae.optim import LM
from bae.utils.pysolvers import PCG, CuDSS

TARGET_DATASET = "ladybug"
TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_PROBLEM = "problem-49-7776-pre"
# TARGET_PROBLEM = "problem-1695-155710-pre"  
# TARGET_PROBLEM = "problem-969-105826-pre"
# TARGET_DATASET = "trafalgar"
# TARGET_PROBLEM = "problem-257-65132-pre"
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"



DEVICE = 'cuda'
OPTIMIZE_INTRINSICS = True

USE_QUATERNIONS = True
REPORT_WARP_MEMPOOL = True

file_name = f'{TARGET_DATASET}.{TARGET_PROBLEM}'
dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET, use_quat=USE_QUATERNIONS)
memory_snapshot_path = None
cuda_device = torch.device(DEVICE) if DEVICE.startswith("cuda") else None
warp_device = None
warp_mempool_start_current = None
warp_mempool_start_high = None


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.2f} {unit}"

if DEVICE.startswith("cuda") and torch.cuda.is_available():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path("memory_traces")
    snapshot_dir.mkdir(exist_ok=True)
    memory_snapshot_path = snapshot_dir / f"{file_name}_cuda_memory_{timestamp}.pickle"
    # Record allocator events so we can inspect GPU memory usage after the run.
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        device=cuda_device,
        clear_history=True,
    )

if REPORT_WARP_MEMPOOL and DEVICE.startswith("cuda"):
    try:
        if wp.is_cuda_available():
            warp_device = wp.get_device("cuda:0" if DEVICE == "cuda" else DEVICE)
            if not wp.is_mempool_enabled(warp_device):
                wp.set_mempool_enabled(warp_device, True)
            warp_mempool_start_current = wp.get_mempool_used_mem_current(warp_device)
            warp_mempool_start_high = wp.get_mempool_used_mem_high(warp_device)
    except Exception as e:
        print(f"Warning: failed to query Warp mempool stats: {e}")

if OPTIMIZE_INTRINSICS:
    NUM_CAMERA_PARAMS = 10 if USE_QUATERNIONS else 9
else:
    NUM_CAMERA_PARAMS = 7 if USE_QUATERNIONS else 6

print(f'Fetched {TARGET_PROBLEM} from {TARGET_DATASET}')

trimmed_dataset = dataset
trimmed_dataset = {k: v.to(DEVICE) for k, v in trimmed_dataset.items() if type(v) == torch.Tensor}

input = {
    "points_2d": trimmed_dataset['points_2d'],
    "camera_indices": trimmed_dataset['camera_index_of_observations'],
    "point_indices": trimmed_dataset['point_index_of_observations']
}

model = Reproj(
    trimmed_dataset['camera_params'][:, :NUM_CAMERA_PARAMS].clone(),
    trimmed_dataset['points_3d'].clone()
).to(DEVICE)
strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
solver = PCG(tol=1e-4, maxiter=250)  # or CuDSS()
optimizer = LM(model, matrix_free_normal=True, strategy=strategy, solver=solver, reject=30)



print('Loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
).item())

print("Initial loss", optimizer.model.loss(input, None).item())

if cuda_device is not None and torch.cuda.is_available():
    torch.cuda.synchronize(cuda_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)

start = perf_counter()
for idx in range(20):
    loss = optimizer.step(input)
    print('Iteration', idx, 'loss', loss.item(), 'time', perf_counter() - start)

if memory_snapshot_path:
    torch.cuda.synchronize(cuda_device)
    torch.cuda.memory._dump_snapshot(str(memory_snapshot_path))
    print(f"CUDA memory snapshot saved to {memory_snapshot_path}")

# exit()

if cuda_device is not None and torch.cuda.is_available():
    torch.cuda.synchronize(cuda_device)
end = perf_counter()
print('Time', end - start)

if cuda_device is not None and torch.cuda.is_available():
    peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
    try:
        peak_reserved = torch.cuda.max_memory_reserved(cuda_device)
    except AttributeError:  # older PyTorch
        peak_reserved = torch.cuda.max_memory_cached(cuda_device)
    print(f"Peak CUDA memory allocated: {_format_bytes(peak_allocated)}")
    print(f"Peak CUDA memory reserved: {_format_bytes(peak_reserved)}")

if warp_device is not None and warp_mempool_start_current is not None and warp_mempool_start_high is not None:
    try:
        warp_current = wp.get_mempool_used_mem_current(warp_device)
        warp_high = wp.get_mempool_used_mem_high(warp_device)
        print(f"Warp CUDA mempool current: {_format_bytes(warp_current)} (Δ {_format_bytes(warp_current - warp_mempool_start_current)})")
        print(f"Warp CUDA mempool high-water: {_format_bytes(warp_high)} (Δ {_format_bytes(warp_high - warp_mempool_start_high)})")
    except Exception as e:
        print(f"Warning: failed to query Warp mempool stats: {e}")

print('Ending loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
).item())
