from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


def _bsr_to_torch(
    A,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           Tuple[int, int], Tuple[int, int], torch.dtype, torch.device]:
    
    if isinstance(A, torch.Tensor) and A.layout == torch.sparse_bsr:
        crow = A.crow_indices()
        col = A.col_indices()
        values = A.values()
        BR, BC = values.shape[-2], values.shape[-1]
        nrow = A.shape[0] // BR
        ncol = A.shape[1] // BC
        return crow, col, values, (BR, BC), (nrow, ncol), values.dtype, values.device

    raise TypeError(f"Unsupported BSR matrix type: {type(A)}")


def _flatten_vec(v: torch.Tensor) -> torch.Tensor:
    if not v.is_contiguous():
        raise ValueError("Vector must be contiguous to share memory with Triton kernel")
    return v.view(-1)


def _build_bsr_output(crow, col, vals, nrow, ncol, BR, BC):
    return torch.sparse_bsr_tensor(
        crow_indices=crow.to(torch.int32),
        col_indices=col.to(torch.int32),
        values=vals,
        size=(nrow * BR, ncol * BC),
    )


_TRITON_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.bfloat16: tl.bfloat16,
}


def _triton_dtype(t: torch.dtype):
    try:
        return _TRITON_DTYPES[t]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype for Triton BSR kernels: {t}") from exc


def _next_pow2(n: int) -> int:
    return 1 << max(0, (int(n) - 1)).bit_length()


def _uncompress_rows(crow: torch.Tensor, nnz: int) -> torch.Tensor:
    nrow = crow.numel() - 1

    if nnz == 0:
        return torch.empty(0, dtype=torch.int32, device=crow.device)

    counts = (crow[1:] - crow[:-1]).to(torch.int64)

    return torch.repeat_interleave(torch.arange(nrow, device=crow.device, dtype=torch.int32), counts)


@triton.jit
def _bsr_mv_fwd_kernel(
    A_crow, A_col, A_val,
    X, Y,
    alpha, beta,
    nrow,
    BR: tl.constexpr, BC: tl.constexpr,
    BLK_R: tl.constexpr, BLK_C: tl.constexpr,
    DTYPE: tl.constexpr,
):
    
    row = tl.program_id(0).to(tl.int32)
    if row >= nrow:
        return

    r_idx = tl.arange(0, BLK_R)
    c_idx = tl.arange(0, BLK_C)
    rmask = r_idx < BR
    cmask = c_idx < BC

    y_off = row * BR + r_idx
    if beta == 0.0:
        acc = tl.zeros((BLK_R,), dtype=DTYPE)
    else:
        prev = tl.load(Y + y_off, mask=rmask, other=0.0).to(DTYPE)
        acc = beta * prev

    if alpha != 0.0:
        beg = tl.load(A_crow + row).to(tl.int32)
        end = tl.load(A_crow + row + 1).to(tl.int32)
        block_size = BR * BC
        partial = tl.zeros((BLK_R,), dtype=DTYPE)
        n = end - beg
        for i in tl.range(0, n):
            blk = beg + i
            col = tl.load(A_col + blk).to(tl.int32)
            v_off = blk * block_size + r_idx[:, None] * BC + c_idx[None, :]
            block_vals = tl.load(
                A_val + v_off,
                mask=rmask[:, None] & cmask[None, :], other=0.0,
            ).to(DTYPE)
            x_off = col * BC + c_idx
            x_vals = tl.load(X + x_off, mask=cmask, other=0.0).to(DTYPE)
            partial += tl.sum(block_vals * x_vals[None, :], axis=1)
        acc += alpha * partial

    tl.store(Y + y_off, acc, mask=rmask)


@triton.jit
def _bsr_mv_trans_kernel(
    A_crow, A_col, A_val,
    X, Y,
    alpha,
    nrow,
    BR: tl.constexpr, BC: tl.constexpr,
    BLK_R: tl.constexpr, BLK_C: tl.constexpr,
    DTYPE: tl.constexpr,
):
    
    row = tl.program_id(0).to(tl.int32)

    if row >= nrow:
        return

    if alpha == 0.0:
        return

    r_idx = tl.arange(0, BLK_R)
    c_idx = tl.arange(0, BLK_C)
    rmask = r_idx < BR
    cmask = c_idx < BC
    x_off = row * BR + r_idx
    x_vals = tl.load(X + x_off, mask=rmask, other=0.0).to(DTYPE)
    beg = tl.load(A_crow + row).to(tl.int32)
    end = tl.load(A_crow + row + 1).to(tl.int32)
    block_size = BR * BC
    n = end - beg

    for i in tl.range(0, n):
        blk = beg + i
        col = tl.load(A_col + blk).to(tl.int32)
        v_off = blk * block_size + r_idx[:, None] * BC + c_idx[None, :]
        block_vals = tl.load(A_val + v_off, mask=rmask[:, None] & cmask[None, :], other=0.0).to(DTYPE)
        contrib = tl.sum(block_vals * x_vals[:, None], axis=0) * alpha
        tl.atomic_add(Y + (col * BC + c_idx), contrib, mask=cmask)


@triton.jit
def _gather_transpose_kernel(
    src_ptr, sort_idx_ptr, dst_ptr,
    nnz,
    BR: tl.constexpr, BC: tl.constexpr,
    BLK_R: tl.constexpr, BLK_C: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int32)
    if i >= nnz:
        return
    src_idx = tl.load(sort_idx_ptr + i).to(tl.int64)
    r_idx = tl.arange(0, BLK_R)
    c_idx = tl.arange(0, BLK_C)
    rmask = r_idx < BR
    cmask = c_idx < BC

    src_off = src_idx * (BR * BC) + r_idx[:, None] * BC + c_idx[None, :]
    block = tl.load(
        src_ptr + src_off,
        mask=rmask[:, None] & cmask[None, :], other=0.0,
    )
    i64 = i.to(tl.int64)
    dst_off = i64 * (BC * BR) + c_idx[None, :] * BR + r_idx[:, None]
    tl.store(dst_ptr + dst_off, block, mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _bsr_scale_kernel(Y, beta, n, BLOCK: tl.constexpr, DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n

    if beta == 0.0:
        tl.store(Y + off, tl.zeros((BLOCK,), dtype=DTYPE), mask=mask)
    else:
        v = tl.load(Y + off, mask=mask, other=0.0).to(DTYPE)
        tl.store(Y + off, beta * v, mask=mask)


@triton.jit
def _bsr_mm_numeric_kernel(
    A_crow, A_col, A_val,
    B_crow, B_col, B_val,
    C_blk_row, C_col, C_val,
    alpha,
    BR_A: tl.constexpr, BC_A: tl.constexpr, BC_B: tl.constexpr,
    BLK_R: tl.constexpr, BLK_K: tl.constexpr, BLK_C: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
    DTYPE: tl.constexpr,
):

    c_blk = tl.program_id(0).to(tl.int32)
    c_row = tl.load(C_blk_row + c_blk).to(tl.int32)
    c_col = tl.load(C_col + c_blk).to(tl.int32)
    r_idx = tl.arange(0, BLK_R)
    k_idx = tl.arange(0, BLK_K)
    s_idx = tl.arange(0, BLK_C)
    rmask = r_idx < BR_A
    kmask = k_idx < BC_A
    smask = s_idx < BC_B
    a_block_size = BR_A * BC_A
    b_block_size = BC_A * BC_B
    c_block_size = BR_A * BC_B
    contrib = tl.zeros((BLK_R, BLK_C), dtype=DTYPE)

    if alpha != 0.0:
        a_beg = tl.load(A_crow + c_row).to(tl.int32)
        a_end = tl.load(A_crow + c_row + 1).to(tl.int32)
        n_a = a_end - a_beg
        for i in tl.range(0, n_a):
            a_blk = a_beg + i
            k = tl.load(A_col + a_blk).to(tl.int32)
            b_beg = tl.load(B_crow + k).to(tl.int32)
            b_end = tl.load(B_crow + k + 1).to(tl.int32)
            lo = b_beg
            hi = b_end

            for _ in tl.range(0, BSEARCH_ITERS):
                mid = (lo + hi) // 2
                cond = lo < hi
                safe_mid = tl.where(cond, mid, b_beg)
                v = tl.load(B_col + safe_mid).to(tl.int32)
                go_right = cond & (v < c_col)
                go_left = cond & (v >= c_col)
                lo = tl.where(go_right, mid + 1, lo)
                hi = tl.where(go_left, mid, hi)

            in_range = lo < b_end
            safe_idx = tl.where(in_range, lo, b_beg)
            cur = tl.load(B_col + safe_idx).to(tl.int32)
            exists = in_range & (cur == c_col)
            safe_b_blk = tl.where(exists, lo, 0)
            a_off = a_blk * a_block_size + r_idx[:, None] * BC_A + k_idx[None, :]
            a_block = tl.load(A_val + a_off, mask=rmask[:, None] & kmask[None, :], other=0.0).to(DTYPE)
            b_off = safe_b_blk * b_block_size + k_idx[:, None] * BC_B + s_idx[None, :]
            b_block = tl.load(B_val + b_off,mask=(kmask[:, None] & smask[None, :]) & exists, other=0.0).to(DTYPE)
            contrib += tl.sum(a_block[:, :, None] * b_block[None, :, :], axis=1)

    v_off = c_blk * c_block_size + r_idx[:, None] * BC_B + s_idx[None, :]
    prev = tl.load(C_val + v_off, mask=rmask[:, None] & smask[None, :], other=0.0).to(DTYPE)
    tl.store(C_val + v_off, prev + alpha * contrib, mask=rmask[:, None] & smask[None, :])


def _bsr_mm_topology(
    A_crow: torch.Tensor, A_col: torch.Tensor,
    B_crow: torch.Tensor, B_col: torch.Tensor,
    nrow_C: int, ncol_C: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = A_crow.device
    A_crow_l = A_crow.to(torch.int64)
    A_col_l = A_col.to(torch.int64)
    B_crow_l = B_crow.to(torch.int64)
    B_col_l = B_col.to(torch.int64)
    a_nnz = A_col_l.numel()
    empty = (
        torch.zeros(nrow_C + 1, dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
    )
    if a_nnz == 0 or B_col_l.numel() == 0:
        return empty

    b_lens = B_crow_l[1:] - B_crow_l[:-1]
    counts = b_lens[A_col_l]
    a_blk_offsets = torch.zeros(a_nnz + 1, dtype=torch.int64, device=device)
    a_blk_offsets[1:] = counts.cumsum(0)
    total = int(a_blk_offsets[-1].item())

    if total == 0:
        return empty

    a_blk_per_t = torch.repeat_interleave(torch.arange(a_nnz, device=device, dtype=torch.int64), counts)
    pos_in_row = (
        torch.arange(total, device=device, dtype=torch.int64)
        - a_blk_offsets[a_blk_per_t]
    )
    b_starts = B_crow_l[A_col_l]
    del B_crow_l
    triplet_j = B_col_l[b_starts[a_blk_per_t] + pos_in_row]
    del b_starts, pos_in_row, B_col_l

    nrow_A = A_crow_l.numel() - 1
    a_row_per_blk = torch.repeat_interleave(
        torch.arange(nrow_A, device=device, dtype=torch.int64),
        A_crow_l[1:] - A_crow_l[:-1],
    )
    triplet_i = a_row_per_blk[a_blk_per_t]
    key = triplet_i * ncol_C + triplet_j
    sorted_key, _ = torch.sort(key)
    keep = torch.ones(total, dtype=torch.bool, device=device)
    keep[1:] = sorted_key[1:] != sorted_key[:-1]
    unique_key = sorted_key[keep]
    out_i = (unique_key // ncol_C).to(torch.int64)
    out_j = (unique_key % ncol_C).to(torch.int32)
    counts_per_row = torch.bincount(out_i, minlength=nrow_C)
    crow_l = torch.zeros(nrow_C + 1, dtype=torch.int64, device=device)
    crow_l[1:] = counts_per_row.cumsum(0)

    return crow_l.to(torch.int32), out_j


def sparse_bsr_mv(
    A,
    x,
    y=None,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose: bool = False,
    work_buffer=None,
):
    crow, col, vals3d, (BR, BC), (nrow, ncol), dtype, device = _bsr_to_torch(A)

    if transpose:
        out_blocks, out_block_size = ncol, BC
        in_blocks, in_block_size = nrow, BR
    else:
        out_blocks, out_block_size = nrow, BR
        in_blocks, in_block_size = ncol, BC

    expected_y = out_blocks * out_block_size
    vals3d = vals3d.contiguous()
    crow_i = crow.to(torch.int32).contiguous()
    col_i = col.to(torch.int32).contiguous()

    x_t = _flatten_vec(x)

    if x_t.numel() != in_blocks * in_block_size:
        raise ValueError(f"x has {x_t.numel()} scalars, expected {in_blocks * in_block_size}")

    if x_t.dtype != dtype:
        raise TypeError(f"x dtype {x_t.dtype} != A dtype {dtype}")

    return_obj = y

    if y is None:
        y_t = torch.empty(expected_y, dtype=dtype, device=device)
        return_obj = y_t
        beta = 0.0
    else:
        y_t = _flatten_vec(y)
        if y_t.numel() != expected_y:
            raise ValueError(f"y has {y_t.numel()} scalars, expected {expected_y}")
        if y_t.dtype != dtype:
            raise TypeError(f"y dtype {y_t.dtype} != A dtype {dtype}")

    if x_t.data_ptr() == y_t.data_ptr():
        if work_buffer is None:
            x_t = y_t.clone()
        else:
            wb = _flatten_vec(work_buffer)
            wb.copy_(y_t)
            x_t = wb

    BLK_R = max(_next_pow2(BR), 1)
    BLK_C = max(_next_pow2(BC), 1)
    DTYPE = _triton_dtype(dtype)

    if transpose:
        n = y_t.numel()
        SCALE_BLOCK = 1024
        scale_grid = ((n + SCALE_BLOCK - 1) // SCALE_BLOCK,)
        _bsr_scale_kernel[scale_grid](y_t, float(beta), n, BLOCK=SCALE_BLOCK, DTYPE=DTYPE)

        if alpha != 0.0:
            _bsr_mv_trans_kernel[(nrow,)](
                crow_i, col_i, vals3d,
                x_t, y_t,
                float(alpha),
                nrow,
                BR=BR, BC=BC, BLK_R=BLK_R, BLK_C=BLK_C, DTYPE=DTYPE,
            )
    else:
        _bsr_mv_fwd_kernel[(nrow,)](
            crow_i, col_i, vals3d,
            x_t, y_t,
            float(alpha), float(beta),
            nrow,
            BR=BR, BC=BC, BLK_R=BLK_R, BLK_C=BLK_C, DTYPE=DTYPE,
        )
    return return_obj


def sparse_bsr_mm(A, B, alpha: float = 1.0, *, topology_cache=None):
    A_crow, A_col, A_val, (BR_A, BC_A), (nrow_A, ncol_A), dtype, device = _bsr_to_torch(A)
    B_crow, B_col, B_val, (BR_B, BC_B), (nrow_B, ncol_B), dtype_b, device_b = _bsr_to_torch(B)

    if dtype != dtype_b:
        raise ValueError(f"A and B dtypes differ: {dtype} vs {dtype_b}")

    if device != device_b:
        raise ValueError(f"A and B on different devices: {device} vs {device_b}")

    if BC_A != BR_B:
        raise ValueError(f"Block-shape mismatch: A.block_shape[1]={BC_A}, B.block_shape[0]={BR_B}")

    if ncol_A != nrow_B:
        raise ValueError(f"Block-row/col mismatch: A.ncol={ncol_A}, B.nrow={nrow_B}")

    nrow_C, ncol_C = nrow_A, ncol_B
    BR_C, BC_C = BR_A, BC_B

    A_crow_i = A_crow.to(torch.int32).contiguous()
    A_col_i = A_col.to(torch.int32).contiguous()
    B_crow_i = B_crow.to(torch.int32).contiguous()
    B_col_i = B_col.to(torch.int32).contiguous()

    if alpha == 0.0 or A_col.numel() == 0 or B_col.numel() == 0:
        crow = torch.zeros(nrow_C + 1, dtype=torch.int32, device=device)
        col = torch.empty(0, dtype=torch.int32, device=device)
        vals = torch.empty((0, BR_C, BC_C), dtype=dtype, device=device)
        return _build_bsr_output(crow, col, vals, nrow_C, ncol_C, BR_C, BC_C)

    cached = topology_cache is not None and "C_crow" in topology_cache
    if cached:
        C_crow = topology_cache["C_crow"]
        C_col = topology_cache["C_col"]
        C_blk_row = topology_cache["C_blk_row"]
    else:
        C_crow, C_col = _bsr_mm_topology(A_crow_i, A_col_i, B_crow_i, B_col_i, nrow_C, ncol_C)
        out_nnz_tmp = int(C_col.numel())
        C_blk_row = _uncompress_rows(C_crow, out_nnz_tmp)
        if topology_cache is not None:
            topology_cache["C_crow"] = C_crow
            topology_cache["C_col"] = C_col
            topology_cache["C_blk_row"] = C_blk_row

    out_nnz = int(C_col.numel())
    C_vals = torch.zeros((out_nnz, BR_C, BC_C), dtype=dtype, device=device)
    if out_nnz == 0:
        return _build_bsr_output(C_crow, C_col, C_vals, nrow_C, ncol_C, BR_C, BC_C)

    BLK_R = max(_next_pow2(BR_C), 1)
    BLK_K = max(_next_pow2(BC_A), 1)
    BLK_C = max(_next_pow2(BC_C), 1)
    DTYPE = _triton_dtype(dtype)
    BSEARCH_ITERS = max(1, (max(1, ncol_C) - 1).bit_length() + 1)

    A_val_c = A_val.contiguous()
    B_val_c = B_val.contiguous()

    _bsr_mm_numeric_kernel[(out_nnz,)](
        A_crow_i, A_col_i, A_val_c,
        B_crow_i, B_col_i, B_val_c,
        C_blk_row, C_col, C_vals,
        float(alpha),
        BR_A=BR_A, BC_A=BC_A, BC_B=BC_B,
        BLK_R=BLK_R, BLK_K=BLK_K, BLK_C=BLK_C,
        BSEARCH_ITERS=BSEARCH_ITERS,
        DTYPE=DTYPE,
    )
    return _build_bsr_output(C_crow, C_col, C_vals, nrow_C, ncol_C, BR_C, BC_C)


def sparse_bsr_transposed(A):
    crow, col, vals3d, (BR, BC), (nrow, ncol), dtype, device = _bsr_to_torch(A)

    crow_i = crow.to(torch.int32).contiguous()
    col_i = col.to(torch.int32).contiguous()
    nnz = int(col_i.numel())

    if nnz == 0:
        new_crow = torch.zeros(ncol + 1, dtype=torch.int32, device=device)
        new_col = torch.empty(0, dtype=torch.int32, device=device)
        new_vals = torch.empty((0, BC, BR), dtype=dtype, device=device)
        return _build_bsr_output(new_crow, new_col, new_vals, ncol, nrow, BC, BR)

    old_row = _uncompress_rows(crow_i, nnz).to(torch.int64)
    new_row = col_i.to(torch.int64)
    new_col = old_row

    new_nrow, new_ncol = ncol, nrow
    key = new_row * new_ncol + new_col
    _, sort_idx = torch.sort(key, stable=True)

    sorted_new_row = new_row[sort_idx]
    sorted_new_col = new_col[sort_idx].to(torch.int32)
    vals3d_c = vals3d.contiguous()
    permuted_vals = torch.empty((nnz, BC, BR), dtype=dtype, device=device)
    sort_idx64 = sort_idx.to(torch.int64).contiguous()
    BLK_R_ = max(_next_pow2(BR), 1)
    BLK_C_ = max(_next_pow2(BC), 1)

    _gather_transpose_kernel[(nnz,)](
        vals3d_c, sort_idx64, permuted_vals,
        nnz, BR=BR, BC=BC, BLK_R=BLK_R_, BLK_C=BLK_C_,
    )

    counts = torch.bincount(sorted_new_row, minlength=new_nrow)
    crow_l = torch.zeros(new_nrow + 1, dtype=torch.int64, device=device)
    crow_l[1:] = counts.cumsum(0)
    new_crow = crow_l.to(torch.int32)

    return _build_bsr_output(new_crow, sorted_new_col, permuted_vals, new_nrow, new_ncol, BC, BR)


def sparse_bsr_axpy(x, y, alpha: float = 1.0, *, topology_cache=None):
    x_crow, x_col, x_vals, (BR, BC), (nrow, ncol), dtype, device = _bsr_to_torch(x)
    y_crow, y_col, y_vals, (BR_y, BC_y), (nrow_y, ncol_y), dtype_y, device_y = _bsr_to_torch(y)

    if (BR, BC) != (BR_y, BC_y):
        raise ValueError(f"Block shapes differ: {(BR, BC)} vs {(BR_y, BC_y)}")
    if (nrow, ncol) != (nrow_y, ncol_y):
        raise ValueError(f"Block-matrix shapes differ: {(nrow, ncol)} vs {(nrow_y, ncol_y)}")
    if dtype != dtype_y:
        raise ValueError(f"Dtypes differ: {dtype} vs {dtype_y}")
    if device != device_y:
        raise ValueError(f"Devices differ: {device} vs {device_y}")

    x_crow_i = x_crow.to(torch.int32).contiguous()
    x_col_i = x_col.to(torch.int32).contiguous()
    y_crow_i = y_crow.to(torch.int32).contiguous()
    y_col_i = y_col.to(torch.int32).contiguous()
    x_nnz = int(x_col_i.numel())
    y_nnz = int(y_col_i.numel())
    x_vals_c = x_vals.contiguous()
    y_vals_c = y_vals.contiguous()

    if x_nnz == 0:
        return _build_bsr_output(
            y_crow_i.clone(), y_col_i.clone(), y_vals_c.clone(),
            nrow, ncol, BR, BC,
        )
    if y_nnz == 0:
        return _build_bsr_output(
            x_crow_i.clone(), x_col_i.clone(),
            (alpha * x_vals_c).contiguous(), nrow, ncol, BR, BC,
        )

    cached = topology_cache is not None and "out_crow" in topology_cache
    
    if cached:
        out_crow = topology_cache["out_crow"]
        out_col = topology_cache["out_col"]
        x_target = topology_cache["x_target"]
        x_pick = topology_cache["x_pick"]
        y_target = topology_cache["y_target"]
        y_pick = topology_cache["y_pick"]
        out_nnz = int(out_col.numel())
    else:
        x_block_row = _uncompress_rows(x_crow_i, x_nnz).to(torch.int64)
        y_block_row = _uncompress_rows(y_crow_i, y_nnz).to(torch.int64)

        all_row = torch.cat([x_block_row, y_block_row])
        all_col = torch.cat([x_col_i.to(torch.int64), y_col_i.to(torch.int64)])
        all_src = torch.cat([
            torch.zeros(x_nnz, dtype=torch.int8, device=device),
            torch.ones(y_nnz, dtype=torch.int8, device=device),
        ])
        all_idx = torch.cat([
            torch.arange(x_nnz, dtype=torch.int64, device=device),
            torch.arange(y_nnz, dtype=torch.int64, device=device),
        ])

        key = all_row * ncol + all_col
        _, sort_idx = torch.sort(key, stable=True)
        sorted_key = key[sort_idx]
        sorted_src = all_src[sort_idx]
        sorted_idx = all_idx[sort_idx]
        keep = torch.ones(sorted_key.numel(), dtype=torch.bool, device=device)
        keep[1:] = sorted_key[1:] != sorted_key[:-1]
        unique_key = sorted_key[keep]
        out_nnz = int(unique_key.numel())

        out_row = (unique_key // ncol).to(torch.int64)
        out_col = (unique_key % ncol).to(torch.int32)
        output_index = keep.long().cumsum(0) - 1

        x_mask = sorted_src == 0
        y_mask = ~x_mask
        x_pick = sorted_idx[x_mask].contiguous()
        x_target = output_index[x_mask].contiguous()
        y_pick = sorted_idx[y_mask].contiguous()
        y_target = output_index[y_mask].contiguous()

        counts = torch.bincount(out_row, minlength=nrow)
        crow_l = torch.zeros(nrow + 1, dtype=torch.int64, device=device)
        crow_l[1:] = counts.cumsum(0)
        out_crow = crow_l.to(torch.int32)

        if topology_cache is not None:
            topology_cache["out_crow"] = out_crow
            topology_cache["out_col"] = out_col
            topology_cache["x_target"] = x_target
            topology_cache["x_pick"] = x_pick
            topology_cache["y_target"] = y_target
            topology_cache["y_pick"] = y_pick

    out_vals = torch.zeros((out_nnz, BR, BC), dtype=dtype, device=device)
    if x_pick.numel() > 0:
        out_vals.index_add_(0, x_target, alpha * x_vals_c[x_pick])
    if y_pick.numel() > 0:
        out_vals.index_add_(0, y_target, y_vals_c[y_pick])

    return _build_bsr_output(out_crow, out_col, out_vals, nrow, ncol, BR, BC)


class BlockJacobi:
    def __init__(self, A, ridge: float = 1e-12):
        crow, col, vals3d, (BR, BC), (nrow, _), dtype, device = _bsr_to_torch(A)
        if BR != BC:
            raise ValueError("Block-Jacobi requires square diagonal blocks")
        self.BR = BR
        self.nrow = nrow
        self.dtype = dtype
        self.device = device
        self.ridge = ridge
        nnz = int(col.numel())
        if nnz > 0:
            crow_l = crow.to(torch.int64)
            col_l = col.to(torch.int64)
            block_row = torch.repeat_interleave(
                torch.arange(nrow, device=device, dtype=torch.int64),
                crow_l[1:] - crow_l[:-1],
            )
            is_diag = block_row == col_l
            self._diag_idx = is_diag.nonzero(as_tuple=False).flatten()
            self._diag_rows = block_row[self._diag_idx]
        else:
            self._diag_idx = torch.empty(0, dtype=torch.int64, device=device)
            self._diag_rows = torch.empty(0, dtype=torch.int64, device=device)
        self._diag_blocks = torch.zeros(nrow, BR, BR, dtype=dtype, device=device)
        self._eye = torch.eye(BR, dtype=dtype, device=device)
        self._out_buf = torch.empty(nrow, BR, 1, dtype=dtype, device=device)
        self.diag_inv = None
        self.refresh(A)

    def refresh(self, A):
        _, _, vals3d, _, _, _, _ = _bsr_to_torch(A)
        self._diag_blocks.zero_()
        if self._diag_idx.numel() > 0:
            self._diag_blocks[self._diag_rows] = vals3d[self._diag_idx]
        self.diag_inv = torch.linalg.inv(self._diag_blocks + self.ridge * self._eye)

    def __call__(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.reshape(-1, self.BR, 1)
        torch.matmul(self.diag_inv, x, out=self._out_buf)
        return self._out_buf.view(-1)


def cg(matvec, b, x=None, M=None, tol: float = 1e-5, maxiter: Optional[int] = None, *, r_buf=None, p_buf=None):
    if x is None:
        x = torch.zeros_like(b)

    if maxiter is None or maxiter == 0:
        maxiter = b.numel()

    b_flat = b.reshape(-1)
    b_norm_sq = torch.dot(b_flat, b_flat).item()

    if b_norm_sq == 0.0:
        return x

    atol_sq = (tol ** 2) * b_norm_sq

    Ax = matvec(x)
    if r_buf is None:
        r = b - Ax
    else:
        r = r_buf
        torch.sub(b, Ax, out=r)
    r_flat = r.reshape(-1)

    z = M(r) if M is not None else r
    z_flat = z.reshape(-1)

    if p_buf is None:
        p = z.clone()
    else:
        p = p_buf
        p.copy_(z)

    rz = torch.dot(r_flat, z_flat)
    r_norm_sq = torch.dot(r_flat, r_flat)

    for _ in range(maxiter):
        if r_norm_sq.item() <= atol_sq:
            break

        Ap = matvec(p)
        Ap_flat = Ap.reshape(-1)
        alpha = (rz / torch.dot(p.reshape(-1), Ap_flat)).item()
        x.add_(p, alpha=alpha)
        r.add_(Ap, alpha=-alpha)

        if M is not None:
            z = M(r)
            z_flat = z.reshape(-1)

        rz_new = torch.dot(r_flat, z_flat)
        beta = (rz_new / rz).item()
        p.mul_(beta).add_(z)
        rz = rz_new
        r_norm_sq = torch.dot(r_flat, r_flat)

    return x


__all__ = [
    "sparse_bsr_mv", "sparse_bsr_mm",
    "sparse_bsr_transposed", "sparse_bsr_axpy",
    "BlockJacobi", "cg",
]
