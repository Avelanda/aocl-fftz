// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file packed_solver_rdft.c
 *
 *  @brief Real-packed R2C/C2R solver.
 *
 *  setup_real_packed_solver copies the parent decomp onto the child,
 *  reconfigures it as a complex FFT(M) with packed-safe placement, and
 *  binds the recombine (forward) / separate (backward) kernel. The selector
 *  then models that child and attaches next_sol. The recombine/separate
 *  twiddle table is built afterwards by the common selector twiddle setup.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "selector/selector.h"
#include "utils/allocator.h"

FFTZ_INT32 setup_real_packed_solver(aoclfftz_solution_t *sol,
                                    aoclfftz_solution_t *complex_sol,
                                    kernel_tables_t *kernel_tables)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_decomp_scheme_t *parent_decomp = sol->decomp_scheme;
    FFTZ_INT32 dt_prec = DT_PRECISION_FLAG(parent_decomp->flags);
    FFTZ_INT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    FFTZ_UINT8 direction = FFT_DIR(parent_decomp->flags);
    FFTZ_INTP n = parent_decomp->dims[0].n;
    FFTZ_INTP m = n >> 1; // child complex FFT length (N real values -> N/2 complex values)

    FFTZ_INT32 ret =
        copy_decomp_scheme(complex_sol->decomp_scheme, parent_decomp);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_decomp_scheme failed: %s", get_status_string(ret));
        return ret;
    }

    // Reconfigure the child as a complex, 1D, single-batch, unit-stride
    // FFT(M). Inherit flags / control / thread info from the copy, then
    // override dims, vector layout, the complex flag and placement.
    aoclfftz_decomp_scheme_t *complex_decomp = complex_sol->decomp_scheme;
    complex_decomp->dim_rank = 1;
    complex_decomp->vec_rank = 1;
    complex_decomp->dims[0].n = m;
    complex_decomp->dims[0].in_stride = 1;
    complex_decomp->dims[0].out_stride = 1;
    complex_decomp->vecs[0].n = 1;
    complex_decomp->vecs[0].in_stride = m;
    complex_decomp->vecs[0].out_stride = m;

    SET_COMPLEX(complex_decomp->flags);
    SET_FFT_DIR(complex_decomp->flags, direction);
    complex_decomp->thread_info->avl_threads = 1;
    complex_decomp->thread_info->n_threads = 1;

    // Child CFFT buffers are chosen so overwrites stay safe.
    // R2C recombine may overwrite the child output, so placement matches the
    // parent. C2R separate must not overwrite the Hermitian input, so placement
    // is flipped: in-place parent -> out-of-place child (scratch);
    // out-of-place parent -> in-place child (on out).
    FFTZ_UINT8 parent_oop = IS_OUT_OF_PLACE(parent_decomp->flags);
    FFTZ_UINT8 child_oop =
        (direction == FORWARD_FFT_DIR) ? parent_oop : !parent_oop;
    if (child_oop)
    {
        SET_OUTOFPLACE(complex_decomp->flags);
        complex_decomp->in_real = NULL;
        complex_decomp->out_real = NULL;
        complex_decomp->in_imag = NULL;
        complex_decomp->out_imag = NULL;
    }
    else
    {
        SET_INPLACE(complex_decomp->flags);
        FFTZ_VOID *buf = (direction == FORWARD_FFT_DIR)
                             ? parent_decomp->in_real
                             : parent_decomp->out_real;
        complex_decomp->in_real = buf;
        complex_decomp->out_real = buf;
        complex_decomp->in_imag =
            (buf != NULL) ? MOVE_ADDR(buf, dt_bytes) : NULL;
        complex_decomp->out_imag = complex_decomp->in_imag;
    }

    sol->dft_bufs->pack_rdft = (direction == FORWARD_FFT_DIR)
                                 ? kernel_tables->packed_rdft.recombine
                                 : kernel_tables->packed_rdft.separate;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

static FFTZ_INT32 execute_real_packed_r2c(aoclfftz_solution_t *sol,
                                          aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *complex_sol = sol->next_sol;
    FFTZ_INT32 dt_bytes = CTX_DT_SIZE(ctx);
    FFTZ_INTP m = sol->decomp_scheme->dims[0].n >> 1;
    FFTZ_VOID *in_real = ctx->in_real;
    FFTZ_VOID *out_real = ctx->out_real;

    // The N contiguous real values are the M interleaved complex child input.
    aoclfftz_mutable_ctx_t c2c_child_ctx = *ctx;
    c2c_child_ctx.in_real = in_real;
    c2c_child_ctx.in_imag = MOVE_ADDR(in_real, dt_bytes);
    c2c_child_ctx.out_real = out_real;
    c2c_child_ctx.out_imag = MOVE_ADDR(out_real, dt_bytes);
    c2c_child_ctx.flags = complex_sol->decomp_scheme->flags;
    FFTZ_INT32 status =
        complex_sol->solver->execute_solver(complex_sol, &c2c_child_ctx);
    if (status != SOLVER_SUCCESS)
    {
        return status;
    }

    // R2C recombine may overwrite: child spectrum and Hermitian result share
    // out_real.
    sol->dft_bufs->pack_rdft(out_real, out_real, sol->twiddle->twiddle_buf_ptr,
                           m);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

static FFTZ_INT32 execute_real_packed_c2r(aoclfftz_solution_t *sol,
                                          aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *complex_sol = sol->next_sol;
    FFTZ_INT32 dt_bytes = CTX_DT_SIZE(ctx);
    FFTZ_INTP m = sol->decomp_scheme->dims[0].n >> 1;
    FFTZ_VOID *in_real = ctx->in_real;
    FFTZ_VOID *out_real = ctx->out_real;
    // C2R separate must not overwrite the Hermitian input.
    // Out-of-place C2R writes the double-length complex sequence into out_real.
    // In-place C2R still holds HC in in==out, so that sequence goes to packed
    // scratch and the child is out-of-place scratch -> out.
    FFTZ_VOID *cout = out_real;
    if (!IS_OUT_OF_PLACE(ctx->flags))
    {
        FFTZ_INTP slot_bytes = GET_PADDED_SIZE(2 * m * dt_bytes);
        cout = MOVE_ADDR(ctx->real_packed_buf_base,
                         (FFTZ_INTP)ctx->slot_idx * slot_bytes);
    }

    sol->dft_bufs->pack_rdft(cout, in_real, sol->twiddle->twiddle_buf_ptr, m);

    aoclfftz_mutable_ctx_t c2c_child_ctx = *ctx;
    c2c_child_ctx.in_real = cout;
    c2c_child_ctx.in_imag = MOVE_ADDR(cout, dt_bytes);
    c2c_child_ctx.out_real = out_real;
    c2c_child_ctx.out_imag = MOVE_ADDR(out_real, dt_bytes);
    c2c_child_ctx.flags = complex_sol->decomp_scheme->flags;
    FFTZ_INT32 status =
        complex_sol->solver->execute_solver(complex_sol, &c2c_child_ctx);
    if (status != SOLVER_SUCCESS)
    {
        return status;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_real_packed_r2c(FFTZ_VOID)
{
    return execute_real_packed_r2c;
}

dft_solver_ register_execute_real_packed_c2r(FFTZ_VOID)
{
    return execute_real_packed_c2r;
}
