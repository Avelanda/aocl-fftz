// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file packed_rdft_common.h
 *
 * @brief Definitions shared by both real-packed kernel families.
 *
 * recombine (R2C, DIT last stage) and separate (C2R, DIF first stage) share
 * one conjugate-pair walk over the packed M-point spectrum, so the pair count
 * and the scalar butterflies live here once instead of in every ISA variant.
 *
 * @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_PACKED_RDFT_COMMON_H
#define AOCLFFTZ_PACKED_RDFT_COMMON_H

#include "core/kernels/kernel.h"

// Pair count for child length m (bins 1 .. floor((m-1)/2)).
// DC, Nyquist, and the even-m middle bin are done outside this loop.
#define REAL_PACKED_NUM_PAIRS(m) (((m) - 1) / 2)

// recombine (R2C): DIT last stage. Recombine child pair Z[k], Z[M-k] with one
// twiddle multiply and write Hermitian bins X[k], X[M-k] (N = 2M).
// Twiddle table is pre-broadcast with the 0.5 scale and conjugation baked in
// (tw_re = 0.5*cos, tw_im = -0.5*sin):
//   sum_re     = z_k_re + z_mk_re ; diff_im    = z_k_im - z_mk_im
//   rot_arg_re = z_k_im + z_mk_im ; rot_arg_im = z_mk_re - z_k_re
//   rot_re = tw_re*rot_arg_re - tw_im*rot_arg_im
//   rot_im = tw_re*rot_arg_im + tw_im*rot_arg_re
//   X[k]   = (0.5*sum_re + rot_re,  0.5*diff_im + rot_im)
//   X[M-k] = (0.5*sum_re - rot_re,  rot_im - 0.5*diff_im)
//
// `out` may overwrite `cout` (in-place child FFT): every input is read before
// any write, so that overwrite is safe. `static inline` so every ISA variant's
// main loop and SIMD tail share one copy with no call overhead.

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_C_S fp32 pairs starting at k and mirrored pairs ending at
 * m-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_fp32(
    FFTZ_FLOAT *out, const FFTZ_FLOAT *cout, const FFTZ_FLOAT *twr,
    const FFTZ_FLOAT *twi, FFTZ_INTP m, FFTZ_INTP k)
{
    FFTZ_FLOAT z_k_re = cout[2 * k];
    FFTZ_FLOAT z_k_im = cout[2 * k + 1];
    FFTZ_FLOAT z_mk_re = cout[2 * (m - k)];
    FFTZ_FLOAT z_mk_im = cout[2 * (m - k) + 1];
    FFTZ_FLOAT sum_re = z_k_re + z_mk_re;
    FFTZ_FLOAT diff_im = z_k_im - z_mk_im;
    FFTZ_FLOAT rot_arg_re = z_k_im + z_mk_im;
    FFTZ_FLOAT rot_arg_im = z_mk_re - z_k_re;
    FFTZ_FLOAT tw_re = twr[2 * k];
    FFTZ_FLOAT tw_im = twi[2 * k];
    FFTZ_FLOAT rot_re = tw_re * rot_arg_re - tw_im * rot_arg_im;
    FFTZ_FLOAT rot_im = tw_re * rot_arg_im + tw_im * rot_arg_re;
    FFTZ_FLOAT half_re = (FFTZ_FLOAT)0.5 * sum_re;
    FFTZ_FLOAT half_im = (FFTZ_FLOAT)0.5 * diff_im;
    out[2 * k] = half_re + rot_re;
    out[2 * k + 1] = half_im + rot_im;
    out[2 * (m - k)] = half_re - rot_re;
    out[2 * (m - k) + 1] = rot_im - half_im;
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_C_D fp64 pairs starting at k and mirrored pairs ending at
 * m-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_fp64(
    FFTZ_DOUBLE *out, const FFTZ_DOUBLE *cout, const FFTZ_DOUBLE *twr,
    const FFTZ_DOUBLE *twi, FFTZ_INTP m, FFTZ_INTP k)
{
    FFTZ_DOUBLE z_k_re = cout[2 * k];
    FFTZ_DOUBLE z_k_im = cout[2 * k + 1];
    FFTZ_DOUBLE z_mk_re = cout[2 * (m - k)];
    FFTZ_DOUBLE z_mk_im = cout[2 * (m - k) + 1];
    FFTZ_DOUBLE sum_re = z_k_re + z_mk_re;
    FFTZ_DOUBLE diff_im = z_k_im - z_mk_im;
    FFTZ_DOUBLE rot_arg_re = z_k_im + z_mk_im;
    FFTZ_DOUBLE rot_arg_im = z_mk_re - z_k_re;
    FFTZ_DOUBLE tw_re = twr[2 * k];
    FFTZ_DOUBLE tw_im = twi[2 * k];
    FFTZ_DOUBLE rot_re = tw_re * rot_arg_re - tw_im * rot_arg_im;
    FFTZ_DOUBLE rot_im = tw_re * rot_arg_im + tw_im * rot_arg_re;
    FFTZ_DOUBLE half_re = (FFTZ_DOUBLE)0.5 * sum_re;
    FFTZ_DOUBLE half_im = (FFTZ_DOUBLE)0.5 * diff_im;
    out[2 * k] = half_re + rot_re;
    out[2 * k + 1] = half_im + rot_im;
    out[2 * (m - k)] = half_re - rot_re;
    out[2 * (m - k) + 1] = rot_im - half_im;
}

// separate (C2R): DIF first stage. Separate Hermitian bins X[k], X[M-k] into
// the child pair that the M-point CFFT then transforms. Reconstructs
// 2*Z[k], 2*Z[M-k] from X[k], X[M-k] (N = 2M):
//   tw_re = p_tw_re[2k] ; tw_im = p_tw_im[2k]
//   sum_re   = x_k_re + x_mk_re ; diff_im  = x_k_im - x_mk_im
//   delta_re = x_k_re - x_mk_re ; delta_im = x_k_im + x_mk_im
//   corr_re = 2*( tw_re*delta_re + tw_im*delta_im)
//   corr_im = 2*(-tw_im*delta_re + tw_re*delta_im)
//   Z[k]   = (sum_re - corr_im, corr_re + diff_im)
//   Z[M-k] = (sum_re + corr_im, corr_re - diff_im)
//
// `cout` must not overwrite `in`: the solver uses a separate scratch spectrum
// since each pair is written only after both `in` entries are read. `static
// inline` so every ISA variant's main loop and SIMD tail share one copy with
// no call overhead.

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_C_S
 * fp32 pairs starting at k and mirrored pairs ending at m-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_fp32(
    FFTZ_FLOAT *cout, const FFTZ_FLOAT *in, const FFTZ_FLOAT *twr,
    const FFTZ_FLOAT *twi, FFTZ_INTP m, FFTZ_INTP k)
{
    FFTZ_FLOAT x_k_re = in[2 * k];
    FFTZ_FLOAT x_k_im = in[2 * k + 1];
    FFTZ_FLOAT x_mk_re = in[2 * (m - k)];
    FFTZ_FLOAT x_mk_im = in[2 * (m - k) + 1];
    FFTZ_FLOAT sum_re = x_k_re + x_mk_re;
    FFTZ_FLOAT diff_im = x_k_im - x_mk_im;
    FFTZ_FLOAT delta_re = x_k_re - x_mk_re;
    FFTZ_FLOAT delta_im = x_k_im + x_mk_im;
    FFTZ_FLOAT tw_re = twr[2 * k];
    FFTZ_FLOAT tw_im = twi[2 * k];
    FFTZ_FLOAT corr_re = (FFTZ_FLOAT)2 * (tw_re * delta_re + tw_im * delta_im);
    FFTZ_FLOAT corr_im = (FFTZ_FLOAT)2 * (-tw_im * delta_re + tw_re * delta_im);
    cout[2 * k] = sum_re - corr_im;
    cout[2 * k + 1] = corr_re + diff_im;
    cout[2 * (m - k)] = sum_re + corr_im;
    cout[2 * (m - k) + 1] = corr_re - diff_im;
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_C_D
 * fp64 pairs starting at k and mirrored pairs ending at m-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_fp64(
    FFTZ_DOUBLE *cout, const FFTZ_DOUBLE *in, const FFTZ_DOUBLE *twr,
    const FFTZ_DOUBLE *twi, FFTZ_INTP m, FFTZ_INTP k)
{
    FFTZ_DOUBLE x_k_re = in[2 * k];
    FFTZ_DOUBLE x_k_im = in[2 * k + 1];
    FFTZ_DOUBLE x_mk_re = in[2 * (m - k)];
    FFTZ_DOUBLE x_mk_im = in[2 * (m - k) + 1];
    FFTZ_DOUBLE sum_re = x_k_re + x_mk_re;
    FFTZ_DOUBLE diff_im = x_k_im - x_mk_im;
    FFTZ_DOUBLE delta_re = x_k_re - x_mk_re;
    FFTZ_DOUBLE delta_im = x_k_im + x_mk_im;
    FFTZ_DOUBLE tw_re = twr[2 * k];
    FFTZ_DOUBLE tw_im = twi[2 * k];
    FFTZ_DOUBLE corr_re =
        (FFTZ_DOUBLE)2 * (tw_re * delta_re + tw_im * delta_im);
    FFTZ_DOUBLE corr_im =
        (FFTZ_DOUBLE)2 * (-tw_im * delta_re + tw_re * delta_im);
    cout[2 * k] = sum_re - corr_im;
    cout[2 * k + 1] = corr_re + diff_im;
    cout[2 * (m - k)] = sum_re + corr_im;
    cout[2 * (m - k) + 1] = corr_re - diff_im;
}

#endif // AOCLFFTZ_PACKED_RDFT_COMMON_H
