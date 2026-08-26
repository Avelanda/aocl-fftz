// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file packed_rdft_avx512_common.h
 *
 * @brief 512-bit conjugate-pair butterflies shared by the real-packed
 * recombine and separate kernels.
 *
 * Same bodies as the 128-bit helpers at four times the width; see
 * packed_rdft_avx128_common.h for the algebra. AVX512 kernel sources
 * include the narrower helpers they use for tail descent.
 *
 * @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_PACKED_RDFT_AVX512_COMMON_H
#define AOCLFFTZ_PACKED_RDFT_AVX512_COMMON_H

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

/**
 * @brief Reverses the order of the complex numbers held in a 512-bit register
 * for single precision floating point: [c0..c7] -> [c7..c0].
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define REV_CPLX_512_S(v)                                                      \
    _mm512_permutexvar_ps(_mm512_set_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11,   \
                                           10, 13, 12, 15, 14),                \
                          (v))

/**
 * @brief Reverses the order of the complex numbers held in a 512-bit register
 * for double precision floating point: [c0..c3] -> [c3..c0].
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define REV_CPLX_512_D(v)                                                      \
    _mm512_permutexvar_pd(_mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6), (v))

static const __m512 v_half_512_s = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                    0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                                    0.5f, 0.5f, 0.5f, 0.5f};
static const __m512d v_half_512_d = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
static const __m512 v_sign_512_s = {-2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f,
                                    -2.0f, 2.0f, -2.0f, 2.0f, -2.0f, 2.0f,
                                    -2.0f, 2.0f, -2.0f, 2.0f};
static const __m512d v_sign_512_d = {-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0,
                                     2.0};

/**
 * Packed twiddle: SWAP_RI of operand `d`, then complex mul by (tw_re, tw_im).
 * Direction is conjugation of `d` in the caller, not a sign in this helper.
 * AVX-512 has no addsub; negate even lanes of (tw_im * d) then add.
 */
static inline __m512 apply_packed_twiddle_512_s(__m512 d, __m512 tw_re,
                                                __m512 tw_im)
{
    __m512 rot_arg = SWAP_RI_512_S(d);
    __m512 rot_tmp = _mm512_mul_ps(tw_im, d);
    return _mm512_add_ps(_mm512_mul_ps(tw_re, rot_arg),
                         _mm512_xor_ps(rot_tmp, SWAP_RI_512_S(_conj_512_f.s)));
}

static inline __m512d apply_packed_twiddle_512_d(__m512d d, __m512d tw_re,
                                                 __m512d tw_im)
{
    __m512d rot_arg = SWAP_RI_512_D(d);
    __m512d rot_tmp = _mm512_mul_pd(tw_im, d);
    return _mm512_add_pd(_mm512_mul_pd(tw_re, rot_arg),
                         _mm512_xor_pd(rot_tmp, SWAP_RI_512_D(_conj_512_d.d)));
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_512_S fp32 pairs starting at k and mirrored pairs ending
 * at m-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_avx512_fp32(
    FFTZ_FLOAT *p_out, const FFTZ_FLOAT *p_cout, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP m, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (m - k - NUM_SETS_512_S + 1);
    __m512 v_cout_k = _mm512_loadu_ps(p_cout + 2 * k);
    __m512 v_cout_mk_raw = _mm512_loadu_ps(p_cout + mk_base);
    __m512 v_cout_mk = REV_CPLX_512_S(v_cout_mk_raw);
    __m512 v_cout_mk_conj = CONJ_512_S(v_cout_mk);
    __m512 v_sum = _mm512_add_ps(v_cout_k, v_cout_mk_conj);
    __m512 v_diff_conj = CONJ_512_S(_mm512_sub_ps(v_cout_mk_conj, v_cout_k));
    __m512 v_tw_re = _mm512_loadu_ps(p_tw_re + 2 * k);
    __m512 v_tw_im = _mm512_loadu_ps(p_tw_im + 2 * k);
    __m512 v_rot =
        apply_packed_twiddle_512_s(v_diff_conj, v_tw_re, v_tw_im);
    __m512 v_out_k = _mm512_add_ps(_mm512_mul_ps(v_half_512_s, v_sum), v_rot);
    __m512 v_out_mk =
        CONJ_512_S(_mm512_sub_ps(_mm512_mul_ps(v_half_512_s, v_sum), v_rot));
    _mm512_storeu_ps(p_out + 2 * k, v_out_k);
    _mm512_storeu_ps(p_out + mk_base, REV_CPLX_512_S(v_out_mk));
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_512_D fp64 pairs starting at k and mirrored pairs ending
 * at m-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_avx512_fp64(
    FFTZ_DOUBLE *p_out, const FFTZ_DOUBLE *p_cout, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP m, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (m - k - NUM_SETS_512_D + 1);
    __m512d v_cout_k = _mm512_loadu_pd(p_cout + 2 * k);
    __m512d v_cout_mk_raw = _mm512_loadu_pd(p_cout + mk_base);
    __m512d v_cout_mk = REV_CPLX_512_D(v_cout_mk_raw);
    __m512d v_cout_mk_conj = CONJ_512_D(v_cout_mk);
    __m512d v_sum = _mm512_add_pd(v_cout_k, v_cout_mk_conj);
    __m512d v_diff_conj = CONJ_512_D(_mm512_sub_pd(v_cout_mk_conj, v_cout_k));
    __m512d v_tw_re = _mm512_loadu_pd(p_tw_re + 2 * k);
    __m512d v_tw_im = _mm512_loadu_pd(p_tw_im + 2 * k);
    __m512d v_rot =
        apply_packed_twiddle_512_d(v_diff_conj, v_tw_re, v_tw_im);
    __m512d v_out_k = _mm512_add_pd(_mm512_mul_pd(v_half_512_d, v_sum), v_rot);
    __m512d v_out_mk =
        CONJ_512_D(_mm512_sub_pd(_mm512_mul_pd(v_half_512_d, v_sum), v_rot));
    _mm512_storeu_pd(p_out + 2 * k, v_out_k);
    _mm512_storeu_pd(p_out + mk_base, REV_CPLX_512_D(v_out_mk));
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_512_S
 * fp32 pairs starting at k and mirrored pairs ending at m-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_avx512_fp32(
    FFTZ_FLOAT *p_cout, const FFTZ_FLOAT *p_in, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP m, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (m - k - NUM_SETS_512_S + 1);
    __m512 v_in_k = _mm512_loadu_ps(p_in + 2 * k);
    __m512 v_in_mk_raw = _mm512_loadu_ps(p_in + mk_base);
    __m512 v_in_mk = REV_CPLX_512_S(v_in_mk_raw);
    __m512 v_in_mk_conj = CONJ_512_S(v_in_mk);
    __m512 v_base = _mm512_add_ps(v_in_k, v_in_mk_conj);
    __m512 v_delta = _mm512_sub_ps(v_in_k, v_in_mk_conj);
    __m512 v_tw_re = _mm512_loadu_ps(p_tw_re + 2 * k);
    __m512 v_tw_im = _mm512_loadu_ps(p_tw_im + 2 * k);
    __m512 v_corr_half =
        apply_packed_twiddle_512_s(v_delta, v_tw_re, v_tw_im);
    __m512 v_corr = _mm512_mul_ps(v_corr_half, v_sign_512_s);
    __m512 v_cout_mk = CONJ_512_S(_mm512_sub_ps(v_base, v_corr));
    _mm512_storeu_ps(p_cout + 2 * k, _mm512_add_ps(v_base, v_corr));
    _mm512_storeu_ps(p_cout + mk_base, REV_CPLX_512_S(v_cout_mk));
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_512_D
 * fp64 pairs starting at k and mirrored pairs ending at m-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_avx512_fp64(
    FFTZ_DOUBLE *p_cout, const FFTZ_DOUBLE *p_in, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP m, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (m - k - NUM_SETS_512_D + 1);
    __m512d v_in_k = _mm512_loadu_pd(p_in + 2 * k);
    __m512d v_in_mk_raw = _mm512_loadu_pd(p_in + mk_base);
    __m512d v_in_mk = REV_CPLX_512_D(v_in_mk_raw);
    __m512d v_in_mk_conj = CONJ_512_D(v_in_mk);
    __m512d v_base = _mm512_add_pd(v_in_k, v_in_mk_conj);
    __m512d v_delta = _mm512_sub_pd(v_in_k, v_in_mk_conj);
    __m512d v_tw_re = _mm512_loadu_pd(p_tw_re + 2 * k);
    __m512d v_tw_im = _mm512_loadu_pd(p_tw_im + 2 * k);
    __m512d v_corr_half =
        apply_packed_twiddle_512_d(v_delta, v_tw_re, v_tw_im);
    __m512d v_corr = _mm512_mul_pd(v_corr_half, v_sign_512_d);
    __m512d v_cout_mk = CONJ_512_D(_mm512_sub_pd(v_base, v_corr));
    _mm512_storeu_pd(p_cout + 2 * k, _mm512_add_pd(v_base, v_corr));
    _mm512_storeu_pd(p_cout + mk_base, REV_CPLX_512_D(v_cout_mk));
}

#endif // AOCLFFTZ_PACKED_RDFT_AVX512_COMMON_H
