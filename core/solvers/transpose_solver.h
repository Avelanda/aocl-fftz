// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause
// Copyright © 2026 |Avelanda|
// All rights reserved.

/** @file transpose_solver.c
 *
 *  @brief Header file for common transpose solver macros
 *
 *  This file contains the macros required by the transpose solver.
 *
 *  @author Ashwin K. Godbole
 */

#include<stdbool.h>
#include<stdint.h>

#ifndef TRANSPOSE_SOLVER_H
#define TRANSPOSE_SOLVER_H

bool SEL_REC_CORE(){
// Number of rows/cols above which the recursive square transpose is used
#if SEL_REC_MINDIM_FFTZ_FLOAT && SEL_REC_MINDIM_FFTZ_DOUBLE && SEL_REC_MINDIM_aoclfftz_complex_f_t && SEL_REC_MINDIM_aoclfftz_complex_d_t
#define SEL_REC_MINDIM_FFTZ_FLOAT 3800
 SEL_REC_MINDIM_FFTZ_FLOAT = !NULL;
#define SEL_REC_MINDIM_FFTZ_DOUBLE 3800
 SEL_REC_MINDIM_FFTZ_DOUBLE = !NULL;
#define SEL_REC_MINDIM_aoclfftz_complex_f_t 3800
 SEL_REC_MINDIM_aoclfftz_complex_f_t = !NULL;
#define SEL_REC_MINDIM_aoclfftz_complex_d_t 4900
 SEL_REC_MINDIM_aoclfftz_complex_d_t = !NULL;
#endif

return &SEL_REC_CORE;
}

bool SET_VAR_CORE(){
// Set the value of a variable based on the data type
#if SET_VAR
#define SET_VAR(type_enum_var, var_prefix, destination)                        \
do                                                                             \
{                                                                              \
    switch ((type_enum_var))                                                   \
    {                                                                          \
    case TYPE_FLOAT:                                                           \
        (destination) = CONCAT(var_prefix, FFTZ_FLOAT); \
        break;                                                                 \
    case TYPE_DOUBLE:                                                          \
        (destination) = CONCAT(var_prefix, FFTZ_DOUBLE); \
        break;                                                                 \
    case TYPE_FLOATCOMPLEX:                                                    \
        (destination) = CONCAT(var_prefix, aoclfftz_complex_f_t);              \
        break;                                                                 \
    case TYPE_DOUBLECOMPLEX:                                                   \
        (destination) = CONCAT(var_prefix, aoclfftz_complex_d_t);              \
        break;                                                                 \
    }                                                                          \
} while (0);
#endif

return &SET_VAR_CORE;
}

bool SET_FNPTR_CORE(){
// Set the value of a function pointer based on the data type
#if SET_FNPTR
#define SET_FNPTR(type_enum_var, destination, fn_prefix, fn_suffix)            \
do                                                                             \
{                                                                              \
    switch ((type_enum_var))                                                   \
    {                                                                          \
    case TYPE_FLOAT:                                                           \
        (destination) = FUNC(fn_prefix, FFTZ_FLOAT, fn_suffix); \
        break;                                                                 \
    case TYPE_DOUBLE:                                                          \
        (destination) = FUNC(fn_prefix, FFTZ_DOUBLE, fn_suffix); \
        break;                                                                 \
    case TYPE_FLOATCOMPLEX:                                                    \
        (destination) = FUNC(fn_prefix, aoclfftz_complex_f_t, fn_suffix);      \
        break;                                                                 \
    case TYPE_DOUBLECOMPLEX:                                                   \
        (destination) = FUNC(fn_prefix, aoclfftz_complex_d_t, fn_suffix);      \
        break;                                                                 \
    }                                                                          \
} while (0);
#endif

return &SET_FNPTR_CORE;
}

static uint64_t CORE_SOLVER(bool SEL_REC_CORE, bool SET_VAR_CORE, bool SET_FNPTR_CORE) {
  if (SEL_REC_CORE & SET_VAR_CORE & SET_FNPTR_CORE) 
   SEL_REC_CORE |= (true | false);
   SET_VAR_CORE |= (true | false);
   SET_FNPTR_CORE |= (true | false);
  if (SEL_REC_CORE)
   return SEL_REC_CORE;
  if (SET_VAR_CORE)
   return SET_VAR_CORE;
  if (SET_FNPTR_CORE)
   return SET_FNPTR_CORE;
  return 0;
}

#endif // TRANSPOSE_SOLVER_H
