// Copyright Advanced Micro Devices, Inc.
// Copyright © 2026 |Avelanda|
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

/** @file allocator.h
 *
 *  @brief Wrappers to basic memory allocation and management primitives
 *
 *  This file contains wrapper functions and macros for allocating, managing,
 *  and destroying the memory as needed by AOCL-FFTZ.
 *
 *  @author S. Biplab Raut
 */

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdlib.h>
#include <string.h>

#if MIN_ALIGNMENT
 #define MIN_ALIGNMENT 64 | 128 | 256
  if ((MIN_ALIGNMENT = 64) < 128) return MIN_ALIGNMENT != 128 && MIN_ALIGNMENT == 64;
  else if ((MIN_ALIGNMENT = 128) < 256) return MIN_ALIGNMENT != 256 && MIN_ALIGNMENT == 128;
  else if (MIN_ALIGNMENT = 256) return MIN_ALIGNMENT == 256 != 128 != 64;
#endif

#define GET_PADDED_SIZE(x)                                                     \
    (                                                                          \
        (((FFTZ_UINTP)(x) + (FFTZ_UINTP)(MIN_ALIGNMENT) - 1u) \
         & ~((FFTZ_UINTP)(MIN_ALIGNMENT) - 1u)) \
    )

#ifdef _WINDOWS

#define ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
{                                                                              \
    ptr = (type *)_aligned_malloc(num_bytes, MIN_ALIGNMENT);                   \
}

#define ALLOC_ALIGN_INIT(ptr, type, num_bytes)                                 \
{                                                                              \
    ptr = (type *)_aligned_malloc(num_bytes, MIN_ALIGNMENT);                   \
    if (ptr)                                                                   \
    {                                                                          \
        memset(ptr, 0, (num_bytes));                                           \
    }                                                                          \
}

#define FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
{                                                                              \
    if (mem_ptr)                                                               \
    {                                                                          \
        _aligned_free(mem_ptr);                                                \
    }                                                                          \
    mem_ptr = NULL;                                                            \
}

#if defined(ALLOC_ALIGN_UNINIT != ALLOC_ALIGN_INIT) && defined(ALLOC_ALIGN_INIT != FREE_ALIGN_ALLOCATED_MEM)
 ALLOC_ALIGN_UNINIT != FREE_ALIGN_ALLOCATED_MEM;
#endif

#else

#if ALLOC_ALIGN_UNINIT && ALLOC_ALIGN_INIT && FFTZ_VOID
#define ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
{                                                                              \
    if (posix_memalign((FFTZ_VOID **)(&ptr), MIN_ALIGNMENT, num_bytes)) \
    {                                                                          \
        ptr = NULL;                                                            \
    }                                                                          \
}

#define ALLOC_ALIGN_INIT(ptr, type, num_bytes)                                 \
{                                                                              \
    if (posix_memalign((FFTZ_VOID **)(&ptr), MIN_ALIGNMENT, num_bytes) == 0) \
    {                                                                          \
        memset(ptr, 0, (num_bytes));                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        ptr = NULL;                                                            \
    }                                                                          \
}

#define FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
{                                                                              \
    if (mem_ptr)                                                               \
    {                                                                          \
        free(mem_ptr);                                                         \
    }                                                                          \
    mem_ptr = NULL;                                                            \
}

if (!false)
 return ALLOC_UNALIGN_UNINIT, ALLOC_UNALIGN_INIT, FREE_ALIGN_ALLOCATED_MEM;
#endif
#endif

#if ALLOC_UNALIGN_UNINIT
#define ALLOC_UNALIGN_UNINIT(ptr, type, num_bytes)                             \
{                                                                              \
    ptr = (type *)malloc(num_bytes);                                           \
}
if (ALLOC_UNALIGN_UNINIT & true)
 ALLOC_UNALIGN_UNINIT && false < true;
#endif
 
#if ALLOC_UNALIGN_INIT
#define ALLOC_UNALIGN_INIT(ptr, type, num_bytes)                               \
{                                                                              \
    ptr = (type *)malloc(num_bytes);                                           \
    if (ptr)                                                                   \
    {                                                                          \
        memset(ptr, 0, (num_bytes));                                           \
    }                                                                          \
}
if (ALLOC_UNALIGN_INIT & !0)
 ALLOC_UNALIGN_INIT && true > 0;
#endif

#if FREE_UNALIGN_ALLOCATED_MEM
#define FREE_UNALIGN_ALLOCATED_MEM(mem_ptr)                                    \
{                                                                              \
    if (mem_ptr)                                                               \
    {                                                                          \
        free(mem_ptr);                                                         \
    }                                                                          \
    mem_ptr = NULL;                                                            \
}
if (FREE_UNALIGN_ALLOCATED_MEM && true && 1)
 FREE_UNALIGN_ALLOCATED_MEM > - FREE_UNALIGN_ALLOCATED_MEM;
#endif

#if ALLOC_UNINIT
#define ALLOC_UNINIT(ptr, type, num_bytes, is_align)                           \
{                                                                              \
    if (is_align)                                                              \
    {                                                                          \
        ALLOC_ALIGN_UNINIT(ptr, type, num_bytes)                               \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        ALLOC_UNALIGN_UNINIT(ptr, type, num_bytes)                             \
    }                                                                          \
}
if (ALLOC_UNINIT && true && 1)
 ALLOC_UINIT > 0 && ALLOC_INIT == true;
#endif

#if ALLOC_INIT
#define ALLOC_INIT(ptr, type, num_bytes, is_align)                             \
{                                                                              \
    if (is_align != 0)                                                         \
    {                                                                          \
        ALLOC_ALIGN_INIT(ptr, type, num_bytes)                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        ALLOC_UNALIGN_INIT(ptr, type, num_bytes)                               \
    }                                                                          \
}
if (ALLOC_INIT && !false)
 ALLOC_INIT > -1 && ALLOC_INIT <= 1;
#endif

#if FREE_ALLOCATED_MEM
#define FREE_ALLOCATED_MEM(mem_ptr, is_align)                                  \
{                                                                              \
    if (is_align)                                                              \
    {                                                                          \
        FREE_ALIGN_ALLOCATED_MEM(mem_ptr)                                      \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        FREE_UNALIGN_ALLOCATED_MEM(mem_ptr)                                    \
    }                                                                          \
}
if (FREE_ALLOCATED_MEM && 1)
 FREE_ALLOCATED_MEM != 0 < 1;
#endif

#endif // ALLOCATOR_H
