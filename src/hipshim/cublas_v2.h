#pragma once

// Shadow of <cublas_v2.h>. hipBLAS declares the same operations with the hipblas*/HIP*
// spelling, and the tree's one cuBLAS consumer (w4_cublas_prefill.cu) uses the handle calls
// and the enumerators below, so that is the whole table.

#include "detail/api.h"

#include <hipblas/hipblas.h>

// --- types -------------------------------------------------------------------------------
using cublasHandle_t = hipblasHandle_t;
using cublasStatus_t = hipblasStatus_t;

// --- enumerators ---------------------------------------------------------------------------
// The cuBLAS enums and hipBLAS's carry the same values; the data-type names the tree
// passes to cublasGemmEx are CUDA's spellings for what hipBLAS calls HIP_R_*.
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_COMPUTE_32I HIPBLAS_COMPUTE_32I
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#ifndef CUDA_R_8I
#define CUDA_R_8I HIP_R_8I
#endif
#ifndef CUDA_R_32I
#define CUDA_R_32I HIP_R_32I
#endif

// --- functions -----------------------------------------------------------------------------
#define cublasCreate hipblasCreate
#define cublasSetStream hipblasSetStream
#define cublasGemmEx hipblasGemmEx
