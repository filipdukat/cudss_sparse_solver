#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudss.h"
#include "matrix_loader.h"

#define CUDA_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        cudaError_t cuda_error = call;                                                   \
        if (cuda_error != cudaSuccess) {                                                 \
            printf("CUDA ERROR: %s\n", msg);                                             \
            return -1;                                                                   \
        }                                                                                \
    } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                          \
    do {                                                                                 \
        status = call;                                                                   \
        if (status != CUDSS_STATUS_SUCCESS) {                                            \
            printf("CUDSS ERROR: %s\n", msg);                                            \
            return -2;                                                                   \
        }                                                                                \
    } while (0);

int main()
{
    printf("========================================\n");
    printf("cuDSS Sparse Solver\n");
    printf("========================================\n");

    // =========================================
    // LOAD COO MATRIX
    // =========================================

    printf("Loading COO matrix...\n");

    SparseMatrixCOO coo;

    bool ok = loadMatrixCOO(
        "combline_Filter_6pole_coarse_L1_A.bin",
        coo);

    if (!ok)
    {
        return -1;
    }

    // =========================================
    // COO -> CSR
    // =========================================

    printf("Converting COO -> CSR...\n");

    SparseMatrixCSR csr;

    convertCOOtoCSR(coo, csr);

    // =========================================
    // LOAD RHS VECTOR
    // =========================================

    printf("Loading RHS vector...\n");

    std::vector<double> b_host;

    if (!loadVector(
        "combline_Filter_6pole_coarse_L1_b.bin",
        b_host))
    {
        return -1;
    }

    // BASIC INFO
    int64_t n = csr.nrows;
    int64_t nnz = csr.nnz;
    int64_t nrhs = 1;

    printf("Rows: %lld\n", n);
    printf("NNZ : %lld\n", nnz);

    // HOST POINTERS
    int64_t* csr_offsets_h =
        csr.rowOffsets.data();

    int64_t* csr_columns_h =
        csr.colIndices.data();

    double* csr_values_h =
        csr.values.data();

    double* b_values_h =
        b_host.data();

    double* x_values_h =
        (double*)malloc(n * sizeof(double));


    // DEVICE POINTERS
    int64_t* csr_offsets_d = NULL;
    int64_t* csr_columns_d = NULL;

    double* csr_values_d = NULL;
    double* b_values_d = NULL;
    double* x_values_d = NULL;


    // GPU MEMORY ALLOCATION
    printf("Allocating GPU memory...\n");

    CUDA_CALL_AND_CHECK(
        cudaMalloc(
            &csr_offsets_d,
            (n + 1) * sizeof(int64_t)),
        "csr_offsets");

    CUDA_CALL_AND_CHECK(
        cudaMalloc(
            &csr_columns_d,
            nnz * sizeof(int64_t)),
        "csr_columns");

    CUDA_CALL_AND_CHECK(
        cudaMalloc(
            &csr_values_d,
            nnz * sizeof(double)),
        "csr_values");

    CUDA_CALL_AND_CHECK(
        cudaMalloc(
            &b_values_d,
            n * sizeof(double)),
        "b_values");

    CUDA_CALL_AND_CHECK(
        cudaMalloc(
            &x_values_d,
            n * sizeof(double)),
        "x_values");


    // COPY TO GPU
    printf("Copying data to GPU...\n");

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            csr_offsets_d,
            csr_offsets_h,
            (n + 1) * sizeof(int64_t),
            cudaMemcpyHostToDevice),
        "csr_offsets memcpy");

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            csr_columns_d,
            csr_columns_h,
            nnz * sizeof(int64_t),
            cudaMemcpyHostToDevice),
        "csr_columns memcpy");

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            csr_values_d,
            csr_values_h,
            nnz * sizeof(double),
            cudaMemcpyHostToDevice),
        "csr_values memcpy");

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            b_values_d,
            b_values_h,
            n * sizeof(double),
            cudaMemcpyHostToDevice),
        "b_values memcpy");

 
    // CUDA STREAM
    cudaStream_t stream = NULL;

    CUDA_CALL_AND_CHECK(
        cudaStreamCreate(&stream),
        "stream create");

    // cuDSS HANDLE
    cudssHandle_t handle;
    cudssStatus_t status;

    CUDSS_CALL_AND_CHECK(
        cudssCreate(&handle),
        status,
        "cudssCreate");

    CUDSS_CALL_AND_CHECK(
        cudssSetStream(handle, stream),
        status,
        "cudssSetStream");

    // SOLVER CONFIG
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(
        cudssConfigCreate(&solverConfig),
        status,
        "config create");

    CUDSS_CALL_AND_CHECK(
        cudssDataCreate(handle, &solverData),
        status,
        "data create");

    // DENSE MATRICES
    cudssMatrix_t A, b, x;

    int64_t ldb = n;
    int64_t ldx = n;

    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateDn(
            &b,
            n,
            nrhs,
            ldb,
            b_values_d,
            CUDA_R_64F,
            CUDSS_LAYOUT_COL_MAJOR),
        status,
        "matrix b");

    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateDn(
            &x,
            n,
            nrhs,
            ldx,
            x_values_d,
            CUDA_R_64F,
            CUDSS_LAYOUT_COL_MAJOR),
        status,
        "matrix x");

    // SPARSE MATRIX
    cudssMatrixType_t mtype =
        CUDSS_MTYPE_GENERAL;

    cudssMatrixViewType_t mview =
        CUDSS_MVIEW_FULL;

    cudssIndexBase_t base =
        CUDSS_BASE_ZERO;

    CUDSS_CALL_AND_CHECK(
        cudssMatrixCreateCsr(
            &A,
            n,
            n,
            nnz,
            csr_offsets_d,
            NULL,
            csr_columns_d,
            csr_values_d,
            CUDA_R_64I,
            CUDA_R_64F,
            mtype,
            mview,
            base),
        status,
        "matrix A");

    // ANALYSIS
    printf("Running analysis...\n");

    CUDSS_CALL_AND_CHECK(
        cudssExecute(
            handle,
            CUDSS_PHASE_ANALYSIS,
            solverConfig,
            solverData,
            A,
            x,
            b),
        status,
        "analysis");

    // FACTORIZATION
    printf("Running factorization...\n");

    CUDSS_CALL_AND_CHECK(
        cudssExecute(
            handle,
            CUDSS_PHASE_FACTORIZATION,
            solverConfig,
            solverData,
            A,
            x,
            b),
        status,
        "factorization");

    // SOLVE
    printf("Running solve...\n");

    CUDSS_CALL_AND_CHECK(
        cudssExecute(
            handle,
            CUDSS_PHASE_SOLVE,
            solverConfig,
            solverData,
            A,
            x,
            b),
        status,
        "solve");

    CUDA_CALL_AND_CHECK(
        cudaStreamSynchronize(stream),
        "sync");

    // COPY X BACK
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(
            x_values_h,
            x_values_d,
            n * sizeof(double),
            cudaMemcpyDeviceToHost),
        "x memcpy");

    printf("========================================\n");
    printf("SOLVE FINISHED\n");
    printf("========================================\n");

    printf("First 10 x values:\n");

    for (int i = 0; i < 10; i++)
    {
        printf("x[%d] = %.10f\n",
            i,
            x_values_h[i]);
    }


    // CLEANUP
    cudssMatrixDestroy(A);
    cudssMatrixDestroy(b);
    cudssMatrixDestroy(x);

    cudssDataDestroy(handle, solverData);
    cudssConfigDestroy(solverConfig);
    cudssDestroy(handle);

    cudaFree(csr_offsets_d);
    cudaFree(csr_columns_d);
    cudaFree(csr_values_d);
    cudaFree(b_values_d);
    cudaFree(x_values_d);

    free(x_values_h);

    return 0;
}