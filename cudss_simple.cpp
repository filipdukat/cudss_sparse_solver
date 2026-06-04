#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cudss.h"
#include "matrix_loader.h"
#include <chrono>

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

// Extract only lower triangle from full CSR matrix.
// Needed because symmetric cuDSS mode expects only one half of the matrix.
void extractLowerTriangleCSR(
    const SparseMatrixCSR & input,
    SparseMatrixCSR & output)
{
    output.nrows = input.nrows;
    output.ncols = input.ncols;

    output.rowOffsets.clear();
    output.colIndices.clear();
    output.values.clear();

    output.rowOffsets.resize(input.nrows + 1);
    output.rowOffsets[0] = 0;

    for (int64_t row = 0; row < input.nrows; row++)
    {
        for (int64_t idx = input.rowOffsets[row];
            idx < input.rowOffsets[row + 1];
            idx++)
        {
            int64_t col = input.colIndices[idx];

            if (row >= col)
            {
                output.colIndices.push_back(col);
                output.values.push_back(input.values[idx]);
            }
        }

        output.rowOffsets[row + 1] =
            output.colIndices.size();
    }

    output.nnz = output.values.size();
}

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

    // Create lower-triangle version for symmetric solver mode.
    SparseMatrixCSR csr_lower;
    extractLowerTriangleCSR(csr, csr_lower);

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

    // Switch between general LU solver and symmetric LDL^T solver.
    // Let user choose solver mode at runtime.
    int solverChoice = 0;

    printf("Choose solver mode:\n");
    printf("1 - General LU\n");
    printf("2 - Symmetric LDL^T\n");
    printf("Selection: ");

    scanf_s("%d", &solverChoice);

    // Number of benchmark repetitions.
    int numberOfRuns = 1;

    printf("Number of runs: ");
    scanf_s("%d", &numberOfRuns);

    // RHS vector type selection.
    int rhsChoice = 1;

    printf("\n");
    printf("Choose RHS vector:\n");
    printf("1 - Original vector from file\n");
    printf("2 - Dense vector (all ones)\n");
    printf("3 - Random vector\n");
    printf("Selection: ");

    scanf_s("%d", &rhsChoice);

    printf("Selected RHS: ");

    if (rhsChoice == 1)
    {
        printf("Original vector\n");
    }
    else if (rhsChoice == 2)
    {
        printf("All ones\n");
    }
    else if (rhsChoice == 3)
    {
        printf("Random vector\n");
    }

    if (numberOfRuns < 1)
    {
        numberOfRuns = 1;
    }

    // Replace RHS vector depending on benchmark mode.
    if (rhsChoice == 2)
    {
        for (int64_t i = 0; i < (int64_t)b_host.size(); i++)
        {
            b_host[i] = 1.0;
        }
    }
    else if (rhsChoice == 3)
    {
        srand(12345);

        for (int64_t i = 0; i < (int64_t)b_host.size(); i++)
        {
            b_host[i] =
                (double)rand() / RAND_MAX;
        }
    }

    bool useSymmetricSolver = (solverChoice == 2);

    printf("Selected mode: %s\n",
        useSymmetricSolver
        ? "Symmetric LDL^T"
        : "General LU");

    // Select correct number of non-zero elements depending on solver mode.
    int64_t nnz =
        useSymmetricSolver
        ? csr_lower.nnz
        : csr.nnz;

    int64_t nrhs = 1;

    printf("Rows: %lld\n", n);
    printf("NNZ : %lld\n", nnz);

    // HOST POINTERS
    // 
    // Select correct matrix representation depending on solver mode.
    int64_t* csr_offsets_h =
        useSymmetricSolver
        ? csr_lower.rowOffsets.data()
        : csr.rowOffsets.data();

    int64_t* csr_columns_h =
        useSymmetricSolver
        ? csr_lower.colIndices.data()
        : csr.colIndices.data();

    double* csr_values_h =
        useSymmetricSolver
        ? csr_lower.values.data()
        : csr.values.data();

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

    // Measure available GPU memory before allocations.
    size_t freeMemBefore = 0;
    size_t totalMem = 0;

    CUDA_CALL_AND_CHECK(
        cudaMemGetInfo(&freeMemBefore, &totalMem),
        "mem before");

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

    // Timing variables for performance tests.
    std::chrono::high_resolution_clock::time_point t1, t2;
    double analysisTimeMs = 0.0;
    double factorizationTimeMs = 0.0;
    double solveTimeMs = 0.0;

    // Benchmark statistics.
    double totalAnalysisTimeMs = 0.0;
    double totalFactorizationTimeMs = 0.0;
    double totalSolveTimeMs = 0.0;

    double minAnalysisTimeMs = 1e100;
    double minFactorizationTimeMs = 1e100;
    double minSolveTimeMs = 1e100;

    double maxAnalysisTimeMs = 0.0;
    double maxFactorizationTimeMs = 0.0;
    double maxSolveTimeMs = 0.0;

    double analysisMemoryMB = 0.0;
    double factorizationMemoryMB = 0.0;
    double solveMemoryMB = 0.0;
    double peakMemoryMB = 0.0;
    double factorStorageMB = 0.0;

    double totalAnalysisMemoryMB = 0.0;
    double totalFactorizationMemoryMB = 0.0;
    double totalSolveMemoryMB = 0.0;
    double totalPeakMemoryMB = 0.0;
    double totalFactorStorageMB = 0.0;

    std::vector<double> analysisTimes;
    std::vector<double> factorizationTimes;
    std::vector<double> solveTimes;

    // SOLVER CONFIG
    cudssConfig_t solverConfig;

    CUDSS_CALL_AND_CHECK(
        cudssConfigCreate(&solverConfig),
        status,
        "config create");

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
    // Configure solver type depending on selected performance tests mode.
    cudssMatrixType_t mtype =
        useSymmetricSolver
        ? CUDSS_MTYPE_SYMMETRIC
        : CUDSS_MTYPE_GENERAL;

    cudssMatrixViewType_t mview =
        useSymmetricSolver
        ? CUDSS_MVIEW_LOWER
        : CUDSS_MVIEW_FULL;

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

    for (int run = 0; run < numberOfRuns; run++)
    {
        printf("\n");
        printf("========================================\n");
        printf("RUN %d / %d\n", run + 1, numberOfRuns);
        printf("========================================\n");

        cudssData_t solverData;

        CUDSS_CALL_AND_CHECK(
            cudssDataCreate(handle, &solverData),
            status,
            "data create");

        // ANALYSIS
        printf("Running analysis...\n");

        t1 = std::chrono::high_resolution_clock::now();

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

        CUDA_CALL_AND_CHECK(
            cudaStreamSynchronize(stream),
            "analysis sync");

        t2 = std::chrono::high_resolution_clock::now();

        analysisTimeMs =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        size_t freeAfterAnalysis = 0;

        CUDA_CALL_AND_CHECK(
            cudaMemGetInfo(&freeAfterAnalysis, &totalMem),
            "analysis mem");

        analysisMemoryMB =
            (freeMemBefore - freeAfterAnalysis) / (1024.0 * 1024.0);

        peakMemoryMB = analysisMemoryMB;

        // FACTORIZATION
        printf("Running factorization...\n");

        t1 = std::chrono::high_resolution_clock::now();

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

        CUDA_CALL_AND_CHECK(
            cudaStreamSynchronize(stream),
            "factorization sync");

        t2 = std::chrono::high_resolution_clock::now();

        factorizationTimeMs =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        size_t freeAfterFactorization = 0;

        CUDA_CALL_AND_CHECK(
            cudaMemGetInfo(&freeAfterFactorization, &totalMem),
            "factor mem");

        factorizationMemoryMB =
            (freeMemBefore - freeAfterFactorization) / (1024.0 * 1024.0);

        if (factorizationMemoryMB > peakMemoryMB)
            peakMemoryMB = factorizationMemoryMB;

       factorStorageMB =
    factorizationMemoryMB - analysisMemoryMB;

        // SOLVE
        printf("Running solve...\n");

        t1 = std::chrono::high_resolution_clock::now();

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
            "solve sync");

        t2 = std::chrono::high_resolution_clock::now();

        solveTimeMs =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        totalAnalysisTimeMs += analysisTimeMs;
        totalFactorizationTimeMs += factorizationTimeMs;
        totalSolveTimeMs += solveTimeMs;

        analysisTimes.push_back(analysisTimeMs);
        factorizationTimes.push_back(factorizationTimeMs);
        solveTimes.push_back(solveTimeMs);

        if (analysisTimeMs < minAnalysisTimeMs)
            minAnalysisTimeMs = analysisTimeMs;

        if (analysisTimeMs > maxAnalysisTimeMs)
            maxAnalysisTimeMs = analysisTimeMs;

        if (factorizationTimeMs < minFactorizationTimeMs)
            minFactorizationTimeMs = factorizationTimeMs;

        if (factorizationTimeMs > maxFactorizationTimeMs)
            maxFactorizationTimeMs = factorizationTimeMs;

        if (solveTimeMs < minSolveTimeMs)
            minSolveTimeMs = solveTimeMs;

        if (solveTimeMs > maxSolveTimeMs)
            maxSolveTimeMs = solveTimeMs;


        // Measure total GPU memory usage including internal cuDSS allocations.
        size_t freeMemAfterSolver = 0;

        CUDA_CALL_AND_CHECK(
            cudaMemGetInfo(&freeMemAfterSolver, &totalMem),
            "mem after solver");

        solveMemoryMB =
            (freeMemBefore - freeMemAfterSolver) / (1024.0 * 1024.0);

        if (solveMemoryMB > peakMemoryMB)
            peakMemoryMB = solveMemoryMB;

        totalAnalysisMemoryMB += analysisMemoryMB;
        totalFactorizationMemoryMB += factorizationMemoryMB;
        totalSolveMemoryMB += solveMemoryMB;
        totalPeakMemoryMB += peakMemoryMB;
        totalFactorStorageMB += factorStorageMB;

        cudssDataDestroy(handle, solverData);
    }

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

// =========================================
// RESIDUAL CHECK ||Ax - b||
// =========================================

    double residual_norm = 0.0;
    double b_norm = 0.0;

    for (int64_t row = 0; row < n; row++)
    {
        double ax = 0.0;

        for (int64_t idx = csr.rowOffsets[row];
            idx < csr.rowOffsets[row + 1];
            idx++)
        {
            int64_t col = csr.colIndices[idx];
            ax += csr.values[idx] * x_values_h[col];
        }

        double residual = ax - b_values_h[row];

        residual_norm += residual * residual;
        b_norm += b_values_h[row] * b_values_h[row];
    }

    residual_norm = sqrt(residual_norm);
    b_norm = sqrt(b_norm);

    double relative_residual = residual_norm / b_norm;

    double averageAnalysisTimeMs =
        totalAnalysisTimeMs / numberOfRuns;

    double averageFactorizationTimeMs =
        totalFactorizationTimeMs / numberOfRuns;

    double averageSolveTimeMs =
        totalSolveTimeMs / numberOfRuns;

    double averageAnalysisMemoryMB =
        totalAnalysisMemoryMB / numberOfRuns;

    double averageFactorizationMemoryMB =
        totalFactorizationMemoryMB / numberOfRuns;

    double averageSolveMemoryMB =
        totalSolveMemoryMB / numberOfRuns;

    double averagePeakMemoryMB =
        totalPeakMemoryMB / numberOfRuns;

    double averageFactorStorageMB =
        totalFactorStorageMB / numberOfRuns;

    printf("========================================\n");
    printf("INDIVIDUAL RUN TIMES\n");
    printf("========================================\n");

    for (size_t i = 0; i < solveTimes.size(); i++)
    {
        printf(
            "Run %zu -> Analysis: %.3f ms | Factorization: %.3f ms | Solve: %.3f ms\n",
            i + 1,
            analysisTimes[i],
            factorizationTimes[i],
            solveTimes[i]);
    }

    printf("\n");

    printf("========================================\n");
    printf("RESIDUAL CHECK\n");
    printf("========================================\n");
    printf("||Ax - b|| = %.12e\n", residual_norm);
    printf("||b||= %.12e\n", b_norm);
    printf("relative residual = %.12e\n", relative_residual);

    printf("========================================\n");
    printf("PERFORMANCE TESTS RESULTS\n");
    printf("========================================\n");
    printf("Average analysis time      : %.3f ms\n",
        averageAnalysisTimeMs);
    printf("Min analysis time          : %.3f ms\n",
        minAnalysisTimeMs);
    printf("Max analysis time          : %.3f ms\n",
        maxAnalysisTimeMs);

    printf("Average factorization time : %.3f ms\n",
        averageFactorizationTimeMs);
    printf("Min factorization time     : %.3f ms\n",
        minFactorizationTimeMs);
    printf("Max factorization time     : %.3f ms\n",
        maxFactorizationTimeMs);

    printf("Average solve time         : %.3f ms\n",
        averageSolveTimeMs);
    printf("Min solve time             : %.3f ms\n",
        minSolveTimeMs);
    printf("Max solve time             : %.3f ms\n",
        maxSolveTimeMs);

    printf("Average memory after analysis     : %.3f MB\n",
        averageAnalysisMemoryMB);

    printf("Average memory after factorization: %.3f MB\n",
        averageFactorizationMemoryMB);

    printf("Average memory after solve        : %.3f MB\n",
        averageSolveMemoryMB);

    printf("Average factor storage            : %.3f MB\n",
        averageFactorStorageMB);

    printf("Average peak GPU memory           : %.3f MB\n",
        averagePeakMemoryMB);


    // CLEANUP
    cudssMatrixDestroy(A);
    cudssMatrixDestroy(b);
    cudssMatrixDestroy(x);

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