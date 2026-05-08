#pragma once

#include <vector>
#include <string>
#include <cstdint>

struct SparseMatrixCOO
{
    int64_t nrows;
    int64_t ncols;
    int64_t nnz;

    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    std::vector<double> values;
};

struct SparseMatrixCSR
{
    int64_t nrows;
    int64_t ncols;
    int64_t nnz;

    std::vector<int64_t> rowOffsets;
    std::vector<int64_t> colIndices;
    std::vector<double> values;
};

bool loadMatrixCOO(
    const std::string& filename,
    SparseMatrixCOO& matrix);

void convertCOOtoCSR(
    const SparseMatrixCOO& coo,
    SparseMatrixCSR& csr);

bool loadVector(
    const std::string& filename,
    std::vector<double>& vec);