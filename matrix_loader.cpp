#include "matrix_loader.h"

#include <iostream>
#include <fstream>

bool loadMatrixCOO(
    const std::string& filename,
    SparseMatrixCOO& matrix)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cout << "Cannot open matrix file!"
            << std::endl;

        return false;
    }

    uint8_t complexFlag;

    file.read((char*)&matrix.nrows, sizeof(int64_t));
    file.read((char*)&matrix.ncols, sizeof(int64_t));
    file.read((char*)&matrix.nnz, sizeof(int64_t));
    file.read((char*)&complexFlag, sizeof(uint8_t));

    matrix.rows.resize(matrix.nnz);
    matrix.cols.resize(matrix.nnz);
    matrix.values.resize(matrix.nnz);

    file.read(
        (char*)matrix.rows.data(),
        matrix.nnz * sizeof(int64_t));

    file.read(
        (char*)matrix.cols.data(),
        matrix.nnz * sizeof(int64_t));

    file.read(
        (char*)matrix.values.data(),
        matrix.nnz * sizeof(double));

    file.close();

    return true;
}

void convertCOOtoCSR(
    const SparseMatrixCOO& coo,
    SparseMatrixCSR& csr)
{
    csr.nrows = coo.nrows;
    csr.ncols = coo.ncols;
    csr.nnz = coo.nnz;

    csr.rowOffsets.resize(csr.nrows + 1, 0);
    csr.colIndices.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    for (int64_t i = 0; i < coo.nnz; i++)
    {
        csr.rowOffsets[coo.rows[i] + 1]++;
    }

    for (int64_t i = 0; i < csr.nrows; i++)
    {
        csr.rowOffsets[i + 1] += csr.rowOffsets[i];
    }

    std::vector<int64_t> currentOffset =
        csr.rowOffsets;

    for (int64_t i = 0; i < coo.nnz; i++)
    {
        int64_t row = coo.rows[i];

        int64_t dest =
            currentOffset[row]++;

        csr.colIndices[dest] =
            coo.cols[i];

        csr.values[dest] =
            coo.values[i];
    }
}

bool loadVector(
    const std::string& filename,
    std::vector<double>& vec)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cout << "Cannot open vector file!"
            << std::endl;

        return false;
    }

    uint64_t size;
    uint8_t isComplex;

    file.read(
        (char*)&size,
        sizeof(uint64_t));

    file.read(
        (char*)&isComplex,
        sizeof(uint8_t));

    std::cout << "Vector size: "
        << size
        << std::endl;

    std::cout << "isComplex: "
        << (int)isComplex
        << std::endl;

    vec.resize(size);

    file.read(
        (char*)vec.data(),
        size * sizeof(double));

    int nonzeroCount = 0;

    for (size_t i = 0; i < vec.size(); i++)
    {
        if (vec[i] != 0.0)
        {
            nonzeroCount++;

            if (nonzeroCount <= 10)
            {
                printf("nonzero b[%zu] = %.10f\n",
                    i,
                    vec[i]);
            }
        }
    }

    printf("Total nonzero b values: %d\n",
        nonzeroCount);

    file.close();

    return true;
}