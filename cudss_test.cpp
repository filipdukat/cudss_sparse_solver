#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

struct SparseMatrixCOO
{
    uint64_t nrows;
    uint64_t ncols;
    uint64_t nnz;

    std::vector<uint64_t> rows;
    std::vector<uint64_t> cols;
    std::vector<double> values;
};

struct SparseMatrixCSR
{
    uint64_t nrows;
    uint64_t ncols;
    uint64_t nnz;

    std::vector<uint64_t> rowOffsets;
    std::vector<uint64_t> colIndices;
    std::vector<double> values;
};

bool loadMatrixCOO(
    const std::string& filename,
    SparseMatrixCOO& matrix)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cout << "Cannot open file!"
            << std::endl;

        return false;
    }

    uint8_t complexFlag;

    //wiersze
    file.read((char*)&matrix.nrows, sizeof(uint64_t));

    //kolumny
    file.read((char*)&matrix.ncols, sizeof(uint64_t));

    //liczba elementow niezerowych
    file.read((char*)&matrix.nnz, sizeof(uint64_t));

    //czy macierz jest zespolona
    file.read((char*)&complexFlag, sizeof(uint8_t));

    matrix.rows.resize(matrix.nnz);
    matrix.cols.resize(matrix.nnz);
    matrix.values.resize(matrix.nnz);

    file.read(
        (char*)matrix.rows.data(),
        matrix.nnz * sizeof(uint64_t));

    file.read(
        (char*)matrix.cols.data(),
        matrix.nnz * sizeof(uint64_t));

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


    // count elements per row
    for (uint64_t i = 0; i < coo.nnz; i++)
    {
        csr.rowOffsets[coo.rows[i] + 1]++;
    }

    // prefix sum
    for (uint64_t i = 0; i < csr.nrows; i++)
    {
        csr.rowOffsets[i + 1] += csr.rowOffsets[i];
    }

    // temp copy
    std::vector<uint64_t> currentOffset =
        csr.rowOffsets;

    // fill csr
    for (uint64_t i = 0; i < coo.nnz; i++)
    {
        uint64_t row = coo.rows[i];

        uint64_t dest =
            currentOffset[row]++;

        csr.colIndices[dest] =
            coo.cols[i];

        csr.values[dest] =
            coo.values[i];
    }
}

int main()
{
    SparseMatrixCOO coo;

    bool ok = loadMatrixCOO(
        "combline_Filter_6pole_coarse_A.bin",
        coo);

    if (!ok)
    {
        return 1;
    }

    std::cout << "COO loaded!"
        << std::endl;

    SparseMatrixCSR csr;

    std::cout << "Converting COO to CSR..."
        << std::endl;

    convertCOOtoCSR(coo, csr);

    std::cout << "CSR genereted successfully"
        << std::endl;

    std::cout << std::endl;

    std::cout << "Rows: "
        << csr.nrows
        << std::endl;

    std::cout << "NNZ: "
        << csr.nnz
        << std::endl;

    std::cout << std::endl;

    std::cout << "First 10 rowOffsets:"
        << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout
            << csr.rowOffsets[i]
            << std::endl;
    }

    return 0;
}