#include <immintrin.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>


#ifndef MATRIX_H    
#define MATRIX_H

typedef struct Matrix {
    float* data;
    int m;
    int n;
    int stride;
} Matrix;

Matrix matCreate(int m, int n);
int matSize(Matrix mat);
int matSizeBytes(Matrix mat);
void matFree(Matrix mat);

Matrix matReadFromFile(int m, int n, FILE* file);

Matrix matSubmatrix(Matrix mat, int r1, int r2, int c1, int c2);

Matrix matSliceCols(Matrix mat, int c1, int c2);
Matrix matSliceRows(Matrix mat, int r1, int r2);
Matrix matSelectRow(Matrix mat, int rowIdx);

void matFillZeros(Matrix mat);
void matInplaceScalarProd(Matrix mat, float scalar);
void matInplaceSigmoid(Matrix mat);
void matInplaceTanh(Matrix mat);
void matCopy(Matrix src, Matrix dst);
void matSum(Matrix mat_a, Matrix mat_b, Matrix out);
void matHProduct(Matrix mat_a, Matrix mat_b, Matrix out);
void matVecProduct(Matrix mat_a, Matrix mat_b, Matrix out);
void matSlerp(Matrix mat_a, Matrix mat_b, Matrix t, Matrix out);
void matSlerpZero(Matrix mat_a, Matrix t, Matrix out);
int matVecArgmax(Matrix a);

#endif