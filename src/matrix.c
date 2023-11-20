#include <math.h>
#include <stdlib.h>

#include <avx_mathfun/avx_mathfun.h>

#include "matrix.h"


Matrix matCreate(int m, int n) {
    assert(n % 8 == 0);

    Matrix mat; 
    mat.data = aligned_alloc(32, m * n * sizeof(float));
    mat.m = m;
    mat.n = n;
    mat.stride = n;

    return mat;
}

void matFree(Matrix mat) {
    free(mat.data);
}

Matrix matReadFromFile(int m, int n, FILE* file) {
    Matrix mat = matCreate(m, n);
    fread(mat.data, sizeof(float), matSize(mat), file);

    return mat;
}

int matSize(Matrix mat) { return mat.n * mat.m; }
int matSizeBytes(Matrix mat) { return mat.n * mat.m * sizeof(float); }

Matrix matSubmatrix(Matrix mat, int r1, int r2, int c1, int c2) {
    Matrix submat;

    assert(0 <= r1 && r1 < r2 && r2 <= mat.m && 
           0 <= c1 && c1 < c2 && c2 <= mat.n);
    assert(c1 % 8 == 0 && c2 % 8 == 0);

    submat.data = mat.data + r1 * mat.stride + c1;
    submat.m = r2 - r1;
    submat.n = c2 - c1;
    submat.stride = mat.stride;

    return submat;
}

Matrix matSliceCols(Matrix mat, int c1, int c2) {
    return matSubmatrix(mat, 0, mat.m, c1, c2);
}

Matrix matSliceRows(Matrix mat, int r1, int r2) {
    return matSubmatrix(mat, r1, r2, 0, mat.n);
}

Matrix matSelectRow(Matrix mat, int rowIdx) {
    return matSubmatrix(mat, rowIdx, rowIdx + 1, 0, mat.n);
}

void matFillZeros(Matrix mat) {
    __m256 zeros = _mm256_setzero_ps();

    for (int row = 0; row < mat.m; row++) {
        for (int col = 0; col < mat.n; col+=8) {
            _mm256_store_ps(&mat.data[row * mat.stride + col], zeros);
        }
    }
}

void matInplaceScalarProd(Matrix mat, float scalar) {
    __m256 vec_scalar = _mm256_set1_ps(scalar); 

    for (int row = 0; row < mat.m; row++) {
        for (int col = 0; col < mat.n; col += 8) {
            __m256 vec_mat = _mm256_load_ps(&mat.data[row * mat.stride + col]);
            vec_mat = _mm256_mul_ps(vec_mat, vec_scalar);
            _mm256_store_ps(&mat.data[row * mat.stride + col], vec_mat);
        }
    }
}

void matCopy(Matrix src, Matrix dst) {
    for (int row = 0; row < src.m; row++) {
        for (int col = 0; col < src.n; col += 8) {
            __m256 vec_a = _mm256_load_ps(&src.data[row * src.stride + col]);
            _mm256_store_ps(&dst.data[row * dst.stride + col], vec_a);
        }
    }
}

void matSum(Matrix mat_a, Matrix mat_b, Matrix out) {
    for (int row = 0; row < mat_a.m; row++) {
        for (int col = 0; col < mat_a.n; col += 8) {
            __m256 vec_a = _mm256_load_ps(&mat_a.data[row * mat_a.stride + col]);
            __m256 vec_b = _mm256_load_ps(&mat_b.data[row * mat_b.stride + col]);
            __m256 vec_c = _mm256_add_ps(vec_a, vec_b);
            _mm256_store_ps(&out.data[row * out.stride + col], vec_c);
        }
    }
}

void matHProduct(Matrix mat_a, Matrix mat_b, Matrix out) { 
    for (int row = 0; row < mat_a.m; row++) {
        for (int col = 0; col < mat_a.n; col += 8) {
            __m256 vec_a = _mm256_load_ps(&mat_a.data[row * mat_a.stride + col]);
            __m256 vec_b = _mm256_load_ps(&mat_b.data[row * mat_b.stride + col]);
            __m256 vec_c = _mm256_mul_ps(vec_a, vec_b);
            _mm256_store_ps(&out.data[row * out.stride + col], vec_c);
        }
    }
}

/*
 * c = a * (1 - t) + b * t
 */
void matSlerp(Matrix mat_a, Matrix mat_b, Matrix t, Matrix out) {
    __m256 vec_ones = _mm256_set1_ps(1.0f);

    for (int row = 0; row < mat_a.m; row++) {
        for (int col = 0; col < mat_a.n; col += 8) {
            __m256 vec_a = _mm256_load_ps(&mat_a.data[row * mat_a.stride + col]);
            __m256 vec_b = _mm256_load_ps(&mat_b.data[row * mat_b.stride + col]);
            __m256 vec_t = _mm256_load_ps(&t.data[row * t.stride + col]);

            vec_a = _mm256_mul_ps(_mm256_sub_ps(vec_ones, vec_t), vec_a);
            vec_a = _mm256_add_ps(vec_a, _mm256_mul_ps(vec_t, vec_b));

            _mm256_store_ps(&out.data[row * out.stride + col], vec_a);
        }
    }
}

void matSlerpZero(Matrix mat_a, Matrix t, Matrix out) {
    __m256 vec_ones = _mm256_set1_ps(1.0f);

    for (int row = 0; row < mat_a.m; row++) {
        for (int col = 0; col < mat_a.n; col += 8) {
            __m256 vec_a = _mm256_load_ps(&mat_a.data[row * mat_a.stride + col]);
            __m256 vec_t = _mm256_load_ps(&t.data[row * t.stride + col]);

            vec_a = _mm256_mul_ps(_mm256_sub_ps(vec_ones, vec_t), vec_a);
            _mm256_store_ps(&out.data[row * out.stride + col], vec_a);
        }
    }
}

void matInplaceSigmoid(Matrix a) {
#ifdef USE_AVX_EXP

    __m256 ones = _mm256_set1_ps(1.0f);
    __m256 zeros = _mm256_setzero_ps();
    __m256 x;

    for (int row = 0; row < a.m; row++) {
        for (int col = 0; col < a.n; col += 8) {
            x = _mm256_load_ps(&a.data[row * a.stride + col]);
            x = _mm256_sub_ps(zeros, x);
            x = exp256_ps(x);
            x = _mm256_add_ps(ones, x);
            x = _mm256_div_ps(ones, x);

            _mm256_store_ps(&a.data[row * a.stride + col], x);
        }
    }

#else

    for (int row = 0; row < a.m; row++) {
        for (int col = 0; col < a.n; col++) {
            float x = a.data[row * a.stride + col];
            a.data[row * a.stride + col] = 1 / (1 + exp(-x));
        }
    }

#endif
}

void matInplaceTanh(Matrix a) {
#ifdef USE_AVX_EXP

    __m256 minus_two = _mm256_set1_ps(-2.0f);
    __m256 ones = _mm256_set1_ps(1.0f);
    __m256 x, nom, denom;

    for (int row = 0; row < a.m; row++) {
        for (int col = 0; col < a.n; col += 8) {
            x = _mm256_load_ps(&a.data[row * a.stride + col]);
            x = _mm256_mul_ps(minus_two, x);
            x = exp256_ps(x);
            nom = _mm256_sub_ps(ones, x);
            denom = _mm256_add_ps(ones, x);
            x = _mm256_div_ps(nom, denom);
            
            _mm256_store_ps(&a.data[row * a.stride + col], x);
        }
    }

#else

    for (int row = 0; row < a.m; row++) {
        for (int col = 0; col < a.n; col++) {
            float x = a.data[row * a.stride + col];
            float e = exp(-2*x);
            a.data[row * a.stride + col] = (1 - e) / (1 + e);
        }
    }

#endif
}

void matVecProduct(Matrix mat, Matrix vec, Matrix out) {
    float result[8] __attribute__((aligned(32)));

    for (int row = 0; row < mat.m; row++) {
        __m256 sum = _mm256_setzero_ps();

        for (int col = 0; col < mat.n; col += 8) {
            __m256 vec_a = _mm256_load_ps(&mat.data[row * mat.stride + col]);
            __m256 vec_b = _mm256_load_ps(&vec.data[col]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(vec_a, vec_b));
        }

        _mm256_store_ps(result, sum);

        for (int i = 0; i < 8; i++) {
            out.data[row] += result[i];
        }
    }
}

int matVecArgmax(Matrix a) {
    float max_val = a.data[0];
    int argmax = 0;

    for (int i = 0; i < a.n; i++) {
        float val = a.data[i];

        if (max_val < val) {
            max_val = val;
            argmax = i;
        }
    }

    return argmax;
}

