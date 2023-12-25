#ifndef GEMM_H
#define GEMM_H

extern float gemm_time;        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm_nn_fast(int M, int N, int K, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#endif
