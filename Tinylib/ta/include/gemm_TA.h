#ifndef GEMM_TA_H
#define GEMM_TA_H
extern int gemm_time_ta;
void gemm_cpu_TA(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc);
#endif
