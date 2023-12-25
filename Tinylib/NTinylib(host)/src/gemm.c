#include "gemm.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

float gemm_time=0;

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

//void gemm_nn(int M, int N, int K, float ALPHA, 
      //  float *A, int lda, 
    //    float *B, int ldb,
     //   float *C, int ldc)
//{
    //int i,j,k;
    //#pragma omp parallel for
   // for(i = 0; i < M; ++i){
      //  for(k = 0; k < K; ++k){
            //register float A_PART = ALPHA*A[i*lda+k];
        //    register float A_PART = A[i*lda+k];
        //    for(j = 0; j < N; ++j){
         //       C[i*ldc+j] += A_PART*B[k*ldb+j];
         //   }
        //}
   // }

//}

void gemm_nn_fast(int M, int N, int K, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
   int i, j, k;

    for (i = 0; i < M; ++i) {
        int a = i * lda;
        int mcc = i * ldc;
        for (k = 0; k < K; k += 4) {
            register float A_PART0 = A[a + k];
            register float A_PART1 = A[a + k + 1];
            register float A_PART2 = A[a + k + 2];
            register float A_PART3 = A[a + k + 3];
            int mb0 = k * ldb;
            int mb1 = (k + 1) * ldb;
            int mb2 = (k + 2) * ldb;
            int mb3 = (k + 3) * ldb;
            int mc = mcc;
            for (j = 0; j < N; j++) {
                C[mc + j] += A_PART0 * B[mb0 + j] + A_PART1 * B[mb1 + j] + A_PART2 * B[mb2 + j] + A_PART3 * B[mb3 + j];
                //C[mc + j+1] += A_PART0 * B[mb0 + j+1] + A_PART1 * B[mb1 + j+1] + A_PART2 * B[mb2 + j+1] + A_PART3 * B[mb3 + j+1];
                //C[mc + j+2] += A_PART0 * B[mb0 + j+2] + A_PART1 * B[mb1 + j+2] + A_PART2 * B[mb2 + j+2] + A_PART3 * B[mb3 + j+2];
                //C[mc + j+3] += A_PART0 * B[mb0 + j+3] + A_PART1 * B[mb1 + j+3] + A_PART2 * B[mb2 + j+3] + A_PART3 * B[mb3 + j+3];
            }
        }
    }

}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            //register float A_PART = ALPHA*A[i*lda+k];
            register float A_PART = A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}


void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //fprintf(stderr, "cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    clock_t ttt;
             ttt=clock();
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }

    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);


    gemm_time+=sec(clock() - ttt);


}


