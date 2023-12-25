#include "gemm_TA.h"
#include "utils_TA.h"
#include "network_TA.h"
#include "STinylib.h"

#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}
int gemm_time_ta=0;

void gemm_nn_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{

    //printf("1=%d,2=%d,3=%d,ALPHA=%d,1=%d,2=%d,3=%d\n",M,K,N,ALPHA,lda,ldb,ldc);
    //debug_plot("A",A, K*M);
    //debug_plot("B",B, K*N);
    ///debug_plot("C",C, N*M);

    //printf("stoping");

    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            //printf("///////////////////\n");
            //printf("sum");
            for(k = 0; k < K; ++k){
                //if(k % 1 == 0 & roundnum == 2){
                    //printf("+ %f",ALPHA*A[i*lda+k]*B[j*ldb + k]);
                    //printf("1=%f,2=%d,3=%f,4=%d,5=%f\n",ALPHA*A[i*lda+k]*B[j*ldb + k], i*lda+k, A[i*lda+k], j*ldb + k, B[j*ldb + k]);
                //}
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            //printf("\n");
            //printf("j=%d,sum=%f\n",j,sum);

            //printf("///////////////////\n");
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
            //printf("M=%d,N=%d,K=%d,lda=%d,ldb=%d,ldc=%d\n",M,N,K,lda,ldb,ldc);
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt_TA(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
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


void gemm_cpu_TA(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }

    if(!TA && !TB)
        gemm_nn_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt_TA(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);

TEE_GetSystemTime(&p2);

        gemm_time_ta+= time_diff(&p1, &p2);
        
}
