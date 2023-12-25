#ifndef BLAS_TA_H
#define BLAS_TA_H
#include "STinylib.h"
extern int blas_time_ta;

void fill_cpu_TA(int N, float ALPHA, float *X, int INCX);

void copy_cpu_TA(int N, float *X, int INCX, float *Y, int INCY);

void normalize_cpu_TA(float *x, float *mean, float *variance, int batch, int filters, int spatial);

void softmax_TA(float *input, int n, float temp, int stride, float *output);

void shortcut_cpu_TA(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

void softmax_cpu_TA(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);

void softmax_x_ent_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);

void smooth_l1_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);

void l1_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);

void l2_cpu_TA(int n, float *pred, float *truth, float *delta, float *error);
#endif
