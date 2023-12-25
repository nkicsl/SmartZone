#ifndef UTILS_H_TA
#define UTILS_H_TA
#include <stdio.h>
#include "STinylib.h"

float sum_array_TA(float *a, int n);
float rand_uniform_TA(float min, float max);
//float rand_normal_TA(float mu, float sigma);
float rand_normal_TA();
#define TWO_PI_TA 6.2831853071795864769252866f
#endif
