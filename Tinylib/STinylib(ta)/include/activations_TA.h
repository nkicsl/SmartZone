#ifndef ACTIVATIONS_TA_H
#define ACTIVATIONS_TA_H
#include "STinylib.h"
#include <fdlibm.h>
#include <stdio.h>
extern int active_time_ta;

ACTIVATION_TA get_activation_TA(char *s);

float activate_TA(float x, ACTIVATION_TA a);

float*  activate_array_TA(float *x, const int n, const ACTIVATION_TA a);

static inline float stair_activate_TA(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float hardtan_activate_TA(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate_TA(float x){return x;}
static inline float logistic_activate_TA(float x){return 1./(1. + exp(-x));}
static inline float loggy_activate_TA(float x){return 2./(1. + exp(-x)) - 1;}
static inline float relu_activate_TA(float x){return x*(x>0);}
static inline float elu_activate_TA(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float selu_activate_TA(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline float relie_activate_TA(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate_TA(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate_TA(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate_TA(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float plse_activate_TA(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

static inline float lhtan_activate_TA(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}

#endif
