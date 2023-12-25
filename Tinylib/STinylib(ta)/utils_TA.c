#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fdlibm.h>
#include "utils_TA.h"

float sum_array_TA(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    
    return sum;
}

float rand_uniform_TA(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }    
    //return ((float)rand() * (max - min)) + min;
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

float rand_normal_TA()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    if(haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI_TA;

    return sqrt(rand1) * cos(rand2);
}

/*
float rand_normal_TA(float mu, float sigma){
    float U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;
    
    if (call == 1)
    {
        call = !call;
        return (mu + sigma * (float) X2);
    }
    
    do
    {
        U1 = -1 + (float) rand () * 2;
        U2 = -1 + (float) rand () * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);
    
    mult = sqrt ((-2 * log (W)) / W); //ta_log
    X1 = U1 * mult;
    X2 = U2 * mult;
    
    call = !call;
    
    return (mu + sigma * (float) X1);
}
*/