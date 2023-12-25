#include "STinylib.h"

#include "batchnorm_layer_TA.h"
#include "blas_TA.h"
#include "convolutional_layer_TA.h"
#include <stdio.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}

void forward_batchnorm_layer_TA(layer_TA l, network_TA net)
{
    /*
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_Time p3 = { };
    TEE_Time p4 = { };
    TEE_Time p5 = { };
    TEE_Time p6 = { };

TEE_GetSystemTime(&p1);
  */ 
    if(l.type == BATCHNORM_TA) copy_cpu_TA(l.outputs*l.batch, net.input, 1, l.output, 1);
  
    //TEE_GetSystemTime(&p2);
    
    copy_cpu_TA(l.outputs*l.batch, l.output, 1, l.x, 1);

//TEE_GetSystemTime(&p3);

    normalize_cpu_TA(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
   
   // TEE_GetSystemTime(&p4);
    scale_bias_TA(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    //TEE_GetSystemTime(&p5);
    add_bias_TA(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
    //TEE_GetSystemTime(&p6);

/*
    uint32_t pp = time_diff(&p1, &p2);
    DMSG("bat0     ==%d \n",pp);

    pp = time_diff(&p2, &p3);
    DMSG("bat1     ==%d \n",pp);

    pp = time_diff(&p3, &p4);
    DMSG("bat2     ==%d \n",pp);

    pp = time_diff(&p4, &p5);
    DMSG("bat3     ==%d \n",pp);

    pp = time_diff(&p5, &p6);
    DMSG("bat4     ==%d \n",pp);
*/

}
