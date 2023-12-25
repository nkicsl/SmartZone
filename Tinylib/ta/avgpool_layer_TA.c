#include "avgpool_layer_TA.h"

#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

int avg_time_ta;
static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}
avgpool_layer_TA make_avgpool_layer_TA(int batch, int w, int h, int c)
{
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    avgpool_layer_TA l = {0};
    l.type = AVGPOOL_TA;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward_TA = forward_avgpool_layer_TA_new;
    TEE_GetSystemTime(&p2);
    avg_time_ta+=time_diff(&p1, &p2);

    return l;
}

void forward_avgpool_layer_TA_new(const avgpool_layer_TA l, network_TA net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
}