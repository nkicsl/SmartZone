#include "softmax_layer_TA.h"
#include "blas_TA.h"
#include "utils_TA.h"

//#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

int soft_time_ta=0;
static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}
softmax_layer_TA make_softmax_layer_TA_new(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss)
{
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    assert(inputs%groups == 0);
    //IMSG("softmax_TA                                     %4d\n",  inputs);
    softmax_layer_TA l = {0};
    l.type = SOFTMAX_TA;
    l.batch = batch;
    l.groups = groups;

    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.temperature = temperature;
    l.w = w;
    l.h = h;
    l.c = c;
    l.spatial = spatial;
    l.noloss = noloss;

    l.forward_TA = forward_softmax_layer_TA;

    TEE_GetSystemTime(&p2);
    soft_time_ta+=time_diff(&p1, &p2);

    return l;
}

void forward_softmax_layer_TA(softmax_layer_TA l, network_TA net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu_TA(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu_TA(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu_TA(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array_TA(l.loss, l.batch*l.inputs);
    }
}

