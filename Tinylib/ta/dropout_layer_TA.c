#include "dropout_layer_TA.h"
#include "utils_TA.h"

#include <stdlib.h>
#include <stdio.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

dropout_layer_TA make_dropout_layer_TA_new(int batch, int inputs, float probability, int w, int h, int c, int netnum)
{
    dropout_layer_TA l = {0};
    l.type = DROPOUT_TA;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);

    l.netnum = netnum;

    l.output = malloc(sizeof(float) * inputs*batch);
    l.delta = malloc(sizeof(float) * inputs*batch);

    l.forward_TA = forward_dropout_layer_TA_new;

    l.w = w;
    l.h = h;
    l.c = c;

    //char prob[20];
    //ftoa(probability,prob,3);
    return l;
}

void forward_dropout_layer_TA_new(dropout_layer_TA l, network_TA net)
{
    int i;
    if (!net.train) return;

    float *pter;
    if(l.netnum == 0){
        for(i = 0; i < l.batch * l.inputs; ++i){
            l.output[i] = net.input[i];
        }

        pter = l.output;
    }else{
        pter = net.input;
    }

    for(i = 0; i < l.batch * l.inputs; ++i){
        //printf("i = %d; total = %d\n",i, l.batch * l.inputs);
        float r = rand_uniform_TA(0, 1);
        l.rand[i] = r;
        if(r < l.probability)   pter[i] = 0;
        else    pter[i] *= l.scale;
    }
}
