#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "utils_TA.h"
#include "STinylib.h"
#include "gemm_TA.h"
#include "blas_TA.h"
#include "activations_TA.h"
#include "batchnorm_layer_TA.h"
#include "connected_layer_TA.h"
#include "convolutional_layer_TA.h"

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

void forward_connected_layer_TA_new(layer_TA l, network_TA net)
{
    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;

    gemm_cpu_TA(0,1,m,n,k,1,a,k,b,k,1,c,n);

    if(l.batch_normalize){
        forward_batchnorm_layer_TA(l, net);
    } else {
        add_bias_TA(l.output, l.biases, l.batch, l.outputs, 1);
    }
    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
}

layer_TA make_connected_layer_TA_new(int batch, int inputs, int outputs, ACTIVATION_TA activation, int batch_normalize, int adam)
{
    int i;
    layer_TA l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED_TA;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));
    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));
    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));
    l.forward_TA = forward_connected_layer_TA_new;

    if(adam){
        l.m = calloc(l.inputs*l.outputs, sizeof(float));
        l.v = calloc(l.inputs*l.outputs, sizeof(float));
        l.bias_m = calloc(l.outputs, sizeof(float));
        l.scale_m = calloc(l.outputs, sizeof(float));
        l.bias_v = calloc(l.outputs, sizeof(float));
        l.scale_v = calloc(l.outputs, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(outputs, sizeof(float));
        l.scale_updates = calloc(outputs, sizeof(float));

        l.mean = calloc(outputs, sizeof(float));
        l.mean_delta = calloc(outputs, sizeof(float));
        l.variance = calloc(outputs, sizeof(float));
        l.variance_delta = calloc(outputs, sizeof(float));

        l.rolling_mean = calloc(outputs, sizeof(float));
        l.rolling_variance = calloc(outputs, sizeof(float));

        l.x = calloc(batch*outputs, sizeof(float));
        l.x_norm = calloc(batch*outputs, sizeof(float));
    }

    l.activation = activation;
    //IMSG("connected_TA                         %4d  ->  %4d\n", inputs, outputs);

    return l;
}
