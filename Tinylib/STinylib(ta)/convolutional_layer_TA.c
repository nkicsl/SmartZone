#include "convolutional_layer_TA.h"
#include "batchnorm_layer_TA.h"

#include "utils_TA.h"
#include "gemm_TA.h"
#include "blas_TA.h"
#include "im2col_TA.h"
#include "STinylib.h"
#include "activations_TA.h"
#include "malloc.h"
#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

int conv_time_ta=0;
int add_bias_time_ta=0;
int scale_bias_time_ta=0;
static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}
void swap_binary_TA(convolutional_layer_TA *l)
{
    float *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;
}

void add_bias_TA(float *output, float *biases, int batch, int n, int size)
{
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
    TEE_GetSystemTime(&p2);
   add_bias_time_ta+= time_diff(&p1, &p2);
}


void scale_bias_TA(float *output, float *scales, int batch, int n, int size)
{
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
    TEE_GetSystemTime(&p2);
   scale_bias_time_ta+= time_diff(&p1, &p2);
}

void backward_bias_TA(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array_TA(delta+size*(i+b*n), size);
        }
    }
}

int convolutional_out_height_TA(convolutional_layer_TA l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width_TA(convolutional_layer_TA l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}


void binarize_weights_TA(float *weights, int n, int size, float *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs_ta(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu_TA(float *input, int n, float *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}


static size_t get_workspace_size(layer_TA l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
}


convolutional_layer_TA make_convolutional_layer_TA_new(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION_TA activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot)
{
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    int i;
    convolutional_layer_TA l = {0};
    l.type = CONVOLUTIONAL_TA;
    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

/*
    l.nweights = c/groups*n*size*size;
   // l.weights = calloc(l.nweights, sizeof(float));
    l.weights = malloc(l.nweights*sizeof(float));
    l.weight_updates = malloc(c/groups*n*size*size*sizeof(float));

    //l.biases = calloc(n, sizeof(float));
    l.biases = malloc(n*sizeof(float));
    l.bias_updates = malloc(n*sizeof(float));

    
    l.nbiases = n;
*/

     l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    //float scale = ta_sqrt(2./(size*size*c/l.groups));
    //float scale = sqrt(2./(size*size*c/l.groups));
    //for(i = 0; i < l.nweights; ++i) {
        //l.weights[i] = scale*rand_normal_TA(0,1);
    //}


    int out_w = convolutional_out_width_TA(l);
    int out_h = convolutional_out_height_TA(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    //l.output = malloc(l.batch*l.outputs*sizeof(float));
    //l.delta  = malloc(l.batch*l.outputs* sizeof(float));
    l.forward_TA = forward_convolutional_layer_TA_new;
    //l.backward_TA = backward_convolutional_layer_TA_new;
    //l.update_TA = update_convolutional_layer_TA_new;
    if(binary){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }

    if(xnor){
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
    }

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }

    if(adam){
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    l.flipped = flipped;
    l.dot = dot;

    TEE_GetSystemTime(&p2);
    conv_time_ta += time_diff(&p1, &p2);
    return l;
}


void forward_convolutional_layer_TA_new(convolutional_layer_TA l, network_TA net)
{
    int i, j;

    /*
            TEE_Time p1 = { };
    TEE_Time p2 = { };
                TEE_Time p10 = { };
    TEE_Time p3 = { };
                TEE_Time p11 = { };
    TEE_Time p4 = { };
                TEE_Time p12 = { };
    TEE_Time p5 = { };
    TEE_Time p6 = { };
    TEE_Time p7 = { };
    TEE_Time p8 = { };
    TEE_Time p9 = { };
TEE_GetSystemTime(&p1);

*/
    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);
//TEE_GetSystemTime(&p2);
    if(l.xnor){
        binarize_weights_TA(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary_TA(&l);
        binarize_cpu_TA(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }
   // TEE_GetSystemTime(&p3);
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;


    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output + (i*l.groups + j)*n*m;
            float *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
               //TEE_GetSystemTime(&p4);
                im2col_cpu_TA(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            //TEE_GetSystemTime(&p5);
            }

            //TEE_Time t0 = { };
    //TEE_Time t1 = { };
//TEE_GetSystemTime(&t0);

            //DMSG("gemm   m=%d  n=%d  k=%d   \n",m,n,k);
            gemm_cpu_TA(0,0,m,n,k,1,a,k,b,n,1,c,n);
//TEE_GetSystemTime(&t1);
        //uint32_t tt = time_diff(&t0, &t1);
       //DMSG("conv_gemm     ==%d \n",tt);


        }
    }
//TEE_GetSystemTime(&p6);
    if(l.batch_normalize){
        //DMSG("----------------------------0 \n");
        forward_batchnorm_layer_TA(l, net);
    } else {
       // DMSG("----------------------------1 \n");
        add_bias_TA(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }
//TEE_GetSystemTime(&p7);
    activate_array_TA(l.output, l.outputs*l.batch, l.activation);
   // TEE_GetSystemTime(&p8);
    if(l.binary || l.xnor) swap_binary_TA(&l);
//TEE_GetSystemTime(&p9);

/*
uint32_t pp = time_diff(&p1, &p2);
DMSG("t0     ==%d \n",pp);

pp = time_diff(&p2, &p3);
DMSG("t1     ==%d \n",pp);

pp = time_diff(&p4, &p5);
DMSG("t2     ==%d \n",pp);

pp = time_diff(&p3, &p6);
DMSG("t3     ==%d \n",pp);

pp = time_diff(&p6, &p7);
DMSG("t4     ==%d \n",pp);

pp = time_diff(&p7, &p8);
DMSG("t5     ==%d \n",pp);

pp = time_diff(&p8, &p9);
DMSG("t6     ==%d \n",pp);
*/
}
