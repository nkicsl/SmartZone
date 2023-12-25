#include "depthwise_convolutional_layer_TA.h"
#include "utils_TA.h"
#include "convolutional_layer_TA.h"
#include "batchnorm_layer_TA.h"
#include "im2col_TA.h"
#include "blas_TA.h"
#include "STinylib.h"
#include "gemm_TA.h"
#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
int depthwise_convolutional_out_height_TA(depthwise_convolutional_layer_TA l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int depthwise_convolutional_out_width_TA(depthwise_convolutional_layer_TA l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}


//��ʱ���ݿռ���?
static size_t get_workspace_size_TA(layer_TA l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}
static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}
int dw_time_ta=0;
depthwise_convolutional_layer_TA make_depthwise_convolutional_layer_TA_new(int batch, int h, int w, int c,int size, int stride, int padding, ACTIVATION_TA activation, int batch_normalize, int flipped, float dot)
{
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    int i;
	depthwise_convolutional_layer_TA l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL_TA;

    l.h = h;
    l.w = w;
    l.n = c;
	l.c = c;

    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
   l.nweights = l.n*size*size;
    l.weights = calloc(l.nweights, sizeof(float));
    l.weight_updates = calloc(l.n*size*size, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));

 
    l.nbiases = l.n;


    //float scale = ta_sqrt(2./(size*size*c));
 
    //for(i = 0; i < l.n*l.size*l.size; ++i) l.weights[i] = scale*rand_normal_TA(0,1);//whx_ ci chu bu ying you can shu

    int out_w = depthwise_convolutional_out_width_TA(l);
    int out_h = depthwise_convolutional_out_height_TA(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.forward_TA = forward_depthwise_convolutional_layer_TA;

    if(batch_normalize){
        l.scales = calloc(c, sizeof(float));
        l.scale_updates = calloc(c, sizeof(float));

        l.mean = calloc(c, sizeof(float));
        l.variance = calloc(c, sizeof(float));

        l.mean_delta = calloc(c, sizeof(float));
        l.variance_delta = calloc(c, sizeof(float));

        l.rolling_mean = calloc(c, sizeof(float));
        l.rolling_variance = calloc(c, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    l.workspace_size = get_workspace_size_TA(l);
    l.activation = activation;
    l.flipped = flipped;
    l.dot = dot;

TEE_GetSystemTime(&p2);
dw_time_ta+=time_diff(&p1, &p2);
    return l;
}
void forward_depthwise_convolutional_layer_TA(depthwise_convolutional_layer_TA l, network_TA net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;
    fill_cpu_TA(l.outputs*l.batch, 0, l.output, 1);
    int k = l.size*l.size;
    int n = out_h*out_w;

    for(int b = 0; b < l.batch; ++b){
		for (int c=0;c<l.c;c++)
		{
			float *aoffset = l.weights+c*l.size*l.size;
			float *boffset = net.workspace;
			float *coffset = l.output+c*l.out_h*l.out_w+b*l.n*l.out_h*l.out_w;
			float *intput_offset = net.input + c*l.h*l.w+ b*l.c*l.h*l.w;
			im2col_cpu_TA(intput_offset, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm_cpu_TA(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);

		}
    }
    if(l.batch_normalize){
        forward_batchnorm_layer_TA(l, net);
    } else {
        add_bias_TA(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

	int m = l.n;
    activate_array_TA(l.output, m*n*l.batch, l.activation);//�����ǰ�򴫵�
}
