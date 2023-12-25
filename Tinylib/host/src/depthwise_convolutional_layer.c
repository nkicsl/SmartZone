#include "depthwise_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include "parser.h"
int depthwise_convolutional_out_height(depthwise_convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int depthwise_convolutional_out_width(depthwise_convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

//��ʱ���ݿռ���?
static size_t get_workspace_size(layer l){
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c*sizeof(float);
}
float dw_time=0;
depthwise_convolutional_layer make_depthwise_convolutional_layer(int batch, int h, int w, int c,int size, int stride, int padding, ACTIVATION activation, int batch_normalize)
{
    clock_t time;
    time=clock();
    int i;
	depthwise_convolutional_layer l = {0};
    l.type = DEPTHWISE_CONVOLUTIONAL;

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


    if(!tee_flag){
         //fprintf(stderr, "------flag\n");
        l.weights = calloc(l.nweights, sizeof(float));
    }else{
        l.weights = calloc(1, sizeof(float));
    }


    
    
    l.weight_updates = calloc(l.n*size*size, sizeof(float));

    l.biases = calloc(l.n, sizeof(float));
    l.bias_updates = calloc(l.n, sizeof(float));
    layer_mem+=l.nweights+l.n*size*size+l.n*2;
    //l.nweights = l.n*size*size;
    l.nbiases = l.n;


    //float scale = sqrt(2./(size*size*c));
    //for(i = 0; i < l.n*l.size*l.size; ++i) l.weights[i] = scale*rand_normal();




    int out_w = depthwise_convolutional_out_width(l);
    int out_h = depthwise_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = l.n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));
    layer_mem+=l.batch*l.outputs*2;

    l.forward = forward_depthwise_convolutional_layer;
    l.backward = backward_depthwise_convolutional_layer;
    l.update = update_depthwise_convolutional_layer;

    if(batch_normalize){
        l.scales = calloc(c, sizeof(float));
        l.scale_updates = calloc(c, sizeof(float));
        for(i = 0; i < c; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(c, sizeof(float));
        l.variance = calloc(c, sizeof(float));

        l.mean_delta = calloc(c, sizeof(float));
        l.variance_delta = calloc(c, sizeof(float));

        l.rolling_mean = calloc(c, sizeof(float));
        l.rolling_variance = calloc(c, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
        layer_mem+=c*8+l.batch*l.outputs*2;
    }
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;
    dw_time+= sec(clock() - time);
    fprintf(stderr, "dw conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", c, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c,(2.0 * l.n * l.size*l.size* l.out_h*l.out_w)/1000000000.);

    return l;
}

void resize_depthwise_convolutional_layer(depthwise_convolutional_layer *l, int w, int h)
{
	l->w = w;
	l->h = h;
	int out_w = depthwise_convolutional_out_width(*l);
	int out_h = depthwise_convolutional_out_height(*l);

	l->out_w = out_w;
	l->out_h = out_h;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));
	if (l->batch_normalize) {
		l->x = realloc(l->x, l->batch*l->outputs * sizeof(float));
		l->x_norm = realloc(l->x_norm, l->batch*l->outputs * sizeof(float));
	}

	l->workspace_size = get_workspace_size(*l);
}

void add_bias_depthwise(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias_depthwise(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias_depthwise(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    int out_h = l.out_h;
    int out_w = l.out_w;
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
   
    int k = l.size*l.size;
    int n = out_h*out_w;

    for(int b = 0; b < l.batch; ++b){
		for (int c=0;c<l.c;c++)
		{
			float *aoffset = l.weights+c*l.size*l.size;
			float *boffset = net.workspace;
			float *coffset = l.output+c*l.out_h*l.out_w+b*l.n*l.out_h*l.out_w;
			float *intput_offset = net.input + c*l.h*l.w+ b*l.c*l.h*l.w;
			im2col_cpu(intput_offset, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);

			gemm_cpu(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);
		}
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }

	int m = l.n;
    activate_array(l.output, m*n*l.batch, l.activation);//�����ǰ�򴫵�

}

void backward_depthwise_convolutional_layer(depthwise_convolutional_layer l, network net)
{
    int i;
    int m = l.n;
    int n = l.size*l.size;
    int k = l.out_w*l.out_h;
	//�����������
    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

	for (int b = 0; b < l.batch; ++b) {
		for (int c = 0; c<l.c; c++)
		{
			//��Ȩ����
			float *aoffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
			float *boffset = net.workspace;
			float *coffset = l.weight_updates + c*l.size*l.size;


			float *im = net.input + c*l.h*l.w + b*l.c*l.h*l.w;


			im2col_cpu(im, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm(0, 1, 1, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);
			//�Ա������������󵼣�Ҳ������ԭʼ��Ȩ�أ������������΢������ͼ���о������

			if (net.delta) {
				aoffset = l.weights+ c*l.size*l.size;
				boffset = l.delta + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
				coffset = net.workspace;

				gemm(1, 0, n, k, 1, 1, aoffset, n, boffset, k, 0, coffset, k);

				col2im_cpu(net.workspace, 1, l.h, l.w, l.size, l.stride, l.pad, net.delta + c*l.h*l.w + b*l.n*l.h*l.w);
			}
		}
	}

}

void update_depthwise_convolutional_layer(depthwise_convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, momentum, l.weight_updates, 1);
}
void denormalize_depthwise_convolutional_layer(depthwise_convolutional_layer l)
{
	int i, j;
	for (i = 0; i < l.n; ++i) {
		float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .00001);
		for (j = 0; j < l.size*l.size; ++j) {
			l.weights[i*l.size*l.size + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}