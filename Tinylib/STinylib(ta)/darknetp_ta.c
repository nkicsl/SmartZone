#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <compiler.h>
#include <stdbool.h>
#include <string.h>
#include <sys/queue.h>
#include <tee_api.h>
#include <tee_ta_api.h>
#include <user_ta_header.h>
#include <utee_syscalls.h>
#include <tee_arith_internal.h>
#include <malloc.h>
#include "crop_layer_TA.h"
#include "STinylib_ta.h"
#include "depthwise_convolutional_layer_TA.h"
#include "convolutional_layer_TA.h"
#include "maxpool_layer_TA.h"
#include "avgpool_layer_TA.h"
#include "dropout_layer_TA.h"
#include "blas_TA.h"
#include "gemm_TA.h"
#include "connected_layer_TA.h"
#include "softmax_layer_TA.h"
#include "cost_layer_TA.h"
#include "network_TA.h"
#include "shortcut_layer_TA.h"
#include "activations_TA.h"
#include "STinylib.h"
#include "route_layer_TA.h"
#include "im2col_TA.h"

#define LOOKUP_SIZE 4096

float *netta_truth;
int netnum = 0;
int norm_output = 1;
int need_weight[80];
int mov_index=0;
int begin_index=0;

TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("has been called");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("has been called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    /* Unused parameters */
    (void)&params;
    (void)&sess_ctx;

    return TEE_SUCCESS;
}


void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    (void)&sess_ctx; /* Unused parameter */
    IMSG("Goodbye!\n");
}

static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}


static TEE_Result make_netowork_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE );

  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;


//uint8_t ta_heap[160* 1024 * 1024];
//const size_t ta_heap_size = sizeof(ta_heap);
//malloc_add_pool(ta_heap, ta_heap_size);

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;

    int n = params0[0];
    int time_steps = params0[1];
    int notruth = params0[2];
    int batch = params0[3];
    int subdivisions = params0[4];
    int random = params0[5];
    int adam = params0[6];
    int h = params0[7];
    int w = params0[8];
    int c = params0[9];
    int inputs = params0[10];
    int max_crop = params0[11];
    int min_crop = params0[12];
    int center = params0[13];
    int burn_in = params0[14];
    int max_batches = params0[15];
    begin_index=params0[16];

    float learning_rate = params1[0];
    float momentum = params1[1];
    float decay = params1[2];
    float B1 = params1[3];
    float B2 = params1[4];
    float eps = params1[5];
    float max_ratio = params1[6];
    float min_ratio = params1[7];
    float clip = params1[8];
    float angle = params1[9];
    float aspect = params1[10];
    float saturation = params1[11];
    float exposure = params1[12];
    float hue = params1[13];
    float power = params1[14];

    make_network_TA(n, learning_rate, momentum, decay, time_steps, notruth, batch, subdivisions, random, adam, B1, B2, eps, h, w, c, inputs, max_crop, min_crop, max_ratio, min_ratio, center, clip, angle, aspect, saturation, exposure, hue, burn_in, power, max_batches);

    return TEE_SUCCESS;
}

static TEE_Result make_convolutional_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float params1 = params[1].value.a;
    char *params2 = params[2].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int n = params0[4];
    int groups = params0[5];
    int size = params0[6];
    int stride = params0[7];
    int padding = params0[8];
    int batch_normalize = params0[9];
    int binary = params0[10];
    int xnor = params0[11];
    int adam = params0[12];
    int flipped = params0[13];
    float dot = params1;
    char *acti = params2;

    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_convolutional_layer_TA_new(batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize, binary, xnor, adam, flipped, dot);
    netta.layers[netnum] = lta;
    need_weight[netnum]=1;
    if (lta.workspace_size > netta.workspace_size) netta.workspace_size = lta.workspace_size;
    netnum++;

    return TEE_SUCCESS;
}


static TEE_Result make_route_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    

    int *params0 = params[0].memref.buffer;

    int *params1 = params[1].memref.buffer;
    int *params2 = params[2].memref.buffer;

    int batch = params0[0];
    int n = params0[1];
    int w = params0[2];
    int h = params0[3];
    int c = params0[4];
     int *input_layers=params1;
     int *input_sizes=params2;

    //DMSG( "ta   batch=%d n=%d   w=%d  h=%d c= %d     \n",batch, n, w, h, c);
    //for(int i=0;i<n;i++){
        //DMSG("a_layers[%d]=%d   size[%d]=%d  \n ",i,input_layers[i],i,input_sizes[i]);
    //}

    layer_TA lta = make_route_layer_TA_new( batch, n, input_layers, input_sizes);
    //DMSG("-----5 \n");
    lta.out_w=w;
    lta.out_h=h;
    lta.out_c=c;
    //DMSG("-----6 \n");
    netta.layers[netnum] = lta;
    netnum++;
    //DMSG("-----7 \n");
    return TEE_SUCCESS;

}

static TEE_Result make_maxpool_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int size = params0[4];
    int stride = params0[5];
    int padding = params0[6];

    layer_TA lta = make_maxpool_layer_TA(batch, h, w, c, size, stride, padding);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_depthwise_convolutional_layer_CA(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_NONE);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float params1 = params[1].value.a;
    char *params2 = params[2].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];
    int size = params0[4];
    int stride = params0[5];
    int padding = params0[6];
    int batch_normalize = params0[7];
    int flipped = params0[8];
    float dot = params1;
    char *acti = params2;

    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_depthwise_convolutional_layer_TA_new(batch,h,w,c,size,stride,padding,activation, batch_normalize, flipped, dot);
    netta.layers[netnum] = lta;
    need_weight[netnum]=1;
    if (lta.workspace_size > netta.workspace_size) netta.workspace_size = lta.workspace_size;
    netnum++;

    return TEE_SUCCESS;
}
static TEE_Result make_avgpool_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");
    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;

    int batch = params0[0];
    int h = params0[1];
    int w = params0[2];
    int c = params0[3];

    layer_TA lta = make_avgpool_layer_TA(batch, h, w, c);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_dropout_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_MEMREF_INPUT);

  //DMSG("has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
    float *params2 = params[2].memref.buffer;
    float *params3 = params[3].memref.buffer;
    int buffersize = params[2].memref.size / sizeof(float);

    int *passint;
    passint = params0;
    int batch = passint[0];
    int inputs = passint[1];
    int w = passint[2];
    int h = passint[3];
    int c = passint[4];
    float probability = params1[0];

    float *net_prev_output = params2;
    float *net_prev_delta = params3;

    layer_TA lta = make_dropout_layer_TA_new(batch, inputs, probability, w, h, c, netnum);

    if(netnum == 0){
        for(int z=0; z<buffersize; z++){
            lta.output[z] = net_prev_output[z];
            lta.delta[z] = net_prev_delta[z];
        }
    }else{
        lta.output = netta.layers[netnum-1].output;
        lta.delta = netta.layers[netnum-1].delta;
    }

    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_shortcut_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *passarg;
    float *passflo;

    passarg = params[0].memref.buffer;
    passflo = params[1].memref.buffer;

    int batch = passarg[0];
    int index = passarg[1];
    int w = passarg[2];
    int h = passarg[3];
    int c = passarg[4];
    int w2 = passarg[5];
    int h2 = passarg[6];
    int c2 = passarg[7];

    float alpha=passflo[0];
    float beta=passflo[1];

    char *acti;
    acti = params[2].memref.buffer;
    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_shortcut_layer_TA_new(batch, index, w, h,c,w2,h2,c2);
    lta.activation=activation;
    lta.alpha=alpha;
    lta.beta=beta;
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}


static TEE_Result make_connected_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *passarg;
    passarg = params[0].memref.buffer;
    int batch = passarg[0];
    int inputs = passarg[1];
    int outputs = passarg[2];
    int batch_normalize = passarg[3];
    int adam = passarg[4];

    char *acti;
    acti = params[1].memref.buffer;
    ACTIVATION_TA activation = get_activation_TA(acti);

    layer_TA lta = make_connected_layer_TA_new(batch, inputs, outputs, activation, batch_normalize, adam);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_softmax_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    //DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];
    int groups = params0[2];
    int w = params0[3];
    int h = params0[4];
    int c = params0[5];
    int spatial = params0[6];
    int noloss = params0[7];
    float temperature = params[1].value.a;

    layer_TA lta = make_softmax_layer_TA_new(batch, inputs, groups, temperature, w, h, c, spatial, noloss);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_cost_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE);


    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    int batch = params0[0];
    int inputs = params0[1];

    float *params1 = params[1].memref.buffer;
    float scale = params1[0];
    float ratio = params1[1];
    float noobject_scale = params1[2];
    float thresh = params1[3];

    char *cost_t;
    cost_t = params[2].memref.buffer;
    
    COST_TYPE_TA cost_type = get_cost_type_TA(cost_t);


    layer_TA lta = make_cost_layer_TA_new(batch, inputs, cost_type, scale, ratio, noobject_scale, thresh);
    netta.layers[netnum] = lta;
    netnum++;

    return TEE_SUCCESS;
}

static TEE_Result make_crop_layer_TA_params(uint32_t param_types,
                                       TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);


    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    int *params0 = params[0].memref.buffer;
    float *params1 = params[1].memref.buffer;
 
    int  batch=params0[0];
    int  h=params0[1];
    int  w=params0[2];
    int  c=params0[3];
    int  crop_height=params0[4];
    int  crop_width=params0[5];
    int  flip=params0[6];
    int  noadjust=params0[7];

    float  angle=params1[0];
    float  saturation=params1[1];
    float  exposure=params1[2];
    float  shift=params1[3];

    //DMSG("crop_ta    %d %d %d %d %d %d %d  %d   \n", batch,h,w,c,crop_height,crop_width,flip,noadjust);

    layer_TA lta = make_crop_layer_TA_new(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);

    lta.shift=shift;
    lta.noadjust=noadjust;
    //DMSG("----c6 \n");
    netta.layers[netnum] = lta;
    netnum++;
   // DMSG("----c7 \n");
    return TEE_SUCCESS;
}


static TEE_Result forward_network_TA_params(uint32_t param_types,
                                          TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    float *net_input = params[0].memref.buffer;

    netta.input = net_input;
    forward_network_TA();

    return TEE_SUCCESS;
}

static TEE_Result forward_network_back_TA_params(uint32_t param_types,
                                           TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);
    for(int z=0; z<buffersize; z++){
        params0[z] = netta.layers[netta.n-1].output[z];
    }

    return TEE_SUCCESS;
}

static TEE_Result mov_weight_CA(uint32_t param_types,
                                              TEE_Param params[4])
{
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT,
                                               TEE_PARAM_TYPE_VALUE_INPUT);

    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;
        
    float *params0 ;
    int params1=params[1].value.a;
    int params2=params[2].value.a;
    int params3=params[3].value.a;
    int par_size=params[0].memref.size;

    params0 = (float *)TEE_Malloc(par_size, TEE_MALLOC_FILL_ZERO);
    TEE_MemMove(params0, params[0].memref.buffer, params[0].memref.size);
    int par_index =0;
    
    for(int i=0; i<netta.n; i++){
        if(need_weight[i]==1){
            if(params3)need_weight[i]=0;
            if(netta.layers[i].type==CONVOLUTIONAL_TA||netta.layers[i].type==DEPTHWISE_CONVOLUTIONAL_TA){
                int n=netta.layers[i].n;
                int p=0;
                if(params1==0){
                    TEE_MemMove(netta.layers[i].biases,params0,n*sizeof(float));
                }
                    if (netta.layers[i].batch_normalize &&params1==0){
                        p=3;
                        TEE_MemMove(netta.layers[i].scales,&params0[n],n*sizeof(float));
                        TEE_MemMove(netta.layers[i].rolling_mean,&params0[2*n],n*sizeof(float));
                        TEE_MemMove(netta.layers[i].rolling_variance,&params0[3*n],n*sizeof(float));
                    }
                if(params1==0){
                    mov_index=(par_size-(((1+p)*n)*sizeof(float)))/sizeof(float);
                }
                int mem_start=(params1-1)*params2;
                if(params1==0){
                    TEE_MemMove(netta.layers[i].weights,&params0[n+p*n],par_size-((n+p*n)*sizeof(float)));
                }else{
                    TEE_MemMove(&netta.layers[i].weights[mem_start+mov_index],params0,par_size);
                }
                if(params3){
                    mov_index=0;
                }
            }
            break;
        }  
    }
    TEE_Free(params0);

    return TEE_SUCCESS;
}


static TEE_Result net_output_return_TA_params(uint32_t param_types,
                                              TEE_Param params[4])
{//add 18 lines
    uint32_t exp_param_types = TEE_PARAM_TYPES( TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_MEMREF_OUTPUT,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);
    if (param_types != exp_param_types)
        return TEE_ERROR_BAD_PARAMETERS;

    int p,i,j,top=5;
    int *indexes = calloc(top, sizeof(int));
    for(p = netta.n - 1; p >= 0; --p){
        if(netta.layers[p].type != COST_TA) break;
    }

    for(j = 0; j < top; ++j) indexes[j] = -1;

    for(i = 0; i < netta.layers[p].outputs; ++i){
        int curr = i;
        for(j = 0; j < top; ++j){
            if((indexes[j] < 0) || netta.layers[netta.n-1].output[curr] > netta.layers[netta.n-1].output[indexes[j]]){
                int swap = curr;
                curr = indexes[j];
                indexes[j] = swap;
    }  }  }    



    float *params0 = params[0].memref.buffer;
    int buffersize = params[0].memref.size / sizeof(float);

    int *params1 = params[1].memref.buffer;
    int buffersize1 = params[1].memref.size / sizeof(int);
 
    for(int z=0; z<buffersize; z++){
        int index = indexes[z];
         params0[z] = netta.layers[netta.n-1].output[index]*100;
         params1[z] = index;
     }
    
    DMSG("---soft_time_ta    ==%d \n",soft_time_ta);
    DMSG("---avg_time_ta    ==%d \n",avg_time_ta);
    DMSG("---conv_time_ta    ==%d \n",conv_time_ta);
    DMSG("---dw_time_ta    ==%d \n",dw_time_ta);
    DMSG("---scale_bias_time_ta    ==%d \n",scale_bias_time_ta);
    DMSG("---add_bias_time_ta    ==%d \n",add_bias_time_ta);
    
    DMSG("---blas_time_ta    ==%d \n",blas_time_ta);
    DMSG("gemm_time_ta     ==%d \n",gemm_time_ta);
    DMSG("im2col_time_ta     ==%d \n",im2col_time_ta);
    DMSG("active_time_ta     ==%d \n",active_time_ta);
    
    return TEE_SUCCESS;
}

TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */
    switch (cmd_id) {
        case MAKE_NETWORK_CMD:
        return make_netowork_TA_params(param_types, params);

        case MAKE_CONV_CMD:
        return make_convolutional_layer_TA_params(param_types, params);

        case MOV_WEIGHT:
        return mov_weight_CA(param_types, params);

	    case MAKE_DW_CONV_CMD:
	    return make_depthwise_convolutional_layer_CA(param_types, params);

        case MAKE_MAX_CMD:
        return make_maxpool_layer_TA_params(param_types, params);

        case MAKE_AVG_CMD:
        return make_avgpool_layer_TA_params(param_types, params);

        case MAKE_DROP_CMD:
        return make_dropout_layer_TA_params(param_types, params);

        case MAKE_CONNECTED_CMD:
        return make_connected_layer_TA_params(param_types, params);

        case MAKE_SHROTCUT_CMD:
        return make_shortcut_layer_TA_params(param_types, params);

        case MAKE_SOFTMAX_CMD:
        return make_softmax_layer_TA_params(param_types, params);

        case MAKE_COST_CMD:
        return make_cost_layer_TA_params(param_types, params);

        case MAKE_CROP_CMD:
        return make_crop_layer_TA_params(param_types, params);

        case MAKE_ROUTE_CMD:
        return make_route_layer_TA_params(param_types, params);

        case FORWARD_CMD:
        return forward_network_TA_params(param_types, params);

        case OUTPUT_RETURN_CMD:
        return net_output_return_TA_params(param_types, params);

        case FORWARD_BACK_CMD:
        return forward_network_back_TA_params(param_types, params);


        
        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
