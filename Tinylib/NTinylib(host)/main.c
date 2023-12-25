#include <err.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include<time.h>
#include "NTinylib.h"
#include "activations.h"
#include "cost_layer.h"
#include "parser.h"
#include "main.h"

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* TEE resources */
TEEC_Context ctx;
TEEC_Session sess;

float *net_input_back;
float *net_delta_back;
float *net_output_back;
int *index_output_back;

void make_dropout_layer_CA(int batch, int inputs, float probability, int w, int h, int c, float *net_prev_output, float *net_prev_delta)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[5];
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = w;
    passint[3] = h;
    passint[4] = c;
    float passflo[1];
    passflo[0] = probability;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(float)*1;

    op.params[2].tmpref.buffer = net_prev_output;
    op.params[2].tmpref.size = sizeof(float)*inputs*batch;
    op.params[3].tmpref.buffer = net_prev_delta;
    op.params[3].tmpref.size = sizeof(float)*inputs*batch;
////////////////////////

    res = TEEC_InvokeCommand(&sess, MAKE_DROP_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(DROP) failed 0x%x origin 0x%x",
         res, origin);
}


void make_shortcut_layer_CA(int batch, int index, int w, int h, int c, int w2, int h2, int c2,ACTIVATION activation,float alpha,float beta)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passarg[8];
    float passflo[2];
    passarg[0] = batch;
    passarg[1] = index;
    passarg[2] = w;
    passarg[3] = h;
    passarg[4] = c;
    passarg[5] = w2;
    passarg[6] = h2;
    passarg[7] = c2;

    passflo[0] = alpha;
    passflo[1] = beta;

    char *actv = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passarg;
    op.params[0].tmpref.size = sizeof(passarg);

    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    op.params[2].tmpref.buffer = actv;
    op.params[2].tmpref.size = strlen(actv)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_SHROTCUT_CMD,
                             &op, &origin);


    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(FC) failed 0x%x origin 0x%x",
         res, origin);
}



void make_connected_layer_CA(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passarg[5];
    passarg[0] = batch;
    passarg[1] = inputs;
    passarg[2] = outputs;
    passarg[3] = batch_normalize;
    passarg[4] = adam;

    char *actv = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passarg;
    op.params[0].tmpref.size = sizeof(passarg);

    op.params[1].tmpref.buffer = actv;
    op.params[1].tmpref.size = strlen(actv)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_CONNECTED_CMD,
                             &op, &origin);


    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(FC) failed 0x%x origin 0x%x",
         res, origin);
}

void make_crop_layer_CA(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure,float shift,int noadjust)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[8];
    float passflo[4];

    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = crop_height;
    passint[5] = crop_width;
    passint[6] = flip;
    passint[7] = noadjust;

    passflo[0] = angle;
    passflo[1] = saturation;
    passflo[2] = exposure;
    passflo[3] = shift;

fprintf(stderr, "crop0    %d %d %d %d %d %d %d      \n", batch,h,w,c,crop_height,crop_width,flip,noadjust);
fprintf(stderr, "crop1    %f %f %f %f   \n", angle,saturation,exposure,shift);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    res = TEEC_InvokeCommand(&sess, MAKE_CROP_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(CROP) failed 0x%x origin 0x%x",
         res, origin);
}


void make_cost_layer_CA(int batch, int inputs, COST_TYPE cost_type, float scale, float ratio, float noobject_scale, float thresh)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[2];
    float passflo[4];
    char *passcost;

    passint[0] = batch;
    passint[1] = inputs;
    passflo[0] = scale;
    passflo[1] = ratio;
    passflo[2] = noobject_scale;
    passflo[3] = thresh;

    passcost = get_cost_string(cost_type);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT,
        TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passflo;
    op.params[1].tmpref.size = sizeof(passflo);

    op.params[2].tmpref.buffer = passcost;
    op.params[2].tmpref.size = strlen(passcost)+1;

    res = TEEC_InvokeCommand(&sess, MAKE_COST_CMD,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(COST) failed 0x%x origin 0x%x",
         res, origin);
}

void make_maxpool_layer_CA(int batch, int h, int w, int c, int size, int stride, int padding)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[7];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = size;
    passint[5] = stride;
    passint[6] = padding;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    res = TEEC_InvokeCommand(&sess, MAKE_MAX_CMD,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAX) failed 0x%x origin 0x%x",
         res, origin);
}
void make_convolutional_layer_CA(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int flipped, float dot)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[14];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = n;
    passint[5] = groups;
    passint[6] = size;
    passint[7] = stride;
    passint[8] = padding;
    passint[9] = batch_normalize;
    passint[10] = binary;
    passint[11] = xnor;
    passint[12] = adam;
    passint[13] = flipped;

    float passflo = dot;
    char *acti = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].value.a = passflo;

    op.params[2].tmpref.buffer = acti;
    op.params[2].tmpref.size = strlen(acti)+1;

    //clock_t time;
    //time=clock();
    res = TEEC_InvokeCommand(&sess, MAKE_CONV_CMD,
                             &op, &origin);
//fprintf(stderr, "----------conv_TA_cost_time_main  ==%f      \n", sec(clock()-time));
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(CONV) failed 0x%x origin 0x%x",
         res, origin);
}

void make_depthwise_convolutional_layer_CA(int batch,int h,int w,int c,int size,int stride,int padding,ACTIVATION activation, int batch_normalize,int flipped,float dot)
{
    
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[9];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;
    passint[4] = size;
    passint[5] = stride;
    passint[6] = padding;
    passint[7] = batch_normalize;
    passint[8] = flipped;

    float passflo = dot;
    char *acti = get_activation_string(activation);

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].value.a = passflo;

    op.params[2].tmpref.buffer = acti;
    op.params[2].tmpref.size = strlen(acti)+1;
    //clock_t time;
   // time=clock();

    res = TEEC_InvokeCommand(&sess, MAKE_DW_CONV_CMD,
                             &op, &origin);
//fprintf(stderr, "----------dep_TA_cost_time_main  ==%f      \n", sec(clock()-time));

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(CONV) failed 0x%x origin 0x%x",
         res, origin);
}

void make_network_CA(int begin_index,int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[17];

    passint[0] = n;
    passint[1] = time_steps;
    passint[2] = notruth;
    passint[3] = batch;
    passint[4] = subdivisions;
    passint[5] = random;
    passint[6] = adam;
    passint[7] = h;
    passint[8] = w;
    passint[9] = c;
    passint[10] = inputs;
    passint[11] = max_crop;
    passint[12] = min_crop;
    passint[13] = center;
    passint[14] = burn_in;
    passint[15] = max_batches;
    passint[16] = begin_index;

    float passfloat[15];
    passfloat[0] = learning_rate;
    passfloat[1] = momentum;
    passfloat[2] = decay;
    passfloat[3] = B1;
    passfloat[4] = B2;
    passfloat[5] = eps;
    passfloat[6] = max_ratio;
    passfloat[7] = min_ratio;
    passfloat[8] = clip;
    passfloat[9] = angle;
    passfloat[10] = aspect;
    passfloat[11] = saturation;
    passfloat[12] = exposure;
    passfloat[13] = hue;
    passfloat[14] = power;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);

    op.params[1].tmpref.buffer = passfloat;
    op.params[1].tmpref.size = sizeof(passfloat);
   // clock_t time;
    //time=clock();

    res = TEEC_InvokeCommand(&sess, MAKE_NETWORK_CMD,
                             &op, &origin);
//fprintf(stderr, "----------make_network_TA_cost_time_main  ==%f      \n", sec(clock()-time));
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(MAKE_NET) failed 0x%x origin 0x%x",
         res, origin);
}

void make_route_layer_CA(int batch, int n, int *input_layers, int *input_sizes,int out_w,int out_h,int out_c)
{
   TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[5];
    passint[0] = batch;
    passint[1] = n;
    passint[2] = out_w;
    passint[3] = out_h;
    passint[4] = out_c;


    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_MEMREF_TEMP_INPUT, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].tmpref.buffer = input_layers;
    op.params[1].tmpref.size = sizeof(int)*n;
    op.params[2].tmpref.buffer = input_sizes;
    op.params[2].tmpref.size = sizeof(int)*n;

    //clock_t time;
    //time=clock();

    res = TEEC_InvokeCommand(&sess,MAKE_ROUTE_CMD,
                             &op, &origin);
//fprintf(stderr, "----------route_TA_cost_time_main  ==%f      \n", sec(clock()-time));

    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(ROUTE) failed 0x%x origin 0x%x",
         res, origin);
}

void make_softmax_layer_CA(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int passint[8];
    float passflo = temperature;
    passint[0] = batch;
    passint[1] = inputs;
    passint[2] = groups;
    passint[3] = w;
    passint[4] = h;
    passint[5] = c;
    passint[6] = spatial;
    passint[7] = noloss;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
        TEEC_VALUE_INPUT,
        TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
    op.params[1].value.a = passflo;
    //clock_t time;
    //time=clock();

    res = TEEC_InvokeCommand(&sess, MAKE_SOFTMAX_CMD,
                             &op, &origin);
//fprintf(stderr, "----------soft_TA_cost_time_main  ==%f      \n", sec(clock()-time));                             
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(SOFTMAX) failed 0x%x origin 0x%x",
         res, origin);
}


void make_avgpool_layer_CA(int batch, int h, int w, int c)
{
  //invoke op and transfer paramters
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

    int passint[4];
    passint[0] = batch;
    passint[1] = h;
    passint[2] = w;
    passint[3] = c;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = passint;
    op.params[0].tmpref.size = sizeof(passint);
   // clock_t time;
   // time=clock();

    res = TEEC_InvokeCommand(&sess, MAKE_AVG_CMD,
                             &op, &origin);
//fprintf(stderr, "----------avg_TA_cost_time_main  ==%f      \n", sec(clock()-time));
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(AVG) failed 0x%x origin 0x%x",
         res, origin);
}

void net_output_return_CA(int net_output_size)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    int top=5;
    net_output_back = malloc(sizeof(float) * top);
    index_output_back = malloc(sizeof(int) * top);
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_MEMREF_TEMP_OUTPUT,
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = net_output_back;
    op.params[0].tmpref.size = sizeof(float) * top;
    op.params[1].tmpref.buffer = index_output_back;
    op.params[1].tmpref.size = sizeof(int) * top;


   // clock_t time;
 //   time=clock();

    res = TEEC_InvokeCommand(&sess, OUTPUT_RETURN_CMD,
                             &op, &origin);
//fprintf(stderr, "----------return_TA_cost_time_main  ==%f      \n", sec(clock()-time));                      
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(return_result) failed 0x%x origin 0x%x",
             res, origin);
}

void forward_network_CA(float *net_input, int input_size)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INPUT,
                                     TEEC_NONE, TEEC_NONE);

    float *params0 = malloc(sizeof(float)*input_size);
    for(int z=0; z<input_size; z++){
        params0[z] = net_input[z];
    }

    op.params[0].tmpref.buffer = params0;
    op.params[0].tmpref.size = sizeof(float) *input_size;

    //clock_t time;
    /////time=clock();

    res = TEEC_InvokeCommand(&sess, FORWARD_CMD,
                             &op, &origin);
//fprintf(stderr, "----------forward_TA_cost_time_main  ==%f      \n", sec(clock()-time));
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InvokeCommand(forward) failed 0x%x origin 0x%x",
         res, origin);

    free(params0);
}

void move_weights_CA(float *weights,int size_weight)
{
    //invoke op and transfer paramters
    TEEC_Operation op;
    uint32_t origin;
    TEEC_Result res;
int buf_size=262144;
buf_size=mov_size;
int times_weight=size_weight/buf_size;
int i;
for(i=0;i<times_weight;i++){
    memset(&op, 0, sizeof(op));

    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_VALUE_INPUT,
                                     TEEC_VALUE_INPUT, TEEC_VALUE_INPUT);
    op.params[0].tmpref.buffer = &weights[i*buf_size];
    op.params[0].tmpref.size = sizeof(float) * buf_size;
    //fprintf(stderr, "==============move_weights_CA       p = %d      \n", op.params[0].tmpref.size);
    op.params[1].value.a=i;
    op.params[2].value.a=buf_size;
    op.params[3].value.a=0;
    res = TEEC_InvokeCommand(&sess, MOV_WEIGHT,
                             &op, &origin);

    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(return) failed 0x%x origin 0x%x",
             res, origin);

}

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
                                     TEEC_VALUE_INPUT,
                                     TEEC_VALUE_INPUT, TEEC_VALUE_INPUT);
    op.params[0].tmpref.buffer = &weights[i*buf_size];
    op.params[0].tmpref.size = sizeof(float) *(size_weight%buf_size) ;
    if(times_weight==0){
        op.params[1].value.a=0;
    }else{
        op.params[1].value.a=i;
    }
    
    op.params[2].value.a=buf_size;
    op.params[3].value.a=1;
    //fprintf(stderr, "==============move_weights_CA       p = %d      \n", op.params[0].tmpref.size);


    res = TEEC_InvokeCommand(&sess, MOV_WEIGHT,
                             &op, &origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand(return) failed 0x%x origin 0x%x",
             res, origin);          

}

void forward_network_back_CA(float *l_output, int net_inputs, int net_batch)
{
  TEEC_Operation op;
  uint32_t origin;
  TEEC_Result res;

  net_input_back = malloc(sizeof(float) * net_inputs*net_batch);


  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_NONE,
                                   TEEC_NONE, TEEC_NONE);



   op.params[0].tmpref.buffer = net_input_back;
   op.params[0].tmpref.size = sizeof(float) * net_inputs*net_batch;

   res = TEEC_InvokeCommand(&sess, FORWARD_BACK_CMD,
                            &op, &origin);

   for(int z=0; z<net_inputs * net_batch; z++){
       l_output[z] = net_input_back[z];
   }

   free(net_input_back);

   if (res != TEEC_SUCCESS)
   errx(1, "TEEC_InvokeCommand(forward_add) failed 0x%x origin 0x%x",
        res, origin);
}
int main(int argc, char **argv)
{
    TEEC_Result res;
    TEEC_UUID uuid = TA_DARKNETP_UUID;//需要修改UUID
    uint32_t err_origin;
    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InitializeContext failed with code 0x%x", res);
    /* Open a TA session. */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
             res, err_origin);

    printf("Begin darknet\n");
    darknet_main(argc, argv);
    
    printf("finishing\n");
    /* Close TA */
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
    return 0;
}
