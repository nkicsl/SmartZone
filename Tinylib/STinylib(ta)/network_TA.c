#include <stdio.h>

#include "STinylib.h"
#include "blas_TA.h"
#include "network_TA.h"

#include "STinylib_ta.h"
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

network_TA netta;
int roundnum = 0;
float err_sum = 0;
float avg_loss = -1;

float *ta_net_input;
float *ta_net_delta;
float *ta_net_output;

static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}

void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches)
{
    netta.n = n;
    
    //netta.seen = calloc(1, sizeof(size_t));
    netta.seen = calloc(1, sizeof(uint64_t));
    netta.layers = calloc(netta.n, sizeof(layer_TA));
    netta.t    = calloc(1, sizeof(int));
    netta.cost = calloc(1, sizeof(float));

    netta.learning_rate = learning_rate;
    netta.momentum = momentum;
    netta.decay = decay;
    netta.time_steps = time_steps;
    netta.notruth = notruth;
    netta.batch = batch;
    netta.subdivisions = subdivisions;
    netta.random = random;
    netta.adam = adam;
    netta.B1 = B1;
    netta.B2 = B2;
    netta.eps = eps;
    netta.h = h;
    netta.w = w;
    netta.c = c;
    netta.inputs = inputs;
    netta.max_crop = max_crop;
    netta.min_crop = min_crop;
    netta.max_ratio = max_ratio;
    netta.min_ratio = min_ratio;
    netta.center = center;
    netta.clip = clip;
    netta.angle = angle;
    netta.aspect = aspect;
    netta.saturation = saturation;
    netta.exposure = exposure;
    netta.hue = hue;
    netta.burn_in = burn_in;
    netta.power = power;
    netta.max_batches = max_batches;
    netta.workspace_size = 0;
    netta.truth=0;
    for(int i=0;i<80;i++){
        need_weight[i]=0;
    }
    //netta.truth = net->truth; ////// ing network.c train_network
}

void forward_network_TA()
{
    TEE_Time t0 = { };
    TEE_Time t1 = { };
    //TEE_GetSystemTime(&t0);
    if(roundnum == 0){
        if(netta.workspace_size){
            netta.workspace = calloc(1, netta.workspace_size);
        }
    }
    roundnum++;
    int i;
    for(i = 0; i < netta.n; ++i){
        //DMSG("---i= %d \n",i);
        /*
        TEE_Time t0 = { };
    TEE_Time t1 = { };
    TEE_Time t2 = { };
    TEE_Time t3 = { };
TEE_GetSystemTime(&t0);

*/
        TEE_GetSystemTime(&t0);
        netta.index = i;
        layer_TA l = netta.layers[i];
       // DMSG("--l.type  = %d   \n",l.type);
        if(l.delta){
            //DMSG("=====0   \n");
            fill_cpu_TA(l.outputs * l.batch, 0, l.delta, 1);
        }

        //TEE_GetSystemTime(&t2);
        //DMSG("=====1   \n");
        l.forward_TA(l, netta);
       // TEE_GetSystemTime(&t3);
        //DMSG("=====2   \n");
        netta.input = l.output;
        //DMSG("=====3   \n");
        if(l.truth) {
            //DMSG("=====4   \n");
            netta.truth = l.output;
        }

       // DMSG("======5   \n");

        TEE_GetSystemTime(&t1);
        uint32_t tt = time_diff(&t0, &t1);
        //DMSG("ta  part i=%d   cost=  %d \n",i,tt);
        //uint32_t ttt = time_diff(&t2, &t3);
       //DMSG("ta  i== %d    time  ==%d   part==  %d \n",i,tt,ttt);
       
    }
    //TEE_GetSystemTime(&t1);
   // uint32_t tt = time_diff(&t0, &t1);
    //DMSG("ta  all cost=  %d \n",tt);
}

