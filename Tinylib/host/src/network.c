#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "main.h"
#include "blas.h"
#include "parser.h"  //wang

network *load_network(char *cfg, char *weights, int clear)
{
    network *net = parse_network_cfg(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    return net;
}


void calc_network_cost(network *netp)
{
    network net = *netp;
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    *net.cost = sum/count;
}


void forward_network(network *netp,int r_size)
{
    network net = *netp;
    int i;
    clock_t time;
                 
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        int input_size;
        
        if(i > tee_layer_start && i <= tee_layer_end)
        {
             
            if(i==0){
                input_size=r_size;
            }else{
                input_size=net.layers[i-1].outputs*net.layers[i-1].batch;
            }

            forward_network_CA(net.input,input_size);


            i = tee_layer_end;
            if(tee_layer_end < net.n - 1)
            {
                layer l_pp2 = net.layers[tee_layer_end];
                forward_network_back_CA(l_pp2.output, l_pp2.outputs, net.batch);
                net.input = l_pp2.output;
            }

        }else
        {
            time=clock();
            if(l.delta){
                fill_cpu(l.outputs * l.batch, 0, l.delta, 1);
            }
            l.forward(l, net);
            net.input = l.output;
            if(l.truth) {
                net.truth = l.output;
            }
           // fprintf(stderr, "----------the i====   %d  cost time====%f      \n", i,sec(clock()-time));   
        }
    }
   //  fprintf(stderr, "----------the all  ca  cost time====%f      \n", sec(clock()-time));   
    calc_network_cost(netp);
}

float *network_predict(network *net, float *input,int r_size)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
   clock_t time;
   time=clock();
    forward_network(net,r_size);
    fprintf(stderr, "time_total  cost ====%f      \n", sec(clock()-time));   
    //net_output_return_CA(net->inputs);
    float *out;
    if(tee_layer_start >= net->n-1){
        out = net->output;
    }else if(tee_layer_start < net->n-1){
         if(tee_layer_end < net->n-1){
             out = net->output;
         }else{
             net_output_return_CA(net->inputs);
             out = net_output_back;
         }
     }
     *net = orig;
     return out;
}

