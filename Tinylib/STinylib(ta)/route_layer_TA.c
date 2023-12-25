#include "route_layer_TA.h"
#include "blas_TA.h"
#include "STinylib_ta.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>


route_layer_TA make_route_layer_TA_new(int batch, int n, int *input_layers, int *input_sizes)
{
    
    route_layer_TA l = {0};
    l.type = ROUTE_TA;
    l.batch = batch;
    l.n = n;
    l.input_layers=calloc(n,sizeof(int));
    l.input_sizes=calloc(n,sizeof(int));
    for(int p=0;p<n;p++){
        l.input_layers[p] = input_layers[p];
        l.input_sizes[p] = input_sizes[p];
        //DMSG("layers[%d]=%d   size[%d]=%d  \n ",p,input_layers[p],p,input_sizes[p]);
    }
    int i;
    int outputs = 0;
    //DMSG("-----1 \n");
    for(i = 0; i < n; i++){
        //DMSG(" input_sizes[i]=%d  \n", input_sizes[i]);
        outputs += input_sizes[i];
    }
    l.outputs = outputs;
    l.inputs = outputs;
    //DMSG("-----2 \n");
    l.delta =  calloc(outputs*batch, sizeof(float));
    l.output = calloc(outputs*batch, sizeof(float));
    //DMSG("-----3 \n");
    l.forward_TA = forward_route_layer_TA_new;
    //DMSG("-----4 \n");

    return l;
}

void forward_route_layer_TA_new(route_layer_TA l, network_TA net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        //DMSG("-----rf1 \n");
        int index = l.input_layers[i]-begin_index;
       // DMSG("----index   %d \n",index);
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        //DMSG("-----rf2 \n");
        for(j = 0; j < l.batch; ++j){
            //DMSG(" input_size==%d \n ",input_size);
            //copy_cpu_TA(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
            float *X=input + j*input_size;
            float *Y=l.output + offset + j*l.outputs;
            for(int m = 0; m < input_size; ++m) {
                Y[m] = X[m];
            }
        }
        //DMSG("-----rf3 \n");
        offset += input_size;
    }
}
