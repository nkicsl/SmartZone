#include "crop_layer_TA.h"

#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

crop_layer_TA make_crop_layer_TA_new(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
{
    crop_layer_TA l = {0};
    
    l.type = CROP_TA;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.scale = (float)crop_height / h;
    l.exposure = exposure;
    l.angle = angle;
    l.saturation = saturation;
    l.flip = flip;
    l.out_w = crop_width;
    
   // l.out_h = crop_height;//error 
    l.out_h = 227;//error 

    l.out_c = c;
    l.inputs = l.w * l.h * l.c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.output = calloc(l.outputs*batch, sizeof(float));
    l.forward_TA = forward_crop_layer_TA_new;
    
    //DMSG("----c1 \n");
    return l;
}


void forward_crop_layer_TA_new(const crop_layer_TA l, network_TA net)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
     int  flip = 0;
    int dh = (l.h - l.out_h)/2;
    int dw = (l.w - l.out_w)/2;
    float scale = 2;
    float trans = -1;
    if(l.noadjust){
        scale = 1;
        trans = 0;
    }
 
    for(b = 0; b < l.batch; ++b){
        for(c = 0; c < l.c; ++c){
            for(i = 0; i < l.out_h; ++i){
                for(j = 0; j < l.out_w; ++j){
                    if(flip){
                        col = l.w - dw - j - 1;    
                    }else{
                        col = j + dw;
                    }
                    row = i + dh;
                    index = col+l.w*(row+l.h*(c + l.c*b)); 
                    l.output[count++] = net.input[index]*scale + trans;
                }
            }
        }
    }
}

