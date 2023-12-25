#ifndef DEPTHWISE_CONVOLUTIONAL_LAYER_H
#define DEPTHWISE_CONVOLUTIONAL_LAYER_H

#include "activations_TA.h"
#include "network_TA.h"

typedef layer_TA depthwise_convolutional_layer_TA;
extern int dw_time_ta;

depthwise_convolutional_layer_TA make_depthwise_convolutional_layer_TA_new(int batch, int h, int w, int c, int size, int stride, int padding, ACTIVATION_TA activation, int batch_normalize, int flipped, float dot);
//void resize_depthwise_convolutional_layer(depthwise_convolutional_layer_TA *layer, int w, int h);
void forward_depthwise_convolutional_layer_TA(const depthwise_convolutional_layer_TA layer, network_TA net);

int depthwise_convolutional_out_height_TA(depthwise_convolutional_layer_TA layer);
int depthwise_convolutional_out_width_TA(depthwise_convolutional_layer_TA layer);

#endif

