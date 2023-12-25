#ifndef ISEG_LAYER_H
#define ISEG_LAYER_H

#include "NTinylib.h"
#include "layer.h"
#include "network.h"

layer make_iseg_layer(int batch, int w, int h, int classes, int ids);
void forward_iseg_layer(const layer l, network net);
void backward_iseg_layer(const layer l, network net);
void resize_iseg_layer(layer *l, int w, int h);
int iseg_num_detections(layer l, float thresh);

#endif
