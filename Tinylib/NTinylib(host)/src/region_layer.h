#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "NTinylib.h"
#include "layer.h"
#include "network.h"

layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);
void forward_region_layer(const layer l, network net);
void backward_region_layer(const layer l, network net);
void resize_region_layer(layer *l, int w, int h);

#endif
