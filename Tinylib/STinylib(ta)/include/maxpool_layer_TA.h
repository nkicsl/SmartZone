#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "STinylib.h"

typedef layer_TA maxpool_layer_TA;

maxpool_layer_TA make_maxpool_layer_TA(int batch, int h, int w, int c, int size, int stride, int padding);
void forward_maxpool_layer_TA_new(const maxpool_layer_TA l, network_TA net);

#endif
