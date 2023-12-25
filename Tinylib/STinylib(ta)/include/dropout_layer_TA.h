#ifndef DROPOUT_LAYER_TA_H
#define DROPOUT_LAYER_TA_H

#include "STinylib.h"

typedef layer_TA dropout_layer_TA;

dropout_layer_TA make_dropout_layer_TA_new(int batch, int inputs, float probability, int w, int h, int c, int netnum);

void forward_dropout_layer_TA_new(dropout_layer_TA l, network_TA net);

#endif
