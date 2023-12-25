#ifndef SOFTMAX_LAYER_TA_H
#define SOFTMAX_LAYER_TA_H
#include "STinylib.h"

typedef layer_TA softmax_layer_TA;

extern int soft_time_ta;
softmax_layer_TA make_softmax_layer_TA_new(int batch, int inputs, int groups, float temperature, int w, int h, int c, int spatial, int noloss);
void forward_softmax_layer_TA(softmax_layer_TA l, network_TA net);

#endif
