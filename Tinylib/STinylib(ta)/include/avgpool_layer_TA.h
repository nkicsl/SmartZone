#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "STinylib.h"

typedef layer_TA avgpool_layer_TA;
extern int avg_time_ta;
avgpool_layer_TA make_avgpool_layer_TA(int batch, int w, int h, int c);
void forward_avgpool_layer_TA_new(const avgpool_layer_TA l, network_TA net);

#endif
