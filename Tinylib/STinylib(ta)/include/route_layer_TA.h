#ifndef ROUTE_LAYER_TA_H
#define ROUTE_LAYER_TA_H
#include "STinylib.h"
typedef layer_TA route_layer_TA;

route_layer_TA make_route_layer_TA_new(int batch, int n, int *input_layers, int *input_sizes);
void forward_route_layer_TA_new(route_layer_TA l, network_TA net);

#endif
