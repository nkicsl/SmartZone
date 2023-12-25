// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "NTinylib.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"


network *make_network(int n);

void calc_network_cost(network *net);

#endif

