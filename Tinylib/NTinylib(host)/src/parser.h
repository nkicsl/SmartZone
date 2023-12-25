#ifndef PARSER_H
#define PARSER_H
#include "NTinylib.h"
#include "network.h"

extern int total_layer_num;
extern int tee_layer_start;
extern int tee_layer_end;
extern int tee_flag;
extern int mov_size;
//extern float *weight_to_TA;
//extern int weight_size;

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#endif
