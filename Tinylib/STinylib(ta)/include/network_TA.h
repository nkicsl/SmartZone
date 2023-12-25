#ifndef NETWORK_TA_H
#define NETWORK_TA_H

extern network_TA netta;

extern float *ta_net_input;
extern float *ta_net_delta;
extern float *ta_net_output;

void make_network_TA(int n, float learning_rate, float momentum, float decay, int time_steps, int notruth, int batch, int subdivisions, int random, int adam, float B1, float B2, float eps, int h, int w, int c, int inputs, int max_crop, int min_crop, float max_ratio, float min_ratio, int center, float clip, float angle, float aspect, float saturation, float exposure, float hue, int burn_in, float power, int max_batches);


void forward_network_TA();
#endif
