#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "STinylib.h"
typedef layer_TA crop_layer_TA;
crop_layer_TA make_crop_layer_TA_new(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void forward_crop_layer_TA_new(const crop_layer_TA l, network_TA net);


#endif

