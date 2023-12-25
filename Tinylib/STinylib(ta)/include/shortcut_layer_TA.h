#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "STinylib.h"
typedef layer_TA shortcut_layer_TA;
shortcut_layer_TA make_shortcut_layer_TA_new(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_shortcut_layer_TA_new(shortcut_layer_TA l, network_TA net);

#endif
