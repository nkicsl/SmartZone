#ifndef IM2COL_H
#define IM2COL_H
extern float im2col_time;
void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#endif
