#ifndef IM2COL_TA_H
#define IM2COL_TA_H
extern int im2col_time_ta;
void im2col_cpu_TA(float* data_im,
                int channels, int height, int width,
                int ksize, int stride, int pad, float* data_col);
#endif
