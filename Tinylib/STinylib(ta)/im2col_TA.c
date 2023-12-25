#include "im2col_TA.h"
#include <stdio.h>
#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
int im2col_time_ta=0;
static uint32_t time_diff(TEE_Time *time0, TEE_Time *time1)
{
    return (time1->seconds * 1000 + time1->millis) - (time0->seconds * 1000 + time0->millis);
}
float im2col_get_pixel_TA(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;
    
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_TA(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    TEE_Time p1 = { };
    TEE_Time p2 = { };
    TEE_GetSystemTime(&p1);
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_TA(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
    TEE_GetSystemTime(&p2);
        im2col_time_ta+= time_diff(&p1, &p2);
}

