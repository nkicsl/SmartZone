#include "NTinylib.h"
#include "main.h"
#include "parser.h"
#include <sys/time.h>
#include <assert.h>
#include <sys/stat.h>
#include<stdio.h>
#include<time.h>
#include<sys/types.h>
#include"blas.h"
#include"gemm.h"
#include "im2col.h"
#include "convolutional_layer.h"
#include "depthwise_convolutional_layer.h"
#include "avgpool_layer.h"
#include "softmax_layer.h"
#include "activations.h"
#include "layer.h"

 float make_time_ca;
 float make_time_ta;
 float im2col_time;
 int layer_mem =0;
void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    clock_t time;
    time=clock();
    make_time_ca=0;
    make_time_ta=0;
    im2col_time=0;
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "load_network %f seconds.\n", sec(clock()-time));
    fprintf(stderr, "make_time_ca %f.\n", make_time_ca);
    fprintf(stderr, "make_time_ta %f.\n", make_time_ta);

    fprintf(stderr, "layer_mem   ==  %d.\n", layer_mem*4);
    fprintf(stderr, "sizeof(layer)  ==  %d.\n", sizeof(layer));
    

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);



    int i = 0;
    char **names = get_labels(name_list);
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    char *img_path;
    FILE *pic_list = fopen(filename, "r");
    if (!pic_list) printf("open %s failed!\n", filename);
    int label;
    for(int ti=0;ti<1;ti++){
        //img_path = calloc(128, sizeof(char));
        //fscanf(pic_list, "%d %s", &label, img_path);
        //printf("----img_path==%s!\n", img_path);
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = letterbox_image(im, net->w, net->h);

        float *X = r.data;
        int r_size=r.c*r.h*r.w;
        time=clock();
        float *predictions = network_predict(net, X,r_size);
        free(img_path);
        //if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
       // top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        if(tee_layer_end==999||tee_layer_end==0){
            top_k(predictions, net->outputs, top, indexes);
            for(i = 0; i < top; ++i){
                int index = indexes[i];
                printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
            }
        }else{
            for(i = 0; i < top; ++i){
                int index = index_output_back[i];
                printf("%5.2f%%: %s\n", net_output_back[i], names[index]);
            }
        }
        // for(i = 0; i < top; ++i){
        //     int index = indexes[i];
        //     printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        // }




        printf("normalize_time     ==%f\n", normalize_time);
        printf("im2col_time     ==%f\n", im2col_time);
        
        printf("gemm_time     ==%f\n", gemm_time);
        printf("conv_time     ==%f\n", conv_time);
        printf("dw_time     ==%f\n", dw_time);
        printf("avg_time     ==%f\n", avg_time);
        printf("soft_time     ==%f\n", soft_time);
        printf("scale_time     ==%f\n", scale_time);
        printf("add_time     ==%f\n", add_time);
        printf("active_time     ==%f\n", active_time);
        

        if(r.data != im.data) free_image(r);
        free_image(im);
        //if (filename) break;


        

    }
    
}

void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    tee_layer_start = find_int_arg(argc, argv, "-tee_start", 999)-1;
    tee_layer_end = find_int_arg(argc, argv, "-tee_end", 999);

    mov_size=find_int_arg(argc, argv, "-mov_size", 1024);

    int top = find_int_arg(argc, argv, "-t", 0);
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;

    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
}


