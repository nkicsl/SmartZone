CC      ?= $(CROSS_COMPILE)gcc
LD      ?= $(CROSS_COMPILE)ld
AR      ?= $(CROSS_COMPILE)ar
NM      ?= $(CROSS_COMPILE)nm
OBJCOPY ?= $(CROSS_COMPILE)objcopy
OBJDUMP ?= $(CROSS_COMPILE)objdump
READELF ?= $(CROSS_COMPILE)readelf

#ALIB=libdarknet.a
#ARFLAGS = -rcs
LDFLAGS= -lm  -pthread
#LDFLAGS= ./fdlibm/lib/libm.a  -pthread

OBJSCR = src/
EXECSCR = examples/

OBJ = activation_layer.o activations.o avgpool_layer.o batchnorm_layer.o blas.o box.o col2im.o connected_layer.o convolutional_layer.o cost_layer.o crnn_layer.o crop_layer.o data.o deconvolutional_layer.o demo.o depthwise_convolutional_layer.o detection_layer.o dropout_layer.o gemm.o gru_layer.o im2col.o image.o iseg_layer.o l2norm_layer.o layer.o list.o local_layer.o logistic_layer.o lstm_layer.o matrix.o maxpool_layer.o network.o normalization_layer.o option_list.o parser.o region_layer.o reorg_layer.o rnn_layer.o route_layer.o shortcut_layer.o softmax_layer.o tree.o upsample_layer.o utils.o yolo_layer.o

EXECOBJ = classifier.o darknet.o

OBJS = $(addprefix $(OBJSCR), $(OBJ))
EXECOBJS = $(addprefix $(EXECSCR), $(EXECOBJ))

EXECMAIN = main.o

CFLAGS += -Wall   -I../ta/include -I$(TEEC_EXPORT)/include -I./include
CFLAGS += -Iinclude/ -Isrc/ -Iexamples/  -I./src/ -I./examples/

#Add/link other required libraries here
LDADD += -lteec -L$(TEEC_EXPORT)/lib 
#-L/home/whx/rpi/optee_examples/mobilenet_rpi/host/fdlibm/lib
#DEPS = $(wildcard src/*.h) inlcude/NTinylib.h inlcude/main.h

BINARY = tinylib

.PHONY: all
#all: $(ALIB) $(BINARY)
all: $(BINARY)

$(BINARY): $(EXECMAIN) $(EXECOBJS) $(OBJS)
	$(CC)  $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LDADD) 

#$(ALIB): $(OBJS)
#	$(AR) $(ARFLAGS) $@ $^





.PHONY: clean
clean:
	rm -f $(OBJS) $(EXECOBJS) $(EXECMAIN) $(BINARY)
