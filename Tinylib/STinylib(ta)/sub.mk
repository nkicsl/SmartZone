global-incdirs-y += include
cflags-y += -Ofast
srcs-y += $(wildcard *.c)
#srcs-y += darknetp_ta.c
libdirs += ./Tinylibm
libnames += m
libdeps += ./Tinylibm/libm.a
# To remove a certain compiler flag, add a line like this
#cflags-template_ta.c-y += -Wno-strict-prototypes
