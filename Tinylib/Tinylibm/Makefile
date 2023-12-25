#
#  @(#)Makefile 1.4 95/01/18 
# 
#  ====================================================
#  Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# 
#  Developed at SunSoft, a Sun Microsystems, Inc. business.
#  Permission to use, copy, modify, and distribute this
#  software is freely granted, provided that this notice 
#  is preserved.
#  ====================================================
# 
# 

#
# There are two options in making libm at fdlibm compile time:
# 	_IEEE_LIBM 	--- IEEE libm; smaller, and somewhat faster
#	_MULTI_LIBM	--- Support multi-standard at runtime by 
#			    imposing wrapper functions defined in 
#			    fdlibm.h:
#				_IEEE_MODE 	-- IEEE
#				_XOPEN_MODE 	-- X/OPEN
#				_POSIX_MODE 	-- POSIX/ANSI
#				_SVID3_MODE 	-- SVID
#
# Here is how to set up CFLAGS to create the desired libm at 
# compile time:
#
# 	CFLAGS = -D_IEEE_LIBM		... IEEE libm (recommended)
#	CFLAGS = -D_SVID3_MODE	... Multi-standard supported
#					    libm with SVID as the 
#					    default standard
#	CFLAGS = -D_XOPEN_MODE	... Multi-standard supported
#					    libm with XOPEN as the 
#					    default standard
#	CFLAGS = -D_POSIX_MODE	... Multi-standard supported
#					    libm with POSIX as the 
#					    default standard
#	CFLAGS = 			... Multi-standard supported
#					    libm with IEEE as the 
#					    default standard
# 
# NOTE: if scalb's second arguement is an int, then one must
# define _SCALB_INT in CFLAGS. The default prototype of scalb
# is double scalb(double, double)
#


#
# Default IEEE libm
#
CFLAGS = -D_IEEE_LIBM  -O3

CC = /home/whx/rpi/toolchains/aarch64/bin/aarch64-linux-gnu-gcc

INCFILES = fdlibm.h
.INIT: $(INCFILES)
.KEEP_STATE:
src =   e_sqrt.c   e_exp.c  s_sin.c s_cos.c s_floor.c  s_fabs.c       \
	e_log.c s_isnan.c k_sin.c k_cos.c e_rem_pio2.c k_standard.c k_rem_pio2.c s_scalbn.c s_copysign.c

obj =   e_sqrt.o   e_exp.o  s_sin.o s_cos.o s_floor.o  s_fabs.o       \
	e_log.o s_isnan.o k_sin.o k_cos.o e_rem_pio2.o k_standard.o k_rem_pio2.o s_scalbn.o s_copysign.o


all: libm.a 

libm.a : $(obj) 
	ar cru libm.a $(obj)
	ranlib libm.a
	/bin/rm -f $(obj) a.out

source: $(src) README

clean: 
	/bin/rm -f $(obj) a.out libm.a
