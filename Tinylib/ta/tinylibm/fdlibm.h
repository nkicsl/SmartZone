
/* @(#)fdlibm.h 1.5 04/04/22 */
/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */

/* Sometimes it's necessary to define __LITTLE_ENDIAN explicitly
   but these catch some common cases. */

//#if defined(i386) || defined(i486) || \
//	defined(intel) || defined(x86) || defined(i86pc) || \
//	defined(__alpha) || defined(__osf__)
//#endif

#define __HI(x) *(1+(int*)&x)
#define __LO(x) *(int*)&x
#define __HIp(x) *(1+(int*)x)
#define __LOp(x) *(int*)x



#define	__P(p)	()

/*
 * ANSI/POSIX
 */


#define	MAXFLOAT	((float)3.40282346638528860e+38)

enum fdversion {fdlibm_ieee = -1, fdlibm_svid, fdlibm_xopen, fdlibm_posix};

#define _LIB_VERSION_TYPE enum fdversion
#define _LIB_VERSION _fdlib_version  

/* if global variable _LIB_VERSION is not desirable, one may 
 * change the following to be a constant by: 
 *	#define _LIB_VERSION_TYPE const enum version
 * In that case, after one initializes the value _LIB_VERSION (see
 * s_lib_version.c) during compile time, it cannot be modified
 * in the middle of a program
 */ 
extern  _LIB_VERSION_TYPE  _LIB_VERSION;


#define _SVID_  fdlibm_svid

#define _POSIX_ fdlibm_posix

struct exception {
	int type;
	char *name;
	double arg1;
	double arg2;
	double retval;
};

#define	HUGE		MAXFLOAT

/* 
 * set X_TLOSS = pi*2**52, which is possibly defined in <values.h>
 * (one may replace the following line by "#include <values.h>")
 */

#define X_TLOSS		1.41484755040568800000e+16 

#define	DOMAIN		1
#define	SING		2
#define	OVERFLOW	3
#define	UNDERFLOW	4
#define	TLOSS		5
#define	PLOSS		6

/*
 * ANSI/POSIX
 */
extern double copysign __P((double, double));
extern double scalbn __P((double, int));
extern double cos __P((double));
extern double sin __P((double));

extern double exp __P((double));
extern double log __P((double));
extern double fabs __P((double));
extern double sqrt __P((double));
extern double floor __P((double));

extern int matherr __P((struct exception *));

/*
 * IEEE Test Vector
 */

/*
 * Functions callable from C, intended to support IEEE arithmetic.
 */


/* ieee style elementary functions */
extern int    __ieee754_rem_pio2 __P((double,double*));



/* fdlibm kernel function */
extern double __kernel_standard __P((double,double,int));	
extern double __kernel_sin __P((double,double,int));
extern double __kernel_cos __P((double,double));
extern int    __kernel_rem_pio2 __P((double*,double*,int,int,int,const int*));
