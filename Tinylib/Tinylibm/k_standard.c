
/* @(#)k_standard.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 *
 */

#include "fdlibm.h"
#include <errno.h>

#include <stdio.h>			/* fputs(), stderr */
#define	WRITE2(u,v)	fputs(u, stderr)

static double zero = 0.0;	/* used as const */

/* 
 * Standard conformance (non-IEEE) on exception cases.
 * Mapping:
 *	1 -- acos(|x|>1)
 *	2 -- asin(|x|>1)
 *	3 -- atan2(+-0,+-0)
 *	4 -- hypot overflow
 *	5 -- cosh overflow
 *	6 -- exp overflow
 *	7 -- exp underflow
 *	8 -- y0(0)
 *	9 -- y0(-ve)
 *	10-- y1(0)
 *	11-- y1(-ve)
 *	12-- yn(0)
 *	13-- yn(-ve)
 *	14-- lgamma(finite) overflow
 *	15-- lgamma(-integer)
 *	16-- log(0)
 *	17-- log(x<0)
 *	18-- log10(0)
 *	19-- log10(x<0)
 *	20-- pow(0.0,0.0)
 *	21-- pow(x,y) overflow
 *	22-- pow(x,y) underflow
 *	23-- pow(0,negative) 
 *	24-- pow(neg,non-integral)
 *	25-- sinh(finite) overflow
 *	26-- sqrt(negative)
 *      27-- fmod(x,0)
 *      28-- remainder(x,0)
 *	29-- acosh(x<1)
 *	30-- atanh(|x|>1)
 *	31-- atanh(|x|=1)
 *	32-- scalb overflow
 *	33-- scalb underflow
 *	34-- j0(|x|>X_TLOSS)
 *	35-- y0(x>X_TLOSS)
 *	36-- j1(|x|>X_TLOSS)
 *	37-- y1(x>X_TLOSS)
 *	38-- jn(|x|>X_TLOSS, n)
 *	39-- yn(x>X_TLOSS, n)
 *	40-- gamma(finite) overflow
 *	41-- gamma(-integer)
 *	42-- pow(NaN,0.0)
 */



	double __kernel_standard(x,y,type) 
	double x,y; int type;
{
	struct exception exc;
#ifndef HUGE_VAL	/* this is the only routine that uses HUGE_VAL */ 
#define HUGE_VAL inf
	double inf = 0.0;

	__HI(inf) = 0x7ff00000;	/* set inf to infinite */
#endif

	exc.arg1 = x;
	exc.arg2 = y;
	switch(type) {
	    case 6:
		/* exp(finite) overflow */
		exc.type = OVERFLOW;
		exc.name = "exp";
		if (_LIB_VERSION == _SVID_)
		  exc.retval = HUGE;
		else
		  exc.retval = HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  errno = ERANGE;
		else if (!matherr(&exc)) {
			errno = ERANGE;
		}
		break;
	    case 7:
		/* exp(finite) underflow */
		exc.type = UNDERFLOW;
		exc.name = "exp";
		exc.retval = zero;
		if (_LIB_VERSION == _POSIX_)
		  errno = ERANGE;
		else if (!matherr(&exc)) {
			errno = ERANGE;
		}
		break;
	    case 16:
		/* log(0) */
		exc.type = SING;
		exc.name = "log";
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  errno = ERANGE;
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("log: SING error\n", 16);
		      }
		  errno = EDOM;
		}
		break;
	    case 17:
		/* log(x<0) */
		exc.type = DOMAIN;
		exc.name = "log";
		if (_LIB_VERSION == _SVID_)
		  exc.retval = -HUGE;
		else
		  exc.retval = -HUGE_VAL;
		if (_LIB_VERSION == _POSIX_)
		  errno = EDOM;
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("log: DOMAIN error\n", 18);
		      }
		  errno = EDOM;
		}
		break;
	    case 26:
		/* sqrt(x<0) */
		exc.type = DOMAIN;
		exc.name = "sqrt";
		if (_LIB_VERSION == _SVID_)
		  exc.retval = zero;
		else
		  exc.retval = zero/zero;
		if (_LIB_VERSION == _POSIX_)
		  errno = EDOM;
		else if (!matherr(&exc)) {
		  if (_LIB_VERSION == _SVID_) {
			(void) WRITE2("sqrt: DOMAIN error\n", 19);
		      }
		  errno = EDOM;
		}
		break;
	}
	return exc.retval; 
}
