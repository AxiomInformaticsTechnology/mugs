#include "stdafx.cuh"
#include "randafx.cuh"

#include "cuda_gsmu2.cuh"

#define MIN(x,y) ((x<=y) ? x : y)

double _PI       = 4.0*atan(1.0);
double _SQRT2PI  = sqrt(8.0*atan(1.0));
double _SQRT2    = sqrt(2.0);
double _SQRTPI   = sqrt(4.0*atan(1.0));
double _1_SQRTPI = 1.0/sqrt(4.0*atan(1.0));


void uniform_random_array(int P, int V, double*Z) 
{     
	for(int I = 0; I < V; I++) {
		rnunp(P, (double*)&Z[I*P]);
	}  
	return;
}

void normal_random_array(int P, int V, double*Z)
{     
	for(int I = 0; I < V; I++) {
		rnnorinv(P, (double*)&Z[I*P]);
	}  
	return;
}


double random_normal() {
	static int i = 1;
	static double u[2] = {0.0, 0.0};
	register double r[2];
	if (i == 1) {
	    unsigned int temp = (unsigned)rand()+(unsigned)1;
		r[0] = sqrt(-2*log((double)(temp)/(double)(unsigned)((unsigned)1+(unsigned)RAND_MAX)));
		r[1] =       2*_PI*(double)(rand()+(unsigned)1)/(double)(unsigned)((unsigned)1+(unsigned)RAND_MAX);
		u[0] = r[0]*sin(r[1]);
		u[1] = r[0]*cos(r[1]);
		i = 0;
	} else {
		i = 1;
	}
	return u[i];
}

double random_uniform_pos()
{
redo:;
	unsigned int rnd = (unsigned)rand() + (unsigned)1;  
	unsigned int den = (unsigned)((unsigned)1+(unsigned)RAND_MAX);
	if(rnd == den) goto redo;
	return (double) ((double) rnd / (double) den); 
}

double random_uniform()
{
redo:;
	unsigned int rnd = (unsigned)rand() + (unsigned)1;
	unsigned int den = (unsigned)((unsigned)1+(unsigned)RAND_MAX);	
	if(rnd == den) goto redo;
	return -1 + 2 * (double) ((double) rnd / (double) den); 
}

unsigned int random_bernoulli(double p)
{
	double u = random_uniform_pos();
	if (u > p) {
		return 1;
	}
	else {
		return 0;
	}
}



double random_gaussian_tail(const double a, const double sigma)
{
	/* Returns a gaussian random variable larger than a
	*  This implementation does one-sided upper-tailed deviates.
	*/

	double s = a / sigma;

	if (s < 1)
	{
		/* For small s, use a direct rejection method. The limit s < 1
		can be adjusted to optimise the overall efficiency */

		double x;

		do
		{
			x = random_normal();
		}
		while (x < s);
		return x * sigma;
	}
	else
	{
		/* Use the "supertail" deviates from the last two steps
		* of Marsaglia's rectangle-wedge-tail method, as described
		* in Knuth, v2, 3rd ed, pp 123-128.  (See also exercise 11, p139,
		* and the solution, p586.)
		*/

		double u, v, x;

		do
		{
			u = random_uniform();
			do
			{
				v = random_uniform();
			}
			while (v <= 0.0);
			x = sqrt (s * s - 2 * log (v));
			//   fprintf(stderr,"X " "%40.30f " " S " "%40.30f " " V " "%40.30f " " log(v)" "%40.30f " "\n",x,s,v,log(v));
		}
		while (x * u > s);
		return x * sigma;
	}
}

void rnbin(int NR, int N, double P, unsigned int *IR)
{	
	for(int n = 0; n < NR; n++) {
		unsigned int trials;
		unsigned int tally = 0;
		for(trials = 0; trials < (unsigned int) N; trials++) {
			tally += random_bernoulli(P);
		}
		IR[n] = tally;
	}
}


void rnbin_(int*pNR, int*pN, double *pP, unsigned int *IR)
{
	int NR = *pNR;
	int N = *pN;
	double P = *pP;
	for(int n = 0; n < NR; n++) {
		unsigned int trials;
		unsigned int tally = 0;
		for(trials = 0; trials < (unsigned int) N; trials++) {
			tally += random_bernoulli(P);
		}
		IR[n] = tally;
	}
}

void random_bernoulli(int NR, int N, double P, unsigned int *IR)
{	
	for(int cnt = 0; cnt < NR; cnt++) {
		unsigned int trials;
		unsigned int tally = 0;
		for(trials = 0; trials < (unsigned int) N; trials++) {
			tally += random_bernoulli(P);
		}
		IR[cnt] = tally;
	}
}

void rnunp_(int *pNR, double *IR)
{
	int NR = *pNR;
	for(int n = 0; n < NR; n++) {
		IR[n] = (double) random_uniform_pos();
	}
}

void rnunp(int NR, double *IR)
{
	for(int n = 0; n < NR; n++) {
		IR[n] = random_uniform_pos();
	}
}

void rnun_(int *pNR, double *IR)
{
	int NR = *pNR;
	for(int n = 0; n < NR; n++) {
		IR[n] = (double)random_uniform();
	}
}
void rnnor_(int *NRp, double* IR)
{
	int NR = *NRp;
	for(int n = 0; n < NR; n++) {
		IR[n] = (double) random_normal();
	}
}
void rnnor(int NR, double* IR)
{
	for(int n = 0; n < NR; n++) {
		IR[n] = (double) random_normal();
	}
}
void rnnora_(int *NRp, double *CDA, double* IR)
{
	int NR = *NRp;
	double cda = *CDA;
	for(int n = 0; n < NR; n++) {
		if (cda>3.0||cda<-3.0) {
			if(cda>=0.0) {
				IR[n] = (double) random_gaussian_tail(cda, 1.0);
			}
			else {
				IR[n] = -(double) random_gaussian_tail(-cda, 1.0);
			}
		}
		else {
			IR[n] = 0.0;
		}
	}
}
void rnnorinv(int NR, double* IR)
{
	for(int n = 0; n < NR; n++) {
		IR[n] = (double) random_normal();
	}
}
void rnnor1(int NR, double* IR)
{
	for(int n = 0; n < NR; n++) {
		IR[n] = (double) random_normal();
	}
}



double anorin_(double *ths)
{
	return (double) stdnormal_inv((*ths));
}
double dnorin_(double *ths)
{
	return (double) stdnormal_inv((*ths));
}
double anordf_(double *ths)
{
	return (double) stdnormal_cdf((*ths));
}


/*
* The standard normal PDF, for one random variable.
*/
double stdnormal_pdf(double u) 
{
	return exp(-u*u/2)/_SQRT2PI;
};

/*
* An implementation of adaptive, recursive Newton-Cotes integration.
* Based on the MATLAB implementation, but covered in a lot of books...
*
* This only does integration over the standard normal PDF.  It's just
* here to check the error function approximations.
*/
#define LEVMAX 10
double quad8_stdnormal_pdf(double a, double b, double Q /*= 1.0*/)
{
	/* The magic Newton-Cotes weights */
	const int w[9] = {3956, 23552, -3712, 41984, -18160, 41984, -3712, 23552, 3956};
	const int dw = 14175;
	static int level = -1;
	static double tol = 1e-30;
	register double h, Q1 = 0.0, Q2 = 0.0;
	register int i;

	level++;
	h = (b-a)/16.0;
	for (i = 0; i < 9; i++) {
		Q1 += h*w[i]*stdnormal_pdf(a+  i  *h)/dw;
		Q2 += h*w[i]*stdnormal_pdf(a+(i+8)*h)/dw;
	};
	/* This is the adaptive recursive bit.  We only recurse if we can improve... */
	if (fabs(Q1+Q2-Q) > tol*fabs(Q1+Q2) && level <= LEVMAX) {
		tol = tol/2;
		Q1 = quad8_stdnormal_pdf(a,(a+b)/2,Q1);
		Q2 = quad8_stdnormal_pdf((a+b)/2,b,Q2);
		tol = tol*2;
	}
	level--;
	return Q1 + Q2;
}

/*
* The standard normal CDF, for one random variable.
*
*   Author:  W. J. Cody
*   URL:   http://www.netlib.org/specfun/erf
*
* This is the erfc() routine only, adapted by the
* transform stdnormal_cdf(u)=(erfc(-u/sqrt(2))/2;
*/
double stdnormal_cdf(double u)
{
	static const double a[5] = {
		1.161110663653770e-002,3.951404679838207e-001,2.846603853776254e+001,
		1.887426188426510e+002,3.209377589138469e+003
	};
	static const double b[5] = {
		1.767766952966369e-001,8.344316438579620e+000,1.725514762600375e+002,
		1.813893686502485e+003,8.044716608901563e+003
	};
	static const double c[9] = {
		2.15311535474403846e-8,5.64188496988670089e-1,8.88314979438837594e00,
		6.61191906371416295e01,2.98635138197400131e02,8.81952221241769090e02,
		1.71204761263407058e03,2.05107837782607147e03,1.23033935479799725E03
	};
	static const double d[9] = {
		1.00000000000000000e00,1.57449261107098347e01,1.17693950891312499e02,
		5.37181101862009858e02,1.62138957456669019e03,3.29079923573345963e03,
		4.36261909014324716e03,3.43936767414372164e03,1.23033935480374942e03
	};
	static const double p[6] = {
		1.63153871373020978e-2,3.05326634961232344e-1,3.60344899949804439e-1,
		1.25781726111229246e-1,1.60837851487422766e-2,6.58749161529837803e-4
	};
	static const double q[6] = {
		1.00000000000000000e00,2.56852019228982242e00,1.87295284992346047e00,
		5.27905102951428412e-1,6.05183413124413191e-2,2.33520497626869185e-3
	};
	register double y, z;

	if (isnan(u))
		return u;
	if (!isfinite(u))
		return (u < 0 ? 0.0 : 1.0);//neg inf : pos inf
	y = fabs(u); 
	// double Y = y;
	if (y <= (double)0.46875 * _SQRT2) {
		/* evaluate erf() for |u| <= sqrt(2)*0.46875 */
		z = y*y;

		y = u * (  ((((a[0]*z+a[1])*z+a[2])*z+a[3])*z+a[4]) / ((((b[0]*z+b[1])*z+b[2])*z+b[3])*z+b[4]) );   //az^4 + az^3

		//    if(y>=0.5)
		//    fprintf(stddoc,"<< y %f z %f y %f Y %f z %32.28f %32.28f %32.28f\n",y,z,z*(_1_SQRTPI-y), Y, exp(-Y*Y/2)/2, exp(32.0),0.46875*_SQRT2);

		return (double)0.5+y;
	}
	z = exp(-(y*y)/2.0)/2.0;

	if (y <= 4.0) {
		/* evaluate erfc() for sqrt(2)*0.46875 <= |u| <= sqrt(2)*4.0 */
		y = y/_SQRT2;
		y =
			((((((((c[0]*y+c[1])*y+c[2])*y+c[3])*y+c[4])*y+c[5])*y+c[6])*y+c[7])*y+c[8])/
			((((((((d[0]*y+d[1])*y+d[2])*y+d[3])*y+d[4])*y+d[5])*y+d[6])*y+d[7])*y+d[8]);

		y = z*y;
		//  if( y==1.0 || y==0.0)
		//   fprintf(stddoc,"<4 y %f z %f y %f Y %f z %32.28f %32.28f\n",y,z,z*(_1_SQRTPI-y), Y, exp(-Y*Y/2)/2, exp(32.0));

	} else {
		/* evaluate erfc() for |u| > sqrt(2)*4.0 */
		z = z*_SQRT2/y;
		y = 2.0/(y*y);
		y = y*(((((p[0]*y+p[1])*y+p[2])*y+p[3])*y+p[4])*y+p[5])/
			(((((q[0]*y+q[1])*y+q[2])*y+q[3])*y+q[4])*y+q[5]); 
		y = z*(_1_SQRTPI-y);
		//if( y==1.0 || y==0.0)
		//fprintf(stddoc,"!Y y %f z %f y %f Y %f z %32.28f %32.28f\n",y,z,z*(_1_SQRTPI-y), Y, exp(-Y*Y/2)/2, exp(32.0));
	}

	double res = (u < 0.0 ? (double)y : (double)((double)1.0 - y));

	return res;
};

/*
* The inverse standard normal distribution.
*
*   Author:      Peter John Acklam <pjacklam@online.no>
*   URL:         http://home.online.no/~pjacklam
*
* This function is based on the MATLAB code from the address above,
* translated to C, and adapted for our purposes.
*/


double stdnormal_inv(double p)
{
	static const double a[6] = {
		-3.969683028665376e+01,  2.209460984245205e+02,
		-2.759285104469687e+02,  1.383577518672690e+02,
		-3.066479806614716e+01,  2.506628277459239e+00
	};
	static const double b[5] = {
		-5.447609879822406e+01,  1.615858368580409e+02,
		-1.556989798598866e+02,  6.680131188771972e+01,
		-1.328068155288572e+01
	};
	static const double c[6] = {
		-7.784894002430293e-03, -3.223964580411365e-01,
		-2.400758277161838e+00, -2.549732539343734e+00,
		4.374664141464968e+00,  2.938163982698783e+00
	};
	static const double d[4] = {
		7.784695709041462e-03,  3.224671290700398e-01,
		2.445134137142996e+00,  3.754408661907416e+00
	};

	register double q, t, u;

	if (isnan(p) || p > 1.0 || p < 0.0)
		return p;
	if (p == 0.0)
		return -HUGE_VAL;
	if (p == 1.0)
		return  HUGE_VAL;
	q = MIN(p,1-p);
	if (q > 0.02425) {
		/* Rational approximation for central region. */
		u = q-0.5;
		t = u*u;
		u = u*( (((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5])
			/ (((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4])*t+1   ) );
	} else {
		/* Rational approximation for tail region. */
		t = sqrt(-2*log(q));
		u = (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5])
			/((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1);
	}
	/* The relative error of the approximation has absolute value less
	than 1.15e-9.  One iteration of Halley's rational method (third
	order) gives full machine precision... */
	t = stdnormal_cdf(u)-q;    /* error */
	t = t*_SQRT2PI*exp(u*u/2);   /* f(u)/df(u) */
	u = u-t/(1+u*t/2);     /* Halley's method */

	return (p > 0.5 ? -u : u);
};

///////////////////

/*
* Lower tail quantile for standard normal distribution function.
*
* This function returns an approximation of the inverse cumulative
* standard normal distribution function.  I.e., given P, it returns
* an approximation to the X satisfying P = Pr{Z <= X} where Z is a
* random variable from the standard normal distribution.
*
* The algorithm uses a minimax approximation by rational functions
* and the result has a relative error whose absolute value is less
* than 1.15e-9.
*
* Author:      Peter John Acklam
* Time-stamp:  2002-06-09 18:45:44 +0200
* E-mail:      jacklam@math.uio.no
* WWW URL:     http://www.math.uio.no/~jacklam
*
* C implementation adapted from Peter's Perl version
*/
/* Coefficients in rational approximations. */

#define LOW 0.02425
#define HIGH 0.97575

double ltqnorm(double p)
{
	double q, r;
	static const double a[] =
	{
		-3.969683028665376e+01,
		2.209460984245205e+02,
		-2.759285104469687e+02,
		1.383577518672690e+02,
		-3.066479806614716e+01,
		2.506628277459239e+00
	};

	static const double b[] =
	{
		-5.447609879822406e+01,
		1.615858368580409e+02,
		-1.556989798598866e+02,
		6.680131188771972e+01,
		-1.328068155288572e+01
	};

	static const double c[] =
	{
		-7.784894002430293e-03,
		-3.223964580411365e-01,
		-2.400758277161838e+00,
		-2.549732539343734e+00,
		4.374664141464968e+00,
		2.938163982698783e+00
	};

	static const double d[] =
	{
		7.784695709041462e-03,
		3.224671290700398e-01,
		2.445134137142996e+00,
		3.754408661907416e+00
	};

	errno = 0;

	if (p < 0 || p > 1)
	{
		errno = EDOM;
		return 0.0;
	}
	else if (p == 0)
	{
		errno = ERANGE;
		return -HUGE_VAL /* minus "infinity" */;
	}
	else if (p == 1)
	{
		errno = ERANGE;
		return HUGE_VAL /* "infinity" */;
	}
	else if (p < LOW)
	{
		/* Rational approximation for lower region */
		q = sqrt(-2*log(p));
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
	}
	else if (p > HIGH)
	{
		/* Rational approximation for upper region */
		q  = sqrt(-2*log(1-p));
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
	}
	else
	{
		/* Rational approximation for central region */
		q = p - 0.5;
		r = q*q;
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
	}
}