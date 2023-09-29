#include <math_functions.h>

#define MATMUL(x,y,z,bs) { matmul(sizeof(x)/sizeof(x[0]), sizeof(x[0])/sizeof(x[0][0]), (double*)&x, \
								  sizeof(y)/sizeof(y[0]), sizeof(y[0])/sizeof(y[0][0]), (double*)&y, \
								  sizeof(z)/sizeof(z[0]), sizeof(z[0])/sizeof(z[0][0]), (double*)&z, int bs); }

#define MATADD(x,y,z) { matadd(sizeof(x)/sizeof(x[0]), sizeof(x[0])/sizeof(x[0][0]), (double*)&x, \
							   sizeof(y)/sizeof(y[0]), sizeof(y[0])/sizeof(y[0][0]), (double*)&y, \
							   sizeof(z)/sizeof(z[0]), sizeof(z[0])/sizeof(z[0][0]), (double*)&z); }

#define TRANSPOSE(x,y,bs) { transpose( sizeof(x)/sizeof(x[0]), sizeof(x[0])/sizeof(x[0][0]), (double*)&x, \
									   sizeof(y)/sizeof(y[0]), sizeof(y[0])/sizeof(y[0][0]), (double*)&y, int bs); }

#define KETMUL(x,y,z) { matmul(sizeof(x)/sizeof(x[0]), sizeof(x[0])/sizeof(x[0][0]), (double*)&x, \
							   sizeof(y)/sizeof(y[0]), 1, (double*)&y, \
							   sizeof(z)/sizeof(z[0]), 1, (double*)&z); }
	
#define BRAMUL(y,x,z) { matmul(1, sizeof(y)/sizeof(y[0]), (double*)&y, \
							   sizeof(x)/sizeof(x[0]), sizeof(x[0])/sizeof(x[0][0]), (double*)&x, \
							   1, sizeof(z)/sizeof(z[0]), (double*)&z); }


#define CHFAC(total_subtest_count,PVAR,RSIG,gCHFAC) chfac(total_subtest_count, PVAR, RSIG, gCHFAC)

#define MATINV(SRC,DST,SIZE,DIM)  { memcpy(DST,SRC,SIZE); matinv(DIM, (double*)DST); }

#define MATKETMUL(KET,MAT,DST) matmul(1, sizeof(KET)/sizeof(KET[0]), (double(*)[1][sizeof(KET)/sizeof(KET[0])])&KET, \
									  sizeof(MAT)/sizeof(MAT[0]), sizeof(MAT[0])/sizeof(MAT[0][0]), &MAT, \
									  1, sizeof(DST)/sizeof(DST[0]), (double(*)[1][sizeof(DST)/sizeof(DST[0])])&DST)
	
#define MATVECMUL(MAT,VEC,DST) matmul(sizeof(MAT)/sizeof(MAT[0]), sizeof(MAT[0])/sizeof(MAT[0][0]), &MAT, \
									  sizeof(VEC)/sizeof(VEC[0]), 1, (double(*)[sizeof(VEC)/sizeof(VEC[0])][1])&VEC, \
									  sizeof(DST)/sizeof(DST[0]), 1, (double(*)[sizeof(DST)/sizeof(DST[0])][1])&DST)

void matadd(int nr1, int nc1, double *m1a, int nr2, int nc2, double*m1b, int nr3, int nc3, double *m3a);
void matmul(int nr1, int nc1, double *m1a, int nr2, int nc2, double*m1b, int nr3, int nc3, double *m3a, int bs);

void transpose(int nr1, int nc1, double *m1a, int nr2, int nc2, double*m2a, int bs);

void matinv(int N, double*A);

int chfac(int Np, double *Ap, double *Rp, double *L);

void sadd(int *LDA, double *A, double *R, int* LDR);
void sscal(int *LDA, double *A, double *R, int* LDR);

void sadd(int LDA, double *A, double *R, int* LDR);
void sscal(int LDA, double *A, double *R, int* LDR);


double mean(double *X, int count);
double correlation(double *X, double *Y, int count);


void CTOD(int total_subtest_count, int person_count, double *C, double *THT, double *D);

/*
C          THT              CTHT                      D
c 0 0 0  t t .. t .. t t   ct ct c.c. ct c.c. ct ct  cc cd ce cf
0 d 0 0  t t .. t .. t t   dt dt d.d. dt d.d. dt dt  dc dd de df
0 0 e 0  t t .. t .. t t   et et e.e. et e.e. et et  ec ed ee ef
0 0 0 f  t t .. t .. t t   ft ft f.f. ft f.f. ft ft  fc fd fe ff
*/
