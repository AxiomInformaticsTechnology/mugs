#include "gsmu2.cuh"
#include "cuda_gsmu2.cuh"
#include "mpi_gsmu2.cuh"

#include "gendata.cuh"
#include "number.cuh"
#include "mathafx.cuh"
#include "stdafx.cuh"
#include "randafx.cuh"

int theta_gen(double* CORR, int SUBTESTS, int PERSONS, double* THS)
{
	int IRES;
	
	double *RSIG = (double*) malloc(sizeof(double)*SUBTESTS*SUBTESTS);	
	double *TH   = (double*) malloc(sizeof(double)*PERSONS*SUBTESTS);

	double *CHF = (double*)malloc(sizeof(double)*SUBTESTS*SUBTESTS);
	
	IRES = CHFAC(SUBTESTS, CORR, RSIG, CHF);
		
	if(IRES != 0) {
		showmatr(CORR,SUBTESTS,SUBTESTS,"CH4 ");
		showmatr(RSIG,SUBTESTS,SUBTESTS,"RSIG ");
		return 1;
	}
	
	for(int ith = 0; ith < PERSONS; ith++) {
		for(int mth = 0; mth < SUBTESTS; mth++) {
		
			TH[ith*SUBTESTS + mth] = random_normal();
		}
	}
	
	matmul(PERSONS, SUBTESTS, TH, SUBTESTS, SUBTESTS, RSIG, PERSONS, SUBTESTS, THS, 1);
	
	
	free(RSIG);
	free(TH);
	free(CHF);
	
	return 0;
}

int gen_data(int PERSONS, int ITEMS, int SUBTESTS, int* subtestcounts, double* CORR, double* AS,double* GS,double* THS,int* Y)
{
	srand(rand());
	
	int* offset_array = (int*)malloc(SUBTESTS*sizeof(int));
	
	int idst = ITEMS / SUBTESTS;
	int imst = ITEMS % SUBTESTS;

	if(0) printf("Number of persons: %d\nNumber of items: %d\nNumber of subtests: %d\nItems divided by number of subtests: %d\nItems mod subtests: %d\n\n", PERSONS, ITEMS, SUBTESTS, idst, imst);

	int n; for (n = 0; n < SUBTESTS; n++) {
		subtestcounts[n] = idst;
	}
	
	n = 0;
	while (imst > 0) {
		subtestcounts[n] += 1;
		n++; imst--;
	}

	int one    =  1;
	double LL  = -1.0;
	double UL  =  1.0;
	double DIF =  UL - LL;

	int *ord    = (int *)malloc(sizeof(int)*ITEMS);	
	int *places = (int *)malloc(sizeof(int)*SUBTESTS);	

	for(int subtest = SUBTESTS - 1; subtest >= 0; subtest--) {
		places[subtest] = subtestcounts[subtest];
		for(int subtest0 = 0; subtest0 < subtest; subtest0++)
			places[subtest] += subtestcounts[subtest0];
	}	

	for(int item = 0, subtest = 0; item < ITEMS; item++) {
		if(item >= places[subtest]) {
			subtest++;
		}
		ord[item] = subtest;
	}

	rnunp(ITEMS, AS);
	
	
	rnunp(ITEMS, GS);
	

	sscal(ITEMS,&DIF,GS,&one);
	sadd(ITEMS,&LL,GS,&one);

	if(0) showmatr(AS,1,ITEMS,"AS");
	if(0) showmatr(GS,1,ITEMS,"GS");

	do {
		for(int ith = 0; ith < SUBTESTS; ith++) {
			CORR[ith*SUBTESTS+ith] = 1.0;
			for(int jth = ith + 1; jth < SUBTESTS; jth++) {
				CORR[ith*SUBTESTS+jth] = CORR[jth*SUBTESTS+ith] = 0.5;
			}
		}
	} while(theta_gen(CORR,SUBTESTS,PERSONS,THS) == 1);

	if(0) showmatrt(THS,PERSONS,SUBTESTS,"THS");
	
	
	for(int ith = 0; ith < PERSONS; ith++) {
		for(int jth = 0; jth < ITEMS; jth++) {
				
			if( random_uniform_pos() < stdnormal_cdf((GS[jth] - AS[jth] * THS[ith*SUBTESTS + ord[jth]])) ) {
				Y[ith*ITEMS + jth] = 0;
			}
			else {
				Y[ith*ITEMS + jth] = 1;
			}
		}
		if (0) fprintf(stderr, "%6d\r", ith);
	}
	
	
	free(ord);	
	free(places);
	
	return 0;
}