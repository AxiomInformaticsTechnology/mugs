#include "cuda_kernels.cuh"

#include "stdafx.cuh"
#include "mathafx.cuh"
#include "randafx.cuh"


//#define CUDA_AATS
//#define CUDA_TH
//#define CUDA_SUBTEST
//#define CUDA_STATS



void cuda_gsmu2(int *Y, int *subtest_counts, int person_count, int item_count, int total_subtest_count, int iteration_count, int BURNIN, int BATCHES, int uniform, double *ITEM, double *PERSON, double *CORR);

void mcuda_gsmu2(int *Y, int *subtest_counts, int person_count, int item_count, int total_subtest_count, int iteration_count, int BURNIN, int BATCHES, int uniform, double *ITEM, double *PERSON, double *CORR);