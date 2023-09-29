#include <omp.h>
#include <mpi.h>

#define TAG 0
#define ROOT 0

void master_gsmu2(int *Y, int *subtest_counts, int size, int person_count, int item_count, int total_subtest_count, int iteration_count, int burnin, int batches, int uniform, double *ITEM, double *PERSON, double *CORR);

void slave_gsmu2(int rank, int size, int person_count, int item_count, int total_subtest_count, int iteration_count, int burnin, int batches, int uniform);

void master_cuda_gsmu2(int *Y, int *subtest_counts, int size, int person_count, int item_count, int total_subtest_count, int iteration_count, int burnin, int batches, int uniform, double *ITEM, double *PERSON, double *CORR);

void slave_cuda_gsmu2(int rank, int size, int person_count, int item_count, int total_subtest_count, int iteration_count, int burnin, int batches, int uniform);
