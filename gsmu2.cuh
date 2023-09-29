#include <omp.h>

void gsmu2(int *Y, int *subtest_counts, int person_count, int item_count, int total_subtest_count, int iteration_count, int burnin, int batches, int uniform, double *ITEM, double *PERSON, double *CORR);

void Subtest(int person_count, int item_count, int total_subtest_count, int unif_flag, int subtest, int *subtest_counts, double *Z, double *TH, double *C, double *A, double *G);
