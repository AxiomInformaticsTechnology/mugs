#include <cublas_v2.h>

#include <math_functions.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>


//#define PRXZ
//#define PRXX
//#define PRD
//#define PRPVAR


cudaError_t				setupRng(curandState_t *rngStates, int gx, int gy, int bx, int by, unsigned long long offset, unsigned long long seed);

cudaError_t				calcZ(cudaStream_t stream, double *dev_U, int gx, int gy, int bx, int by, int *dev_Y, double *dev_A, double *dev_G, double *dev_TH, double *dev_Z, int *dev_ORD, int item_count, int person_count, int total_subtest_count);

cudaError_t				calcTH(cudaStream_t stream, double *dev_WZ, double *dev_AAT, double *dev_RSIG, double *dev_PVAR, double *dev_Z, double *dev_G, double *dev_TH, int GRID_Y, int THREAD_Y, int item_count, int person_count, int total_subtest_count);

cudaError_t				initAA(double* dev_AAT, double* dev_AA, int item_count, int total_subtest_count, int gx, int gy, int bx, int by);

cudaError_t				stripeAA(double* dev_AAT, double* dev_AA, double* dev_A, int* dev_SUBTEST_COUNTS, int item_count, int total_subtest_count, int GRID_Z, int THREAD_Z);

cudaError_t				calcAATS(double* dev_AAT, double* dev_AA, double* dev_D_AATS, int item_count, int total_subtest_count, int gx, int gy, int bx, int by);

cudaError_t				copyRN(double2* dev_CTH_WZ, double *dev_WZ, int person_count, int total_subtest_count, int gx, int gy, int bx, int by);

cudaError_t				cudaSubtest(curandState_t *rngStates, double *dev_A, double *dev_G, double *dev_Z, double *dev_TH, int person_count, int item_count, int total_subtest_count, int unif_flag, int *dev_SUBTEST_COUNTS, int GRID_Z, int THREAD_Z, int BLOCKS, int THREADS);

cudaError_t				calcCTH(double* dev_RSIG_C, double* dev_TH, double2* dev_CTH_WZ, int person_count, int total_subtest_count, int gx, int gy, int bx, int by);

cudaError_t				calcD(double2* dev_CTH_WZ, double* dev_D_AATS, int person_count, int total_subtest_count, int gx, int gy, int bx, int by, int BLOCKS, int THREADS);

cudaError_t				calcC_ZVAR(double2* dev_CTH_WZ, double* dev_PVAR_ZVAR, int* dev_SUBTEST_COUNTS, double* dev_RSIG_C, double* dev_A, int person_count, int total_subtest_count, int GRID_Z, int THREAD_Z, int BLOCKS, int THREADS);

cudaError_t				trackAG(double* dev_AV, double* dev_GV, int item_count, int GRID_X, int THREAD_X, int BSIZE);

cudaError_t				trackTH(double* dev_THV, int total_subtest_count, int persons_by_subtests, int gx, int gy, int bx, int by, int BSIZE);

cudaError_t				trackRHO(double* dev_RHO, int total_subtest_count, int subtests_by_subtests, int gx, int gy, int bx, int by, int BSIZE);

cudaError_t				copyAG(double* dev_AV, double* dev_GV, double* dev_A, double* dev_G, int GRID_X, int THREAD_X);

cudaError_t				copyTH(double* dev_THV, double* dev_TH, int total_subtest_count, int gx, int gy, int bx, int by);

cudaError_t				copySIGMA(double* dev_RHO, double* dev_SIGMA, int total_subtest_count, int gx, int gy, int bx, int by);

cudaError_t				sumAG(double* dev_AV, double* dev_GV, double* dev_A, double* dev_G, int GRID_X, int THREAD_X);

cudaError_t				sumTH(double* dev_THV, double* dev_TH, int total_subtest_count, int gx, int gy, int bx, int by);

cudaError_t				sumSIGMA(double* dev_RHO, double* dev_SIGMA, int total_subtest_count, int gx, int gy, int bx, int by);