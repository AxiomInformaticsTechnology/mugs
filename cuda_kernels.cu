#include "cuda_kernels.cuh"

#include "stdafx.cuh"


#ifndef RandomGenerators
static __global__ void			rngSetupStates(curandState_t *rngStates, unsigned long long offset, unsigned long long seed)
{
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);
	int tid = x + y * blockDim.x * gridDim.x;
	curand_init(seed, tid, offset, &rngStates[tid]);
}
cudaError_t				setupRng(curandState_t *rngStates, int gx, int gy, int bx, int by, unsigned long long offset, unsigned long long seed)
{
	dim3 grid(gx, gy, 1);
	dim3 block(bx, by, 1);
	rngSetupStates<<<grid, block>>>(rngStates, offset, seed);
	cudaError_t cudaStatus = cudaGetLastError(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "rngSetupStates kernel failed.\n", cudaGetErrorString(cudaStatus)); }
	return cudaStatus;
}
#endif

#ifndef UpdateZ
static __device__ void			calcZ_device(int person, int item, double *dev_U, int *dev_Y, double *dev_A, double *dev_G, double *dev_TH, double *dev_Z, int *dev_ORD, int item_count, int person_count, int total_subtest_count)
{
	double LP  = dev_G[item] - dev_A[item] * dev_TH[person*total_subtest_count+dev_ORD[item]];
	double BB  = normcdf(LP);
	double U   = dev_U[person*item_count+item];
	double TMP = (dev_Y[person*item_count+item] == 0) ? (BB * U) : (U  + BB * (1.0 - U));
	TMP        = normcdfinv(TMP) - LP;
	dev_Z[person*item_count+item] = (double)(isfinite(TMP) ? TMP : 0.0);
}
static __global__ void			calcZ_kernel(double *dev_U, int *dev_Y, double *dev_A, double *dev_G, double *dev_TH, double *dev_Z, int *dev_ORD, int item_count, int person_count, int total_subtest_count)
{ 
	int item   = blockIdx.x*blockDim.x + threadIdx.x;
	int person = blockIdx.y*blockDim.y + threadIdx.y;	
	if(item < item_count && person < person_count)
	{
		calcZ_device(person, item, dev_U, dev_Y, dev_A, dev_G, dev_TH, dev_Z, dev_ORD, item_count, person_count, total_subtest_count);
	}
}
cudaError_t				calcZ(cudaStream_t stream, double *dev_U, int gx, int gy, int bx, int by, int *dev_Y, double *dev_A, double *dev_G, double *dev_TH, double *dev_Z, int *dev_ORD, int item_count, int person_count, int total_subtest_count)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx,gy,1);
		dim3 block(bx,by,1);
		calcZ_kernel<<<grid, block, 0, stream>>>(dev_U, dev_Y, dev_A, dev_G, dev_TH, dev_Z, dev_ORD, item_count, person_count, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "calcZ launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcZ_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return  (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef UpdateTH
static __global__ void			calcTH_kernel(double *dev_WZ, double *dev_AAT, double *dev_RSIG, double *dev_PVAR, double *dev_Z, double *dev_G, double *dev_TH, int item_count, int person_count, int total_subtest_count)
{ 
	int person = blockIdx.x*blockDim.x + threadIdx.x;  
	if(person < person_count)
	{	
		double *lcl_RTH    = (double*)malloc(total_subtest_count * sizeof(double));
		double *lcl_PMEAN  = (double*)malloc(total_subtest_count * sizeof(double));
		double *lcl_PMEAN1 = (double*)malloc(total_subtest_count * sizeof(double));

		for(int r = 0; r < total_subtest_count; r++)
		{
			/////////////////////////////////////////////////////////////////////////////////
			double val = 0.0;
			for(int n = 0; n < item_count; n++)
			{
				val += dev_AAT[r*item_count + n] * (dev_Z[person*item_count + n] + dev_G[n]);
			}
			lcl_PMEAN1[r] = (double)val;
			/////////////////////////////////////////////////////////////////////////////////
		}
           
		for(int r = 0; r < total_subtest_count; r++)
		{
			double val1 = 0.0;
			for(int n = 0; n < total_subtest_count; n++)
			{
				val1 += dev_PVAR[r*total_subtest_count + n] * lcl_PMEAN1[n];
			}
			lcl_PMEAN[r] = (double)val1;
			
			double val2 = 0.0;
			for(int n = 0; n < total_subtest_count; n++)
			{
				val2 += dev_WZ[person*total_subtest_count + n] * dev_RSIG[n*total_subtest_count + r];
			}
			lcl_RTH[r] = (double)val2;
		}
		
		for(int subtest = 0; subtest < total_subtest_count; subtest++)
		{
			dev_TH[person*total_subtest_count + subtest] = lcl_RTH[subtest] + lcl_PMEAN[subtest];
		}

		free(lcl_RTH);
		free(lcl_PMEAN);
		free(lcl_PMEAN1);	
	}
}

cudaError_t				calcTH(cudaStream_t stream, double *dev_WZ, double *dev_AAT, double *dev_RSIG, double *dev_PVAR, double *dev_Z, double *dev_G, double *dev_TH, int GRID_Y, int THREAD_Y, int item_count, int person_count, int total_subtest_count)
{	
	cudaError_t cudaStatusA, cudaStatusB;
	{
		calcTH_kernel<<<GRID_Y, THREAD_Y, 0, stream>>>(dev_WZ, dev_AAT, dev_RSIG, dev_PVAR, dev_Z, dev_G, dev_TH, item_count, person_count, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "calcTH launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcTH_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return  (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif


#ifndef ParallelReduction

#ifdef LINUX
/*
__inline__ __device__ double	__shfl_down(double var, unsigned int srcLane, int width = 32) {
	int2 a = *reinterpret_cast<int2*>(&var);
	a.x = __shfl_down(a.x, srcLane, width);
	a.y = __shfl_down(a.y, srcLane, width);
	return *reinterpret_cast<double*>(&a);
}
*/
#endif

static __inline__ __device__ double		warpReduceSum(double val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val += __shfl_down(val, offset);
	}
	return val;
}
static __inline__ __device__ double		blockReduceSum(double val) {
	__shared__ double shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum(val);
	if (lane == 0) shared[wid] = val;
	__syncthreads();
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
	if (wid == 0) val = warpReduceSum(val);
	return val;
}
static __inline__ __device__ double2		warpReduceSum2(double2 val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val.x += __shfl_down(val.x, offset);
		val.y += __shfl_down(val.y, offset);
	}
	return val;
}
static __inline__ __device__ double2		blockReduceSum2(double2 val) {
	__shared__ double2 shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	val = warpReduceSum2(val);
	if (lane == 0) shared[wid] = val;
	__syncthreads();
	double2 zero; zero.x = zero.y = 0.0;
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;
	if (wid == 0) val = warpReduceSum2(val);
	return val;
}
#endif


#ifndef CalculateXX
static __global__ void					device_reduce_stable_XX_kernel(double* dev_TH, double2* dev_XX_block, int person_count, int total_subtest_count, int subtest) {
	double2 sum; sum.x = sum.y = 0.0;
	for (int p = blockIdx.x*blockDim.x + threadIdx.x; p < person_count; p += blockDim.x*gridDim.x) {
		double temp_TH = dev_TH[p*total_subtest_count + subtest];
		sum.x += (temp_TH * temp_TH);
		sum.y -= temp_TH;
	}
	sum = blockReduceSum2(sum);
	if (threadIdx.x == 0) {
		dev_XX_block[blockIdx.x] = sum;
	}
}
static __global__ void					device_reduce_stable_SUM_XX_kernel(double2* dev_XX_block, double* dev_XX, int N) {
	double2 sum; sum.x = sum.y = 0.0;
	for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
		sum.x += dev_XX_block[i].x;
		sum.y += dev_XX_block[i].y;
	}
	sum = blockReduceSum2(sum);
	if (threadIdx.x == 0) {
		dev_XX[0] = sum.x;
		dev_XX[2] = sum.y;
	}
}
#endif


#ifndef CalculateXZ
static __global__ void					device_reduce_stable_XZ_kernel(double* dev_Z, double* dev_TH, double2* dev_XZ_block, int person_count, int item_count, int total_subtest_count, int item, int subtest) {
	double2 sum; sum.x = sum.y = double(0.0);
	for (int p = blockIdx.x*blockDim.x + threadIdx.x; p < person_count; p += blockDim.x*gridDim.x) {
		double temp_Z = dev_Z[p*item_count + item];
		sum.x += (dev_TH[p*total_subtest_count + subtest] * temp_Z);
		sum.y -= temp_Z;
	}
	sum = blockReduceSum2(sum);
	if (threadIdx.x == 0) {
		dev_XZ_block[blockIdx.x] = sum;
	}
}
static __global__ void					device_reduce_stable_SUM_XZ_kernel(double2* dev_XZ_block, double* dev_XZ, int N, int item) {
	double2 sum; sum.x = sum.y = double(0.0);
	for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
		sum.x += dev_XZ_block[i].x;
		sum.y += dev_XZ_block[i].y;
	}
	sum = blockReduceSum2(sum);
	if (threadIdx.x == 0) {
		dev_XZ[0] = sum.x;
		dev_XZ[1] = sum.y;
	}
}
#endif



#ifndef UpdateAG
static __global__ void   calcAG_kernel(curandState_t *rngStates, double *dev_A, double *dev_G, double *dev_Z, double *dev_TH, double *dev_SIGMA2, double *dev_IX, double *dev_AGMU, double *dev_AMAT, int person_count, int item_count, int total_subtest_count, int unif_flag, int subitem, int subtest, int BLOCKS, int THREADS)
{
	int item = subitem + (blockIdx.x*blockDim.x + threadIdx.x);

	double BZ[2];
	double* XZ = (double*)malloc(2 * sizeof(double));

#ifdef PRXZ
	double2* XZ_block = (double2*)malloc(1024 * sizeof(double2));
	device_reduce_stable_XZ_kernel << <BLOCKS, THREADS >> >(dev_Z, dev_TH, XZ_block, person_count, item_count, total_subtest_count, item, subtest);
	device_reduce_stable_SUM_XZ_kernel << <1, 1024 >> >(XZ_block, XZ, BLOCKS, item);
	cudaDeviceSynchronize();
	free(XZ_block);
#else
	double val[2]; val[0] = val[1] = 0.0;
	for (int p = 0; p < person_count; p++) {
		double temp_Z = dev_Z[p*item_count + item];
		val[0] += dev_TH[p*total_subtest_count + subtest] * temp_Z;
		val[1] -= temp_Z;
	}
	XZ[0] = (double)val[0];
	XZ[1] = (double)val[1];
#endif
	if (unif_flag == 0) {
		XZ[0] += (dev_SIGMA2[0] * dev_AGMU[0]) + (dev_SIGMA2[1] * dev_AGMU[1]);
		XZ[1] += (dev_SIGMA2[2] * dev_AGMU[0]) + (dev_SIGMA2[3] * dev_AGMU[1]);
	}

	BZ[0] = (dev_IX[0] * XZ[0]) + (dev_IX[1] * XZ[1]);
	BZ[1] = (dev_IX[2] * XZ[0]) + (dev_IX[3] * XZ[1]);

	free(XZ);

	int tick = 0;

	dev_A[item] = 0.0;
	do {
		double2 beta2 = curand_normal2_double(&rngStates[item]);

		dev_A[item] = ((beta2.x * dev_AMAT[0]) + (beta2.x * dev_AMAT[2])) + BZ[0];
		dev_G[item] = ((beta2.y * dev_AMAT[1]) + (beta2.y * dev_AMAT[3])) + BZ[1];

		tick++;

		if (tick >= MAXTICKS) {
			break;
		}
	} while (dev_A[item] <= 0.0);
}
#endif


#ifndef Subtests
static __global__ void			subtest_kernel(curandState_t* rngStates, double *dev_A, double *dev_G, double* dev_Z, double* dev_TH, int person_count, int item_count, int total_subtest_count, int unif_flag, int *dev_SUBTEST_COUNTS, int BLOCKS, int THREADS)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;

	int subitem = 0;
	for (int ith = subtest; ith > 0; ith--) subitem += dev_SUBTEST_COUNTS[ith - 1];

	double det_A, A_00, L_00, A_10, A_11, L_10, diag, L_11;

	double *SIGMA2 = (double*)malloc(4 * sizeof(double));
	double *XX = (double*)malloc(4 * sizeof(double));
	double *IX = (double*)malloc(4 * sizeof(double));
	double *AGMU = (double*)malloc(4 * sizeof(double));
	double *AMAT = (double*)malloc(4 * sizeof(double));

	XX[0] = XX[1] = XX[2] = XX[3] = double(0.0);
#ifdef PRXX
	double2* XX_block = (double2*)malloc(1024 * sizeof(double2));
	device_reduce_stable_XX_kernel << <BLOCKS, THREADS >> >(dev_TH, XX_block, person_count, total_subtest_count, subtest);
	device_reduce_stable_SUM_XX_kernel << <1, 1024 >> >(XX_block, XX, BLOCKS);
	cudaDeviceSynchronize();
	free(XX_block);
#else
	for (int p = 0; p < person_count; p++) {
		float temp_TH = dev_TH[p*total_subtest_count + subtest];
		XX[0] += temp_TH * temp_TH;
		XX[2] -= temp_TH;
	}
#endif
	XX[3] = (double)person_count;
	XX[1] = XX[2];

	if (unif_flag == 0) {
		double  AGVAR[2][2];
		AGMU[0] = AMU_I;
		AGMU[2] = GMU_I;
		AGVAR[0][0] = AVAR_I;
		AGVAR[1][1] = GVAR_I;
		AGVAR[0][1] = 0.0;
		AGVAR[1][0] = 0.0;

		det_A = AGVAR[0][0] * AGVAR[1][1] - AGVAR[1][0] * AGVAR[0][1];
		SIGMA2[0] = AGVAR[1][1] / det_A;
		SIGMA2[1] = -AGVAR[0][1] / det_A;
		SIGMA2[2] = -AGVAR[1][0] / det_A;
		SIGMA2[3] = AGVAR[0][0] / det_A;

		XX[0] += SIGMA2[0];
		XX[1] += SIGMA2[1];
		XX[2] += SIGMA2[2];
		XX[3] += SIGMA2[3];
	}

	det_A = XX[0] * XX[3] - XX[2] * XX[1];
	IX[0] = XX[3] / det_A;
	IX[1] = -XX[1] / det_A;
	IX[2] = -XX[2] / det_A;
	IX[3] = XX[0] / det_A;

	A_00 = IX[0],
		L_00 = sqrt(A_00),
		A_10 = IX[2],
		A_11 = IX[3],
		L_10 = A_10 / L_00,
		diag = A_11 - L_10 * L_10,
		L_11 = sqrt(diag);

	AMAT[0] = L_00;
	AMAT[2] = L_10;
	AMAT[1] = 0.0;
	AMAT[3] = L_11;

	calcAG_kernel << <dev_SUBTEST_COUNTS[subtest], 1 >> >(rngStates, dev_A, dev_G, dev_Z, dev_TH, SIGMA2, IX, AGMU, AMAT, person_count, item_count, total_subtest_count, unif_flag, subitem, subtest, BLOCKS, THREADS);
	cudaDeviceSynchronize();

	free(XX);
	free(SIGMA2);
	free(IX);
	free(AGMU);
	free(AMAT);
}
cudaError_t				cudaSubtest(curandState_t *rngStates, double *dev_A, double *dev_G, double* dev_Z, double* dev_TH, int person_count, int item_count, int total_subtest_count, int unif_flag, int *dev_SUBTEST_COUNTS, int GRID_Z, int THREAD_Z, int BLOCKS, int THREADS)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		subtest_kernel << <GRID_Z, THREAD_Z >> >(rngStates, dev_A, dev_G, dev_Z, dev_TH, person_count, item_count, total_subtest_count, unif_flag, dev_SUBTEST_COUNTS, BLOCKS, THREADS);
	}

	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "subtest launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching subtest_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif



#ifndef CreateAATS
static __global__ void			initAA_kernel(double* dev_AAT, double* dev_AA, int item_count, int total_subtest_count)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;
	int item = blockIdx.y*blockDim.y + threadIdx.y;
	dev_AAT[subtest*item_count + item] = dev_AA[item*total_subtest_count + subtest] = 0.0;
}
cudaError_t				initAA(double* dev_AAT, double* dev_AA, int item_count, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		initAA_kernel<<<grid, block>>>(dev_AAT, dev_AA, item_count, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "initAA_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching initAA_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
static __global__ void			stripeAA_dynamic_kernel(double* dev_AAT, double* dev_AA, double* dev_A, int subtest, int subtest_count, int subtest_ord, int item_count, int total_subtest_count)
{
	int subtest_item = blockIdx.x*blockDim.x + threadIdx.x;
	dev_AAT[subtest*item_count + (subtest_item + subtest_ord)] = dev_AA[(subtest_item + subtest_ord)*total_subtest_count + subtest] = dev_A[(subtest_item + subtest_ord)];
}
static __global__ void			stripeAA_kernel(double* dev_AAT, double* dev_AA, double* dev_A, int* dev_SUBTEST_COUNTS, int item_count, int total_subtest_count)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;
	int subtest_count = dev_SUBTEST_COUNTS[subtest];
	stripeAA_dynamic_kernel<<<subtest_count, 1>>>(dev_AAT, dev_AA, dev_A, subtest, subtest_count, (subtest*subtest_count), item_count, total_subtest_count);
}
cudaError_t				stripeAA(double* dev_AAT, double* dev_AA, double* dev_A, int* dev_SUBTEST_COUNTS, int item_count, int total_subtest_count, int GRID_Z, int THREAD_Z)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		stripeAA_kernel<<<GRID_Z, THREAD_Z>>>(dev_AAT, dev_AA, dev_A, dev_SUBTEST_COUNTS, item_count, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "stripeAA_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching stripeAA_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
static __global__ void			calcAATS_kernel(double* dev_AAT, double* dev_AA, double* dev_D_AATS, int item_count, int total_subtest_count)
{
	int subtest1 = blockIdx.x*blockDim.x + threadIdx.x;
	int subtest0 = blockIdx.y*blockDim.y + threadIdx.y;
	///////////////////////////////////////////////////////////////////////////////////////
	double val = 0.0;
	for(int n = 0; n < item_count; n++)
	{
		val += dev_AAT[subtest1*item_count + n] * dev_AA[n*total_subtest_count + subtest0];
	}
	dev_D_AATS[subtest1*total_subtest_count + subtest0] = (double)val;
	///////////////////////////////////////////////////////////////////////////////////////
}
cudaError_t				calcAATS(double* dev_AAT, double* dev_AA, double* dev_D_AATS, int item_count, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		calcAATS_kernel<<<grid, block>>>(dev_AAT, dev_AA, dev_D_AATS, item_count, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "calcAATS_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcAATS_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return  (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif


#ifndef CopyRandomNormal
static __global__ void			copyRN_kernel(double2* dev_CTH_WZ, double *dev_WZ, int person_count, int total_subtest_count)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;
	int person = blockIdx.y*blockDim.y + threadIdx.y;
	dev_CTH_WZ[subtest*person_count + person].x = dev_CTH_WZ[person*total_subtest_count + subtest].y = dev_WZ[person*total_subtest_count + subtest];
}
cudaError_t				copyRN(double2* dev_CTH_WZ, double *dev_WZ, int person_count, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		copyRN_kernel<<<grid, block>>>(dev_CTH_WZ, dev_WZ, person_count, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "copyRN launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyRN_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif


#ifndef CalculateD
static __global__ void			calcCTH_kernel(double* dev_RSIG_C, double* dev_TH, double2* dev_CTH_WZ, int person_count, int total_subtest_count)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;
	int person = blockIdx.y*blockDim.y + threadIdx.y;
	double val = 0.0;
	for(int n = 0; n < total_subtest_count; n++)
	{
		val += (dev_RSIG_C[subtest*total_subtest_count + n] * dev_TH[person*total_subtest_count + n]);
	}
	dev_CTH_WZ[person*total_subtest_count + subtest].y = dev_CTH_WZ[subtest*person_count + person].x = (double)val;
}
cudaError_t				calcCTH(double* dev_RSIG_C, double* dev_TH, double2* dev_CTH_WZ, int person_count, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		calcCTH_kernel<<<grid, block>>>(dev_RSIG_C, dev_TH, dev_CTH_WZ, person_count, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "calcCTH_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcCTH_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
__global__ void					device_reduce_stable_D_AATS_kernel(double2* dev_CTH_WZ, double* dev_D_AATS_block, int person_count, int total_subtest_count, int subtest0, int subtest1) {
	double sum = 0.0;
	for(int p = blockIdx.x*blockDim.x + threadIdx.x; p < person_count; p += blockDim.x*gridDim.x) {
		sum += (dev_CTH_WZ[subtest0*person_count + p].x * dev_CTH_WZ[p*total_subtest_count + subtest1].y);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		dev_D_AATS_block[blockIdx.x*total_subtest_count*total_subtest_count + subtest0*total_subtest_count + subtest1] = sum;
	}
}
__global__ void					device_reduce_stable_SUM_D_AATS_kernel(double* dev_D_AATS_block, double* dev_D_AATS, int total_subtest_count, int BLOCKS, int subtest0, int subtest1) {
	double sum = 0.0;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < BLOCKS; i += blockDim.x*gridDim.x) {
		sum += dev_D_AATS_block[i*total_subtest_count*total_subtest_count + subtest0*total_subtest_count + subtest1];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		dev_D_AATS[subtest0*total_subtest_count + subtest1] = sum;
	}
}
static __global__ void			calcD_kernel(double2* dev_CTH_WZ, double* dev_D_AATS, int person_count, int total_subtest_count, int BLOCKS, int THREADS)
{
	int subtest1 = blockIdx.x*blockDim.x + threadIdx.x;
	int subtest0 = blockIdx.y*blockDim.y + threadIdx.y;
#ifdef PRD	
	double* D_AATS_block = (double*)malloc(1024 * sizeof(double));
	device_reduce_stable_D_AATS_kernel<<<BLOCKS, THREADS>>>(dev_CTH_WZ, D_AATS_block, person_count, total_subtest_count, subtest0, subtest1);
	device_reduce_stable_SUM_D_AATS_kernel<<<1, 1024>>>(D_AATS_block, dev_D_AATS, total_subtest_count, BLOCKS, subtest0, subtest1);
	cudaDeviceSynchronize();	
	free(D_AATS_block);
#else
	double val = 0.0;
	for(int p = 0; p < person_count; p++)
	{
		val += dev_CTH_WZ[subtest0*person_count + p].x * dev_CTH_WZ[p*total_subtest_count + subtest1].y;
	}
	dev_D_AATS[subtest0*total_subtest_count + subtest1] = (double) val;
#endif
}
cudaError_t				calcD(double2* dev_CTH_WZ, double* dev_D_AATS, int person_count, int total_subtest_count, int gx, int gy, int bx, int by, int BLOCKS, int THREADS)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		calcD_kernel<<<grid, block>>>(dev_CTH_WZ, dev_D_AATS, person_count, total_subtest_count, BLOCKS, THREADS);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "calcD launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcD_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif


#ifndef CalculateC_ZVAR
__global__ void					device_reduce_stable_PVAR_ZVAR_kernel(double2* dev_CTH_WZ, double* dev_PVAR_ZVAR_block, int person_count, int total_subtest_count, int subtest0, int subtest1) {
	double sum = 0.0;
	for(int p = blockIdx.x*blockDim.x + threadIdx.x; p < person_count; p += blockDim.x*gridDim.x) {
		sum += (dev_CTH_WZ[subtest0*person_count + p].x * dev_CTH_WZ[p*total_subtest_count + subtest1].y);
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		dev_PVAR_ZVAR_block[blockIdx.x*total_subtest_count*total_subtest_count + subtest0*total_subtest_count + subtest1] = sum;
	}
}
__global__ void					device_reduce_stable_SUM_PVAR_ZVAR_kernel(double* dev_PVAR_ZVAR_block, double* dev_PVAR_ZVAR, int total_subtest_count, int N, int subtest0, int subtest1) {
	double sum = 0.0;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x) {
		sum += dev_PVAR_ZVAR_block[i*total_subtest_count*total_subtest_count + subtest0*total_subtest_count + subtest1];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0) {
		dev_PVAR_ZVAR[subtest0*total_subtest_count + subtest1] = sum;
	}
}
static __global__ void			calcPVAR_ZVAR_dynamic_kernel(double2* dev_CTH_WZ, double* dev_PVAR_ZVAR, int person_count, int total_subtest_count, int subtest0, int BLOCKS, int THREADS)
{
	int subtest1 = blockIdx.x*blockDim.x + threadIdx.x;	
#ifdef PRPVAR
	double* PVAR_ZVAR_block = (double*)malloc(1024 * sizeof(double));
	device_reduce_stable_PVAR_ZVAR_kernel<<<BLOCKS, THREADS>>>(dev_CTH_WZ, PVAR_ZVAR_block, person_count, total_subtest_count, subtest0, subtest1);
	device_reduce_stable_SUM_PVAR_ZVAR_kernel<<<1, 1024>>>(PVAR_ZVAR_block, dev_PVAR_ZVAR, total_subtest_count, BLOCKS, subtest0, subtest1);
	cudaDeviceSynchronize();
	free(PVAR_ZVAR_block);
#else
	double val = 0.0;
	for(int p = 0; p < person_count; p++)
	{
		val += dev_CTH_WZ[subtest0*person_count + p].x * dev_CTH_WZ[p*total_subtest_count + subtest1].y;
	}
	dev_PVAR_ZVAR[subtest0*total_subtest_count + subtest1] = (double)val;
#endif
}
static __global__ void			calcC_ZVAR_kernel(double2* dev_CTH_WZ, double* dev_PVAR_ZVAR, int* dev_SUBTEST_COUNTS, double* dev_RSIG_C, double* dev_A, int person_count, int total_subtest_count, int GRID_Z, int THREAD_Z, int BLOCKS, int THREADS)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;

	int subtest_count = dev_SUBTEST_COUNTS[subtest];
	for(int subtest0 = 0; subtest0 < total_subtest_count; subtest0++)
	{
		dev_RSIG_C[subtest0*total_subtest_count + subtest] = 0.0;
	}
	int subitem = 0;
	for(int ith = subtest; ith > 0; ith--) subitem += dev_SUBTEST_COUNTS[ith - 1];
	double PRODA = 0.0; // 1.0
	for(int item = subitem; item < subitem + subtest_count; item++)
	{
		PRODA += log(dev_A[item]); //*= dev_A[item];
	}
	double fp = 1.0 / (double)subtest_count;

	dev_RSIG_C[subtest*total_subtest_count + subtest] = (double)pow((double)exp(PRODA), fp); // = (double)pow((double)PRODA, fp);

	calcPVAR_ZVAR_dynamic_kernel<<<GRID_Z, THREAD_Z>>>(dev_CTH_WZ, dev_PVAR_ZVAR, person_count, total_subtest_count, subtest, BLOCKS, THREADS);
#ifdef PRPVAR	
	cudaDeviceSynchronize();
#endif
}
cudaError_t				calcC_ZVAR(double2* dev_CTH_WZ, double* dev_PVAR_ZVAR, int* dev_SUBTEST_COUNTS, double* dev_RSIG_C, double* dev_A, int person_count, int total_subtest_count, int GRID_Z, int THREAD_Z, int BLOCKS, int THREADS)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		calcC_ZVAR_kernel<<<GRID_Z, THREAD_Z>>>(dev_CTH_WZ, dev_PVAR_ZVAR, dev_SUBTEST_COUNTS, dev_RSIG_C, dev_A, person_count, total_subtest_count, GRID_Z, THREAD_Z, BLOCKS, THREADS);
	}

	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "calcC_ZVAR launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcC_ZVAR_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif



#ifndef TrackItemStatistics
static __global__ void			trackAG_kernel(double* dev_AV, double* dev_GV, int item_count, int BSIZE)
{
	int item = blockIdx.x*blockDim.x + threadIdx.x;
	double M1 = (dev_AV[item] / BSIZE);
	double M2 = (dev_GV[item] / BSIZE);
	dev_AV[item_count + item] += M1;
	dev_GV[item_count + item] += M2;
	dev_AV[2 * item_count + item] += (M1 * M1);
	dev_GV[2 * item_count + item] += (M2 * M2);
}
cudaError_t				trackAG(double* dev_AV, double* dev_GV, int item_count, int GRID_X, int THREAD_X, int BSIZE)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		trackAG_kernel<<<GRID_X, THREAD_X>>>(dev_AV, dev_GV, item_count, BSIZE);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "trackAG_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching trackAG_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef TrackPersonStatistics
static __global__ void			trackTH_kernel(double* dev_THV, int total_subtest_count, int persons_by_subtests, int BSIZE)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;
	int person = blockIdx.y*blockDim.y + threadIdx.y;
	int i = person*total_subtest_count + subtest;
	double M1 = (dev_THV[i] / BSIZE);
	dev_THV[persons_by_subtests + i] += M1;
	dev_THV[2 * persons_by_subtests + i] += (M1 * M1);
}
cudaError_t				trackTH(double* dev_THV, int total_subtest_count, int persons_by_subtests, int gx, int gy, int bx, int by, int BSIZE)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		trackTH_kernel<<<grid, block>>>(dev_THV, total_subtest_count, persons_by_subtests, BSIZE);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "trackTH_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching trackTH_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef TrackCorrelationStatistics
static __global__ void			trackRHO_kernel(double* dev_RHO, int total_subtest_count, int subtests_by_subtests, int BSIZE)
{
	int subtest1 = blockIdx.x*blockDim.x + threadIdx.x;
	int subtest0 = blockIdx.y*blockDim.y + threadIdx.y;
	int i = subtest0*total_subtest_count + subtest1;
	double M1 = (dev_RHO[i] / BSIZE);
	dev_RHO[subtests_by_subtests + i] += M1;
	dev_RHO[2*subtests_by_subtests + i] += (M1 * M1);
}
cudaError_t				trackRHO(double* dev_RHO, int total_subtest_count, int subtests_by_subtests, int gx, int gy, int bx, int by, int BSIZE)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		trackRHO_kernel<<<grid, block>>>(dev_RHO, total_subtest_count, subtests_by_subtests, BSIZE);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "trackRHO launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching trackRHO_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef CopyItemStatistics
static __global__ void			copyAG_kernel(double* dev_AV, double* dev_GV, double* dev_A, double* dev_G)
{
	int item = blockIdx.x*blockDim.x + threadIdx.x;
	dev_AV[item] = dev_A[item];
	dev_GV[item] = dev_G[item];
}
cudaError_t				copyAG(double* dev_AV, double* dev_GV, double* dev_A, double* dev_G, int GRID_X, int THREAD_X)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		copyAG_kernel<<<GRID_X, THREAD_X>>>(dev_AV, dev_GV, dev_A, dev_G);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "copyAG_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyAG_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef CopyPersonStatistics
static __global__ void			copyTH_kernel(double* dev_THV, double* dev_TH, int total_subtest_count)
{
	int subtest = blockIdx.x*blockDim.x + threadIdx.x;
	int person = blockIdx.y*blockDim.y + threadIdx.y;
	int i = person*total_subtest_count + subtest;
	dev_THV[i] = dev_TH[i];
}
cudaError_t				copyTH(double* dev_THV, double* dev_TH, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		copyTH_kernel<<<grid, block>>>(dev_THV, dev_TH, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "copyTH_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copyTH_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef CopyCorrelationStatistics
static __global__ void			copySIGMA_kernel(double* dev_RHO, double* dev_SIGMA, int total_subtest_count)
{
	int subtest1 = blockIdx.x*blockDim.x + threadIdx.x;
	int subtest0 = blockIdx.y*blockDim.y + threadIdx.y;
	int i = subtest0*total_subtest_count + subtest1;
	dev_RHO[i] = dev_SIGMA[i];
}
cudaError_t				copySIGMA(double* dev_RHO, double* dev_SIGMA, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		copySIGMA_kernel<<<grid, block>>>(dev_RHO, dev_SIGMA, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "copySIGMA launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching copySIGMA_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef SumItemStatistics
static __global__ void			sumAG_kernel(double* dev_AV, double* dev_GV, double* dev_A, double* dev_G)
{
	int item = blockIdx.x*blockDim.x + threadIdx.x;
	dev_AV[item] += dev_A[item];
	dev_GV[item] += dev_G[item];
}
cudaError_t				sumAG(double* dev_AV, double* dev_GV, double* dev_A, double* dev_G, int GRID_X, int THREAD_X)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		sumAG_kernel<<<GRID_X, THREAD_X>>>(dev_AV, dev_GV, dev_A, dev_G);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "sumAG_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sumAG_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef SumPersonStatistics
static __global__ void			sumTH_kernel(double* dev_THV, double* dev_TH, int total_subtest_count)
{
	int person = blockIdx.x*blockDim.x + threadIdx.x;
	int subtest = blockIdx.y*blockDim.y + threadIdx.y;
	int i = person*total_subtest_count + subtest;
	dev_THV[i] += dev_TH[i];
}
cudaError_t				sumTH(double* dev_THV, double* dev_TH, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		sumTH_kernel<<<grid, block>>>(dev_THV, dev_TH, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "sumTH_kernel launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sumTH_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif

#ifndef SumCorrelationStatistics
static __global__ void			sumSIGMA_kernel(double* dev_RHO, double* dev_SIGMA, int total_subtest_count)
{
	int subtest1 = blockIdx.x*blockDim.x + threadIdx.x;
	int subtest0 = blockIdx.y*blockDim.y + threadIdx.y;
	int i = subtest0*total_subtest_count + subtest1;
	dev_RHO[i] += dev_SIGMA[i];
}
cudaError_t				sumSIGMA(double* dev_RHO, double* dev_SIGMA, int total_subtest_count, int gx, int gy, int bx, int by)
{
	cudaError_t cudaStatusA;
	cudaError_t cudaStatusB;
	{
		dim3 grid(gx, gy, 1);
		dim3 block(bx, by, 1);
		sumSIGMA_kernel<<<grid, block>>>(dev_RHO, dev_SIGMA, total_subtest_count);
	}
	cudaStatusA = cudaGetLastError(); if (cudaStatusA != cudaSuccess) { fprintf(stderr, "sumSIGMA launch failed: %s\n", cudaGetErrorString(cudaStatusA)); }
	cudaStatusB = cudaDeviceSynchronize(); if (cudaStatusB != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sumSIGMA_kernel!\n", cudaStatusB); goto Error; }
	;
Error:
	;
	return (cudaStatusA != cudaSuccess) ? cudaStatusA : (cudaStatusB != cudaSuccess) ? cudaStatusB : cudaSuccess;
}
#endif
