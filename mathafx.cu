#include "stdafx.cuh"
#include "mathafx.cuh"

#include "cuda_gsmu2.cuh"

#define BLOCK_SIZE 8

#define TILE_DIM   32
#define BLOCK_ROWS 8

__global__ void CUDA_MATMUL(double *A, double *B, double *C,
							int rows_A, int cols_A,
							int rows_B, int cols_B,
							int rows_C, int cols_C) 
{   
	// BLOCK_SIZE = TOTAL_SUBTEST_COUNT

	__shared__ double shared_M[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double shared_N[BLOCK_SIZE][BLOCK_SIZE];
	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		row = by * BLOCK_SIZE + ty,
		col = bx * BLOCK_SIZE + tx;
	float pval = 0;

	for (int m = 0; m < (cols_A-1)/BLOCK_SIZE+1; ++m) {
		if (row < rows_A && m*BLOCK_SIZE+tx < cols_A)
			shared_M[ty][tx] = A[row*cols_A + m*BLOCK_SIZE+tx];
		else
			shared_M[ty][tx] = 0;
		if (col < cols_B && m*BLOCK_SIZE+ty < rows_B)
			shared_N[ty][tx] = B[(m*BLOCK_SIZE+ty)*cols_B+col];
		else
			shared_N[ty][tx] = 0;

		__syncthreads();
		for (int k=0;k<BLOCK_SIZE;++k)
			pval += shared_M[ty][k] * shared_N[k][tx];
		__syncthreads();
	}
	if (row < rows_C && col < cols_C)
		C[row*cols_C+col] = pval;
}


cudaError_t CUBLAS_MATRIX_MULTIPLY(double *A, double *B, double *C, int2 dimA, int2 dimB)
{
	if(dimA.y != dimB.x) { fprintf(stderr, "A.y != B.x"); exit(-1); }
	
	cudaError_t cudaStatus;

	double *dev_A = 0;
	double *dev_B = 0;
	double *dev_C = 0;

	cublasHandle_t handle;

    cublasCreate(&handle);

    const double alf = 1;
    const double bet = 0;
    const double *alpha = &alf;
    const double *beta = &bet;

    cudaStatus = cudaMalloc((void**)&dev_A, dimA.x * dimA.y * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_B, dimB.x * dimB.y * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_C,  dimA.x * dimB.y * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_A, A, dimA.x * dimA.y * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_B, B, dimB.x * dimB.y * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimA.x, dimB.y, dimA.y, alpha, dev_A, dimA.x, dev_B, dimB.x, beta, dev_C, dimA.x);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(C, dev_C, dimA.x * dimB.y * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	;
Error:
	;

    cudaFree(dev_A);
    cudaFree(dev_B);
	cudaFree(dev_C);

	cublasDestroy(handle);

    return cudaStatus;
}

__global__ void CUDA_TRANSPOSE(const double *idata, double *odata, int width, int height)
{	
	__shared__ double block[BLOCK_SIZE][BLOCK_SIZE+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

void matadd(int nr1, int nc1, double *m1a, int nr2, int nc2, double*m1b, int nr3, int nc3, double *m3a)
{
	//assert row & cols correct
	int i; for (i = 0; i <  nr1; i++)
	{
		int j; for (j = 0; j < nc1; j++)
		{
			m3a[i*nc3+j] = m1a[i*nc1+j] + m1b[i*nc2+j];
		}
	}
}
void matmul(int nr1, int nc1, double *m1a, int nr2, int nc2, double *m2a, int nr3, int nc3, double *m3a, int bs)
{
	switch(bs) {
	case 0: // SERIAL
		{
			int r; for(r=0;r<nr1;r++)
			{
				int c; for(c=0;c<nc2;c++)
				{
					double val = 0.0;
					int n; for(n=0;n<nc1;n++)
					{
						double fp = m1a[r*nc1+n];
						double gp = m2a[n*nc2+c];
						val = val + fp * gp;
					}

					m3a[r*nc3+c] = (double)val;   //floor(val*10000)/10000.0;
				}
			}
		} break;
	case 1: // CUDA
		{
			const int mem_size_1 = nr1 * nc1 * sizeof(double);
			const int mem_size_2 = nr2 * nc2 * sizeof(double);
			const int mem_size_3 = nr3 * nc3 * sizeof(double);
			
			double *dev_A;
			double *dev_B;
			double *dev_C;

			cudaStream_t stream = (cudaStream_t) malloc(sizeof(cudaStream_t));
			cudaStreamCreate(&(stream));
			
			cudaMalloc(&dev_A, mem_size_1);
			cudaMalloc(&dev_B, mem_size_2);
			cudaMalloc(&dev_C, mem_size_3);

			dim3 dimGrid((nc3-1)/BLOCK_SIZE+1, (nr3-1)/BLOCK_SIZE+1, 1);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
			
			cudaMemcpyAsync(dev_A, m1a, mem_size_1, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(dev_B, m2a, mem_size_2, cudaMemcpyHostToDevice, stream);
			
			CUDA_MATMUL<<<dimGrid, dimBlock, BLOCK_SIZE*BLOCK_SIZE*2, stream>>>(dev_A, dev_B, dev_C, nr1, nc1, nr2, nc2, nr3, nc3);
			
			cudaMemcpyAsync(m3a, dev_C, mem_size_3, cudaMemcpyDeviceToHost, stream);
			
			cudaFree(dev_A);
			cudaFree(dev_B);
			cudaFree(dev_C);

			cudaStreamDestroy(stream);

		} break;
	case 2: // CUBLAS
		{
			int2 dimA = { nc1, nr1 };
			int2 dimB = { nc2, nr2 };
			CUBLAS_MATRIX_MULTIPLY(m1a, m2a, m3a, dimA, dimB);
		} break;
	}

}

void transpose(int nr1, int nc1, double *m1a, int nr2, int nc2, double*m2a, int bs)
{	
	switch(bs) {
	case 0: // SERIAL
		{
			//assert row & cols correct
			int i; for (i = 0; i < nr1; i++)
			{
				int j; for (j = 0; j < nc1; j++)
				{
					m2a[j*nc2+i] = m1a[i*nc1+j];
				}
			}
		} break;
	case 1: // CUDA
		{
			const int mem_size_1 = nr1 * nc1 * sizeof(double);
			
			double *dev_A;
			double *dev_B;

			cudaStream_t stream = (cudaStream_t) malloc(sizeof(cudaStream_t));
			cudaStreamCreate(&(stream));
			
			cudaMalloc(&dev_A, mem_size_1);
			cudaMalloc(&dev_B, mem_size_1);

			dim3 dimGrid((nc1-1)/BLOCK_SIZE+1, (nr1-1)/BLOCK_SIZE+1, 1);
			dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
			
			cudaMemcpyAsync(dev_A, m1a, mem_size_1, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(dev_B, m2a, mem_size_1, cudaMemcpyHostToDevice, stream);
			
			CUDA_TRANSPOSE<<<dimGrid, dimBlock, BLOCK_SIZE*(BLOCK_SIZE+1), stream>>>(dev_A, dev_B, nc1, nr1);
			
			cudaMemcpyAsync(m2a, dev_B, mem_size_1, cudaMemcpyDeviceToHost, stream);
			
			cudaFree(dev_A);
			cudaFree(dev_B);

			cudaStreamDestroy(stream);
		} break;
	}
}

void matinv(int N, double*A)
{
	double Z,D;
	int I;
	if(N==2)
	{ 
		int I;
		D = (double)((double)A[0*2+0] * (double)A[1*2+1])- (double)((double)A[0*2+1] * (double)A[1*2+0]);
		Z = A[0*2+0];
		A[0*2+0] = A[1*2+1]; 
		A[1*2+1] = (double)Z;
		Z = A[0*2+1];
		A[0*2+1] = -A[1*2+0]; 
		A[1*2+0] = (double)-Z;
		for(I=0;I<N;I++)
		{     
			int J;for(J=0;J<N;J++)
			{
				Z = A[I*2+J];
				A[I*2+J] =(double) ( Z/D);
			}
		}
		return;
	}
	for(I=0;I<N;I++)
	{ 
		int J;
		int K;
		Z = A[I*N+I];	
		A[I*N+I] = 1.0;
		for(J=0;J<N;J++)  
			A[I*N+J] = (double)( A[I*N+J] / Z);

		for(K=0;K<N;K++)   
		{
			if(K!=I)
			{
				int J; 
				Z = A[K*N+I];
				A[K*N+I] = 0.0;
				for(J=0;J<N;J++) 
					A[K*N+J] = (double)(A[K*N+J] - Z * A[I*N+J]);
			}   
		}
	}
	return;
}

int chfac(int Np, double *Ap, double *Rp, double *L)
{
	int N = Np;
	for (int I = 0; I<N; I++)
		for (int K = 0; K<N; K++)
			L[I*N + K] = 0.0;
	int flag = 0;
	for (int j = 0; j<N; j++)
	{
		double COL = 0;
		for (int i = 0; i <= j - 1; i++)
		{
			COL = 0;
			for (int K = 0; K <= i - 1; K++)
			{
				COL = COL + L[K*N + i] * L[K*N + j];
			}
			if (L[i*N + i] == 0.0)
			{
				flag = 1;
				L[i*N + j] = 0.0;
			}
			else
				L[i*N + j] = (Ap[i*N + j] - COL) / L[i*N + i];
		}
		COL = 0;
		for (int K = 0; K <= j - 1; K++)
		{
			COL = COL + L[K*N + j] * L[K*N + j];
		}
		if (Ap[j*N + j] <= COL)
		{
			L[j*N + j] = 0.0;
			flag = 2;
		}
		else
			L[j*N + j] = sqrt(Ap[j*N + j] - COL);

	}
	for (int I = 0; I<N; I++)
		for (int K = 0; K<N; K++)
			Rp[I*N + K] = L[I*N + K];
	return flag;
}

//void sadd(int* LDA, double *A, double *R, int* LDR)
//{
//	int lda = *LDA;
//	if ( *LDR==1)
//	{
//		for(int I=0;I<lda;I++)
//			R[I] = R[I] + A[0];
//	}
//}

//void sscal(int *LDA, double *A, double *R, int* LDR)
//{
//	int lda = *LDA;
//	if ( *LDR==1)
//	{
//		for(int I=0;I<lda;I++)
//			R[I] = R[I] * A[0];
//	}
//}

void sadd(int lda, double *A, double *R, int* LDR)
{
	if (*LDR == 1)
	{
		for(int I=0;I<lda;I++)
			R[I] = R[I] + A[0];
	}
}

void sscal(int lda, double *A, double *R, int* LDR)
{
	if (*LDR == 1)
	{
		for(int I=0;I<lda;I++)
			R[I] = R[I] * A[0];
	}
}


double mean(double *X, int count)
{
	double sum = 0.0; 
	for(int i = 0; i < count; i++)
	{
		sum += X[i];
	}
	return sum/count;
}

double correlation(double *X, double *Y, int count) 
{
	double mean_X = mean(X, count);
	double mean_Y = mean(Y, count);
	double sum_XY = 0.0;	
	double sum_XS = 0.0;
	double sum_YS = 0.0;
	for(int i = 0; i < count; i++)
	{
		double mx = (X[i]-mean_X);
		double my = (Y[i]-mean_Y);
		sum_XS += mx*mx;
		sum_XY += mx*my;
		sum_YS += my*my;
	}
	double ms = sum_XS*sum_YS;
	return (ms!=0) ? (sum_XY / sqrt(ms)) : 0; 
}



/*
C          THT              CTHT                      D
c 0 0 0  t t .. t .. t t   ct ct c.c. ct c.c. ct ct  cc cd ce cf
0 d 0 0  t t .. t .. t t   dt dt d.d. dt d.d. dt dt  dc dd de df
0 0 e 0  t t .. t .. t t   et et e.e. et e.e. et et  ec ed ee ef
0 0 0 f  t t .. t .. t t   ft ft f.f. ft f.f. ft ft  fc fd fe ff
*/

void CTOD(int total_subtest_count, int person_count, double *C, double *THT, double *D)
{
	double *CTH  = (double*)malloc(total_subtest_count*person_count*sizeof(double));
	double *CTHT = (double*)malloc(total_subtest_count*person_count*sizeof(double));
	matmul(total_subtest_count,total_subtest_count,C,  total_subtest_count,person_count,THT,  total_subtest_count,person_count,CTH,1);
	transpose(total_subtest_count,person_count, CTH,   person_count,total_subtest_count,CTHT, 1);
	matmul(total_subtest_count,person_count, CTH,      person_count,total_subtest_count,CTHT, total_subtest_count,total_subtest_count,D,1);
	free(CTH);
	free(CTHT);
}
