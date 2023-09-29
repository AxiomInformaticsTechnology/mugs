#include "gsmu2.cuh"
#include "cuda_gsmu2.cuh"
#include "mpi_gsmu2.cuh"

#include "showdata.cuh"
#include "mathafx.cuh"
#include "number.cuh"
#include "stdafx.cuh"

#include "randafx.cuh"


// TODO: make struct out of inputs
void slave_cuda_gsmu2(int rank, 
					  int size,
					  int person_count,
					  int item_count,
					  int total_subtest_count,
					  int iteration_count,
					  int burnin,
					  int batches,
					  int uniform)
{

	srand(rand() + 7 * rank);


	int persons;

	MPI_Recv(&persons, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


	int persons_by_items = persons * item_count,
		persons_by_subtests = persons * total_subtest_count,
		items_by_subtests = item_count * total_subtest_count,
		subtests_by_subtests = total_subtest_count * total_subtest_count;


	unsigned int subtests_by_szDbl = total_subtest_count * sizeof(double),
				 persons_by_items_by_szDbl = persons_by_items * sizeof(double), 
				 persons_by_subtests_by_szDbl = persons_by_subtests * sizeof(double),
				 items_by_subtests_by_szDbl = items_by_subtests * sizeof(double),
				 subtests_by_subtests_by_szDbl = subtests_by_subtests * sizeof(double),
				 items_by_szDbl = item_count * sizeof(double);



	





	// SETUP GPU	
	unsigned int THREAD_X = 8,
				 THREAD_Y = 8;

	if (item_count % 10 == 0)     THREAD_X = 10;
	else if (item_count % 8 == 0) THREAD_X = 8;
	else if (item_count % 5 == 0) THREAD_X = 5;
	else if (item_count % 4 == 0) THREAD_X = 4;
	else if (item_count % 2 == 0) THREAD_X = 2;
	else					      THREAD_X = 1;

	if (persons % 10 == 0)     THREAD_Y = 10;
	else if (persons % 8 == 0) THREAD_Y = 8;
	else if (persons % 5 == 0) THREAD_Y = 5;
	else if (persons % 4 == 0) THREAD_Y = 4;
	else if (persons % 2 == 0) THREAD_Y = 2;
	else					   THREAD_Y = 1;

	unsigned int GRID_X = ((item_count + THREAD_X - 1) / THREAD_X),
				 GRID_Y = ((persons + THREAD_Y - 1) / THREAD_Y);



	double *dev_Z,
		   *dev_U,
		   *dev_A,
		   *dev_G,
		   *dev_TH;

	int *dev_Y,
		*dev_ORD;


	cudaStream_t stream;

	cudaStreamCreate(&stream);

	curandGenerator_t gen;

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);


	cudaMalloc((void **)&dev_Z, persons_by_items_by_szDbl);
	cudaMalloc((void **)&dev_U, persons_by_items_by_szDbl);
	cudaMalloc((void **)&dev_A, items_by_szDbl);
	cudaMalloc((void **)&dev_G, items_by_szDbl);
	cudaMalloc((void **)&dev_TH, persons_by_subtests_by_szDbl);

	cudaMalloc((void **)&dev_Y, persons*item_count*sizeof(int));
	cudaMalloc((void **)&dev_ORD, item_count*sizeof(int));


#ifdef CUDA_TH

	double *dev_WZ,
		   *dev_RSIG,
		   *dev_PVAR,
		   *dev_AAT;

	cudaMalloc((void **)&dev_WZ, persons_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_RSIG, subtests_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_PVAR, subtests_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_AAT, items_by_subtests_by_szDbl);
#endif



	int* Y = (int*)malloc(sizeof(int)*persons*item_count);

	int* ord = (int*)malloc(sizeof(int)*item_count);

	MPI_Bcast(ord, item_count, MPI_INT, ROOT, MPI_COMM_WORLD);

	MPI_Scatterv(NULL, NULL, NULL, MPI_INT, Y, persons_by_items, MPI_INT, ROOT, MPI_COMM_WORLD);


	cudaMemcpy(dev_Y, Y, persons*item_count * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_ORD, ord, item_count * sizeof(unsigned int), cudaMemcpyHostToDevice);


	double LP, BB, U, TMP, SUM;

	double* A = (double*)malloc(items_by_szDbl);
	double* G = (double*)malloc(items_by_szDbl);
	double* TH = (double*)malloc(persons_by_subtests_by_szDbl);

	double* Z = (double*)malloc(persons_by_items_by_szDbl);

	double* AAT = (double*)malloc(items_by_subtests_by_szDbl);
	double* RSIG = (double*)malloc(subtests_by_subtests_by_szDbl);
	double* PVAR = (double*)malloc(subtests_by_subtests_by_szDbl);
	double* RTH = (double*)malloc(subtests_by_szDbl);
	double* PMEAN = (double*)malloc(subtests_by_szDbl);
	double* PMEAN1 = (double*)malloc(subtests_by_szDbl);


	MPI_Status statusA, statusG;
	MPI_Request requestA, requestG;

	MPI_Ibcast(A, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestA);
	MPI_Ibcast(G, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestG);

	
	// SCATTER TH
	MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, TH, persons_by_subtests, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


	MPI_Wait(&requestA, &statusA);
	MPI_Wait(&requestG, &statusG);


	// BEGIN ITERATE
	for (int iteration = 0; iteration < iteration_count; iteration++) {


		// UPDATE Z

		cudaMemcpyAsync(dev_A, A, items_by_szDbl, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_G, G, items_by_szDbl, cudaMemcpyHostToDevice, stream);

		cudaMemcpyAsync(dev_TH, TH, persons_by_subtests_by_szDbl, cudaMemcpyHostToDevice, stream);

		curandGenerateUniformDouble(gen, dev_U, persons_by_items);



		calcZ(stream, dev_U, GRID_X, GRID_Y, THREAD_X, THREAD_Y, dev_Y, dev_A, dev_G, dev_TH, dev_Z, dev_ORD, item_count, persons, total_subtest_count);


		cudaMemcpy(Z, dev_Z, persons_by_items_by_szDbl, cudaMemcpyDeviceToHost);





		// GATHER Z
		MPI_Gatherv(Z, persons_by_items, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


		MPI_Bcast(AAT, item_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(PVAR, total_subtest_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(RSIG, total_subtest_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(PMEAN, total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(RTH, total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


		// UPDATE TH

#ifdef CUDA_TH
		curandGenerateNormalDouble(gen, dev_WZ, persons_by_subtests, 0.0, 1.0);

		cudaMemcpyAsync(dev_RSIG, RSIG, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_PVAR, PVAR, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, stream);

		cudaMemcpyAsync(dev_AAT, AAT, items_by_subtests_by_szDbl, cudaMemcpyHostToDevice, stream);

		calcTH(stream, dev_WZ, dev_AAT, dev_RSIG, dev_PVAR, dev_Z, dev_G, dev_TH, GRID_Y, THREAD_Y, item_count, persons, total_subtest_count);

		cudaMemcpy(TH, dev_TH, persons_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
#else
		for (int person = 0; person < persons; person++) {

			for (int r = 0; r < total_subtest_count; r++) {
				SUM = 0.0;
				for (int item = 0; item < item_count; item++) {
					SUM += AAT[r*item_count + item] * (Z[person*item_count + item] + G[item]);
				}
				PMEAN1[r] = SUM;
			}
			for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
				SUM = 0.0;
				for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					SUM += PVAR[subtest0*total_subtest_count + subtest1] * PMEAN1[subtest1];
				}
				PMEAN[subtest0] = SUM;
				SUM = 0.0;
				for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					SUM += random_normal() * RSIG[subtest1*total_subtest_count + subtest0];
				}
				RTH[subtest0] = SUM;
			}
			for (int subtest = 0; subtest < total_subtest_count; subtest++) {
				TH[person*total_subtest_count + subtest] = RTH[subtest] + PMEAN[subtest];
			}

		}
#endif



		// GATHER TH
		MPI_Gatherv(TH, persons_by_subtests, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


		MPI_Ibcast(A, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestA);
		MPI_Ibcast(G, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestG);

		MPI_Wait(&requestA, &statusA);
		MPI_Wait(&requestG, &statusG);

	}


	free(Y);
	free(ord);
	free(A);
	free(G);
	free(TH);

	free(Z);

	free(AAT);
	free(RSIG);
	free(PVAR);
	free(RTH);
	free(PMEAN);
	free(PMEAN1);


	cudaFree(dev_Z);
	cudaFree(dev_U);
	cudaFree(dev_A);
	cudaFree(dev_G);
	cudaFree(dev_TH);

	cudaFree(dev_Y);
	cudaFree(dev_ORD);

#ifdef CUDA_TH
	cudaFree(dev_WZ);
	cudaFree(dev_RSIG);
	cudaFree(dev_PVAR);
	cudaFree(dev_AAT);
#endif

	cudaStreamDestroy(stream);
	curandDestroyGenerator(gen);
	cudaDeviceReset();

}
