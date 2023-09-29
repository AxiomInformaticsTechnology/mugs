#include "gsmu2.cuh"
#include "mpi_gsmu2.cuh"

#include "showdata.cuh"
#include "mathafx.cuh"
#include "number.cuh"
#include "stdafx.cuh"

#include "randafx.cuh"


// TODO: make struct out of inputs
void slave_gsmu2(int rank,
				 int size,
				 int person_count,
				 int item_count,
				 int total_subtest_count,
				 int iteration_count,
				 int burnin,
				 int batches,
				 int uniform) 
{
	
	srand(rand()+7*rank);
	
	int persons;

	MPI_Recv(&persons, 1, MPI_INT, ROOT, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


	int* Y = (int*)malloc(sizeof(int)*persons*item_count);

	int* ord = (int*)malloc(sizeof(int)*item_count);

	MPI_Bcast(ord, item_count, MPI_INT, ROOT, MPI_COMM_WORLD);

	MPI_Scatterv(NULL, NULL, NULL, MPI_INT, Y, persons*item_count, MPI_INT, ROOT, MPI_COMM_WORLD);

	double LP, BB, U, TMP, SUM;

	double* A = (double*)malloc(sizeof(double)*item_count);
	double* G = (double*)malloc(sizeof(double)*item_count);
	double* TH = (double*)malloc(sizeof(double)*persons*total_subtest_count);

	double* Z = (double*)malloc(sizeof(double)*persons*item_count);

	double* AAT = (double*)malloc(sizeof(double)*item_count*total_subtest_count);
	double* RSIG = (double*)malloc(sizeof(double)*total_subtest_count*total_subtest_count);
	double* PVAR = (double*)malloc(sizeof(double)*total_subtest_count*total_subtest_count);
	double* RTH = (double*)malloc(sizeof(double)*total_subtest_count);
	double* PMEAN = (double*)malloc(sizeof(double)*total_subtest_count);
	double* PMEAN1 = (double*)malloc(sizeof(double)*total_subtest_count);


	MPI_Status statusA, statusG;
	MPI_Request requestA, requestG;

	MPI_Ibcast(A, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestA);
	MPI_Ibcast(G, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestG);


	// SCATTER TH
	MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, TH, persons*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);



	MPI_Wait(&requestA, &statusA);
	MPI_Wait(&requestG, &statusG);

	
	// BEGIN ITERATE
	for(int iteration = 0; iteration < iteration_count; iteration++) {


		// UPDATE Z

		for(int person = 0; person < persons; person++) {
			for(int item = 0; item < item_count; item++) {
				LP = G[item] - A[item] * TH[person*total_subtest_count + ord[item]];
				BB = stdnormal_cdf(LP);
				U = random_uniform_pos();
				TMP = (Y[person*item_count + item] == 0) ? (BB * U) : (U + BB * (1.0 - U));
				TMP = stdnormal_inv(TMP) - LP;
				Z[person*item_count + item] = isfinite(TMP) ? TMP : 0.0;
			}
		}
		
		
		

		// GATHER Z
		MPI_Gatherv(Z, persons*item_count, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

		

		MPI_Bcast(AAT, item_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(PVAR, total_subtest_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(RSIG, total_subtest_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(PMEAN, total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(RTH, total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


		// UPDATE TH

		for(int person = 0; person < persons; person++) {

			for(int r = 0; r < total_subtest_count; r++) {
				SUM = 0.0;
				for(int item = 0; item < item_count; item++) {
					SUM += AAT[r*item_count + item] * (Z[person*item_count + item] + G[item]);
				}
				PMEAN1[r] = SUM;
			}
			for(int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
				SUM = 0.0;
				for(int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					SUM += PVAR[subtest0*total_subtest_count + subtest1] * PMEAN1[subtest1];
				}
				PMEAN[subtest0] = SUM;
				SUM = 0.0;
				for(int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					SUM += random_normal() * RSIG[subtest1*total_subtest_count + subtest0];
				}
				RTH[subtest0] = SUM;
			}
			for(int subtest = 0; subtest < total_subtest_count; subtest++) {
				TH[person*total_subtest_count + subtest] = RTH[subtest] + PMEAN[subtest];
			}

		}
		

		// GATHER TH
		MPI_Gatherv(TH, persons*total_subtest_count, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);




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
	
}
