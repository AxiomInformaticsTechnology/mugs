#include "gsmu2.cuh"
#include "cuda_gsmu2.cuh"
#include "mpi_gsmu2.cuh"

#include "showdata.cuh"
#include "mathafx.cuh"
#include "number.cuh"
#include "stdafx.cuh"

#include "randafx.cuh"


// TODO: make struct out of inputs
void master_cuda_gsmu2(int *Y,
					   int *subtest_counts,
					   int size,
					   int person_count,
					   int item_count,
					   int total_subtest_count,
					   int iteration_count,
					   int burnin,
					   int batches,
					   int uniform,
					   double *ITEM,
					   double *PERSON,
					   double *CORR) 
{

	srand(rand());


	int bsize = (iteration_count - burnin) / batches;

	int persons_by_items = person_count * item_count,
		persons_by_subtests = person_count * total_subtest_count,
		items_by_subtests = item_count * total_subtest_count,
		subtests_by_subtests = total_subtest_count * total_subtest_count;

	unsigned int items_by_subtests_by_szDbl = items_by_subtests * sizeof(double),
				 persons_by_subtests_by_szDbl = persons_by_subtests * sizeof(double),
				 subtests_by_subtests_by_szDbl = subtests_by_subtests * sizeof(double),
				 subtests_by_szDbl = total_subtest_count * sizeof(double),
				 subtests_by_szInt = total_subtest_count * sizeof(int),
				 items_by_szDbl = item_count * sizeof(double),
				 items_by_szInt = item_count * sizeof(int),
				 persons_by_items_by_szDbl = persons_by_items * sizeof(double);

	double LP, BB, U, TMP, SUM;

	double *A = (double*)malloc(items_by_szDbl);
	double *G = (double*)malloc(items_by_szDbl);
	double *TH = (double*)malloc(persons_by_subtests_by_szDbl);

	double *Z = (double*)malloc(persons_by_items_by_szDbl);

	double *SIGMA = (double*)malloc(subtests_by_subtests_by_szDbl);

	double *AV = (double*)malloc(items_by_szDbl * 3);
	double *GV = (double*)malloc(items_by_szDbl * 3);
	double *THV = (double*)malloc(persons_by_subtests_by_szDbl * 3);
	double *RHO = (double*)malloc(subtests_by_subtests_by_szDbl * 3);
	

	double *AA = (double *)malloc(items_by_subtests_by_szDbl);
	double *AAT = (double *)malloc(items_by_subtests_by_szDbl);
	double *AATS = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *RSIG = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *PVAR = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *PVARI = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *BA = (double*)malloc(item_count * sizeof(double));
	double *RTH = (double*)malloc(subtests_by_szDbl);
	double *PMEAN = (double*)malloc(subtests_by_szDbl);

	double *PMEAN1 = (double*)malloc(subtests_by_szDbl);

	double *C = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *D = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *VAR = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *CTH = (double *)malloc(persons_by_subtests_by_szDbl);
	double *CTHT = (double *)malloc(persons_by_subtests_by_szDbl);

	double *DI = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *WZ = (double *)malloc(persons_by_subtests_by_szDbl);
	double *ZVAR = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *ZVART = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *RSIGT = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *AZ = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *CHF = (double*)malloc(subtests_by_subtests_by_szDbl);

	int *ord = (int *)malloc(items_by_szInt);
	int *places = (int *)malloc(subtests_by_szInt);


	omp_set_num_threads(total_subtest_count);


	// INITIALIZE
	for (int item = 0; item < item_count; item++) {
		int csum = 0;
		for (int person = 0; person < person_count; person++) {
			csum += Y[person*item_count + item];
		}
		A[item] = ALPHA_I;
		G[item] = (double)(stdnormal_inv((double)csum / (double)person_count) * -sqrt(5.0));
	}
	for (int person = 0; person < person_count; person++) {
		for (int subtest = 0; subtest < total_subtest_count; subtest++) {
			TH[person*total_subtest_count + subtest] = TH_I;
		}
	}
	for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
		for (int subtest1 = subtest0 + 1; subtest1 < total_subtest_count; subtest1++) {
			SIGMA[subtest0*total_subtest_count + subtest1] = 0.0;
			SIGMA[subtest1*total_subtest_count + subtest0] = 0.0;
		}
		SIGMA[subtest0*total_subtest_count + subtest0] = 1.0;
	}
	for (int subtest0 = total_subtest_count - 1; subtest0 >= 0; subtest0--) {
		places[subtest0] = subtest_counts[subtest0];
		for (int subtest1 = 0; subtest1 < subtest0; subtest1++) {
			places[subtest0] += subtest_counts[subtest1];
		}
	}
	for (int item = 0, subtest = 0; item < item_count; item++) {
		if (item >= places[subtest]) {
			subtest++;
		}
		ord[item] = subtest;
	}


	// DECOMPOSE
	int* persons = (int*)malloc(sizeof(int) * size);
	int* persons_x_items = (int*)malloc(sizeof(int) * size);
	int* persons_x_subtests = (int*)malloc(sizeof(int) * size);
	int* persons_x_items_displacement = (int*)malloc(sizeof(int) * size);
	int* persons_x_subtests_displacement = (int*)malloc(sizeof(int) * size);

	int portion = person_count / size;
	int remainder = person_count % size;

	int count = 0;


	for (int r = 0; r < size; r++) {
		persons[r] = portion + ((remainder > 0) ? 1 : 0);

		if (remainder > 0) {
			remainder--;
		}

		persons_x_items[r] = persons[r] * item_count;
		persons_x_subtests[r] = persons[r] * total_subtest_count;

		persons_x_items_displacement[r] = count * item_count;
		persons_x_subtests_displacement[r] = count * total_subtest_count;

		count += persons[r];
	}


	// SETUP GPU	
	unsigned int THREAD_X = 8,
				 THREAD_Y = 8,
				 THREAD_Z = 1;

	if (item_count % 10 == 0)     THREAD_X = 10;
	else if (item_count % 8 == 0) THREAD_X = 8;
	else if (item_count % 5 == 0) THREAD_X = 5;
	else if (item_count % 4 == 0) THREAD_X = 4;
	else if (item_count % 2 == 0) THREAD_X = 2;
	else					      THREAD_X = 1;

	if (persons[ROOT] % 10 == 0)     THREAD_Y = 10;
	else if (persons[ROOT] % 8 == 0) THREAD_Y = 8;
	else if (persons[ROOT] % 5 == 0) THREAD_Y = 5;
	else if (persons[ROOT] % 4 == 0) THREAD_Y = 4;
	else if (persons[ROOT] % 2 == 0) THREAD_Y = 2;
	else						     THREAD_Y = 1;

	unsigned int GRID_X = ((item_count + THREAD_X - 1) / THREAD_X),
				 GRID_Y = ((persons[ROOT] + THREAD_Y - 1) / THREAD_Y),
				 GRID_Z = ((total_subtest_count + THREAD_Z - 1) / THREAD_Z);


	cudaSetDevice(0);

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


	cudaMalloc((void **)&dev_Z, persons_x_items[ROOT] * sizeof(double));
	cudaMalloc((void **)&dev_U, persons_x_items[ROOT] * sizeof(double));
	cudaMalloc((void **)&dev_A, items_by_szDbl);
	cudaMalloc((void **)&dev_G, items_by_szDbl);
	cudaMalloc((void **)&dev_TH, persons_x_subtests[ROOT] * sizeof(double));

	cudaMalloc((void **)&dev_Y, persons_by_items*sizeof(int));
	cudaMalloc((void **)&dev_ORD, items_by_szInt);


	cudaMemcpy(dev_Y, Y + persons_x_items_displacement[ROOT], persons_x_items[ROOT] * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_ORD, ord, item_count * sizeof(unsigned int), cudaMemcpyHostToDevice);




#ifdef CUDA_TH
	double *dev_WZ,
		   *dev_RSIG,
		   *dev_PVAR,
		   *dev_AAT;

	cudaMalloc((void **)&dev_WZ, persons_x_subtests[ROOT] * sizeof(double));
	cudaMalloc((void **)&dev_RSIG, subtests_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_PVAR, subtests_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_AAT, items_by_subtests_by_szDbl);
#endif	



	for (int r = 1; r < size; r++) {
		MPI_Send(&persons[r], 1, MPI_INT, r, TAG, MPI_COMM_WORLD);
	}


	MPI_Bcast(ord, item_count, MPI_INT, ROOT, MPI_COMM_WORLD);

	MPI_Scatterv(Y, persons_x_items, persons_x_items_displacement, MPI_INT, MPI_IN_PLACE, person_count * item_count, MPI_INT, ROOT, MPI_COMM_WORLD);


	MPI_Status statusA, statusG;
	MPI_Request requestA, requestG;

	MPI_Ibcast(A, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestA);
	MPI_Ibcast(G, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestG);


	MPI_Scatterv(TH, persons_x_subtests, persons_x_subtests_displacement, MPI_DOUBLE, MPI_IN_PLACE, persons_by_subtests, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);


	MPI_Wait(&requestA, &statusA);
	MPI_Wait(&requestG, &statusG);

	// BEGIN ITERATE
	for (int iteration = 0; iteration < iteration_count; iteration++) {


		// UPDATE Z

		cudaMemcpyAsync(dev_A, A, items_by_szDbl, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_G, G, items_by_szDbl, cudaMemcpyHostToDevice, stream);

		cudaMemcpyAsync(dev_TH, TH + persons_x_subtests_displacement[ROOT], persons_x_subtests[ROOT] * sizeof(double), cudaMemcpyHostToDevice, stream);

		curandGenerateUniformDouble(gen, dev_U, persons_x_items[ROOT]);


		calcZ(stream, dev_U, GRID_X, GRID_Y, THREAD_X, THREAD_Y, dev_Y, dev_A, dev_G, dev_TH, dev_Z, dev_ORD, item_count, persons[ROOT], total_subtest_count);


		cudaMemcpy(Z + persons_x_items_displacement[ROOT], dev_Z, persons_x_items[ROOT] * sizeof(double), cudaMemcpyDeviceToHost);



		// GATHER Z *needs to be on gpus
		MPI_Gatherv(MPI_IN_PLACE, persons_x_items[ROOT], MPI_DOUBLE, Z, persons_x_items, persons_x_items_displacement, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);







		for (int item = 0; item < item_count; item++) {
			for (int subtest = 0; subtest < total_subtest_count; subtest++) {
				AAT[subtest*item_count + item] = AA[item*total_subtest_count + subtest] = 0.0;
			}
		}

		for (int subtest = 0, subitem = 0; subtest < total_subtest_count; subtest++) {
			for (int subtest_item = 0; subtest_item < subtest_counts[subtest]; subtest_item++, subitem++) {
				AAT[subtest*item_count + subitem] = AA[subitem*total_subtest_count + subtest] = A[subitem];
			}
		}

		for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
			for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
				SUM = 0.0;
				for (int item = 0; item < item_count; item++) {
					SUM += AAT[subtest0*item_count + item] * AA[item*total_subtest_count + subtest1];
				}
				AATS[subtest0*total_subtest_count + subtest1] = SUM;
			}
		}





		MATINV(SIGMA, RSIG, subtests_by_subtests_by_szDbl, total_subtest_count);

		for (int i = 0; i < total_subtest_count; i++) {
			for (int j = 0; j < total_subtest_count; j++) {
				PVARI[i*total_subtest_count + j] = RSIG[i*total_subtest_count + j] + AATS[i*total_subtest_count + j];
			}
		}

		MATINV(PVARI, PVAR, subtests_by_subtests_by_szDbl, total_subtest_count);

		CHFAC(total_subtest_count, PVAR, RSIG, CHF);






		MPI_Bcast(AAT, item_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(PVAR, total_subtest_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(RSIG, total_subtest_count*total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(PMEAN, total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
		MPI_Bcast(RTH, total_subtest_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);



		// UPDATE TH

#ifdef CUDA_TH
		curandGenerateNormalDouble(gen, dev_WZ, persons_x_subtests[ROOT], 0.0, 1.0);

		cudaMemcpyAsync(dev_RSIG, RSIG, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_PVAR, PVAR, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, stream);

		cudaMemcpyAsync(dev_AAT, AAT, items_by_subtests_by_szDbl, cudaMemcpyHostToDevice, stream);


		calcTH(stream, dev_WZ, dev_AAT, dev_RSIG, dev_PVAR, dev_Z, dev_G, dev_TH, GRID_Y, THREAD_Y, item_count, persons[ROOT], total_subtest_count);

		cudaMemcpy(TH + persons_x_subtests_displacement[ROOT], dev_TH, persons_x_subtests[ROOT] * sizeof(double), cudaMemcpyDeviceToHost);
#else		
		for (int person = 0; person < persons[ROOT]; person++) {

			for (int subtest = 0; subtest < total_subtest_count; subtest++) {
				SUM = 0.0;
				for (int item = 0; item < item_count; item++) {
					SUM += AAT[subtest*item_count + item] * (Z[person*item_count + item] + G[item]);
				}
				PMEAN1[subtest] = SUM;
			}

			for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
				SUM = 0.0;
				for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					SUM += PVAR[subtest0*total_subtest_count + subtest1] * PMEAN1[subtest1];
				}
				PMEAN[subtest0] = SUM;

				SUM = 0.0;
				for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					double temp;
					temp = random_normal();
					SUM += temp * RSIG[subtest1*total_subtest_count + subtest0];
				}

				RTH[subtest0] = SUM;
			}


			for (int subtest = 0; subtest < total_subtest_count; subtest++) {
				TH[person*total_subtest_count + subtest] = RTH[subtest] + PMEAN[subtest];
			}
		}
#endif






		// GATHER TH *need to be on gpus using device pointers
		MPI_Gatherv(MPI_IN_PLACE, persons_x_subtests[ROOT], MPI_DOUBLE, TH, persons_x_subtests, persons_x_subtests_displacement, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);




		// UPDATE AG
		#pragma omp parallel for num_threads(total_subtest_count)
		for (int subtest = 0; subtest < total_subtest_count; subtest++) {
			Subtest(person_count, item_count, total_subtest_count, uniform, subtest, subtest_counts, Z, TH, C, A, G);
		}





		MPI_Ibcast(A, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestA);
		MPI_Ibcast(G, item_count, MPI_DOUBLE, ROOT, MPI_COMM_WORLD, &requestG);






		for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
			for (int person = 0; person < person_count; person++) {
				SUM = 0.0;
				for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					SUM += C[subtest0*total_subtest_count + subtest1] * TH[person*total_subtest_count + subtest1];
				}
				CTHT[person*total_subtest_count + subtest0] = CTH[subtest0*person_count + person] = SUM;
			}
		}


		for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
			for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
				SUM = 0.0;
				for (int person = 0; person < person_count; person++) {
					SUM += CTH[subtest0*person_count + person] * CTHT[person*total_subtest_count + subtest1];
				}
				D[subtest0*total_subtest_count + subtest1] = SUM;
			}
		}




		//INVERSEWISHART

		MATINV(D, DI, subtests_by_subtests_by_szDbl, total_subtest_count);

		CHFAC(total_subtest_count, DI, RSIG, CHF);

		normal_random_array(total_subtest_count, person_count, WZ);




		for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
			for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
				SUM = 0.0;
				for (int person = 0; person < person_count; person++) {
					SUM += WZ[person*total_subtest_count + subtest0] * WZ[person*total_subtest_count + subtest1];
				}

				ZVAR[subtest0*total_subtest_count + subtest1] = SUM;
			}
		}

		for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
			for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
				SUM = 0.0;
				for (int n = 0; n < total_subtest_count; n++) {
					SUM += RSIG[n*total_subtest_count + subtest0] * ZVAR[n*total_subtest_count + subtest1];
				}
				AZ[subtest0*total_subtest_count + subtest1] = SUM;
			}
		}

		for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
			for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
				SUM = 0.0;
				for (int n = 0; n < total_subtest_count; n++) {
					SUM += AZ[subtest0*total_subtest_count + n] * RSIG[n*total_subtest_count + subtest1];
				}
				RSIGT[subtest0*total_subtest_count + subtest1] = SUM;
			}
		}

		MATINV(RSIGT, VAR, subtests_by_subtests_by_szDbl, total_subtest_count);



		for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
			for (int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
				SIGMA[subtest0*total_subtest_count + subtest1] = (double)(VAR[subtest0*total_subtest_count + subtest1] * sqrt(1.0 / (VAR[subtest0*total_subtest_count + subtest0] * VAR[subtest1*total_subtest_count + subtest1])));
			}
		}


		// TRACK STATISTICS
		if (iteration >= burnin) {
			if (((iteration - burnin) % bsize) == 0) {
				if (iteration == burnin) {
					for (int item = 0; item < item_count; item++) {
						AV[item_count + item] = 0.0;
						GV[item_count + item] = 0.0;
						AV[2 * item_count + item] = 0.0;
						GV[2 * item_count + item] = 0.0;
					}
					for (int subtest = 0; subtest < total_subtest_count; subtest++) {
						for (int person = 0; person < person_count; person++) {
							THV[persons_by_subtests + person*total_subtest_count + subtest] = 0.0;
							THV[2 * persons_by_subtests + person*total_subtest_count + subtest] = 0.0;
						}
						for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
							RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] = 0.0;
							RHO[2 * subtests_by_subtests + subtest*total_subtest_count + subtest0] = 0.0;
						}
					}
				}
				else {
					for (int item = 0; item < item_count; item++) {
						double M1 = AV[item] / bsize;
						double M2 = GV[item] / bsize;
						AV[item_count + item] += M1;
						GV[item_count + item] += M2;
						AV[2 * item_count + item] += M1 * M1;
						GV[2 * item_count + item] += M2 * M2;
					}
					for (int subtest = 0; subtest < total_subtest_count; subtest++) {
						for (int person = 0; person < person_count; person++) {
							double M1 = THV[person*total_subtest_count + subtest] / bsize;
							THV[persons_by_subtests + person*total_subtest_count + subtest] += M1;
							THV[2 * persons_by_subtests + person*total_subtest_count + subtest] += M1 * M1;
						}
						for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
							double M1 = RHO[subtest*total_subtest_count + subtest0] / bsize;
							RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1;
							RHO[2 * subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1 * M1;
						}
					}
				}
				for (int item = 0; item < item_count; item++) {
					AV[item] = A[item];
					GV[item] = G[item];
				}
				for (int subtest = 0; subtest < total_subtest_count; subtest++) {
					for (int person = 0; person < person_count; person++) {
						THV[person*total_subtest_count + subtest] = TH[person*total_subtest_count + subtest];
					}
					for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
						RHO[subtest*total_subtest_count + subtest0] = SIGMA[subtest*total_subtest_count + subtest0];
					}
				}
			}
			else
			{
				for (int j = 0; j < item_count; j++) {
					AV[j] += A[j];
					GV[j] += G[j];
				}
				for (int i = 0; i < person_count; i++) {
					for (int j = 0; j < total_subtest_count; j++) {
						THV[i*total_subtest_count + j] += TH[i*total_subtest_count + j];
					}
				}
				for (int i = 0; i < total_subtest_count; i++) {
					for (int j = 0; j < total_subtest_count; j++) {
						RHO[i*total_subtest_count + j] += SIGMA[i*total_subtest_count + j];
					}
				}
			}
		}


		MPI_Wait(&requestA, &statusA);
		MPI_Wait(&requestG, &statusG);


		fprintf(stderr, "REGUL %12d\r", iteration);

	} // END OF ITERATIONS

	free(Z);
	free(A);
	free(G);
	free(TH);
	free(SIGMA);

	free(RSIG);
	free(AAT);
	free(PVAR);
	free(RTH);
	free(PMEAN);

	free(AA);
	free(AATS);
	free(BA);

#ifndef CUDA_TH
	free(PMEAN1);
#endif

	free(CTH);
	free(CTHT);
	free(DI);
	free(WZ);
	free(AZ);
	free(ZVAR);
	free(D);
	free(VAR);
	free(C);

	free(CHF);

	free(persons);
	free(persons_x_items);
	free(persons_x_subtests);
	free(persons_x_items_displacement);
	free(persons_x_subtests_displacement);


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


	for (int item = 0; item < item_count; item++) {
		double M1 = AV[item] / bsize;
		double M2 = GV[item] / bsize;
		AV[item_count + item] += M1;
		GV[item_count + item] += M2;
		AV[2 * item_count + item] += M1 * M1;
		GV[2 * item_count + item] += M2 * M2;
	}

	for (int subtest = 0; subtest < total_subtest_count; subtest++) {
		for (int person = 0; person < person_count; person++) {
			double M1 = THV[person*total_subtest_count + subtest] / bsize;
			THV[persons_by_subtests + person*total_subtest_count + subtest] += M1;
			THV[2 * persons_by_subtests + person*total_subtest_count + subtest] += M1 * M1;
		}
		for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			double M1 = RHO[subtest*total_subtest_count + subtest0] / bsize;
			RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1;
			RHO[2 * subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1 * M1;
		}
	}

	for (int item = 0; item < item_count; item++) {
		ITEM[item * 4] = AV[item_count + item] / (double)batches;
		ITEM[item * 4 + 1] = (double)(sqrt((AV[2 * item_count + item] - (AV[item_count + item] * AV[item_count + item] / batches)) / (batches - 1)) / sqrt((double)(batches)));
		ITEM[item * 4 + 2] = GV[item_count + item] / (double)batches;
		ITEM[item * 4 + 3] = (double)(sqrt((GV[2 * item_count + item] - (GV[item_count + item] * GV[item_count + item] / batches)) / (batches - 1)) / sqrt((double)(batches)));
	}

	for (int subtest = 0, JK = 0; subtest < total_subtest_count; subtest++) {
		for (int person = 0; person < person_count; person++) {
			PERSON[person*total_subtest_count * 2 + 2 * subtest] = THV[persons_by_subtests + person*total_subtest_count + subtest] / (double)(batches);
			PERSON[person*total_subtest_count * 2 + 2 * subtest + 1] = (double)(sqrt((THV[2 * persons_by_subtests + person*total_subtest_count + subtest] - (THV[persons_by_subtests + person*total_subtest_count + subtest] * THV[persons_by_subtests + person*total_subtest_count + subtest] / batches)) / (batches - 1)) / sqrt((double)(batches)));
		}

		for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			CORR[JK * 2] = RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] / batches;
			CORR[JK * 2 + 1] = (double)(sqrt((RHO[2 * subtests_by_subtests + subtest*total_subtest_count + subtest0] - (RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] * RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] / batches)) / (batches - 1)) / sqrt((double)(batches)));
			JK = JK + 1;
		}
	}

	free(AV);
	free(GV);
	free(THV);
	free(RHO);
}