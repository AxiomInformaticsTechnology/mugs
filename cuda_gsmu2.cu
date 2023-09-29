#include "gsmu2.cuh"
#include "cuda_gsmu2.cuh"


// TODO: make struct out of inputs
void cuda_gsmu2(int *Y, 
				int *subtest_counts,
				int person_count,
				int item_count,
				int total_subtest_count,
				int iteration_count,
				int BURNIN,
				int BATCHES,
				int uniform,
				double *ITEM,
				double *PERSON,
				double *CORR)
{
	int persons_by_items = person_count * item_count,
		persons_by_subtests = person_count * total_subtest_count,
		items_by_subtests = item_count * total_subtest_count,
		subtests_by_subtests = total_subtest_count * total_subtest_count;

	unsigned int subtests_by_szDbl = total_subtest_count * sizeof(double),
				 items_by_szDbl = item_count * sizeof(double),
				 items_by_subtests_by_szDbl = items_by_subtests * sizeof(double),
				 persons_by_items_by_szDbl = persons_by_items * sizeof(double),
				 persons_by_subtests_by_szDbl = persons_by_subtests * sizeof(double),
				 subtests_by_subtests_by_szDbl = subtests_by_subtests * sizeof(double);

	double *A = (double*)malloc(items_by_szDbl);
	double *G = (double*)malloc(items_by_szDbl);
	double *TH = (double*)malloc(persons_by_subtests_by_szDbl);
	double *SIGMA = (double*)malloc(subtests_by_subtests_by_szDbl);

	double *AV = (double*)malloc(items_by_szDbl * 3);
	double *GV = (double*)malloc(items_by_szDbl * 3);
	double *THV = (double*)malloc(persons_by_subtests_by_szDbl * 3);
	double *RHO = (double*)malloc(subtests_by_subtests_by_szDbl * 3);
	
	double *Z = (double *)malloc(persons_by_items_by_szDbl);

	int *ord = (int *)malloc(sizeof(int)*item_count);
	int *places = (int *)malloc(sizeof(int)*total_subtest_count);

	int BSIZE = (iteration_count - BURNIN) / BATCHES;

	double mSR5 = -sqrt(5.0);

	srand(RANDOM_SEED);

	// initialize
	for (int item = 0; item < item_count; item++) {
		int csum = 0;
		for (int person = 0; person < person_count; person++) {
			csum += Y[person*item_count + item];
		}
		A[item] = ALPHA_I;
		G[item] = (double)(stdnormal_inv((double)csum / (double)person_count) * mSR5);
	}
	for (int person = 0; person < person_count; person++) {
		for (int subtest = 0; subtest < total_subtest_count; subtest++) {
			TH[person*total_subtest_count + subtest] = TH_I;
		}
	}
	for (int subtest = 0; subtest < total_subtest_count; subtest++) {
		for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			SIGMA[subtest*total_subtest_count + subtest0] = 0.0;
			SIGMA[subtest0*total_subtest_count + subtest] = 0.0;
		}
		SIGMA[subtest*total_subtest_count + subtest] = 1.0;
	}
	for (int subtest = total_subtest_count - 1; subtest >= 0; subtest--) {
		places[subtest] = subtest_counts[subtest];
		for (int subtest0 = 0; subtest0 < subtest; subtest0++) {
			places[subtest] += subtest_counts[subtest0];
		}
	}
	for (int item = 0, subtest = 0; item < item_count; item++) {
		if (item >= places[subtest]) {
			subtest++;
		}
		ord[item] = subtest;
	}


	unsigned int THREAD_X = 8,
				 THREAD_Y = 8,
				 THREAD_Z = 1;

	if (item_count % 10 == 0)     THREAD_X = 10;
	else if (item_count % 8 == 0) THREAD_X = 8;
	else if (item_count % 5 == 0) THREAD_X = 5;
	else if (item_count % 4 == 0) THREAD_X = 4;
	else if (item_count % 2 == 0) THREAD_X = 2;
	else					      THREAD_X = 1;

	if (person_count % 10 == 0)     THREAD_Y = 10;
	else if (person_count % 8 == 0) THREAD_Y = 8;
	else if (person_count % 5 == 0) THREAD_Y = 5;
	else if (person_count % 4 == 0) THREAD_Y = 4;
	else if (person_count % 2 == 0) THREAD_Y = 2;
	else						    THREAD_Y = 1;

	unsigned int GRID_X = ((item_count + THREAD_X - 1) / THREAD_X),
				 GRID_Y = ((person_count + THREAD_Y - 1) / THREAD_Y),
				 GRID_Z = ((total_subtest_count + THREAD_Z - 1) / THREAD_Z);

	unsigned int THREADS = 512,
				 BLOCKS = (((person_count + THREADS - 1) / THREADS) < 1024) ? 
						  ((person_count + THREADS - 1) / THREADS) : 1024;

	curandState_t *rngStates;

	double *dev_WZ,
		   *dev_U,
		   *dev_Z,
		   *dev_A,
		   *dev_G,
		   *dev_TH,
		   *dev_AA,
		   *dev_AAT,
		   *dev_D_AATS,
		   *dev_RSIG,
		   *dev_RSIG_C,
		   *dev_PVAR,
		   *dev_PVAR_ZVAR,
		   *dev_ZTW,
		   *dev_SIGMA,
		   *dev_AV,
		   *dev_GV,
		   *dev_THV,
		   *dev_RHO;

	int *dev_Y,
		*dev_SUBTEST_COUNTS,
		*dev_ORD;

	double2 *dev_CTH_WZ;

	curandGenerator_t gen;

	cudaStream_t *streams = (cudaStream_t*)malloc(2 * sizeof(cudaStream_t));


	cudaSetDevice(0);

#ifdef CUDA_STATS	
	cudaMalloc((void **)&dev_AV, 3 * items_by_szDbl);
	cudaMalloc((void **)&dev_GV, 3 * items_by_szDbl);
	cudaMalloc((void **)&dev_THV, 3 * persons_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_RHO, 3 * subtests_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_SIGMA, subtests_by_subtests_by_szDbl);
#endif

#ifdef CUDA_SUBTEST
	cudaMalloc((void **)&dev_ZTW, (persons_by_subtests_by_szDbl));
#endif

	{

		for (int i = 0; i < 2; i++) {
			cudaStreamCreate(&(streams[i]));
		}

		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

		cudaMalloc((void **)&rngStates, GRID_X * THREAD_X * sizeof(curandState_t));

		setupRng(rngStates, GRID_X, 1, THREAD_X, 1, 12357, 123571);

		cudaMalloc((void **)&dev_U, persons_by_items_by_szDbl);
		cudaMalloc((void **)&dev_Z, persons_by_items_by_szDbl);
		cudaMalloc((void **)&dev_Y, persons_by_items * sizeof(int));

		cudaMalloc((void **)&dev_A, items_by_szDbl);
		cudaMalloc((void **)&dev_G, items_by_szDbl);

		cudaMalloc((void **)&dev_TH, persons_by_subtests_by_szDbl);

		cudaMalloc((void **)&dev_SUBTEST_COUNTS, total_subtest_count * sizeof(int));

		cudaMalloc((void **)&dev_ORD, item_count * sizeof(int));

		cudaMemcpy(dev_SUBTEST_COUNTS, subtest_counts, total_subtest_count * sizeof(unsigned int), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_Y, Y, persons_by_items * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_ORD, ord, item_count * sizeof(unsigned int), cudaMemcpyHostToDevice);

#ifdef CUDA_TH		
		cudaMalloc((void **)&dev_WZ, persons_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_RSIG, subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_PVAR, subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_AAT, items_by_subtests_by_szDbl);

		cudaMemcpy(dev_TH, TH, persons_by_subtests_by_szDbl, cudaMemcpyHostToDevice);
#endif

#ifdef CUDA_AATS
		cudaMalloc((void **)&dev_AA, items_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_D_AATS, subtests_by_subtests_by_szDbl);
#ifndef CUDA_TH
		cudaMalloc((void **)&dev_AAT, items_by_subtests_by_szDbl);
#endif
#endif

#ifdef CUDA_SUBTEST
		cudaMemcpy(dev_A, A, items_by_szDbl, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_G, G, items_by_szDbl, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&dev_RSIG_C, subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_PVAR_ZVAR, subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_CTH_WZ, persons_by_subtests * sizeof(double2));
#endif
	}


	double *AATS = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *AAT = (double*)malloc(items_by_subtests_by_szDbl);
	double *AA = (double*)malloc(items_by_subtests_by_szDbl);
	double *PVARI = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *RSIG = (double*)malloc(subtests_by_subtests_by_szDbl);

	double *PVAR = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *WZ = (double*)malloc(persons_by_subtests_by_szDbl);
	double *RTH = (double*)malloc(subtests_by_szDbl);
	double *BA = (double*)malloc(items_by_szDbl);
	double *PMEAN = (double*)malloc(subtests_by_szDbl);
	double *PMEAN1 = (double*)malloc(subtests_by_szDbl);
	double *C = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *VAR = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *D = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *CTH = (double*)malloc(persons_by_subtests_by_szDbl);
	double *CTHT = (double*)malloc(persons_by_subtests_by_szDbl);
	double *RSIG2 = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *DI = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *ZVAR = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *RSIGT = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *AZ = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *CHF = (double*)malloc(subtests_by_subtests_by_szDbl);

	// begin iterate
	for (int iteration = 0; iteration < iteration_count; iteration++) {

//	int reiterate = 0;
//restart_iteration:
//	reiterate++;
//	if (reiterate > 10)
//	{
//		fprintf(stderr, "CHF failure during iteration %12d\r", iteration);
//		exit(-5);
//	}

		// Update Z
		{
			curandGenerateUniformDouble(gen, dev_U, persons_by_items);
#ifndef CUDA_SUBTEST
			cudaMemcpyAsync(dev_A, A, items_by_szDbl, cudaMemcpyHostToDevice, streams[0]);
			cudaMemcpyAsync(dev_G, G, items_by_szDbl, cudaMemcpyHostToDevice, streams[0]);
#endif
#ifndef CUDA_TH			
			cudaMemcpyAsync(dev_TH, TH, persons_by_subtests_by_szDbl, cudaMemcpyHostToDevice, streams[0]);
#endif
			calcZ(streams[0], dev_U, GRID_X, GRID_Y, THREAD_X, THREAD_Y, dev_Y, dev_A, dev_G, dev_TH, dev_Z, dev_ORD, item_count, person_count, total_subtest_count);
#ifndef CUDA_SUBTEST
			cudaMemcpyAsync(Z, dev_Z, persons_by_items_by_szDbl, cudaMemcpyDeviceToHost, streams[0]);
#endif
		}

		{
		
#ifdef CUDA_AATS
			{
				initAA(dev_AAT, dev_AA, item_count, total_subtest_count, GRID_Z, GRID_X, THREAD_Z, THREAD_X);
				stripeAA(dev_AAT, dev_AA, dev_A, dev_SUBTEST_COUNTS, item_count, total_subtest_count, GRID_Z, THREAD_Z);
				calcAATS(dev_AAT, dev_AA, dev_D_AATS, item_count, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z);
			}
			cudaMemcpy(AATS, dev_D_AATS, subtests_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
#ifndef CUDA_TH
			cudaMemcpy(AAT, dev_AAT, items_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
#endif			
#else
			
			{
				

				for (int item = 0, itsc = 0; item < item_count; item++, itsc += total_subtest_count) {
					for (int subtest = 0, sic = 0; subtest < total_subtest_count; subtest++, sic += item_count) {
						AAT[sic + item] = AA[itsc + subtest] = 0.0;
					}
				}
				for (int subtest = 0, st = 0, subitem = 0; subtest < total_subtest_count; subtest++, st += item_count) {
					for (int subtestitem = 0, stsc = subitem*total_subtest_count; subtestitem < subtest_counts[subtest]; subtestitem++, subitem++, stsc += total_subtest_count) {
						AAT[st + subitem] = AA[stsc + subtest] = A[subitem];
					}
				}
				for (int r = 0, ric = 0, rtsc = 0; r < total_subtest_count; r++, ric += item_count, rtsc += total_subtest_count) {
					for (int c = 0; c < total_subtest_count; c++) {
						double val = 0.0;
						for (int n = 0, ntsc = 0; n < item_count; n++, ntsc += total_subtest_count) {
							val += AAT[ric + n] * AA[ntsc + c];
						}
						AATS[rtsc + c] = (double)val;
					}
				}
			}
#endif

			{

				MATINV(SIGMA, RSIG, subtests_by_subtests_by_szDbl, total_subtest_count);

				for (int i = 0, itsc = 0; i < total_subtest_count; i++, itsc += total_subtest_count) {
					for (int j = 0; j < total_subtest_count; j++) {
						PVARI[itsc + j] = RSIG[itsc + j] + AATS[itsc + j];
					}
				}

				MATINV(PVARI, PVAR, subtests_by_subtests_by_szDbl, total_subtest_count);

				CHFAC(total_subtest_count, PVAR, RSIG, CHF);

//				int res = CHFAC(total_subtest_count, PVAR, RSIG, CHF);
//				if (res != 0)
//				{
//					fprintf(stdout, "CHFAC not positive definite  %s %d %d\n", (res == 2) ? "negative on diagonal" : "zero on diagonal", iteration, res);
//					showmatr((double*)PVAR, total_subtest_count, total_subtest_count, "PVAR");
//					showmatr((double*)RSIG, total_subtest_count, total_subtest_count, "RSIG");
////goto restart_iteration;
//				}

			}

#ifdef CUDA_TH			
			// Update Theta

//	int rethrow = 0;
//rethrow_dice:
//	rethrow++;
//	if (rethrow > 5 && reiterate < 5) goto restart_iteration;
//	if (rethrow > 5 && reiterate > 5)
//	{
//		fprintf(stderr, "CHFAC failure in iteration %12d\r", iteration);
//
//		exit(-6);
//	}

			{



				curandGenerateNormalDouble(gen, dev_WZ, persons_by_subtests, 0.0, 1.0);

				cudaMemcpyAsync(dev_RSIG, RSIG, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, streams[0]);
				cudaMemcpyAsync(dev_PVAR, PVAR, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, streams[0]);
#ifndef CUDA_AATS
				cudaMemcpyAsync(dev_AAT, AAT, items_by_subtests_by_szDbl, cudaMemcpyHostToDevice, streams[1]);
#endif
				calcTH(streams[0], dev_WZ, dev_AAT, dev_RSIG, dev_PVAR, dev_Z, dev_G, dev_TH, GRID_Y, THREAD_Y, item_count, person_count, total_subtest_count);

#ifndef CUDA_STATS
				cudaMemcpyAsync(TH, dev_TH, persons_by_subtests_by_szDbl, cudaMemcpyDeviceToHost, streams[0]);
#endif
			}
#else
			

//			int rethrow = 0;
//		rethrow_dice:
//			rethrow++;
//
//			if (rethrow > 5 && reiterate < 5) goto restart_iteration;
//			if (rethrow > 5 && reiterate > 5)
//			{
//				fprintf(stderr, "CHFAC failure in iteration %12d\r", iteration);
//
//				exit(-6);
//			}

			normal_random_array(total_subtest_count, person_count, WZ);
			{
				
				for (int person = 0, ptsc = 0, pic = 0; person<person_count; person++, ptsc += total_subtest_count, pic += item_count) {
					for (int r = 0, ric = 0; r < total_subtest_count; r++, ric += item_count) {
						double val = 0.0;
						for (int n = 0; n < item_count; n++) {
							val += AAT[ric + n] * (Z[pic + n] + G[n]);
						}
						PMEAN1[r] = (double)val;
					}
					for (int r = 0, rtsc = 0; r < total_subtest_count; r++, rtsc += total_subtest_count) {
						double val1 = 0.0;
						for (int n = 0; n < total_subtest_count; n++) {
							val1 += PVAR[rtsc + n] * PMEAN1[n];
						}
						PMEAN[r] = (double)val1;

						double val2 = 0.0;
						for (int n = 0, ntsc = 0; n < total_subtest_count; n++, ntsc += total_subtest_count) {
							val2 += WZ[ptsc + n] * RSIG[ntsc + r];
						}
						RTH[r] = (double)val2;
					}
					for (int subtest = 0; subtest < total_subtest_count; subtest++) {
						TH[ptsc + subtest] = RTH[subtest] + PMEAN[subtest];
					}
				}

			}

#endif
		}


#ifdef CUDA_SUBTEST
		
		cudaSubtest(rngStates, dev_A, dev_G, dev_Z, dev_TH, person_count, item_count, total_subtest_count, uniform, dev_SUBTEST_COUNTS, GRID_Z, THREAD_Z, BLOCKS, THREADS);
		
		curandGenerateNormalDouble(gen, dev_ZTW, persons_by_subtests, 0.0, 1.0);

		copyRN(dev_CTH_WZ, dev_ZTW, person_count, total_subtest_count, GRID_Z, GRID_Y, THREAD_Z, THREAD_Y);

		calcC_ZVAR(dev_CTH_WZ, dev_PVAR_ZVAR, dev_SUBTEST_COUNTS, dev_RSIG_C, dev_A, person_count, total_subtest_count, GRID_Z, THREAD_Z, BLOCKS, THREADS);

		calcCTH(dev_RSIG_C, dev_TH, dev_CTH_WZ, person_count, total_subtest_count, GRID_Z, GRID_Y, THREAD_Z, THREAD_Y);

		calcD(dev_CTH_WZ, dev_D_AATS, person_count, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z, BLOCKS, THREADS);

		cudaMemcpy(ZVAR, dev_PVAR_ZVAR, subtests_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
		cudaMemcpy(D, dev_D_AATS, subtests_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);

#ifndef CUDA_STATS
		cudaMemcpy(A, dev_A, items_by_szDbl, cudaMemcpyDeviceToHost);
		cudaMemcpy(G, dev_G, items_by_szDbl, cudaMemcpyDeviceToHost);
#endif


		//INVERSEWISHART
		{
			
			//showmatr((double*)D,total_subtest_count,total_subtest_count,"D");

			MATINV(D, DI, subtests_by_subtests_by_szDbl, total_subtest_count);

			//showmatr((double*)DI,total_subtest_count,total_subtest_count,"DI");

			CHFAC(total_subtest_count, DI, RSIG, CHF);

//			int res = CHFAC(total_subtest_count, DI, RSIG, CHF);
//			if (res != 0)
//			{
//				fprintf(stdout, "CHFAC in INVERSEWISHART not positive definite  %s %d %d\n", (res == 2) ? "negative on diagonal" : "zero on diagonal", iteration, res);
//				//showmatrt(TH, total_subtest_count, person_count, "TH");
//				//showmatr((double*)ggCTH, person_count, total_subtest_count, "CTH");
//				//showmatr((double*)ggCTHT, total_subtest_count, person_count, "CTHT");
//				//showmatr((double*)ggD, total_subtest_count, total_subtest_count, "D");
//				//showmatr((double*)ggDI, total_subtest_count, total_subtest_count, "DI");
//				//showmatr((double*)ggRSIG, total_subtest_count, total_subtest_count, "RSIG");
////goto rethrow_dice;
//			}
			
			{
				
				{
					for (int i = 0; i < total_subtest_count; i++) {
						for (int j = 0; j < total_subtest_count; j++) {
							RSIGT[j*total_subtest_count + i] = RSIG[i*total_subtest_count + j];
						}
					}
				}
				{
					//MATMUL(RSIG, ZVAR, AZ)
					for (int r = 0, rtsc = 0; r < total_subtest_count; r++, rtsc += total_subtest_count) {
						for (int c = 0; c < total_subtest_count; c++) {
							double val = 0.0;
							for (int n = 0, ntsc = 0; n < total_subtest_count; n++, ntsc += total_subtest_count) {
								val += RSIGT[rtsc + n] * ZVAR[ntsc + c];
							}
							AZ[rtsc + c] = (double)val;
						}
					}
				}

				{
					//MATMUL(gAZ, gRSIG, gRSIGT)
					for (int r = 0, rtsc = 0; r < total_subtest_count; r++, rtsc += total_subtest_count) {
						for (int c = 0; c < total_subtest_count; c++) {
							double val = 0.0;
							for (int n = 0, ntsc = 0; n < total_subtest_count; n++, ntsc += total_subtest_count) {
								val += AZ[rtsc + n] * RSIG[ntsc + c];
							}
							RSIGT[rtsc + c] = (double)val;
						}
					}
				}
				
				MATINV(RSIGT, VAR, subtests_by_subtests_by_szDbl, total_subtest_count);

			}
		}

		for (int subtest0 = 0, s0tsc = 0; subtest0 < total_subtest_count; subtest0++, s0tsc += total_subtest_count) {
			for (int subtest1 = 0, s1tsc = 0; subtest1 < total_subtest_count; subtest1++, s1tsc += total_subtest_count) {
				SIGMA[s0tsc + subtest1] = (double)(VAR[s0tsc + subtest1] * sqrt(1.0 / (VAR[s0tsc + subtest0] * VAR[s1tsc + subtest1])));
			}
		}

#ifdef CUDA_STATS
		cudaMemcpy(dev_SIGMA, SIGMA, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice);
#endif
		
#else	

		#pragma omp parallel for num_threads(total_subtest_count)
		for (int subtest = 0; subtest < total_subtest_count; subtest++) {
			Subtest(person_count, item_count, total_subtest_count, uniform, subtest, subtest_counts, Z, TH, C, A, G);
		}

		{
			{

				for (int r = 0, rtsc = 0, rpc = 0; r < total_subtest_count; r++, rtsc += total_subtest_count, rpc += person_count) {
					for (int c = 0, ctsc = 0; c < person_count; c++, ctsc += total_subtest_count) {
						double val = 0.0;
						for (int n = 0; n < total_subtest_count; n++) {
							val += C[rtsc + n] * TH[ctsc + n];
						}
						CTHT[ctsc + r] = CTH[rpc + c] = (double)val;
					}
				}
				for (int r = 0, rtsc = 0, rpc = 0; r < total_subtest_count; r++, rtsc += total_subtest_count, rpc += person_count) {
					for (int c = 0; c < total_subtest_count; c++) {
						double val = 0.0;
						for (int n = 0, ntsc = 0; n < person_count; n++, ntsc += total_subtest_count) {
							val += CTH[rpc + n] * CTHT[ntsc + c];
						}
						D[rtsc + c] = (double)val;
					}
				}
			}

			//INVERSEWISHART
			{
				
				MATINV(D, DI, subtests_by_subtests_by_szDbl, total_subtest_count);

				CHFAC(total_subtest_count, DI, RSIG, CHF);

//				int res = CHFAC(total_subtest_count, DI, RSIG, CHF);
//				if (res != 0)
//				{
//					fprintf(stdout, "CHFAC in INVERSEWISHART not positive definite  %s %d %d\n", (res == 2) ? "negative on diagonal" : "zero on diagonal", iteration, res);
//					//showmatrt(TH, total_subtest_count, person_count, "TH");
//					//showmatr((double*)CTH, person_count, total_subtest_count, "CTH");
//					//showmatr((double*)CTHT, total_subtest_count, person_count, "CTHT");
//					//showmatr((double*)D, total_subtest_count, total_subtest_count, "D");
//					//showmatr((double*)DI, total_subtest_count, total_subtest_count, "DI");
//					//showmatr((double*)RSIG, total_subtest_count, total_subtest_count, "RSIG");
////goto rethrow_dice;
//				}


				normal_random_array(total_subtest_count, person_count, WZ);
				{
					for (int ar = 0, r = 0; r < total_subtest_count; r++, ar += total_subtest_count) {
						for (int c = 0; c < total_subtest_count; c++) {
							double val = 0.0;
							for (int nt = 0, n = 0; n < person_count; n++, nt += total_subtest_count) {
								val += WZ[nt + r] * WZ[nt + c];
							}
							ZVAR[ar + c] = (double)val;
						}
					}
				}

				{
					{
						//MATMUL(RSIG, ZVAR, AZ)
						for (int r = 0, rtsc = 0; r < total_subtest_count; r++, rtsc += total_subtest_count) {
							for (int c = 0; c < total_subtest_count; c++) {
								double val = 0.0;
								for (int n = 0, ntsc = 0; n < total_subtest_count; n++, ntsc += total_subtest_count) {
									val += RSIG[ntsc + r] * ZVAR[ntsc + c];
								}
								AZ[rtsc + c] = (double)val;
							}
						}
					}

					{
						//MATMUL(AZ, RSIG, RSIGT)
						for (int r = 0, rtsc = 0; r < total_subtest_count; r++, rtsc += total_subtest_count) {
							for (int c = 0; c < total_subtest_count; c++) {
								double val = 0.0;
								for (int n = 0, ntsc = 0; n < total_subtest_count; n++, ntsc += total_subtest_count) {
									val += AZ[rtsc + n] * RSIG[ntsc + c];
								}
								RSIGT[rtsc + c] = (double)val;
							}
						}
					}

					MATINV(RSIGT, VAR, subtests_by_subtests_by_szDbl, total_subtest_count);

				}
			}

			for (int subtest0 = 0, s0tsc = 0; subtest0 < total_subtest_count; subtest0++, s0tsc += total_subtest_count) {
				for (int subtest1 = 0, s1tsc = 0; subtest1 < total_subtest_count; subtest1++, s1tsc += total_subtest_count) {
					SIGMA[s0tsc + subtest1] = (double)(VAR[s0tsc + subtest1] * sqrt(1.0 / (VAR[s0tsc + subtest0] * VAR[s1tsc + subtest1])));
				}
			}
		}
#endif

		// track statistics
#ifdef CUDA_STATS
		if (iteration >= BURNIN) {
			if (((iteration - BURNIN) % BSIZE) == 0) {
				if (iteration == BURNIN) {
					for (int item = 0; item < item_count; item++) {
						AV[item_count + item] = 0.0;
						GV[item_count + item] = 0.0;
						AV[2 * item_count + item] = 0.0;
						GV[2 * item_count + item] = 0.0;
					}
					for (int subtest = 0; subtest < total_subtest_count; subtest++) {
						for (int person = 0; person < person_count; person++) {
							int i = person*total_subtest_count + subtest;
							THV[persons_by_subtests + i] = 0.0;
							THV[2 * persons_by_subtests + i] = 0.0;
						}
						for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
							int i = subtest*total_subtest_count + subtest0;
							RHO[subtests_by_subtests + i] = 0.0;
							RHO[2 * subtests_by_subtests + i] = 0.0;
						}
					}
					cudaMemcpy(dev_AV, AV, 3 * items_by_szDbl, cudaMemcpyHostToDevice);
					cudaMemcpy(dev_GV, GV, 3 * items_by_szDbl, cudaMemcpyHostToDevice);
					cudaMemcpy(dev_THV, THV, 3 * persons_by_subtests_by_szDbl, cudaMemcpyHostToDevice);
					cudaMemcpy(dev_RHO, RHO, 3 * subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice);
				}
				else {
					trackAG(dev_AV, dev_GV, item_count, GRID_X, THREAD_X, BSIZE);
					trackTH(dev_THV, total_subtest_count, persons_by_subtests, GRID_Z, GRID_Y, THREAD_Z, THREAD_Y, BSIZE);
					trackRHO(dev_RHO, total_subtest_count, subtests_by_subtests, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z, BSIZE);
				}
				copyAG(dev_AV, dev_GV, dev_A, dev_G, GRID_X, THREAD_X);
				copyTH(dev_THV, dev_TH, total_subtest_count, GRID_Z, GRID_Y, THREAD_Z, THREAD_Y);
				copySIGMA(dev_RHO, dev_SIGMA, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z);
			}
			else {
				sumAG(dev_AV, dev_GV, dev_A, dev_G, GRID_X, THREAD_X);
				sumTH(dev_THV, dev_TH, total_subtest_count, GRID_Y, GRID_Z, THREAD_Y, THREAD_Z);
				sumSIGMA(dev_RHO, dev_SIGMA, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z);
			}
		}
#else
		if (iteration >= BURNIN) {
			if (((iteration - BURNIN) % BSIZE) == 0) {
				if (iteration == BURNIN) {
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
						double M1 = AV[item] / BSIZE;
						double M2 = GV[item] / BSIZE;
						AV[item_count + item] += M1;
						GV[item_count + item] += M2;
						AV[2 * item_count + item] += M1 * M1;
						GV[2 * item_count + item] += M2 * M2;
					}
					for (int subtest = 0; subtest < total_subtest_count; subtest++) {
						for (int person = 0; person < person_count; person++) {
							double M1 = THV[person*total_subtest_count + subtest] / BSIZE;
							THV[persons_by_subtests + person*total_subtest_count + subtest] += M1;
							THV[2 * persons_by_subtests + person*total_subtest_count + subtest] += M1 * M1;
						}
						for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
							double M1 = RHO[subtest*total_subtest_count + subtest0] / BSIZE;
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
			else {
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
#endif


		fprintf(stderr, "REGUL %12d\r", iteration);

	} //end of iterations

	free(AATS);
	free(AAT);
	free(AA);
	free(PVARI);
	free(WZ);
	free(PVAR);
	free(RSIG);
	free(RSIG2);
	free(CTH);
	free(CTHT);
	free(C);
	free(D);
	free(VAR);
	free(DI);
	free(ZVAR);
	free(AZ);
	free(RSIGT);
	free(CHF);

#ifdef CUDA_STATS
	cudaMemcpy(AV, dev_AV, 3 * items_by_szDbl, cudaMemcpyDeviceToHost);
	cudaMemcpy(GV, dev_GV, 3 * items_by_szDbl, cudaMemcpyDeviceToHost);
	cudaMemcpy(THV, dev_THV, 3 * persons_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
	cudaMemcpy(RHO, dev_RHO, 3 * subtests_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);

	cudaFree(dev_AV);
	cudaFree(dev_GV);
	cudaFree(dev_THV);
	cudaFree(dev_RHO);
	cudaFree(dev_SIGMA);
#endif
#ifdef CUDA_SUBTEST
	cudaFree(dev_ZTW);
	cudaFree(dev_ZTW);
#endif

	{

		cudaFree(dev_Y);
		cudaFree(dev_U);
		cudaFree(dev_Z);
		cudaFree(dev_TH);
		cudaFree(dev_A);
		cudaFree(dev_G);
		cudaFree(dev_ORD);

#ifdef CUDA_TH		
		cudaFree(dev_WZ);
		cudaFree(dev_AAT);
		cudaFree(dev_RSIG);
		cudaFree(dev_PVAR);
#endif

#ifdef CUDA_AATS
		cudaFree(dev_AA);
		cudaFree(dev_D_AATS);
#endif
#ifdef CUDA_SUBTEST		
		cudaFree(dev_RSIG_C);
		cudaFree(dev_PVAR_ZVAR);
		cudaFree(dev_CTH_WZ);
#endif
		cudaFree(dev_SUBTEST_COUNTS);

		for (int i = 0; i < 2; i++) {
			cudaStreamDestroy(streams[i]);
		}

		curandDestroyGenerator(gen);

		cudaDeviceReset();
	}

	free(Z);
	free(A);
	free(G);
	free(TH);
	free(SIGMA);

	for (int item = 0; item < item_count; item++) {
		double M1 = AV[item] / BSIZE;
		double M2 = GV[item] / BSIZE;
		AV[item_count + item] += M1;
		GV[item_count + item] += M2;
		AV[2 * item_count + item] += M1 * M1;
		GV[2 * item_count + item] += M2 * M2;
	}

	for (int subtest = 0; subtest < total_subtest_count; subtest++) {
		for (int person = 0; person < person_count; person++) {
			double M1 = THV[person*total_subtest_count + subtest] / BSIZE;
			THV[persons_by_subtests + person*total_subtest_count + subtest] += M1;
			THV[2 * persons_by_subtests + person*total_subtest_count + subtest] += M1 * M1;
		}
		for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			double M1 = RHO[subtest*total_subtest_count + subtest0] / BSIZE;
			RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1;
			RHO[2 * subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1 * M1;
		}
	}

	for (int item = 0; item < item_count; item++) {
		ITEM[item * 4] = AV[item_count + item] / (double)BATCHES;
		ITEM[item * 4 + 1] = (double)(sqrt((AV[2 * item_count + item] - (AV[item_count + item] * AV[item_count + item] / BATCHES)) / (BATCHES - 1)) / sqrt((double)(BATCHES)));
		ITEM[item * 4 + 2] = GV[item_count + item] / (double)BATCHES;
		ITEM[item * 4 + 3] = (double)(sqrt((GV[2 * item_count + item] - (GV[item_count + item] * GV[item_count + item] / BATCHES)) / (BATCHES - 1)) / sqrt((double)(BATCHES)));
	}

	for (int subtest = 0, JK = 0; subtest < total_subtest_count; subtest++) {
		for (int person = 0; person < person_count; person++) {
			PERSON[person*total_subtest_count * 2 + 2 * subtest] = THV[persons_by_subtests + person*total_subtest_count + subtest] / (double)(BATCHES);
			PERSON[person*total_subtest_count * 2 + 2 * subtest + 1] = (double)(sqrt((THV[2 * persons_by_subtests + person*total_subtest_count + subtest] - (THV[persons_by_subtests + person*total_subtest_count + subtest] * THV[persons_by_subtests + person*total_subtest_count + subtest] / BATCHES)) / (BATCHES - 1)) / sqrt((double)(BATCHES)));
		}
		for (int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			CORR[JK * 2] = RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] / BATCHES;
			CORR[JK * 2 + 1] = (double)(sqrt((RHO[2 * subtests_by_subtests + subtest*total_subtest_count + subtest0] - (RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] * RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] / BATCHES)) / (BATCHES - 1)) / sqrt((double)(BATCHES)));
			JK = JK + 1;
		}
	}

	free(AV);
	free(GV);
	free(THV);
	free(RHO);

	return;
}