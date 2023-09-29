#include "gsmu2.cuh"
#include "cuda_gsmu2.cuh"


#define MAX_NUM_DEVICES 4

// TODO: make struct out of inputs
void mcuda_gsmu2(int *Y,
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
	int persons_by_items     = person_count * item_count,
		persons_by_subtests  = person_count * total_subtest_count,
		items_by_subtests    = item_count * total_subtest_count,
		subtests_by_subtests = total_subtest_count * total_subtest_count;

	unsigned int subtests_by_szDbl			   = total_subtest_count * sizeof(double),
				 items_by_szDbl                = item_count * sizeof(double),
				 items_by_subtests_by_szDbl    = items_by_subtests * sizeof(double),
				 persons_by_items_by_szDbl     = persons_by_items * sizeof(double),
				 persons_by_subtests_by_szDbl  = persons_by_subtests * sizeof(double),
				 subtests_by_subtests_by_szDbl = subtests_by_subtests * sizeof(double);
	
	double *A     = (double*)malloc(items_by_szDbl);
	double *G     = (double*)malloc(items_by_szDbl);
	double *TH    = (double*)malloc(persons_by_subtests_by_szDbl);
	double *SIGMA = (double*)malloc(subtests_by_subtests_by_szDbl);

	double *AV    = (double*)malloc(items_by_szDbl*3);
	double *GV    = (double*)malloc(items_by_szDbl*3);
	double *THV   = (double*)malloc(persons_by_subtests_by_szDbl*3);
	double *RHO   = (double*)malloc(subtests_by_subtests_by_szDbl*3);

	double *Z     = (double *)malloc(persons_by_items_by_szDbl);

	double *CHF = (double*)malloc(subtests_by_subtests_by_szDbl);

	int *ord      = (int *)malloc(sizeof(int)*item_count);
	int *places   = (int *)malloc(sizeof(int)*total_subtest_count);

	int BSIZE   = (iteration_count - BURNIN) / BATCHES;

	double mSR5 = -sqrt(5.0);

	srand(RANDOM_SEED);

	// initialize
	for(int item = 0; item < item_count; item++) {
		int csum = 0;
		for(int person = 0; person < person_count; person++) {
			csum += Y[person*item_count+item];
		}
		A[item] = ALPHA_I;
		G[item] = (double)(stdnormal_inv((double)csum / (double)person_count) * mSR5);
	}
	for(int person = 0; person < person_count; person++) {
		for(int subtest = 0; subtest < total_subtest_count; subtest++) {
			TH[person*total_subtest_count + subtest] = TH_I;
		}
	}
	for(int subtest = 0; subtest < total_subtest_count; subtest++) {
		for(int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			SIGMA[subtest*total_subtest_count + subtest0] = 0.0;
			SIGMA[subtest0*total_subtest_count + subtest] = 0.0;
		}
		SIGMA[subtest*total_subtest_count+subtest] = 1.0;
	}
	for(int subtest = total_subtest_count - 1; subtest >= 0; subtest--) {
		places[subtest] = subtest_counts[subtest];
		for (int subtest0 = 0; subtest0 < subtest; subtest0++) {
			places[subtest] += subtest_counts[subtest0];
		}
	}	
	for(int item = 0, subtest = 0; item < item_count; item++) {
		if(item >= places[subtest]) {
			subtest++;
		}
		ord[item] = subtest;
	}
	
	// setup gpu

	int GPU_N;
	cudaGetDeviceCount(&GPU_N);

	GPU_N = min(MAX_NUM_DEVICES, GPU_N);

	unsigned int THREAD_X = 8,
				 THREAD_Y = 8,
				 THREAD_Z = 1;
	
	if(item_count % 10 == 0)     THREAD_X = 10;
	else if(item_count % 8 == 0) THREAD_X = 8;
	else if(item_count % 5 == 0) THREAD_X = 5;
	else if(item_count % 4 == 0) THREAD_X = 4;
	else if(item_count % 2 == 0) THREAD_X = 2;
	else					     THREAD_X = 1;
	
	if(person_count % 10 == 0)     THREAD_Y = 10;
	else if(person_count % 8 == 0) THREAD_Y = 8;
	else if(person_count % 5 == 0) THREAD_Y = 5;
	else if(person_count % 4 == 0) THREAD_Y = 4;
	else if(person_count % 2 == 0) THREAD_Y = 2;
	else						   THREAD_Y = 1;
		
	unsigned int GRID_X = ((item_count + THREAD_X - 1) / THREAD_X),
		         GRID_Y = ((person_count + THREAD_Y - 1) / THREAD_Y),
				 GRID_Z = ((total_subtest_count + THREAD_Z - 1) / THREAD_Z);

	unsigned int *GRID_YS = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *THREAD_YS = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *section = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *offset = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *offset_by_items = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *offset_by_subtests = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *section_by_items = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *section_by_subtests = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *section_by_subtests_by_szDbl = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N),
				 *section_by_items_by_szDbl = (unsigned int *)malloc(sizeof(unsigned int)*GPU_N);

	unsigned int THREADS = 512,
				 BLOCKS = (((person_count + THREADS - 1) / THREADS) < 1024) ? ((person_count + THREADS - 1) / THREADS) : 1024;

	curandState_t *rngStates[MAX_NUM_DEVICES];

	double *dev_WZ[MAX_NUM_DEVICES],
		   *dev_U[MAX_NUM_DEVICES],
		   *dev_THS[MAX_NUM_DEVICES],
		   *dev_RSIG_C[MAX_NUM_DEVICES],
		   *dev_ZS[MAX_NUM_DEVICES],
		   *dev_A[MAX_NUM_DEVICES],
		   *dev_G[MAX_NUM_DEVICES],
		   *dev_AA[MAX_NUM_DEVICES],
		   *dev_AAT[MAX_NUM_DEVICES],
		   *dev_D_AATS[MAX_NUM_DEVICES],
		   *dev_RSIG[MAX_NUM_DEVICES],
		   *dev_PVAR[MAX_NUM_DEVICES],
		   *dev_PVAR_ZVAR[MAX_NUM_DEVICES];
 
	double *dev_Z,
		   *dev_TH,
		   *dev_WZST,
		   *dev_SIGMA,
		   *dev_AV,
		   *dev_GV,
		   *dev_THV,
		   *dev_RHO;
	
	int *dev_SUBTEST_COUNTS[MAX_NUM_DEVICES],
		*dev_Y[MAX_NUM_DEVICES],
		*dev_ORD[MAX_NUM_DEVICES];

	double2 *dev_CTH_WZ[MAX_NUM_DEVICES];

	curandGenerator_t gen[MAX_NUM_DEVICES];
	
	cudaStream_t *streams = (cudaStream_t*)malloc(2 * GPU_N * sizeof(cudaStream_t));

	// formulate sections to decompose number of persons, row-wise	
	int d = person_count / GPU_N;
	int r = person_count % GPU_N;
	int n; for(n = 0; n < GPU_N; n++) {
		section[n] = d;
	}
	n = 0;
	while(r > 0) {
		section[n] += 1;
		n++; r--;
	}
	int _offset = 0;	
	
	if (true) printf("# of GPUs = %d\n", GPU_N);

	for(n = 0; n < GPU_N; n++) {

		if(section[n] % 10 == 0)     THREAD_YS[n] = 10;
		else if(section[n] % 8 == 0) THREAD_YS[n] = 8;
		else if(section[n] % 5 == 0) THREAD_YS[n] = 5;
		else if(section[n] % 4 == 0) THREAD_YS[n] = 4;
		else if(section[n] % 2 == 0) THREAD_YS[n] = 2;
		else					     THREAD_YS[n] = 1;
		
		GRID_YS[n] = ((section[n] + THREAD_YS[n] - 1) / THREAD_YS[n]);
		
		if(n > 0) {
			_offset += section[n];
			offset[n] = _offset;
		}
		else {
			offset[n] = 0;
		}

		offset_by_items[n] = offset[n] * item_count;
		offset_by_subtests[n] = offset[n] * total_subtest_count;
		
		section_by_items[n] = section[n] * item_count;
		section_by_subtests[n] = section[n] * total_subtest_count;

		section_by_items_by_szDbl[n] = section_by_items[n] * sizeof(double);
		section_by_subtests_by_szDbl[n] = section_by_subtests[n] * sizeof(double);
		
		if(true) {
			printf("offset[%d] = %5d, section[%d] = %5d, section_by_subtests[%d] = %5d, section_by_items[%d] = %8d, offset_by_subtests[%d] = %5d, section_by_subtests[%d] = %5d\n", 
				   n, offset[n], n, section[n], n, section_by_subtests[n], n, section_by_items[n], n, offset_by_subtests[n], n, section_by_subtests[n]);
		}
	}
	printf("\n");
	
	omp_set_num_threads(max(GPU_N, total_subtest_count));

	cudaSetDevice(0);

#ifdef CUDA_STATS	
	cudaMalloc((void **)&dev_AV, 3 * items_by_szDbl);
	cudaMalloc((void **)&dev_GV, 3 * items_by_szDbl);
	cudaMalloc((void **)&dev_THV, 3 * persons_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_RHO, 3 * subtests_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_SIGMA, subtests_by_subtests_by_szDbl);
#endif
#ifdef CUDA_SUBTEST
	cudaMalloc((void **)&dev_Z, persons_by_items_by_szDbl);
	cudaMalloc((void **)&dev_TH, persons_by_subtests_by_szDbl);
	cudaMalloc((void **)&dev_WZST, persons_by_subtests_by_szDbl);
#endif
#ifdef CUDA_P2P
	int accessible;
#endif
	#pragma omp parallel for num_threads(GPU_N)
	for(n = 0; n < GPU_N; n++) {
		cudaSetDevice(n);
		
		for(int i = 0; i < 2; i++) {
			cudaStreamCreate(&(streams[i*GPU_N + n]));
		}

		curandCreateGenerator(&gen[n], CURAND_RNG_PSEUDO_DEFAULT); 
		curandSetPseudoRandomGeneratorSeed(gen[n], 1234ULL);

		cudaMalloc((void **)&rngStates[n], GRID_X * THREAD_X * sizeof(curandState_t));
		setupRng(rngStates[n], GRID_X, 1, THREAD_X, 1, 12357, 123671);

		cudaMalloc((void **)&dev_U[n], section_by_items[n] * sizeof(double));
		
		cudaMalloc((void **)&dev_ZS[n], section_by_items_by_szDbl[n]);
		
		cudaMalloc((void **)&dev_Y[n], section_by_items[n] * sizeof(int));
		
		cudaMalloc((void **)&dev_A[n], items_by_szDbl);
		cudaMalloc((void **)&dev_G[n], items_by_szDbl);

		cudaMalloc((void **)&dev_THS[n], section_by_subtests_by_szDbl[n]);

		cudaMalloc((void **)&dev_SUBTEST_COUNTS[n], total_subtest_count * sizeof(int));

		cudaMalloc((void **)&dev_ORD[n], item_count * sizeof(int));

		cudaMemcpy(dev_SUBTEST_COUNTS[n], subtest_counts, total_subtest_count * sizeof(unsigned int), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_Y[n], Y + offset_by_items[n], section_by_items[n] * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_ORD[n], ord, item_count * sizeof(unsigned int), cudaMemcpyHostToDevice);

  #ifdef CUDA_TH		
		cudaMalloc((void **)&dev_WZ[n], section_by_subtests_by_szDbl[n]);
		cudaMalloc((void **)&dev_RSIG[n], subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_PVAR[n], subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_AAT[n], items_by_subtests_by_szDbl);

		cudaMemcpy(dev_THS[n], TH + offset_by_subtests[n], section_by_subtests_by_szDbl[n], cudaMemcpyHostToDevice);
  #endif

  #ifdef CUDA_AATS
		cudaMalloc((void **)&dev_AA[n], items_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_D_AATS[n], subtests_by_subtests_by_szDbl);
    #ifndef CUDA_TH
		cudaMalloc((void **)&dev_AAT[n], items_by_subtests_by_szDbl);
    #endif
  #endif

  #ifdef CUDA_SUBTEST
		cudaMemcpy(dev_A[n], A, items_by_szDbl, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_G[n], G, items_by_szDbl, cudaMemcpyHostToDevice);		
		cudaMalloc((void **)&dev_RSIG_C[n], subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_PVAR_ZVAR[n], subtests_by_subtests_by_szDbl);
		cudaMalloc((void **)&dev_CTH_WZ[n], persons_by_subtests * sizeof(double2));
  #endif
  
  #ifdef CUDA_P2P
		for(int n2 = 0; n2 < GPU_N; n2++) {
			if(n != n2) {
				cudaDeviceCanAccessPeer(&accessible, n, n2);
				if(accessible) {
					cudaDeviceEnablePeerAccess(n,0);
					printf("GPU %d can access GPU %d\n", n, n2);
				}
				else
					printf("GPU %d can not access GPU %d\n", n, n2);
			}
		}
		printf("\n");
  #endif		
	}	


	double *AATS = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *AAT = (double *)malloc(items_by_subtests_by_szDbl);

	double *AA = (double *)malloc(items_by_subtests_by_szDbl);

	double *RSIG = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *PVAR = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *PVARI = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *WZ = (double *)malloc(persons_by_subtests_by_szDbl);

	double *RTH = (double*)malloc(subtests_by_szDbl);
	
	double *PMEAN = (double*)malloc(subtests_by_szDbl);
	double *PMEAN1 = (double*)malloc(subtests_by_szDbl);

	double *D = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *ZVAR = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *VAR = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *DI = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *RSIGT = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *AZ = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *C = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *CTH = (double *)malloc(persons_by_subtests_by_szDbl);
	double *CTHT = (double *)malloc(persons_by_subtests_by_szDbl);


	// begin iterate
	for(int iteration = 0; iteration < iteration_count; iteration++) {	

		// Update Z
		#pragma omp parallel for num_threads(GPU_N)
		for(n = 0; n < GPU_N; n++) {	
			cudaSetDevice(n);
			curandGenerateUniformDouble(gen[n], dev_U[n], section_by_items[n]);
  #ifndef CUDA_SUBTEST
			cudaMemcpyAsync(dev_A[n], A, items_by_szDbl, cudaMemcpyHostToDevice, streams[n]);
			cudaMemcpyAsync(dev_G[n], G, items_by_szDbl, cudaMemcpyHostToDevice, streams[n]);
  #endif
  #ifndef CUDA_TH			
			cudaMemcpyAsync(dev_THS[n], TH + offset_by_subtests[n], section_by_subtests_by_szDbl[n], cudaMemcpyHostToDevice, streams[n]);
  #endif
			calcZ(streams[n], dev_U[n], GRID_X, GRID_YS[n], THREAD_X, THREAD_YS[n], dev_Y[n], dev_A[n], dev_G[n], dev_THS[n], dev_ZS[n], dev_ORD[n], item_count, section[n], total_subtest_count);
  #ifdef CUDA_SUBTEST
			cudaMemcpyAsync(dev_Z + offset_by_items[n], dev_ZS[n], section_by_items_by_szDbl[n], cudaMemcpyDeviceToDevice, streams[n]);
  #else
			cudaMemcpyAsync(Z + offset_by_items[n], dev_ZS[n], section_by_items_by_szDbl[n], cudaMemcpyDeviceToHost, streams[n]);
  #endif
		}

		{
			
#ifdef CUDA_AATS			
			#pragma omp parallel for num_threads(GPU_N)
			for(n = 0; n < GPU_N; n++) {
				cudaSetDevice(n);
				initAA(dev_AAT[n], dev_AA[n], item_count, total_subtest_count, GRID_Z, GRID_X, THREAD_Z, THREAD_X);
				stripeAA(dev_AAT[n], dev_AA[n], dev_A[n], dev_SUBTEST_COUNTS[n], item_count, total_subtest_count, GRID_Z, THREAD_Z);
				calcAATS(dev_AAT[n], dev_AA[n], dev_D_AATS[n], item_count, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z);					
			}
			cudaSetDevice(0);
			cudaMemcpy(AATS, dev_D_AATS[0], subtests_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
  #ifndef CUDA_TH
			
			cudaMemcpy(AAT, dev_AAT[0], items_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
  #endif			
#else
			
			{

				for(int item = 0; item < item_count; item++) {
					for(int subtest = 0; subtest < total_subtest_count; subtest++) {
						AAT[subtest*item_count + item] = AA[item*total_subtest_count + subtest] = 0.0;
					}
				}
				for(int subtest = 0, subitem = 0; subtest < total_subtest_count; subtest++) {	
					for(int subtestitem = 0; subtestitem < subtest_counts[subtest]; subtestitem++, subitem++ ) {
						AAT[subtest*item_count + subitem] = AA[subitem*total_subtest_count + subtest] = A[subitem];
					}
				}
				for(int r = 0; r < total_subtest_count; r++) {
					for(int c = 0; c < total_subtest_count; c++) {
						double val = 0.0;
						for(int n = 0; n < item_count; n++) {
							val += AAT[r*item_count + n] * AA[n*total_subtest_count + c];
						}
						AATS[r*total_subtest_count + c] = (double)val;
					}
				}
				
			}
#endif
			
			{

				MATINV(SIGMA, RSIG, subtests_by_subtests_by_szDbl,total_subtest_count);

				for(int i = 0; i < total_subtest_count; i++) {
					for(int j = 0; j < total_subtest_count; j++) {
						PVARI[i*total_subtest_count + j] = RSIG[i*total_subtest_count + j] + AATS[i*total_subtest_count + j];
					}
				}
								
				MATINV(PVARI, PVAR, subtests_by_subtests_by_szDbl, total_subtest_count);
				
				CHFAC(total_subtest_count, PVAR, RSIG, CHF);
			}

			
#ifdef CUDA_TH			
			// Update Theta
			#pragma omp parallel for num_threads(GPU_N)
			for(n = 0; n < GPU_N; n++) {
				cudaSetDevice(n);
				curandGenerateNormalDouble(gen[n], dev_WZ[n], section_by_subtests[n], 0.0, 1.0);
				cudaMemcpyAsync(dev_RSIG[n], RSIG, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, streams[1*GPU_N+n]);
				cudaMemcpyAsync(dev_PVAR[n], PVAR, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice, streams[1*GPU_N+n]);
  #ifndef CUDA_AATS
				cudaMemcpyAsync(dev_AAT[n], AAT, items_by_subtests_by_szDbl, cudaMemcpyHostToDevice, streams[1*GPU_N+n]);
  #endif
				calcTH(streams[1 * GPU_N + n], dev_WZ[n], dev_AAT[n], dev_RSIG[n], dev_PVAR[n], dev_ZS[n], dev_G[n], dev_THS[n], GRID_YS[n], THREAD_YS[n], item_count, section[n], total_subtest_count);
  #ifdef CUDA_SUBTEST
				cudaMemcpyAsync(dev_TH + offset_by_subtests[n], dev_THS[n], section_by_subtests_by_szDbl[n], cudaMemcpyDeviceToDevice, streams[1*GPU_N+n]);
  #endif				
  #ifndef CUDA_STATS
				cudaMemcpyAsync(TH + offset_by_subtests[n], dev_THS[n], section_by_subtests_by_szDbl[n], cudaMemcpyDeviceToHost, streams[1*GPU_N+n]);
  #endif
			}			
#else
			
			normal_random_array(total_subtest_count, person_count, WZ);
			{
			
				for(int person = 0; person < person_count; person++) {
					for(int r = 0; r < total_subtest_count; r++) {
						double val = 0.0;
						for(int n = 0; n < item_count; n++) {
							val += AAT[r*item_count + n] * (Z[person*item_count + n] + G[n]);
						}
						PMEAN1[r] = (double)val;
					}
					for(int r = 0; r < total_subtest_count; r++) {
						double val1 = 0.0;
						for(int n = 0; n < total_subtest_count; n++) {
							val1 += PVAR[r*total_subtest_count + n] * PMEAN1[n];
						}
						PMEAN[r] = (double)val1;
						
						double val2 = 0.0;
						for(int n = 0; n < total_subtest_count; n++) {
							val2 += WZ[person*total_subtest_count + n] * RSIG[n*total_subtest_count + r];
						}
						RTH[r] = (double)val2;
					}
					for(int subtest = 0; subtest < total_subtest_count; subtest++) {
						TH[person*total_subtest_count + subtest] = RTH[subtest] + PMEAN[subtest];
					}
				}
				
			}
			
#endif

		}		
#ifdef CUDA_SUBTEST
		
		cudaSetDevice(0);
		cudaSubtest(rngStates[0], dev_A[0], dev_G[0], dev_Z, dev_TH, person_count, item_count, total_subtest_count, uniform, dev_SUBTEST_COUNTS[0], GRID_Z, THREAD_Z, BLOCKS, THREADS);
				
		curandGenerateNormalDouble(gen[0], dev_WZST, persons_by_subtests, 0.0, 1.0);

		copyRN(dev_CTH_WZ[0], dev_WZST, person_count, total_subtest_count, GRID_Z, GRID_Y, THREAD_Z, THREAD_Y);

		calcC_ZVAR(dev_CTH_WZ[0], dev_PVAR_ZVAR[0], dev_SUBTEST_COUNTS[0], dev_RSIG_C[0], dev_A[0], person_count, total_subtest_count, GRID_Z, THREAD_Z, BLOCKS, THREADS);

		calcCTH(dev_RSIG_C[0], dev_TH, dev_CTH_WZ[0], person_count, total_subtest_count, GRID_Z, GRID_Y, THREAD_Z, THREAD_Y);

		calcD(dev_CTH_WZ[0], dev_D_AATS[0], person_count, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z, BLOCKS, THREADS);

		cudaMemcpy(ZVAR, dev_PVAR_ZVAR[0], subtests_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);
		cudaMemcpy(D, dev_D_AATS[0], subtests_by_subtests_by_szDbl, cudaMemcpyDeviceToHost);


  #ifndef CUDA_STATS
		cudaMemcpy(A, dev_A[0], items_by_szDbl, cudaMemcpyDeviceToHost);
		cudaMemcpy(G, dev_G[0], items_by_szDbl, cudaMemcpyDeviceToHost);
  #endif

		#pragma omp parallel for num_threads(GPU_N)
		for(n = 1; n < GPU_N; n++) {
			cudaSetDevice(n);
			cudaMemcpy(dev_A[n], dev_A[0], items_by_szDbl, cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_G[n], dev_G[0], items_by_szDbl, cudaMemcpyDeviceToDevice);
		}
				
		
		//INVERSEWISHART(D,VAR)
		{
			
			MATINV(D, DI, subtests_by_subtests_by_szDbl, total_subtest_count);

			CHFAC(total_subtest_count, DI, RSIG, CHF);

			{
				
				{
					for(int i = 0; i < total_subtest_count; i++) {
						for(int j = 0; j < total_subtest_count; j++) {
							RSIGT[j*total_subtest_count + i] = RSIG[i*total_subtest_count + j];
						}
					}
				}
				for(int r = 0; r < total_subtest_count; r++) {
					for(int c = 0; c < total_subtest_count; c++) {
						double val = 0.0;
						for(int n = 0; n < total_subtest_count; n++) {
							val += RSIGT[r*total_subtest_count + n] * ZVAR[n*total_subtest_count + c];
						}
						AZ[r*total_subtest_count + c] = (double) val;
					}
				}				   
				for(int r = 0; r < total_subtest_count; r++) {
					for(int c = 0; c < total_subtest_count; c++) {
						double val = 0.0;
						for(int n = 0; n < total_subtest_count; n++) {
							val += AZ[r*total_subtest_count + n] * RSIG[n*total_subtest_count + c];
						}
						RSIGT[r*total_subtest_count + c] = (double) val;
					}
				}
   
				MATINV(RSIGT, VAR, subtests_by_subtests_by_szDbl, total_subtest_count);					
                   
				
			}

			
			
		}

		

		{
			for(int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
				for(int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					int i = subtest0*total_subtest_count + subtest1;
					SIGMA[i] = (double)(VAR[i] * sqrt(1.0 / (VAR[subtest0*total_subtest_count + subtest0] * VAR[subtest1*total_subtest_count + subtest1])));
				}
			}
  #ifdef CUDA_STATS
			cudaSetDevice(0);
			cudaMemcpy(dev_SIGMA, SIGMA, subtests_by_subtests_by_szDbl, cudaMemcpyHostToDevice);
  #endif
		}
		
#else	
		
		#pragma omp parallel for num_threads(total_subtest_count)
		for(int subtest = 0; subtest < total_subtest_count; subtest++) {
			Subtest(person_count, item_count, total_subtest_count, uniform, subtest, subtest_counts, Z, TH, C, A, G);
		}

		{
			
			{

				

				for(int r = 0; r < total_subtest_count; r++) {
					for(int c = 0; c < person_count; c++) {
						double val = 0.0;
						for(int n = 0; n < total_subtest_count; n++) {
							val += C[r*total_subtest_count + n] * TH[c*total_subtest_count + n];
						}
						CTHT[c*total_subtest_count + r] = CTH[r*person_count + c] = (double)val;
					}
				}
				for(int r = 0; r < total_subtest_count; r++) {
					for(int c = 0; c < total_subtest_count; c++) {
						double val = 0.0;
						for(int n = 0; n < person_count; n++) {
							val += CTH[r*person_count + n] * CTHT[n*total_subtest_count + c];
						}
						D[r*total_subtest_count + c] = (double)val;
					}
				}
				
			}			

			//INVERSEWISHART
			{
				
				MATINV(D, DI, subtests_by_subtests_by_szDbl, total_subtest_count);

				CHFAC(total_subtest_count, DI, RSIG, CHF);

				
				normal_random_array(total_subtest_count, person_count, WZ);				
				{
					for(int r = 0; r < total_subtest_count; r++) {
						for(int c = 0; c < total_subtest_count; c++) {
							double val = 0.0;
							for(int n = 0; n < person_count; n++) {
								val += WZ[n*total_subtest_count + r] * WZ[n*total_subtest_count + c];
							}					
							ZVAR[r*total_subtest_count + c] = (double)val;
						}
					}
				}
				
				

				{
					{
						//MATMUL(RSIG, ZVAR, AZ)
						for(int r = 0; r < total_subtest_count; r++) {
							for(int c = 0; c < total_subtest_count; c++) {
								double val = 0.0;
								for(int n = 0; n < total_subtest_count; n++) {
									val += RSIG[n*total_subtest_count + r] * ZVAR[n*total_subtest_count + c];
								}
								AZ[r*total_subtest_count+c] = (double)val;
							}
						}
					}
					
					{
						//MATMUL(AZ, RSIG, RSIGT)
						for(int r = 0; r < total_subtest_count; r++) {
							for(int c = 0; c < total_subtest_count; c++) {
								double val = 0.0;
								for(int n = 0; n < total_subtest_count; n++) {
									val += AZ[r*total_subtest_count + n] * RSIG[n*total_subtest_count + c];
								}
								RSIGT[r*total_subtest_count + c] = (double)val;
							}
						}
					}

					MATINV(RSIGT,VAR,subtests_by_subtests_by_szDbl,total_subtest_count);					

				}
			}

			for(int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
				for(int subtest1 = 0; subtest1 < total_subtest_count; subtest1++) {
					SIGMA[subtest0*total_subtest_count + subtest1] = (double)(VAR[subtest0*total_subtest_count + subtest1] * sqrt( 1.0 / (VAR[subtest0*total_subtest_count + subtest0]*VAR[subtest1*total_subtest_count + subtest1]))); 
				}
			}

		}		
		
#endif

		// track statistics
#ifdef CUDA_STATS
		if (iteration >= BURNIN) {
			cudaSetDevice(0);
			if (((iteration - BURNIN) % BSIZE) == 0) {
				if (iteration == BURNIN) {
					for(int item = 0; item < item_count; item++) {
						AV[item_count + item] = 0.0;
						GV[item_count + item] = 0.0;
						AV[2 * item_count + item] = 0.0;
						GV[2 * item_count + item] = 0.0;
					}
					for(int subtest = 0; subtest < total_subtest_count; subtest++) {
						for(int person = 0; person < person_count; person++) {
							int i = person*total_subtest_count + subtest;
							THV[persons_by_subtests + i] = 0.0;
							THV[2 * persons_by_subtests + i] = 0.0;
						}
						for(int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
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
				copyAG(dev_AV, dev_GV, dev_A[0], dev_G[0], GRID_X, THREAD_X);
				copyTH(dev_THV, dev_TH, total_subtest_count, GRID_Z, GRID_Y, THREAD_Z, THREAD_Y);
				copySIGMA(dev_RHO, dev_SIGMA, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z);
			}
			else {
				sumAG(dev_AV, dev_GV, dev_A[0], dev_G[0], GRID_X, THREAD_X);
				sumTH(dev_THV, dev_TH, total_subtest_count, GRID_Y, GRID_Z, THREAD_Y, THREAD_Z);
				sumSIGMA(dev_RHO, dev_SIGMA, total_subtest_count, GRID_Z, GRID_Z, THREAD_Z, THREAD_Z);
			}
		}
#else
		if(iteration >= BURNIN) {
			if(((iteration-BURNIN) % BSIZE)==0) {
				if(iteration == BURNIN) {
					for(int item = 0; item < item_count; item++) {
						AV[item_count + item] = 0.0;
						GV[item_count + item] = 0.0;
						AV[2*item_count + item] = 0.0;
						GV[2*item_count + item] = 0.0;
					}					
					for(int subtest = 0; subtest < total_subtest_count; subtest++) {	
						for(int person = 0; person < person_count; person++) {              
							THV[persons_by_subtests + person*total_subtest_count + subtest] = 0.0;
							THV[2*persons_by_subtests + person*total_subtest_count + subtest] = 0.0;
						}						
						for(int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
							RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] = 0.0;
							RHO[2*subtests_by_subtests + subtest*total_subtest_count + subtest0] = 0.0;
						}
					}
				}
				else {
					for(int item = 0; item < item_count; item++) {    
						double M1 = AV[item] / BSIZE;
						double M2 = GV[item] / BSIZE;
						AV[item_count + item] += M1;
						GV[item_count + item] += M2;
						AV[2*item_count + item] += M1 * M1;
						GV[2*item_count + item] += M2 * M2;
					}					
					for(int subtest = 0; subtest < total_subtest_count; subtest++) {
						for(int person = 0; person < person_count; person++) {
							double M1 = THV[person*total_subtest_count + subtest] / BSIZE;
							THV[persons_by_subtests+person*total_subtest_count + subtest] += M1;
							THV[2*persons_by_subtests+person*total_subtest_count + subtest] += M1 * M1;
						}
						for(int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
							double M1 = RHO[subtest*total_subtest_count + subtest0] / BSIZE;
							RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1;
							RHO[2*subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1 * M1;
						}
					}
				}				
				for(int item = 0; item < item_count; item++) {   
					AV[item] = A[item];
					GV[item] = G[item];
				}
				for(int subtest = 0; subtest < total_subtest_count; subtest++) {
					for(int person = 0; person < person_count; person++) {
						THV[person*total_subtest_count + subtest] = TH[person*total_subtest_count + subtest];
					}
					for(int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {	
						RHO[subtest*total_subtest_count + subtest0] = SIGMA[subtest*total_subtest_count + subtest0];
					}
				}
			}
			else {
				for(int j = 0; j < item_count; j++) {
					AV[j] += A[j];
					GV[j] += G[j];
				}				
				for(int i = 0; i < person_count; i++) {
					for(int j = 0; j < total_subtest_count; j++) {
						THV[i*total_subtest_count + j] += TH[i*total_subtest_count + j];
					}
				}				
				for(int i = 0; i < total_subtest_count; i++) {
					for(int j = 0; j < total_subtest_count; j++) {
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

	free(RSIG);
	free(PVAR);
	free(PVARI);

	free(WZ);

	free(RTH);
	
	free(PMEAN);
	free(PMEAN1);

	free(D);
	free(ZVAR);
	free(VAR);

	free(DI);

	free(RSIGT);
	free(AZ);

	free(C);
	free(CTH);
	free(CTHT);



	cudaSetDevice(0);

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
	cudaFree(dev_Z);
	cudaFree(dev_TH);
	cudaFree(dev_WZST);
#endif

	// free memory on gpu
	#pragma omp parallel for num_threads(GPU_N)
	for(n = 0; n < GPU_N; n++) {
		cudaSetDevice(n);

		cudaFree(dev_Y[n]);
		cudaFree(dev_U[n]);
		cudaFree(dev_WZ[n]);
		cudaFree(dev_ZS[n]);
		cudaFree(dev_THS[n]);
		cudaFree(dev_A[n]);
		cudaFree(dev_G[n]);
		cudaFree(dev_AAT[n]);		
		cudaFree(dev_RSIG[n]); 
		cudaFree(dev_PVAR[n]);
		cudaFree(dev_ORD[n]);
#ifdef CUDA_AATS
		cudaFree(dev_AA[n]);
		cudaFree(dev_D_AATS[n]);
#endif
#ifdef CUDA_SUBTEST		
		cudaFree(dev_RSIG_C[n]);
		cudaFree(dev_PVAR_ZVAR[n]);
		cudaFree(dev_CTH_WZ[n]);
#endif
		cudaFree(dev_SUBTEST_COUNTS[n]);

		for(int i = 0; i < 2; i++) {
			cudaStreamDestroy(streams[i*GPU_N + n]);
		}

		curandDestroyGenerator(gen[n]);

		cudaDeviceReset();
	}


	free(Z);
	free(A);
	free(G);
	free(TH);
	free(SIGMA);

	free(CHF);
	
	for(int item = 0; item < item_count; item++) {    
		double M1 = AV[item] / BSIZE;
		double M2 = GV[item] / BSIZE;
		AV[item_count + item] += M1;
		GV[item_count + item] += M2;
		AV[2*item_count + item] += M1 * M1;
		GV[2*item_count + item] += M2 * M2;
	}
	
	for(int subtest = 0; subtest < total_subtest_count; subtest++) {
		for(int person = 0; person < person_count; person++) {
			double M1 = THV[person*total_subtest_count+subtest] / BSIZE;
			THV[persons_by_subtests + person*total_subtest_count + subtest] += M1;
			THV[2*persons_by_subtests + person*total_subtest_count + subtest] += M1 * M1;
		}
		for(int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			double M1 = RHO[subtest*total_subtest_count+subtest0] / BSIZE;
			RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1;
			RHO[2*subtests_by_subtests + subtest*total_subtest_count + subtest0] += M1 * M1;
		}
	}
	
	for(int item = 0; item < item_count; item++) {
		ITEM[item*4] = AV[item_count+item] /(double)BATCHES;
		ITEM[item*4 + 1] = (double)(sqrt((AV[2*item_count + item]-(AV[item_count + item]*AV[item_count + item]/BATCHES)) / (BATCHES-1) ) / sqrt((double)(BATCHES)));
		ITEM[item*4 + 2] = GV[item_count + item] /(double)BATCHES;
		ITEM[item*4 + 3] = (double)(sqrt((GV[2*item_count + item]-(GV[item_count + item]*GV[item_count + item]/BATCHES)) / (BATCHES-1) ) / sqrt((double)(BATCHES)));
	}
	
	for(int subtest = 0, JK = 0; subtest < total_subtest_count; subtest++) {	
		for(int person = 0; person < person_count; person++) {
			PERSON[person*total_subtest_count*2 + 2*subtest] = THV[persons_by_subtests + person*total_subtest_count + subtest]/(double)(BATCHES);
			PERSON[person*total_subtest_count*2 + 2*subtest + 1] = (double)(sqrt((THV[2*persons_by_subtests + person*total_subtest_count + subtest]-(THV[persons_by_subtests+person*total_subtest_count + subtest]*THV[persons_by_subtests + person*total_subtest_count + subtest]/BATCHES))/(BATCHES-1))/sqrt((double)(BATCHES)));
		}
		for(int subtest0 = subtest + 1; subtest0 < total_subtest_count; subtest0++) {
			CORR[JK*2] = RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0]/BATCHES;
			CORR[JK*2 + 1] = (double)(sqrt((RHO[2*subtests_by_subtests + subtest*total_subtest_count + subtest0]-(RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0]*RHO[subtests_by_subtests + subtest*total_subtest_count + subtest0]/BATCHES))/(BATCHES-1))/sqrt((double)(BATCHES)));
			JK = JK + 1;
		}
	}

	free(AV);
	free(GV);
	free(THV);
	free(RHO);

	return;
}