/*
 Windows

	mpiexec -n 4 mugs.exe mpi-cuda 2000 100 2 10000 5000 5 1 > output/mpi_cuda_gsmu2_2000_100_2_4.txt

	mpiexec -n 4 mugs.exe mpi 2000 100 2 10000 5000 5 1 > output/mpi_gsmu2_2000_100_2_4.txt

	mugs.exe multiple-cuda 2000 100 2 10000 5000 5 1 > output/mcuda_gsmu2_2000_100_2.txt

	mugs.exe cuda 2000 100 2 10000 5000 5 1 > output/cuda_gsmu2_2000_100_2.txt

	mugs.exe serial 2000 100 2 10000 5000 5 1 > output/gsmu2_2000_100_2.txt
  
 cheetah.gsu.edu:

	export OMP_NUM_THREADS=8
    
	nvcc --compiler-options "-O3" -w --relocatable-device-code true -gencode arch=compute_35,code=sm_35 --compiler-bindir mpicc -lcurand -lcublas -L /usr/local/cuda/lib64 -lstdc++ -Xcompiler -fopenmp -lgomp -lcudadevrt -DLINUX *.cu -o mugs
		
	mpirun -machinefile mpd.hosts -np 4 ./mugs mpi-cuda 2000 100 2 10000 5000 5 1 > output/mpi_cuda_gsmu2_2000_100_2_4.out

	mpirun -machinefile mpd.hosts -np 4 ./mugs mpi 2000 100 2 10000 5000 5 1 > output/mpi_gsmu2_2000_100_2_4.out

	./mugs multiple-cuda 2000 100 2 10000 5000 5 1 > output/mcuda_gsmu2_2000_100_2.out

	./mugs cuda 2000 100 2 10000 5000 5 1 > output/cuda_gsmu2_2000_100_2.out

	./mugs serial 2000 100 2 10000 5000 5 1 > output/gsmu2_2000_100_2.out
	

	

	sed -i -e 's/\r$//' script.sh
	sed -i -e 's/\r$//' clear.sh
	chmod 755 script.sh
	chmod 755 clear.sh


	./script.sh |& tee -a experiment.out
*/

#include "stdafx.cuh"
#include "mathafx.cuh"
#include "randafx.cuh"
#include "gendata.cuh"
#include "showdata.cuh"
#include "number.cuh"
#include "utilafx.cuh"

#include "gsmu2.cuh"
#include "mpi_gsmu2.cuh"

#include "cuda_gsmu2.cuh"


//#define GPU_PROPERTIES
//#define TEST_RANDOM
//#define SCORE_TESTS


volatile int break_loop = 0;

int main(int argc, char **argv) {

#ifdef GPU_PROPERTIES

	int GPU_N;
	cudaGetDeviceCount(&GPU_N);

	printf("Total number of GPUs: %d\n", GPU_N);

	for(int ith = 0; ith < GPU_N; ith++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, ith);
		printf( "\n GPU                  %d            "
				"\n asyncEngineCount     %d            "
				"\n computeMode          %d            "
				"\n concurrentKernels    %d            "
				"\n maxGridSize[0]       %d            "
				"\n maxGridSize[1]       %d            "
				"\n maxGridSize[2]       %d            "
				"\n multiProcessorCount  %d            "
				"\n name                 %s            "
				"\n totalConstMem        %d            "
				"\n totalGlobalMem       %lu\n\n",
				ith,
				deviceProp.asyncEngineCount,
				deviceProp.computeMode,
				deviceProp.concurrentKernels,
				deviceProp.maxGridSize[0],
				deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2],
				deviceProp.multiProcessorCount,
				deviceProp.name,
				deviceProp.totalConstMem,
				deviceProp.totalGlobalMem);

		printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
		printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);

#if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
	    cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Compute mode:                                  %s\n",
				deviceProp.computeMode == cudaComputeModeDefault ? "Default (multiple host threads can use this device simultaneously)" :
				deviceProp.computeMode == cudaComputeModeExclusive ? "Exclusive (only one host thread at a time can use this device)" :
				deviceProp.computeMode == cudaComputeModeProhibited ? "Prohibited (no host thread can use this device)" : "Unknown");
#endif
#if CUDART_VERSION >= 2000
		printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
		printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
		printf("  Concurrent copy and execution:                 %s\n\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
	}
	
#endif


#ifdef LINUX
	struct rlimit lim = { 2000000000, 4000000000 };
	if (setrlimit(RLIMIT_STACK, &lim) == -1) {
		return 1;
	}
#endif


#ifdef SCORE_TESTS

		unsigned int *places = (unsigned int *)malloc(sizeof(unsigned int)*SUBTESTS);

		for (int subtest0 = SUBTESTS - 1; subtest0 >= 0; subtest0--) {
			places[subtest0] = SUBTEST_COUNTS[subtest0];
			for (int subtest1 = 0; subtest1 < subtest0; subtest1++) {
				places[subtest0] += SUBTEST_COUNTS[subtest1];
			}
		}

		struct GRADE{
			int ord;
			int sum;
		};

		GRADE *sortp = (GRADE *)malloc(sizeof(GRADE)*PERSONS);

		//grade test 
		for (int p = 0; p < PERSONS; p++) {
			int sum = 0;
			for (int i = 0; i < ITEMS; i++) {
				sum += (SCORES[p*ITEMS + i]);
			}
			sortp[p].ord = p;
			sortp[p].sum = sum;
		}
		//sort
		for (int p = 0; p < PERSONS - 1; p++) {
			for (int q = p + 1; q < PERSONS; q++) {
				if (sortp[q].sum < sortp[p].sum) {
					//swap(q,p);
					int ord = sortp[q].ord;
					int sum = sortp[q].sum;
					sortp[q] = sortp[p];
					sortp[p].ord = ord;
					sortp[p].sum = sum;
				}
			}
		}

		for (int p = 0; p < PERSONS; p++) {
			int p2p = sortp[p].ord;
			printf("\n");

			int sum = 0;
			for (int i = 0; i < ITEMS; i++) {
				sum += (SCORES[p2p*ITEMS + i]);
			}

			if (0) {
				int ord = 0;
				printf("%4d : %4d : %4d : %30.20f : ", p, p2p, sum, (THS)[p2p*SUBTESTS + ord]);
			}

			printf("%4d : %4d : %4d : ", p, p2p, sum);
			for (int subtest = 0; subtest < SUBTESTS; subtest++) {
				printf("%30.20f : ", THS[p2p*SUBTESTS + subtest]);
			}

			// make comparisons between persons  
			// tly is the count of similar correct(1s) answers 
			for (int q = 0; q < PERSONS; q++) {
				int p2q = sortp[q].ord;
				int tly = 0;
				for (int i = 0; i < ITEMS; i++) {
					tly += (SCORES[p2p*ITEMS + i] & SCORES[p2q*ITEMS + i]) ? 1 : 0;
				}
				printf("%4d ", tly);
			}
		}

		free(places);
		free(sortp);

#endif

#ifdef TEST_RANDOM

		for (double cdf = -50.0; cdf <= 50.0; cdf += 0.5) {
			printf("cdf  %40.30f %40.30f\n", cdf, stdnormal_cdf(cdf));
		}

		double q = (0.0001);
		for (double w = 0.0, u = -10.0e000, s = stdnormal_cdf(-10.0); u <= 10.0e000; u += q, w += 1.0) {
			double b = floor(u * 00000000000) * (0.00000000001);
			double p = stdnormal_cdf(b);
			double t = stdnormal_pdf(b);
			s += floor(t * 100000000000) * (0.00000000001)*q;

			printf("u=" "%40.30f " "pdf(u)=" "%40.30f "  "calc cdf(u)=" "%40.30f "   "act cdf(u)=" "%40.30f "  "delta cdf(u)=" "%40.30f " "\n", b, floor(t * 100000000000) * (0.00000000001), s, p, (p - (s)));
		}

		double u = 0.0;
		double p = stdnormal_cdf(u);
		double t = stdnormal_pdf(u);
		printf("u=" "%40.30f " "pdf(u)=" "%40.30f "  "act cdf(u)=" "%40.30f "  "\n", u, t, p);

		int testrand[100001][2];
		memset(testrand, 0, 100001 * 2 * sizeof(int));
		for (int rnd = 0; rnd < RANDOM_TEST_COUNT; rnd++) {
			double TH;
			int one = 1;
			rnnor_(&one, &TH, 5, 6);

			fprintf(stderr, "%24.20f\r", TH);
			if (TH < 0.0) {  //-10.0 -- 0
				int ndx = (int)abs(TH * 10000);
				ndx = (ndx<100000) ? ndx : 100000;
				testrand[ndx][0]++;
			}
			else if (TH >= 0.0) { //00 -- 10.0
				int ndx = (int)abs(TH * 10000);
				ndx = (ndx<100000) ? ndx : 100000;
				testrand[ndx][1]++;
			}
		}
		showmat((int*)&testrand, 100001, 2, "TN");

#endif

	srand(RANDOM_SEED);


	if (strcmp(argv[1], "serial") == 0) {

		atexit(exit_function);

		afxsigp();

		printf("Serial Item Response Theory\n\n");

		int PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR;

		int *SCORES, *SUBTEST_COUNTS;

		double *CORRELATIONS, *AS, *GS, *THS, *EST_ITEMS, *EST_PERSONS, *EST_CORRELATIONS;

		if (argc == 7) {
			setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR, &SCORES, &SUBTEST_COUNTS);

			allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

			show_inputs(SCORES, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}
		else {
			setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR);

			allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &SCORES, &SUBTEST_COUNTS, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

			gen_data(PERSONS, ITEMS, SUBTESTS, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, SCORES);

			show_inputs(SCORES, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}

		time_t start_time = time(NULL);

		gsmu2(SCORES, SUBTEST_COUNTS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

		time_t end_time = (int)(time(NULL) - start_time);

		printf("Elapsed Time: %d\n\n", end_time);
		fprintf(stderr, "Elapsed Test Time: %d\n\n", end_time);

		if (argc == 7) {
			show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, PERSONS, ITEMS, SUBTESTS);
		}
		else {
			show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS);
		}

		cleanup(SCORES, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

	}	
	else if (strcmp(argv[1], "cuda") == 0) {
		
		atexit(exit_function);

		afxsigp();

		printf("CUDA Item Response Theory\n\n");
		
		int PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR;

		int *SCORES, *SUBTEST_COUNTS;

		double *CORRELATIONS, *AS, *GS, *THS, *EST_ITEMS, *EST_PERSONS, *EST_CORRELATIONS;

		if (argc == 7) {
			setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR, &SCORES, &SUBTEST_COUNTS);

			allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

			show_inputs(SCORES, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}
		else {
			setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR);

			allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &SCORES, &SUBTEST_COUNTS, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

			gen_data(PERSONS, ITEMS, SUBTESTS, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, SCORES);

			show_inputs(SCORES, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}

		time_t start_time = time(NULL);

		cuda_gsmu2(SCORES, SUBTEST_COUNTS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

		time_t end_time = (int)(time(NULL) - start_time);

		printf("Elapsed Time: %d\n\n", end_time);
		fprintf(stderr, "Elapsed Test Time: %d\n\n", end_time);

		if (argc == 7) {
			show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, PERSONS, ITEMS, SUBTESTS);
		}
		else {
			show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS);
		}

		cleanup(SCORES, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);
	}
	else if (strcmp(argv[1], "multiple-cuda") == 0) {
		
		atexit(exit_function);

		afxsigp();

		printf("Multiple CUDA Item Response Theory\n\n");

		int PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR;

		int *SCORES, *SUBTEST_COUNTS;

		double *CORRELATIONS, *AS, *GS, *THS, *EST_ITEMS, *EST_PERSONS, *EST_CORRELATIONS;

		if (argc == 7) {
			setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR, &SCORES, &SUBTEST_COUNTS);

			allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

			show_inputs(SCORES, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}
		else {
			setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR);

			allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &SCORES, &SUBTEST_COUNTS, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

			gen_data(PERSONS, ITEMS, SUBTESTS, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, SCORES);

			show_inputs(SCORES, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}

		time_t start_time = time(NULL);

		mcuda_gsmu2(SCORES, SUBTEST_COUNTS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

		time_t end_time = (int)(time(NULL) - start_time);

		printf("Elapsed Time: %d\n\n", end_time);
		fprintf(stderr, "Elapsed Test Time: %d\n\n", end_time);

		if (argc == 7) {
			show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, PERSONS, ITEMS, SUBTESTS);
		}
		else {
			show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS);
		}

		cleanup(SCORES, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

	}
	else if (strcmp(argv[1], "mpi") == 0) {

		int RANK, SIZE;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
		MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

		if (RANK == ROOT) {

			atexit(exit_function);

			afxsigp();

			printf("MPI Item Response Theory\n\n");

			int PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR;

			int *SCORES, *SUBTEST_COUNTS;

			double *CORRELATIONS, *AS, *GS, *THS, *EST_ITEMS, *EST_PERSONS, *EST_CORRELATIONS;

			if (argc == 7) {
				setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR, &SCORES, &SUBTEST_COUNTS);

				allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

				show_inputs(SCORES, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
			}
			else {
				setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR);

				allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &SCORES, &SUBTEST_COUNTS, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

				gen_data(PERSONS, ITEMS, SUBTESTS, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, SCORES);

				show_inputs(SCORES, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
			}

			MPI_Bcast(&PERSONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITEMS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&SUBTESTS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITERATIONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BURNIN, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BATCH, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&PRIOR, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

			printf("Number of processors: %d\n\n", SIZE);

			time_t start_time = time(NULL);

			master_gsmu2(SCORES, SUBTEST_COUNTS, SIZE, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

			time_t end_time = (int)(time(NULL) - start_time);

			printf("Elapsed Time: %d\n\n", end_time);
			fprintf(stderr, "Elapsed Test Time: %d\n\n", end_time);

			if (argc == 7) {
				show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, PERSONS, ITEMS, SUBTESTS);
			}
			else {
				show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS);
			}

			cleanup(SCORES, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

		}
		else {

			int PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR;

			MPI_Bcast(&PERSONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITEMS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&SUBTESTS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITERATIONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BURNIN, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BATCH, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&PRIOR, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

			slave_gsmu2(RANK, SIZE, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}

		MPI_Finalize();

	}
	else if (strcmp(argv[1], "mpi-cuda") == 0) {

		int RANK, SIZE;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
		MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

		if (RANK == ROOT) {

			atexit(exit_function);

			afxsigp();

			printf("MPI + CUDA Item Response Theory\n\n");

			int PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR;

			int *SCORES, *SUBTEST_COUNTS;

			double *CORRELATIONS, *AS, *GS, *THS, *EST_ITEMS, *EST_PERSONS, *EST_CORRELATIONS;

			if (argc == 7) {
				setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR, &SCORES, &SUBTEST_COUNTS);

				allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

				show_inputs(SCORES, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
			}
			else {
				setup(argc, argv, &PERSONS, &ITEMS, &SUBTESTS, &ITERATIONS, &BURNIN, &BATCH, &PRIOR);

				allocate(PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, &SCORES, &SUBTEST_COUNTS, &CORRELATIONS, &AS, &GS, &THS, &EST_ITEMS, &EST_PERSONS, &EST_CORRELATIONS);

				gen_data(PERSONS, ITEMS, SUBTESTS, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, SCORES);

				show_inputs(SCORES, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
			}

			MPI_Bcast(&PERSONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITEMS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&SUBTESTS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITERATIONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BURNIN, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BATCH, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&PRIOR, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

			printf("Number of processors: %d\n\n", SIZE);

			time_t start_time = time(NULL);

			master_cuda_gsmu2(SCORES, SUBTEST_COUNTS, SIZE, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

			time_t end_time = (int)(time(NULL) - start_time);

			printf("Elapsed Time: %d\n\n", end_time);
			fprintf(stderr, "Elapsed Test Time: %d\n\n", end_time);

			if (argc == 7) {
				show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, PERSONS, ITEMS, SUBTESTS);
			}
			else {
				show_outputs(EST_ITEMS, EST_PERSONS, EST_CORRELATIONS, CORRELATIONS, AS, GS, THS, PERSONS, ITEMS, SUBTESTS);
			}

			cleanup(SCORES, SUBTEST_COUNTS, CORRELATIONS, AS, GS, THS, EST_ITEMS, EST_PERSONS, EST_CORRELATIONS);

		}
		else {

			int PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR;

			MPI_Bcast(&PERSONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITEMS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&SUBTESTS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&ITERATIONS, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BURNIN, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&BATCH, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
			MPI_Bcast(&PRIOR, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

			slave_cuda_gsmu2(RANK, SIZE, PERSONS, ITEMS, SUBTESTS, ITERATIONS, BURNIN, BATCH, PRIOR);
		}

		MPI_Finalize();
	}
	else {
		printf("Unrecognized implementation: %s\n\n", argv[1]);
		return 1;
	}

	return 0;
}

