#include "utilafx.cuh"
#include "mathafx.cuh"
#include "number.cuh"
#include "showdata.cuh"

#ifndef LINUX
#define LINE_MAX 10000
#endif

char* validate_input(char *files, int *persons, int *items, int *subtests)
{	
	char *arg_string = (char*)malloc(line_item_count(files)*sizeof(char));

	char *file;
	
	int subtest_item_count = -1;
	int last_line_item_count = -1;
	int last_test_person_count = -1;

	persons[0] = 0;
	items[0] = 0;
	subtests[0] = 0;

	file = strtok(files, ",");
	strcpy(arg_string, file);
	strcat(arg_string, ",");
	while (file != NULL)
	{	
		subtests[0]++;
		//printf("%s\n", file);

		char* line = (char*)malloc(LINE_MAX*sizeof(char));
		
		FILE *fh;
		if (!(fh = fopen(file, "r"))) {
			fprintf(stderr, "Can not read file %s\n", file);
			exit(-1);
		}

		persons[0] = 0;
		while (fgets(line, LINE_MAX*sizeof(char), fh) != NULL) {
			//printf(line);

			if (last_line_item_count > -1) {
				subtest_item_count = line_item_count(line);
				if (last_line_item_count != subtest_item_count) {
					fprintf(stderr, "%d != %d\n", last_line_item_count, subtest_item_count);
					fprintf(stderr, "All persons must have same number of items in subtest.\n");
					exit(-1);
				}
			}

			subtest_item_count = line_item_count(line);
			last_line_item_count = subtest_item_count;

			persons[0]++;
		}
		last_line_item_count = -1;		

		items[0] += subtest_item_count;

		//printf("\n");

		if (last_test_person_count > -1) {
			if (last_test_person_count != persons[0]) {
				fprintf(stderr, "All subtests must have same number of persons.\n");
				exit(-1);
			}
		}

		last_test_person_count = persons[0];

		file = strtok(NULL, ",");
		if (file) {
			strcat(arg_string, file);
			strcat(arg_string, ",");
		}	
	}

	arg_string[strlen(arg_string) - 1] = 0;
	return arg_string;
}

void read_input(char* files, int* scores, int* subtest_counts, int *persons, int *items, int *subtests)
{
	char *file;

	int person = 0;
	int item = 0;
	int item_total = 0;
	int subtest = 0;

	file = strtok(files, ",");

	while (file != NULL)
	{
		char* line = (char*)malloc(LINE_MAX*sizeof(char));

		FILE *fh;
		if (!(fh = fopen(file, "r"))) {
			fprintf(stderr, "Can not read file %s\n", file);
			exit(-1);
		}
		person = 0;

		while (fgets(line, LINE_MAX*sizeof(char), fh) != NULL) {
			for (item = 0; line[item]; item++) {
				if (line[item] != '\n' && line[item] != '\r') {
					scores[person*items[0] + item_total + item] = line[item] - '0';
				}	
			}
			person++;
		}

		subtest_counts[subtest] = item;
		subtest++;
		item_total += item;
		
		file = strtok(NULL, ",");
		
	}
}



void check_arguments(int argc, char** args)
{
	if (0) {
		for (int a = 0; a < argc; a++) {
			fprintf(stdout, "args[%d] %s\n", a, args[a]);
		}
		fprintf(stdout, "\n");
	}
	switch (argc) {
		case 7: {
			args_numbers(3, argc, args);
		} break;
		case 9: {
			args_numbers(2, argc, args);
		} break;
		default: {
			fprintf(stderr, "Wrong number of args! %d\n\n", argc);
			fprintf(stderr, "gs [implementation] [\"subtest1,subtest2,...,subtestx\"] [iterations] [burnin] [batches] [prior]\n");
			fprintf(stderr, "gs [implementation] [persons] [items] [subtests] [iterations] [burnin] [batches] [prior]\n");
			exit(1);
		} break;
	}
}

void args_numbers(int s, int argc, char** args)
{
	for (int a = s; a < argc; a++) {
		int i;
		size_t len = strlen(args[a]);
		for (i = 0; i < len; i++) {
			if (!isdigit(args[a][i])) {
				fprintf(stderr, "Argument %s is not a number.\n", args[a]);
				exit(1);
			}
		}
	}
}

bool one_or_zero(char* d) {
	if (d[0] == '0' || d[0] == '1')
		return true;
	else
		return false;
}

int line_item_count(char* line) {
	int count = 0; 
	for (int i = 0; line[i]; i++) {
		if (line[i] != '\n' && line[i] != '\r')
			count++;
	}
	return count;
}


void show_inputs(int* SCORES, int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR) {

	printf("PERSON COUNT: %d\n", PERSONS);
	printf("ITEM COUNT: %d\n", ITEMS);
	printf("SUBTEST COUNT: %d\n", SUBTESTS);
	printf("ITERATIONS: %d\n", ITERATIONS);
	printf("BURNIN: %d\n", BURNIN);
	printf("BATCH SIZE: %d\n", BATCH);
	printf("PRIOR FLAG: %d\n\n", PRIOR);

	if (0) showmatt((int*)SCORES, PERSONS, ITEMS, "SCORES");
	
}

void show_inputs(int* SCORES, double* CORRELATIONS, double* AS, double* GS, double *THS, int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR) {

	printf("PERSON COUNT: %d\n", PERSONS);
	printf("ITEM COUNT: %d\n", ITEMS);
	printf("SUBTEST COUNT: %d\n", SUBTESTS);
	printf("ITERATIONS: %d\n", ITERATIONS);
	printf("BURNIN: %d\n", BURNIN);
	printf("BATCH SIZE: %d\n", BATCH);
	printf("PRIOR FLAG: %d\n\n", PRIOR);

	if (0) showmatt((int*)SCORES, PERSONS, ITEMS, "SCORES");

	if (1) showmatr(CORRELATIONS, SUBTESTS, SUBTESTS, "CORRELATIONS");

	if (1) showmatrt((double*)AS, ITEMS, 1, "AS");
	if (1) showmatrt((double*)GS, ITEMS, 1, "GS");
	if (1) showmatrt((double*)THS, PERSONS, SUBTESTS, "THS");
}

void show_outputs(double* EST_ITEMS, double* EST_PERSONS, double* EST_CORRELATIONS, int PERSONS, int ITEMS, int SUBTESTS) {
	if (1) showmatrt((double*)EST_ITEMS, ITEMS, 4, "ITEMS");
	if (1) showmatrt((double*)EST_PERSONS, PERSONS, 2 * SUBTESTS, "PERSONS");

	double *CORROUT = (double*)malloc(sizeof(double)*SUBTESTS*SUBTESTS);
	int c = 0;
	int pd = 0;
	for (int rc = 0; rc < SUBTESTS; rc++) {
		for (int cc = 0; cc < SUBTESTS; cc++) {
			if (rc == cc) {
				CORROUT[rc*SUBTESTS + cc] = 1.0;
				pd++;
			}
			else if (cc >= pd) {
				CORROUT[rc*SUBTESTS + cc] = CORROUT[cc*SUBTESTS + rc] = EST_CORRELATIONS[c * 2 + 0];
				c++;
			}
		}
	}

	showmatr(CORROUT, SUBTESTS, SUBTESTS, "CORRELATIONS");

	free(CORROUT);
}

void show_outputs(double* EST_ITEMS, double* EST_PERSONS, double* EST_CORRELATIONS, double* CORRELATIONS, double* AS, double* GS, double *THS, int PERSONS, int ITEMS, int SUBTESTS) {

	if (1) showmatrt((double*)EST_ITEMS, ITEMS, 4, "ITEMS");
	if (1) showmatrt((double*)EST_PERSONS, PERSONS, 2 * SUBTESTS, "PERSONS");

	double *CORROUT = (double*)malloc(sizeof(double)*SUBTESTS*SUBTESTS);
	int c = 0;
	int pd = 0;
	for (int rc = 0; rc < SUBTESTS; rc++) {
		for (int cc = 0; cc < SUBTESTS; cc++) {
			if (rc == cc) {
				CORROUT[rc*SUBTESTS + cc] = 1.0;
				pd++;
			}
			else if (cc >= pd) {
				CORROUT[rc*SUBTESTS + cc] = CORROUT[cc*SUBTESTS + rc] = EST_CORRELATIONS[c * 2 + 0];
				c++;
			}
		}
	}

	showmatr(CORROUT, SUBTESTS, SUBTESTS, "CORRELATIONS");

	free(CORROUT);

	double *TEMP_AG = (double*)malloc(sizeof(double)*ITEMS);

	for (int ith = 0; ith < ITEMS; ith++) {
		TEMP_AG[ith] = EST_ITEMS[ith * 4 + 0];
	}
	printf("AS %40.30f\n", correlation((double*)AS, TEMP_AG, ITEMS));
	for (int ith = 0; ith < ITEMS; ith++) {
		TEMP_AG[ith] = EST_ITEMS[ith * 4 + 2];
	}
	printf("GS %40.30f\n", correlation((double*)GS, TEMP_AG, ITEMS));

	free(TEMP_AG);

	for (int subtest = 0; subtest < SUBTESTS; subtest++) {
		double *BS = (double*)malloc(sizeof(double)*PERSONS);
		double *CS = (double*)malloc(sizeof(double)*PERSONS);

		for (int ith = 0; ith < PERSONS; ith++) {
			BS[ith] = THS[ith*SUBTESTS + subtest];
			CS[ith] = EST_PERSONS[ith * 2 * SUBTESTS + subtest * 2];
		}
		printf("TH(%3d) %40.30f\n", subtest, correlation(BS, CS, PERSONS));

		free(BS);
		free(CS);
	}

	if (1) {
		printf("\n");
		double *CORROUT = (double*)malloc(sizeof(double)*(SUBTESTS*(SUBTESTS - 1)) / 2);
		double *CORRACT = (double*)malloc(sizeof(double)*(SUBTESTS*(SUBTESTS - 1)) / 2);

		for (int ith = 0; ith < (SUBTESTS*(SUBTESTS - 1)) / 2; ith++) {
			CORRACT[ith] = EST_CORRELATIONS[ith * 2 + 0];
		}
		int kth = 0;
		for (int ith = 0; ith < SUBTESTS; ith++) {
			for (int jth = ith + 1; jth < SUBTESTS; jth++, kth++) {
				CORROUT[kth] = CORRELATIONS[ith*SUBTESTS + jth];
				printf("%40.30f %40.30f %40.30f\n", CORROUT[kth], CORRACT[kth], CORROUT[kth] - CORRACT[kth]);
			}
		}
		free(CORROUT);
		free(CORRACT);
	}
}


void setup(int argc, char **argv, int* PERSONS, int* ITEMS, int* SUBTESTS, int* ITERATIONS, int* BURNIN, int* BATCH, int* PRIOR, int** SCORES, int** SUBTEST_COUNTS) {
	check_arguments(argc, argv);

	char* validation = validate_input(argv[2], PERSONS, ITEMS, SUBTESTS);

	*SCORES = (int*)malloc(PERSONS[0] * ITEMS[0] * sizeof(int));
	*SUBTEST_COUNTS = (int*)malloc(SUBTESTS[0] * sizeof(int));
	
	read_input(validation, *SCORES, *SUBTEST_COUNTS, PERSONS, ITEMS, SUBTESTS);

	*ITERATIONS = atoi(argv[3]);
	*BURNIN = atoi(argv[4]);
	*BATCH = atoi(argv[5]);
	*PRIOR = atoi(argv[6]);
}


void setup(int argc, char **argv, int *PERSONS, int *ITEMS, int *SUBTESTS, int *ITERATIONS, int *BURNIN, int *BATCH, int *PRIOR) {
	check_arguments(argc, argv);

	*PERSONS = atoi(argv[2]);
	*ITEMS = atoi(argv[3]);
	*SUBTESTS = atoi(argv[4]);
	*ITERATIONS = atoi(argv[5]);
	*BURNIN = atoi(argv[6]);
	*BATCH = atoi(argv[7]);
	*PRIOR = atoi(argv[8]);
}

void allocate(int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR, int** SCORES, int** SUBTEST_COUNTS, double** CORRELATIONS, double** AS, double** GS, double** THS, double** EST_ITEMS, double** EST_PERSONS, double** EST_CORRELATIONS) {

	*SCORES = (int*)malloc(PERSONS*ITEMS*sizeof(int));
	*SUBTEST_COUNTS = (int*)malloc(SUBTESTS*sizeof(int));
	*CORRELATIONS = (double*)malloc(SUBTESTS*SUBTESTS*sizeof(double));

	*AS = (double*)malloc(sizeof(double) * ITEMS);
	*GS = (double*)malloc(sizeof(double) * ITEMS);
	*THS = (double*)malloc(sizeof(double) * PERSONS * SUBTESTS);

	*EST_ITEMS = (double*)malloc(ITEMS * 4 * sizeof(double));
	*EST_PERSONS = (double*)malloc(PERSONS * 2 * SUBTESTS * sizeof(double));
	*EST_CORRELATIONS = (double*)malloc(((SUBTESTS * (SUBTESTS - 1)) / 2) * 2 * sizeof(double));
}

void allocate(int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR, double** CORRELATIONS, double** AS, double** GS, double** THS, double** EST_ITEMS, double** EST_PERSONS, double** EST_CORRELATIONS) {
	*CORRELATIONS = (double*)malloc(SUBTESTS*SUBTESTS*sizeof(double));

	*AS = (double*)malloc(sizeof(double) * ITEMS);
	*GS = (double*)malloc(sizeof(double) * ITEMS);
	*THS = (double*)malloc(sizeof(double) * PERSONS * SUBTESTS);

	*EST_ITEMS = (double*)malloc(ITEMS * 4 * sizeof(double));
	*EST_PERSONS = (double*)malloc(PERSONS * 2 * SUBTESTS * sizeof(double));
	*EST_CORRELATIONS = (double*)malloc(((SUBTESTS * (SUBTESTS - 1)) / 2) * 2 * sizeof(double));
}

void cleanup(int* SCORES, int* SUBTEST_COUNTS, double* CORRELATIONS, double* AS, double* GS, double* THS, double* EST_ITEMS, double* EST_PERSONS, double* EST_CORRELATIONS) {
	free(AS);
	free(GS);
	free(THS);
	free(EST_ITEMS);
	free(EST_PERSONS);
	free(EST_CORRELATIONS);

	free(SCORES);
	free(SUBTEST_COUNTS);
	free(CORRELATIONS);
}
