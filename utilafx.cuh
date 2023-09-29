#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

char* validate_input(char* files, int *persons, int *items, int *subtests);

void read_input(char* files, int* scores, int* subtest_counts, int *persons, int *items, int *subtests);

void check_arguments(int argc, char** args);

void args_numbers(int s, int argc, char** args);

bool one_or_zero(char* d);

int line_item_count(char* line);


void setup(int argc, char **argv, int* PERSONS, int* ITEMS, int* SUBTESTS, int* ITERATIONS, int* BURNIN, int* BATCH, int* PRIOR, int** SCORES, int** SUBTEST_COUNTS);

void allocate(int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR, double** CORRELATIONS, double** AS, double** GS, double** THS, double** EST_ITEMS, double** EST_PERSONS, double** EST_CORRELATIONS);

void show_inputs(int* SCORES, int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR);

void show_outputs(double* EST_ITEMS, double* EST_PERSONS, double* EST_CORRELATIONS, int PERSONS, int ITEMS, int SUBTESTS);


void setup(int argc, char **argv, int *PERSONS, int *ITEMS, int *SUBTESTS, int *ITERATIONS, int *BURNIN, int *BATCH, int *PRIOR);

void allocate(int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR, int** SCORES, int** SUBTEST_COUNTS, double** CORRELATIONS, double** AS, double** GS, double** THS, double** EST_ITEMS, double** EST_PERSONS, double** EST_CORRELATIONS);

void show_inputs(int* SCORES, double* CORRELATIONS, double* AS, double* GS, double *THS, int PERSONS, int ITEMS, int SUBTESTS, int ITERATIONS, int BURNIN, int BATCH, int PRIOR);

void show_outputs(double* EST_ITEMS, double* EST_PERSONS, double* EST_CORRELATIONS, double* CORRELATIONS, double* AS, double* GS, double *THS, int PERSONS, int ITEMS, int SUBTESTS);

void cleanup(int* SCORES, int* SUBTEST_COUNTS, double* CORRELATIONS, double* AS, double* GS, double* THS, double* EST_ITEMS, double *EST_PERSONS, double* EST_CORRELATIONS);