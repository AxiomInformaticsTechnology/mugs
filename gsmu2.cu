#include "gsmu2.cuh"

#include "mathafx.cuh"
#include "number.cuh"
#include "randafx.cuh"
#include "showdata.cuh"
#include "stdafx.cuh"
#include "utilafx.cuh"

// TODO: make struct out of inputs
void gsmu2(int *Y, 
		   int *subtest_counts,
		   int person_count,
		   int item_count,
		   int total_subtest_count,
		   int iteration_count,
		   int burnin,
		   int batches,
		   int uniform,
		   double *ITEM,
		   double *PERSON,
		   double *CORR) {

	srand(rand());

	double LP, BB, U, TMP, SUM;

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

	double *A = (double*)malloc(items_by_szDbl);
	double *G = (double*)malloc(items_by_szDbl);
	double *TH = (double*)malloc(persons_by_subtests_by_szDbl);
	double *SIGMA = (double*)malloc(subtests_by_subtests_by_szDbl);
	double *AV = (double*)malloc(items_by_szDbl * 3);
	double *GV = (double*)malloc(items_by_szDbl * 3);
	double *THV = (double*)malloc(persons_by_subtests_by_szDbl * 3);
	double *RHO = (double*)malloc(subtests_by_subtests_by_szDbl * 3);
	double *Z = (double*)malloc(persons_by_items_by_szDbl);

	double *AA = (double *)malloc(items_by_subtests_by_szDbl);
	double *AAT = (double *)malloc(items_by_subtests_by_szDbl);
	double *AATS = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *RSIG = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *PVAR = (double *)malloc(subtests_by_subtests_by_szDbl);
	double *PVARI = (double *)malloc(subtests_by_subtests_by_szDbl);

	double *BA = (double*)malloc(items_by_szDbl);
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


	double *CHF = (double *)malloc(subtests_by_subtests_by_szDbl);


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
	for (int item = 0; item < item_count; item++) {
		
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

	// BEGIN ITERATE
	for (int iteration = 0; iteration < iteration_count; iteration++) {

		// UPDATE Z

		for (int person = 0; person < person_count; person++) {
			for (int item = 0; item < item_count; item++) {
				LP = G[item] - A[item] * TH[person*total_subtest_count + ord[item]];
				BB = stdnormal_cdf(LP);
				U = random_uniform_pos();
				TMP = Y[person*item_count + item] == 0 ? (BB * U) : (U + BB * (1.0 - U));
				TMP = stdnormal_inv(TMP) - LP;
				Z[person*item_count + item] = (double) isfinite(TMP) ? TMP : 0.0;
			}
		}


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

		// UPDATE TH

		for (int person = 0; person < person_count; person++) {

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


		// UPDATE AG
		#pragma omp parallel for num_threads(total_subtest_count)
		for (int subtest = 0; subtest < total_subtest_count; subtest++) {
			Subtest(person_count, item_count, total_subtest_count, uniform, subtest, subtest_counts, Z, TH, C, A, G);
		}

		

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

	free(PMEAN1);

	free(CTH);
	free(CTHT);
	free(DI);
	free(WZ);
	free(AZ);
	free(ZVAR);
	free(D);
	free(VAR);
	free(C);


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



void Subtest(int person_count, int item_count, int total_subtest_count, int unif_flag, int subtest, int *subtest_counts, double *Z, double *TH, double *C, double *A, double *G)
{
	int subitem = 0;
	for (int ith = subtest; ith > 0; ith--) {
		subitem += subtest_counts[ith - 1];
	}

	double det_A, A_00, L_00, A_10, A_11, L_10, diag, L_11;

	double SIGMA2[2][2];
	double XX[2][2];
	double IX[2][2];
	double AGMU[2][1];
	double AMAT[2][2];

	//double PRODA = 1.0;
	double PRODA = 0.0;
	double fp = 1.0 / (double) subtest_counts[subtest];

	XX[0][0] = XX[1][0] = 0.0;
	for (int person = 0; person < person_count; person++) {
		double temp_TH = TH[person*total_subtest_count + subtest];
		XX[0][0] += temp_TH * temp_TH;
		XX[1][0] -= temp_TH;
	}

	XX[1][1] = (double)person_count;
	XX[0][1] = XX[1][0];

	double XZKET[2];

	if (unif_flag == 0)
	{
		double  AGVAR[2][2];
		AGMU[0][0] = AMU_I;
		AGMU[1][0] = GMU_I;
		AGVAR[0][0] = AVAR_I;
		AGVAR[1][1] = GVAR_I;

		det_A = AGVAR[0][0] * AGVAR[1][1] - AGVAR[1][0] * AGVAR[0][1];
		SIGMA2[0][0] = AGVAR[1][1] / det_A;
		SIGMA2[0][1] = 0.0;
		SIGMA2[1][0] = 0.0;
		SIGMA2[1][1] = AGVAR[0][0] / det_A;

		XZKET[0] = (SIGMA2[0][0] * AGMU[0][0]) + (SIGMA2[0][1] * AGMU[1][0]);
		XZKET[1] = (SIGMA2[1][0] * AGMU[0][0]) + (SIGMA2[1][1] * AGMU[1][0]);

		XX[0][0] += SIGMA2[0][0];
		XX[0][1] += SIGMA2[0][1];
		XX[1][0] += SIGMA2[1][0];
		XX[1][1] += SIGMA2[1][1];
	}

	det_A = XX[0][0] * XX[1][1] - XX[1][0] * XX[0][1];
	IX[0][0] =  XX[1][1] / det_A;
	IX[0][1] = -XX[0][1] / det_A;
	IX[1][0] = -XX[1][0] / det_A;
	IX[1][1] =  XX[0][0] / det_A;

	A_00 = IX[0][0],
	L_00 = sqrt(A_00),
	A_10 = IX[1][0],
	A_11 = IX[1][1],
	L_10 = A_10 / L_00,
	diag = A_11 - L_10 * L_10,
	L_11 = sqrt(diag);

	AMAT[0][0] = L_00;
	AMAT[1][0] = L_10;
	AMAT[0][1] = 0.0;
	AMAT[1][1] = L_11;

	for (int subtest_item = 0; subtest_item < subtest_counts[subtest]; subtest_item++, subitem++) {

		double BETA[2];
		double BZ[2];
		double XZ[2];

		

		double sum1, sum2; sum1 = 0.0; sum2 = 0.0;
		for (int person = 0; person < person_count; person++) {


			//if (subtest == 0) {
			//	printf("%40.30f ", Z[person*item_count + subitem]);
			//}

			double temp_Z = Z[person*item_count + subitem];
			sum1 += TH[person*total_subtest_count + subtest] * temp_Z;
			sum2 -= temp_Z;
		}

		//if (subtest == 0) {
		//	printf("\n");
		//}

		XZ[0] = sum1;
		XZ[1] = sum2;


		if (unif_flag == 0)
		{
			XZ[0] += XZKET[0];
			XZ[1] += XZKET[1];
		}

		BZ[0] = (IX[0][0] * XZ[0]) + (IX[0][1] * XZ[1]);
		BZ[1] = (IX[1][0] * XZ[0]) + (IX[1][1] * XZ[1]);

		int tick = 0;

		A[subitem] = 0;
		do {
			rnnor(2, BETA);

			A[subitem] = ((BETA[0] * AMAT[0][0]) + (BETA[0] * AMAT[1][0])) + BZ[0];

			tick++;

			if (A[subitem] > 0.0) {
				G[subitem] = ((BETA[1] * AMAT[0][1]) + (BETA[1] * AMAT[1][1])) + BZ[1];
				break;
			}

			if (tick >= MAXTICKS) {

				showmatr((double*)&BZ, 2, 1, "BZ");
				showmatr((double*)&AMAT, 2, 2, "AMAT");
				showmatr((double*)&BETA, 2, 1, "BETA");
				showmatr((double*)&IX, 2, 2, "IX");
				showmatr((double*)&XX, 2, 2, "XX");
				showmatr((double*)&XZ, 2, 1, "XZ");


				exit(-23);
			}
		} while (true);

		//PRODA *= A[subitem];
		PRODA += log(A[subitem]);
	}
	
	for (int subtest0 = 0; subtest0 < total_subtest_count; subtest0++) {
		C[subtest0*total_subtest_count + subtest] = 0.0;
	}

	//C[subtest*total_subtest_count+subtest] = (double) pow((double) PRODA, fp);
	C[subtest*total_subtest_count + subtest] = (double)pow((double)exp(PRODA), fp);
}


