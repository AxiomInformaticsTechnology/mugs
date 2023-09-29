#include "stdafx.cuh"
#include "showdata.cuh"

#define stddoc stdout

void showstr(int len, char *str, char *name)
{
	fprintf(stddoc,"SHOW STRING %s %*.*s\n",name,len,len,str);
}

void showmat(int *matrix, int xth, int yth, char *name)
{
	fprintf(stddoc,"SHOW MATRIX %s[%d][%d]\n", name, xth, yth);

	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",xth*yth);			
		fprintf(stddoc,"%8d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int ith=0;ith<xth;ith++)
	{
		for(int jth=0;jth<yth;jth++)
		{
			if (jth==0) fprintf(stddoc,"%8d : ", ith);
			fprintf(stddoc,"%8d ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc,"\n");
	}
	fprintf(stddoc,"\n");
}

void showmatt(int *matrix, int xth, int yth, char *name)
{
	fprintf(stddoc,"SHOW MATRIX %s[%d][%d]\n", name, xth, yth);

	for(int jth=0;jth<xth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",xth*yth);

		fprintf(stddoc,"%8d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int jth=0;jth<yth;jth++)
	{
		for(int ith=0;ith<xth;ith++)
		{
			if (ith==0) fprintf(stddoc,"%8d : ", jth);
			fprintf(stddoc,"%8d ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc,"\n");
	}
	fprintf(stddoc,"\n");
}

void showmatr(double *matrix, int xth, int yth, char *name)
{
	fprintf(stddoc,"SHOW MATRIX %s[%d][%d]\n", name, xth, yth);
	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",xth*yth);

		fprintf(stddoc,"%40d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int ith = 0; ith < xth; ith++)
	{
		for(int jth = 0; jth < yth; jth++)
		{
			if (jth==0) fprintf(stddoc,"%8d : ", ith);
			fprintf(stddoc,"%40.30f ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc,"\n");
	}
	fprintf(stddoc,"\n");
}


void showmatrt(double *matrix, int xth, int yth, char *name)
{
	fprintf(stddoc,"SHOW MATRIX %s[%d][%d]\n", name, xth, yth);
	for(int jth=0;jth<xth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",xth*yth);

		fprintf(stddoc,"%40d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int jth = 0; jth < yth; jth++)
	{
		for(int ith = 0; ith < xth; ith++)
		{
			if (ith == 0) fprintf(stddoc, "%8d : ", jth);
			fprintf(stddoc,"%40.30f ", matrix[ith*yth + jth]);
		}
		fprintf(stddoc,"\n");
	}
	fprintf(stddoc,"\n");
}

void showvec(double *vector, int yth, char *name)
{ 
	fprintf(stddoc,"SHOW VECTOR %s[%d]\n", name, yth);
	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",yth);

		fprintf(stddoc,"%40d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ", yth);
		fprintf(stddoc,"%40.30f ", vector[jth]);
	}
	fprintf(stddoc,"\n");
}

void showint(int var, char *name)
{
	fprintf(stderr,"SHOWINT %3.3s=%d\n", name, var);
}
void showint_(int* var, char *name)
{
	fprintf(stderr,"SHOWINT %3.3s=%d\n", name,* var);
}

void showreal(double var, char *name)
{
	fprintf(stddoc,"SHOWREAL %s=" "%40.30f " "\n",name, var);
}
void showreal_(double* var, char *name)
{
	fprintf(stderr,"SHOWREAL %3.3s=" "%40.30f " "\n",name, *var);
}

void showdouble_(double *var, char *name)
{
	fprintf(stderr,"SHOWDOUBLE %4.4s="  "%40.30f "  "\n",name , *var);
}

void showdouble(double var, char *name)
{
	fprintf(stddoc,"SHOWDOUBLE %s="  "%40.30f "  "\n",name , var);
}

void showveci(int*vector, int yth, char *name)
{ 
	fprintf(stddoc,"SHOW VECTOR %s[%d]\n", name, yth);
	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",yth);

		fprintf(stddoc,"%16d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ", yth);
		fprintf(stddoc,"%16d ", vector[jth]);
	}
	fprintf(stddoc,"\n");
}


void showmat_(int *matrix, int *xP, int *yP, char *name)
{
	int xth=*xP;
	int yth=*yP;

	fprintf(stddoc,"SHOW MATRIX %4.4s[%d][%d]\n", name, xth, yth);

	for(int ith=0;ith<xth;ith++)
	{
		for(int jth=0;jth<yth;jth++)
		{
			fprintf(stddoc,"%d ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc,"\n");
	}
	fprintf(stddoc,"\n");
}

void showmatr_(double *matrix, int*  pxth, int*pyth, char *name)
{
	int xth = *pxth;
	int yth = *pyth;

	fprintf(stddoc,"SHOW MATRIX %4.4s[%d][%d]\n", name, xth, yth);
	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",xth*yth);

		fprintf(stddoc,"%40d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int ith=0;ith<xth;ith++)
	{
		for(int jth=0;jth<yth;jth++)
		{
			if (jth==0) fprintf(stddoc,"%8d : ", ith);
			fprintf(stddoc,"%40.30f ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc,"\n");
	}
	fprintf(stddoc,"\n");
}

void showmattm_(int *matrix, int *xP, int *yP, int *xM, int *yM, char *name)
{
	int xth=*xP;
	int yth=*yP;
	int xmh=*xM;
	int ymh=*yM;

	//	fprintf(stddoc,"SHOW MATRIX %4.4s[%d][%d]\n", name, xth, yth);
	//  fprintf(stddoc,"Y[%d][%d] = \n{ \n",xmh,ymh);
	for(int jth=0;jth<yth&&jth<ymh;jth++)
	{
		fprintf(stddoc,"{ ");
		for(int ith=0;ith<xth&&ith<xmh;ith++)
		{
			fprintf(stddoc,"%d, ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc," },\n");
	}
	//  fprintf(stddoc,"\n };\n");
}

void showmatt_(int *matrix, int *xP, int *yP, char *name)
{
	int xth=*xP;
	int yth=*yP;

	fprintf(stddoc,"SHOW MATRIX %4.4s[%d][%d]\n", name, xth, yth);
	fprintf(stddoc,"Y[xth][yth] = \n{ ");
	for(int jth=0;jth<yth;jth++)
	{
		fprintf(stddoc,"{ ");
		for(int ith=0;ith<xth;ith++)
		{
			fprintf(stddoc,"%d, ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc," },\n");
	}
	fprintf(stddoc," };\n");
}

void showmatrt_(double *matrix, int*  pxth, int*pyth, char *name)
{
	int xth = *pxth;
	int yth = *pyth;
	fprintf(stddoc,"SHOW MATRIX %4.4s[%d][%d]\n", name, xth, yth);
	for(int jth=0;jth<yth;jth++)
	{
		if (jth==0) fprintf(stddoc,"%8d : ",xth*yth);

		fprintf(stddoc,"%40d ", jth);
	}
	fprintf(stddoc,"\n");
	for(int jth=0;jth<yth;jth++)
	{
		for(int ith=0;ith<xth;ith++)
		{
			if (ith==0) fprintf(stddoc,"%8d : ", jth);
			fprintf(stddoc,"%40.30f ", matrix[ith*yth+jth]);
		}
		fprintf(stddoc,"\n");
	}
	fprintf(stddoc,"\n");
}

void showmatrtm_(double *matrix, int *xP, int *yP, int *xM, int *yM, char *name)
{
	int xth=*xP;
	int yth=*yP;
	int xmh=*xM;
	int ymh=*yM;
	fprintf(stderr,"SHOW MATRIX %4.4s[%d][%d]\n", name, xth, yth);
	for(int jth=0;jth<yth&&jth<ymh;jth++)
	{
		if (jth==0) fprintf(stderr,"%8d : ",xth*yth);

		fprintf(stderr,"%40d ", jth);
	}
	fprintf(stderr,"\n");
	for(int jth=0;jth<yth&&jth<ymh;jth++)
	{
		for(int ith=0;ith<xth&&ith<xmh;ith++)
		{
			if (ith==0) fprintf(stderr,"%8d : ", jth);
			fprintf(stderr,"%40.30f ", matrix[ith*yth+jth]);
		}
		fprintf(stderr,"\n");
	}
	fprintf(stderr,"\n");
}

void showmatrm_(double *matrix, int *xP, int *yP, int *xM, int *yM, char *name)
{
	int xth=*xP;
	int yth=*yP;
	int xmh=*xM;
	int ymh=*yM;

	fprintf(stderr,"SHOW MATRIX %4.4s[%d][%d]\n", name, xth, yth);
	for(int jth=0;jth<yth&&jth<ymh;jth++)
	{
		if (jth==0) fprintf(stderr,"%8d : ",xth*yth);

		fprintf(stderr,"%40d ", jth);
	}
	fprintf(stderr,"\n");
	for(int ith=0;ith<xth&&ith<xmh;ith++)
	{
		for(int jth=0;jth<yth&&jth<ymh;jth++)
		{
			if (jth==0) fprintf(stderr,"%8d : ", ith);
			fprintf(stderr,"%40.30f ", matrix[ith*yth+jth]);
		}
		fprintf(stderr,"\n");
	}
	fprintf(stderr,"\n");
}
