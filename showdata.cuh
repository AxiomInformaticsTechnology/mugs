//#include "number.cuh"

void showmat(int* matrix,int xth,int yth,char* name);
void showmatt(int* matrix,int xth,int yth,char* name);
void showmatr(double* matrix,int xth,int yth,char* name);
void showmatrt(double* matrix,int xth,int yth,char* name);
void showvec(double* vector, int xth, char* name);
void showvecd(void* matrix, int xth, char* name);
void showveci(int* matrix, int xth, char* name);
void showint(int var, char* name);
void showreal(double var, char* name);
void showdouble(double var, char* name);
void showveci(int* matrix, int xth, char* name);
void showstr(int len, char* str, char* name);

void showmat_(int* matrix, int* xth, int* yth, char* name);
void showmatr_(double* matrix, int* xth, int* yth, char* name);
void showmatrt_(double* matrix, int* xth, int* yth, char* name);
void showvec_(void* vector, int* xth, char* name);
void showvecd_(void* matrix, int* xth, char* name);
void showveci_(void* matrix, int* xth, char* name);
void showint_(int* var, char* name);
void showreal_(double* var, char* name);
void showdouble_(double* var, char* name);

#define SHOWSTR(STRING,NAME) showstr((int)strlen(STRING),STRING, " " #NAME);
#define SHOWVEC(VECTOR) showvec(VECTOR,sizeof(VECTOR)/sizeof(VECTOR[0])," " #VECTOR);
#define SHOWMATR(MATRIX) showmatr(MATRIX,sizeof(MATRIX)/sizeof(MATRIX[0]),sizeof(MATRIX[0])/sizeof(MATRIX[0][0])," " #MATRIX);
#define SHOWMATRT(MATRIX) showmatrt(MATRIX,sizeof(MATRIX)/sizeof(MATRIX[0]),sizeof(MATRIX[0])/sizeof(MATRIX[0][0])," " #MATRIX);