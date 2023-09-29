double  ltqnorm(double p);
double  stdnormal_cdf(double u);
double  stdnormal_inv(double p);
double  stdnormal_pdf(double u);

double  random_uniform_pos();
double  random_uniform();
double  random_normal();

unsigned int random_bernoulli(double p);
double random_gaussian_tail(const double a, const double sigma);

void rnbin(int NR, int N, double P, unsigned int *IR);
void rnbin_(int*pNR, int*pN, double *pP, unsigned int *IR);
void random_bernoulli(int NR, int N, double P, unsigned int *IR);

void rnunp(int NR, double *IR);
void rnun_(int*pNR, double *IR);
void rnunp_(int*pNR, double *IR);
void rnnor_(int *NRp, double *IR);
void rnnor(int NR, double *IR);
void rnnorinv(int NR, double *IR);
void rnnor1(int NR, double *IR);
void rnnor1_(int *NR, double *IR);

double anorin_(double *ths);
double dnorin_(double *ths);
double anordf_(double *ths);

void normal_random_array(int P, int V, double*Z);
void uniform_random_array(int P, int V, double*Z);


