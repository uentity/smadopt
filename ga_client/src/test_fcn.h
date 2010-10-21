/*-----------------------------------------------------------------
 * Test functions declaration
 *----------------------------------------------------------------*/

//pointer to test function
typedef void (*test_fun_callback)(int, int, const double*, double*);

// globally defined constants
extern int vspNum;
extern double deliver_mult;

/*-----------------------------------------------------------------
 * Well-known test functions
 *----------------------------------------------------------------*/
void RastriginsFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void SphereFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void SphereFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void SphereMultiModFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void RosenbrocksFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void SchwefelFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void AckleyFcn(int nVars, int nPopSize, const double* pPop, double* pScore);

enum test_fun_t {
	rastrigins = 1,
	rosenbrocks = 2,
	sphere = 3,
	sphere_multimod = 4,
	polynom = 5,
	VP = 6,
	schwefel = 7,
	ackley = 8
};

/*-----------------------------------------------------------------
 * Vertical subpops manipulation
 *----------------------------------------------------------------*/
void VspWrapperSum(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f);
void VspWrapperOverlap(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f);
void VspWrapperDep(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f);
void VspWrapperMultInfl(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f);
void PolynomFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void TestVPFcn(int nVars, int nPopSize, const double* pPop, double* pScore);

extern void (*VspWrapper)(int, int, const double*, double*, test_fun_callback);

/*-----------------------------------------------------------------
 * Test functions with VSP
 *----------------------------------------------------------------*/
void RastriginsVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void RosenbrocksVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void SchwefelVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void AckleyVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore);
void SphereMultiModVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore);

