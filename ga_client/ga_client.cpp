// ga_client_win.cpp : Defines the entry point for the console application.
//

#include "ga.h"
#include "matrix.h"
#include "objnet.h"
#include "alg_api.h"
#include "kmeans.h"

#include "kar2words.h"

#include <cmath>
#include <time.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

#define PI 3.1415926535897932384626433832795

using namespace std;
using namespace GA;
using namespace NN;
using namespace KM;

int vspNum = 0;
double deliver_mult = 100;
const double infl_factor = 0.2;

//pointer to test function
typedef void (*test_fun_callback)(int, int, const double*, double*);

void RastriginsFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	//Sleep(10);
	//scores = 10.0 * size(pop,2) + sum(pop .^2 - 10.0 * cos(2 * pi .* pop),2);
	const double a = nVars*10;
	double dSum;
	const double* pTek = pPop;
	for(int i=0; i<nPopSize; ++i) {
		dSum = 0;
		for(int j=0; j<nVars; ++j) {
			dSum += (*pTek)*(*pTek) - 10*std::cos(2*PI*(*pTek));
			++pTek;
		}
		pScore[i] = a + dSum;
	}
}

void SphereFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	//Sleep(10);
	//scores = sum(xi*xi);
	const double* pTek = pPop;
	for(int i=0; i<nPopSize; ++i) {
		pScore[i] = 0;
		for(int j=0; j<nVars; ++j) {
			pScore[i] += (*pTek)*(*pTek);
			++pTek;
		}
	}
}

void SphereMultiModFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	//Sleep(10);
	//scores = sum(xi*xi + 10*(1 - cos(pi*xi));
	double dSum;
	const double* pTek = pPop;
	for(int i=0; i<nPopSize; ++i) {
		dSum = 0;
		for(int j=0; j<nVars; ++j) {
			dSum += (*pTek)*(*pTek) + 10*(1 - std::cos(PI*(*pTek)));
			++pTek;
		}
		pScore[i] = dSum;
	}
}

void RosenbrocksFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	double dSum, dTmp;
	const double* pTek = pPop;
	for(int i=0; i<nPopSize; ++i) {
		dSum = 0;
		for(int j=0; j < nVars - 1; ++j) {
			dTmp = (*pTek)*(*pTek);
			dSum += 100*(dTmp - *(pTek + 1))*(dTmp - *(pTek + 1)) + (1 - dTmp)*(1 - dTmp);
			++pTek;
		}
		++pTek;
		pScore[i] = dSum;
	}
}

void SchwefelFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	const double a = 418.9829 * nVars;
	double dSum;
	const double* pTek = pPop;
	for(int i=0; i<nPopSize; ++i) {
		dSum = 0;
		for(int j=0; j<nVars; ++j) {
			dSum -= (*pTek) * std::sin(sqrt(fabs(*pTek)));
			++pTek;
		}
		pScore[i] = a + dSum;
	}
}

void AckleyFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	const double a = 20 + exp(1.);
	const double mult = 1/(double)nVars;
	double dSum, dSum1;
	const double* pTek = pPop;
	for(int i=0; i<nPopSize; ++i) {
		dSum = 0; dSum1 = 0;
		for(int j=0; j<nVars; ++j) {
			dSum += mult * (*pTek) * (*pTek);
			dSum1 += cos(2*PI*(*pTek));
			++pTek;
		}
		pScore[i] = a - 20*exp(-0.2*sqrt(dSum)) - exp(mult*dSum1);
	}
}

void VspWrapperSum(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f)
{
	const int nspVars = nVars / vspNum;
	Matrix pop(nPopSize, nVars, pPop);
	Matrix score(vspNum, nPopSize);
	//Matrix curScore(nPopSize, 1);
	//Matrix res(nPopSize, 1);
	//res = 0;
	ulong offs = 0, score_offs = 0;
	for(ulong i = 0; i<vspNum; ++i) {
		(*f)(nspVars, nPopSize, &(pop.GetColumns(offs, nspVars) - deliver_mult*i)[0], &score[score_offs]);
		//score.SetColumns(curScore, i);
		//res += curScore;
		offs += nspVars;
		score_offs += nPopSize;
	}
	score &= score.vSum(false);
	score <<= !score;
	//score.SetColumns(res, vspNum);
	memcpy(pScore, &score[0], score.raw_size());
}

void VspWrapperOverlap(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f)
{
	int nspVars = nVars / vspNum;
	const int overlap = nspVars * 0.2;
	Matrix pop(nPopSize, nVars, pPop);
	Matrix score(nPopSize, vspNum + 1);
	Matrix curScore(nPopSize, 1);
	Matrix res(nPopSize, 1);
	res = 0;
	int offs = 0;
	ulong subpop_sz;
	for(ulong i = 0; i < vspNum; ++i) {
		subpop_sz = min(nVars + overlap, nVars - offs);
		(*f)(subpop_sz, nPopSize, &(pop.GetColumns(offs, subpop_sz) - deliver_mult*i)[0], &curScore[0]);
		score.SetColumns(curScore, i);
		res += curScore;
		offs += nspVars;
	}
	score.SetColumns(res, vspNum);
	memcpy(pScore, &score[0], score.raw_size());
}

void VspWrapperDep(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f)
{
	int nspVars = nVars / vspNum;
	Matrix pop(nPopSize, nVars, pPop);
	Matrix score(nPopSize, vspNum + 1);
	Matrix curScore(nPopSize, 1);
	Matrix res(nPopSize, 1);
	Matrix subpop, dep_col;
	res = 0;

	int offs = 0;
	//calc average X over VSP
	for(ulong i = 0; i < vspNum; ++i) {
		//subpop = pop; subpop.DelColumns(offs, nspVars);
		//dep_col |= subpop.vMean(true);
		dep_col |= pop.GetColumns(offs, nspVars).vMean(true) * infl_factor;
		offs += nspVars;
	}
	offs = 0;
	//calc objectives
	for(ulong i = 0; i < vspNum; ++i) {
		subpop <<= pop.GetColumns(offs, nspVars) - deliver_mult*i;
		//subpop |= dep_col.GetColumns(i);
		for(ulong j = 0; j < vspNum; ++j)
			if(j != i) subpop |= dep_col.GetColumns(j);

		(*f)(subpop.col_num(), nPopSize, &subpop[0], &curScore[0]);
		score.SetColumns(curScore, i);
		res += curScore;
		offs += nspVars;
	}
	score.SetColumns(res, vspNum);
	memcpy(pScore, &score[0], score.raw_size());
}

void VspWrapperMultInfl(int nVars, int nPopSize, const double* pPop, double* pScore, test_fun_callback f)
{
	const int nspVars = nVars / vspNum;
	Matrix pop(nPopSize, nVars, pPop);
	Matrix score(vspNum, nPopSize);
	Matrix curScore(nPopSize, 1), tmp(nPopSize, 1);
	Matrix res(nPopSize, 1);

	ulong offs = 0, score_offs = 0;
	for(ulong i = 0; i<vspNum; ++i) {
		(*f)(nspVars, nPopSize, &(pop.GetColumns(offs, nspVars) - deliver_mult*i)[0], &curScore[score_offs]);
		offs += nspVars;
		score_offs += nPopSize;
	}
	score <<= !score;

	//tmp = score;
	//transform(tmp.begin(), tmp.end(), tmp.begin(), bind1st(divides<double>(), infl_factor));

	res = 1;
	for(ulong i = 0; i < score.col_num(); ++i) {
		//curScore = tmp; curScore.DelColumns(i);
		//curScore <<= score.GetColumns(i) + curScore.vSum(true);
		//score.SetColumns(curScore, i);

		curScore <<= score.GetColumns(i);
		offs = 0;
		for(ulong j = 0; j < vspNum; ++j) {
			if(j != i) {
				(*f)(nspVars, nPopSize, &pop.GetColumns(offs, nspVars)[0], &tmp[0]);
				//tmp <<= score.GetColumns(j);
				//tmp *= infl_factor;
				transform(tmp.begin(), tmp.end(), tmp.begin(), bind1st(divides<double>(), infl_factor));

				//curScore += tmp * infl_factor;
			
				curScore += tmp.GetColumns(j);
			}
			offs += nspVars;
		}
		score.SetColumns(curScore, i);
		//if(i == 1)
		//	res /= curScore;
		//else 
			res *= curScore;
	}

	score |= res;
	memcpy(pScore, &score[0], score.raw_size());
}

void PolynomFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	static Matrix g_a, mul_coeffs, add_coeffs;
	int nspVars = nVars / vspNum;
	Matrix pop(nPopSize, nVars, pPop);
	Matrix score(nPopSize, vspNum + 1);
	Matrix curScore(1, vspNum + 1);
	//Matrix res(nPopSize, 1);
	Matrix chrom, pol, pol_row, a_row;
	//read coeffs
	if(g_a.size() == 0) {
		ifstream fsrc("a.txt");
		g_a = Matrix::Read(fsrc);
		fsrc.close();
		mul_coeffs = g_a.GetRows(g_a.row_num() - 2);
		add_coeffs = g_a.GetRows(g_a.row_num() - 1);
	}
	//extract coeffs in row
	for(ulong i=0; i<nVars; ++i)
		a_row |= g_a.GetRows(i).GetColumns(i, g_a.col_num() - i);
	
	int offs;
	double dSum;
	for(ulong i = 0; i<nPopSize; ++i) {
		//calc result for cur chrom
		chrom = pop.GetRows(i) + add_coeffs;
		pol = (!chrom)*chrom;
		//calc each subpop results
		offs = 0;
		for(ulong j=0; j<vspNum; ++j) {
			dSum = 0;
			for(ulong k=0; k<nspVars; ++k)
				dSum += pol.GetRows(offs + k).GetColumns(offs + k, nspVars - k).Mul(g_a.GetRows(offs + k).GetColumns(offs + k, nspVars - k)).Sum();
			dSum += chrom.GetColumns(offs, nspVars).Mul(mul_coeffs.GetColumns(offs, nspVars)).Sum();
			
			curScore[j] = dSum;
			offs += nspVars;
		}
		//calc overall result
		pol_row.NewMatrix(0, 0);
		for(ulong j=0; j<pol.row_num(); ++j)
			pol_row |= pol.GetRows(j).GetColumns(j, pol.col_num() - j);
		curScore[vspNum] = pol_row.Mul(a_row).Sum();
		curScore[vspNum] += chrom.Mul(mul_coeffs).Sum();
		score.SetRows(curScore, i);
	}
	memcpy(pScore, &score[0], score.raw_size());
}

void TestVPFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	//Sleep(10);
	//scores = 10.0 * size(pop,2) + sum(pop .^2 - 10.0 * cos(2 * pi .* pop),2);
	double dSum, dT;
	const double* pTek = pPop;
	double* pSc = pScore;
	for(int i=0; i<nPopSize; ++i) {
		dSum = 0;
		*pSc = 0;
		for(int j=0; j<min(3, nVars); ++j) {
			*pSc = *pSc + *(pTek++);
		}
		*pSc = (*pSc - 14)*(*pSc - 14);
		dSum += *pSc;
		*(++pSc) = 0;
		for(int j=3; j<min(6, nVars); ++j) {
			*pSc = *pSc + *(pTek++);
		}
		*pSc = (*pSc - 8)*(*pSc - 8);
		dSum += *pSc;
		*(++pSc) = 0;
		for(int j=6; j<nVars; ++j) {
			*pSc = *pSc + (*pTek++);
		}
		*pSc = (*pSc - 4)*(*pSc - 4);
		dSum += *pSc;
		*(++pSc) = dSum;
		++pSc;
	}
	ulong nNum = pSc - pScore;
}

void (*VspWrapper)(int, int, const double*, double*, test_fun_callback) = VspWrapperSum;

void RastriginsVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	(*VspWrapper)(nVars, nPopSize, pPop, pScore, &RastriginsFcn);
}

void RosenbrocksVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	(*VspWrapper)(nVars, nPopSize, pPop, pScore, &RosenbrocksFcn);
}

void SchwefelVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	(*VspWrapper)(nVars, nPopSize, pPop, pScore, &SchwefelFcn);
}

void AckleyVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	(*VspWrapper)(nVars, nPopSize, pPop, pScore, &AckleyFcn);
}

void SphereMultiModVspFcn(int nVars, int nPopSize, const double* pPop, double* pScore)
{
	(*VspWrapper)(nVars, nPopSize, pPop, pScore, &SphereMultiModFcn);
}

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

void vsp_correct_init_range(ga_opt& opt)
{
	//correct init range
	Matrix corr_ir(1, opt.vspSize.Sum());
	Matrix::r_iterator pos(corr_ir.begin());
	for(ulong i = 0; i < opt.vspSize.size(); ++i)
		for(ulong j = 0; j < opt.vspSize[i]; ++j) {
			*pos = deliver_mult*i;
			++pos;
		}
	opt.initRange <<= (corr_ir + opt.initRange(0, 0)) & (corr_ir + opt.initRange(1, 0));
	DumpMatrix(opt.initRange);
}

ga_stat TestFcnOpt(int nVars, test_fun_t tf = rastrigins, uint runs_num = 1, 
				   bool read_options = true, Matrix& full_best = Matrix())
{
	//int nVars = 5;
	ulong popSize; //= 100;

	test_fun_callback test_fun;
	GA::ga& ga_obj = *(GA::ga*)GetGAObject();

	if(read_options) ReadOptions();
	gaOptions* pOpt = (gaOptions*)GetGAOptions();
	popSize = pOpt->popSize;
	vspNum = ga_obj.opt_.vspSize.size();
	if(pOpt->subpopT == Vertical)
		nVars = ga_obj.opt_.vspSize.Sum();

	Matrix res(1, nVars);
	Matrix pop(popSize, nVars);
	Matrix score(popSize, vspNum + 1);

	if(vspNum == 0) {
		switch(tf) {
			default:
			case rastrigins:
				test_fun = &RastriginsFcn;
				break;
			case rosenbrocks:
				test_fun = &RosenbrocksFcn;
				break;
			case sphere:
				test_fun = &SphereFcn;
				break;
			case sphere_multimod:
				test_fun = &SphereMultiModFcn;
				break;
			case schwefel:
				test_fun = &SchwefelFcn;
				break;
			case ackley:
				test_fun = &AckleyFcn;
				break;
		}
	}
	else {
		switch(tf) {
			default:
			case rastrigins:
				test_fun = &RastriginsVspFcn;
				break;
			case rosenbrocks:
				test_fun = &RosenbrocksVspFcn;
				break;
			case polynom:
				test_fun = &PolynomFcn;
				break;
			case VP:
				test_fun = &TestVPFcn;
				break;
			case schwefel:
				test_fun = &SchwefelVspFcn;
				break;
			case ackley:
				test_fun = &AckleyVspFcn;
				break;
			case sphere_multimod:
				test_fun = &SphereMultiModVspFcn;
				break;
		}
		//correct init range
		if(read_options) 
			vsp_correct_init_range(ga_obj.opt_);
	}

	clock_t start = clock();
	//Run(RastriginsVspFcn, nVars, res.GetBuffer());

	bool bContinue;
	ga_stat stat;
	const ga_stat* cur_stat;
	for(ulong i = 0; i < runs_num; ++i) {
		Start(pop.GetBuffer(), nVars, false);
		do {
			//calc objective function values
			(*test_fun)(nVars, popSize, pop.GetBuffer(), score.GetBuffer());

			if(pOpt->subpopT == Vertical)
				bContinue = GetNextPop(score.GetBuffer(), pop.GetBuffer(), &popSize);
			else
				bContinue = GetNextPop(score.GetColumns(vspNum).GetBuffer(), pop.GetBuffer(), &popSize);
		} while(bContinue);

		const ga_stat& cur_stat = ga_obj.GetStatistics();
		stat += cur_stat;
		full_best |= cur_stat.best_ff_;
	}
	stat /= runs_num;

	clock_t finish = clock();
	std::cout << "Time elapsed: " << (double)(finish - start)/CLOCKS_PER_SEC << " seconds" << endl;
	return stat;
}

void vsp_test_mig_rates(double start = 0.01, double incr = 0.2, double end = 1.)
{
	GA::ga& ga_obj = *(GA::ga*)GetGAObject();
	ReadOptions();
	const int nVars = ga_obj.opt_.vspSize.Sum();
	//correct init range
	vsp_correct_init_range(ga_obj.opt_);

#ifdef _DEBUG
	const int run_num = 1;
	ga_obj.opt_.generations = 100;
#else
	const int run_num = 10;
#endif

	test_fun_t ft = ackley;
	VspWrapper = &VspWrapperMultInfl;
	
	//test on clear GA
	Matrix best_ff, mean_ff, full_best;
	ga_stat s;
	//disable subpops
	ga_obj.opt_.subpopT = NoSubpops;
	//ga_obj.opt_.vspSize.NewMatrix(0,0);
	s = TestFcnOpt(nVars, ft, run_num, false, full_best);
	best_ff |= s.best_ff_;
	mean_ff |= s.mean_ff_;

	//now test with vertical subpopulations
	//ga_obj.opt_.subpopT = Vertical;
	ReadOptions();
	//correct init range
	vsp_correct_init_range(ga_obj.opt_);
#ifdef _DEBUG
	ga_obj.opt_.generations = 100;
#endif
	clock_t begin = clock();
	for(double mr = start; mr <= end; mr += incr) {
		ga_obj.opt_.vspFract = mr;
		s = TestFcnOpt(nVars, ft, run_num, false, full_best);
		best_ff |= s.best_ff_;
		mean_ff |= s.mean_ff_;
		if(mr == start) mr = 0.2 - incr;
	}

	clock_t finish = clock();
	std::cout << "Overall time: " << (double)(finish - begin)/CLOCKS_PER_SEC << " seconds" << endl;

	ofstream f("best_ff.txt", ios::trunc | ios::out);
	best_ff.Print(f);
	f.close();
	f.open("mean_ff.txt", ios::trunc | ios::out);
	mean_ff.Print(f);
	f.close();
	f.open("full_best.txt", ios::trunc | ios::out);
	full_best.Print(f);
}

void TestKmeans()
{
	kmeans km; 
	ifstream f1("t.txt");
	Matrix t = Matrix::Read(f1);
	f1.close(); f1.clear();
	f1.open("f.txt");
	Matrix f = Matrix::Read(f1);
	km.opt_.ReadOptions();
	km.opt_.seed_t = KM::sample;
	//km.find_clusters(t, 10, 200);

	km.find_clusters_f(t, f, f.size()*0.3, 200);
	const Matrix c = km.get_centers();

	//Matrix c;
	//double quant_mult;
	//vector< Matrix > c_trials;
	//vector< ulong > c_rows;
	//for(double quant_mult = 0.01; quant_mult <= 1; quant_mult+=0.05) {
	//	c <<= km.drops_hetero_simple(t, f, 0.8, 200, quant_mult);
	//	//c <<= km.drops_homo(t, f, 0.8, 200, quant_mult);
	//	c_trials.push_back(c);
	//	c_rows.push_back(c.row_num());
	//}
	//sort(c_rows.begin(), c_rows.end());
	//ulong median = c_rows[c_rows.size()/2];
	//for(ulong i = 0; i < c_trials.size(); ++i) {
	//	if(c_trials[i].row_num() == median) {
	//		c <<= c_trials[i];
	//		break;
	//	}
	//}

	ofstream f2("centers.txt", ios::out | ios::trunc);
	//km.get_centers().Print(f2);
	c.Print(f2);
	f2.close(); f2.clear();
	f2.open("ind.txt", ios::out | ios::trunc);
	km.get_ind().Print(f2);
}

int main(int argc, char* argv[])
{
	if(argc > 1 && strcmp(argv[1], "1") == 0) {
		TestKmeans();
	}
	else
		TestFcnOpt(5, rastrigins);
	//Kar2Words();
	//vsp_test_mig_rates();

	std::cout << "Press Enter to exit" << std::endl;
	std::cin.peek();
	return 0;
}

