#include "test_fcn.h"
#include "matrix.h"

#include "text_table.h"

#include <cmath>
#include <string>
#include <sstream>
#include <fstream>

#define PI 3.1415926535897932384626433832795

using namespace std;
//using namespace GA;
//using namespace NN;
//using namespace KM;
//using namespace DA;

int vspNum = 0;
double deliver_mult = 100;

namespace {
const double infl_factor = 0.2;
}

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
		if(nVars > 1) {
			for(int j=0; j < nVars - 1; ++j) {
				dTmp = (*pTek)*(*pTek);
				dSum += 100*(dTmp - *(pTek + 1))*(dTmp - *(pTek + 1)) + (1 - dTmp)*(1 - dTmp);
				++pTek;
			}
		}
		else {
			dTmp = (*pTek)*(*pTek);
			dSum = 100*(dTmp - 1)*(dTmp - 1) + (1 - dTmp)*(1 - dTmp);
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
	for(ulong i = 0; i < (ulong)vspNum; ++i) {
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
	for(ulong i = 0; i < (ulong)vspNum; ++i) {
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
	for(ulong i = 0; i < (ulong)vspNum; ++i) {
		//subpop = pop; subpop.DelColumns(offs, nspVars);
		//dep_col |= subpop.vMean(true);
		dep_col |= pop.GetColumns(offs, nspVars).vMean(true) * infl_factor;
		offs += nspVars;
	}
	offs = 0;
	//calc objectives
	for(ulong i = 0; i < (ulong)vspNum; ++i) {
		subpop <<= pop.GetColumns(offs, nspVars) - deliver_mult*i;
		//subpop |= dep_col.GetColumns(i);
		for(ulong j = 0; j < (ulong)vspNum; ++j)
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
	for(ulong i = 0; i < (ulong)vspNum; ++i) {
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
		for(ulong j = 0; j < (ulong)vspNum; ++j) {
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
	for(ulong i=0; i < (ulong)nVars; ++i)
		a_row |= g_a.GetRows(i).GetColumns(i, g_a.col_num() - i);

	int offs;
	double dSum;
	for(ulong i = 0; i < (ulong)nPopSize; ++i) {
		//calc result for cur chrom
		chrom = pop.GetRows(i) + add_coeffs;
		pol = (!chrom)*chrom;
		//calc each subpop results
		offs = 0;
		for(ulong j=0; j < (ulong)vspNum; ++j) {
			dSum = 0;
			for(ulong k=0; k < (ulong)nspVars; ++k)
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

