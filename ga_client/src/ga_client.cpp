// ga_client_win.cpp : Defines the entry point for the console application.
//

#include "test_fcn.h"
#include "ga.h"
#include "matrix.h"
#include "m_algorithm.h"
#include "objnet.h"
#include "alg_api.h"
#include "kmeans.h"
#include "determ_annealing.h"

#include "text_table.h"

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
using namespace DA;
//using namespace hybrid_adapt;

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
				   bool read_options = true, Matrix* full_best = NULL)
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
		if(full_best) *full_best |= cur_stat.best_ff_;
	}

	clock_t finish = clock();
	std::cout << "Time elapsed: " << (double)(finish - start)/CLOCKS_PER_SEC << " seconds" << endl;

	stat /= runs_num;
	//print stat
//	ofstream f("ga_stat.txt", ios::out | ios::trunc);
//	stat.print(f);

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
	s = TestFcnOpt(nVars, ft, run_num, false, &full_best);
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
		s = TestFcnOpt(nVars, ft, run_num, false, &full_best);
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

void TestKmeans(int c_num)
{
	ifstream f1("t.txt");
	Matrix t = Matrix::Read(f1);
	f1.close(); f1.clear();
	f1.open("f.txt");
	Matrix f = Matrix::Read(f1);
	Matrix c;
	Matrix::indMatrix ind;

	kmeans km;
	km.opt_.ReadOptions();
	km.opt_.seed_t = KM::sample;
	//km.find_clusters(t, 10, 200);

	km.find_clusters_f(t, f, f.size()*0.5, 200);
	//km.drops_hetero_simple(t, f, 0.7, 200);

	c <<= km.get_centers();
	ind <<= km.get_ind();

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
	c.Print(f2);
	f2.close(); f2.clear();
	f2.open("ind.txt", ios::out | ios::trunc);
	ind.Print(f2);
	//km.get_ind().Print(f2);
}

void TestDA(int c_num) {
	ifstream f1("t.txt");
	Matrix t = Matrix::Read(f1);
	f1.close(); f1.clear();
	f1.open("f.txt");
	Matrix f = Matrix::Read(f1);
	Matrix c;
	Matrix::indMatrix ind;

	determ_annealing da;
	da.find_clusters(t, f, c_num, 200);
	c <<= da.get_centers();
	ind <<= da.get_ind();

	ofstream f2("centers.txt", ios::out | ios::trunc);
	c.Print(f2);
	f2.close(); f2.clear();
	f2.open("ind.txt", ios::out | ios::trunc);
	ind.Print(f2);
}

void test_text_table() {
	text_table tt;
	tt.fmt().wrap = true;
	tt.fmt().align = 2;
	tt.fmt().sep_cols = true;
	tt << text_table::begh() << "## 0 # 0 # 0 # 24 ##" << text_table::endr();
	tt << text_table::begr() << "\\hline" << text_table::endr();
	tt << "Column 1 & Column 2 & Column 3 & Column 4" << text_table::endrh();
	tt << 12 << "&" << 24.5 << "& col3 test & col4 looooooooooooong veeeeeery llllllloooong text"  << text_table::endr();
	tt << "\\hline" << text_table::endr();
	string res = tt.format();
	cout << res << endl;
	res = tt.format();
	cout << res << endl;
}

int main(int argc, char* argv[])
{
	int test_t = 0, c_num = 5;
	test_fun_t fun = rastrigins;

	if(argc > 1) c_num = atoi(argv[1]);
	if(argc > 2) {
		test_t = atoi(argv[1]);
		c_num = atoi(argv[2]);
	}
	if(argc > 3) {
		string s = argv[3];
		if(s == "rastrigins") fun = rastrigins;
		else if(s == "rosenbrocks") fun = rosenbrocks;
		else if(s == "sphere") fun = sphere;
		else if(s == "sphere_multimod") fun = sphere_multimod;
		else if(s == "polynom") fun = polynom;
		else if(s == "VP") fun = VP;
		else if(s == "schwefel") fun = schwefel;
		else if(s == "ackley") fun = ackley;
	}

	switch(test_t) {
		default:
		case 0: TestFcnOpt(c_num, fun); break;
		case 1: TestKmeans(c_num); break;
		case 2: TestDA(c_num); break;
		case 3: test_text_table(); break;
	}

	//std::string s;
	//Kar2Words();
	//vsp_test_mig_rates();

	//std::cout << "Press Enter to exit" << std::endl;
	//std::cin.peek();
	return 0;
}

