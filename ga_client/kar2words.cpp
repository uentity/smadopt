#include "kar2words.h"
#include "matrix.h"
#include "alg_api.h"

#include <time.h>
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

Matrix interp(const Matrix& src, int new_dim)
{
	Matrix dst(new_dim, 1);
	double delta = (double)src.size()/(new_dim - 1);
	double pos = 0;
	ulong old_dim = src.size();
	ulong c;
	for(Matrix::r_iterator p_d(dst.begin()); p_d != dst.end(); ++p_d) {
		c = floor(pos);
		if(c == ceil(pos) || c >= old_dim - 1)
			*p_d = src[c];
		else
			*p_d = src[c] + (src[c + 1] - src[c])*(pos - c);
		pos += delta;
	}

	return dst;
}


Matrix alph, kar;
vector<ulong> word, letlen;

void find_word(int nVars, int nPopSize, double* pPop, double* pScore)
{
	Matrix pop(nPopSize, nVars, pPop);
	Matrix score(nPopSize, 1);
	Matrix chrom, ra, nkar, diff;
	Matrix::indMatrix srt_ind;
	pop = pop.Abs();
	double sk, a;
	ulong wi, k, pos;
	for(int i=0; i<nPopSize; ++i) {
		chrom = pop.GetRows(i);
		sk = chrom.Sum();
		if(sk == 0) {
			score[i] = 100000000;
			continue;
		}
		//wi = 0;
		pos = 0;
		score[i] = 0;
		word.clear(); word.reserve(nVars);
		letlen.clear(); letlen.reserve(nVars);
		for(int j=0; j<nVars; ++j) {
			if(pos >= kar.size() - 1)
				break;
			if(j < nVars - 1)
				k = min<ulong>(round(chrom[j]*kar.size()/sk), kar.size() - pos);
			else k = kar.size() - pos;
			if(k < 2) continue;

			nkar = kar.GetRows(pos, k);
			nkar -= nkar.Min();
			if(nkar.Max() != 0)
				nkar /= nkar.Max();

			diff.NewMatrix(1, alph.col_num());
			for(ulong l=0; l < alph.col_num(); ++l) {
				ra = interp(alph.GetColumns(l), k) - nkar;
				diff[l] = ra.Mul(ra).Sum();
			}
			srt_ind = diff.RawSort();
			score[i] += diff[0];
			word.push_back(srt_ind[0]);
			letlen.push_back(k);
			//word[wi] = ra[0];
			//letlen[wi] = k;
			//++wi;
			pos += k;
		}
	}

	memcpy(pScore, &score[0], score.raw_size());
}

Matrix kar2templ(Matrix& x)
{
	x = x.Abs();
	double score;
	find_word(x.size(), 1, &x[0], &score);
	
	Matrix templ, ra, nkar;
	ulong pos = 0;
	for(ulong i=0; i<word.size(); ++i) {
		ra = interp(alph.GetColumns(word[i]), letlen[i]);
		nkar = kar.GetRows(pos, letlen[i]);
		ra *= nkar.Max() - nkar.Min();
		ra += nkar.Min();
		templ &= ra;
		pos += letlen[i];
	}
	return templ;
}

void Kar2Words()
{
	ifstream fsrc("karotazh.txt");
	Matrix all_kar = Matrix::Read(fsrc);
	fsrc.clear();
	fsrc.close();
	fsrc.open("alph.txt");
	alph = Matrix::Read(fsrc);


	//debug interp
	//ofstream it("interp_down.txt", ios::out | ios::trunc);
	//Matrix ra = interp(alph.GetColumns(0), 37);
	//PrintMatrix(ra, it);
	//it.clear(); it.close();
	//it.open("interp_up.txt", ios::out | ios::trunc);
	//ra = interp(alph.GetColumns(0), 256);
	//PrintMatrix(ra, it);
	//it.close();

	int nVars;
	cout << "How many segments will be used? ";
	cin >> nVars;
	ulong popSize = 100;

	//gaOptions opt;
	//opt.popSize = popSize;
	//opt.scheme = MuLambda;
	//opt.generations = 1000;
	//opt.stallGenLimit = -1;
	//opt.scalingT = Rank;
	//opt.mutProb = 0.05;
	//opt.mutNormLawSigma2 = 1;
	//opt.xoverFraction = 0.8;
	//opt.xoverHeuRatio = 1.2;
	//opt.logEveryPop = true;
	//opt.selectionT = StochasticUniform;
	//opt.mutationT = UniformMutation;
	//SetGAOptions(&opt);

	ReadOptions();
	
	Matrix res(1, nVars);

	//Matrix pop(popSize, nVars);
	//Matrix score(popSize, 1);

	cout << "How many curves to process? ";
	ulong nCurves;
	cin >> nCurves;
	if(nCurves == 0 || nCurves > all_kar.col_num()) nCurves = all_kar.col_num();

	clock_t start = clock();

	Matrix words(all_kar.row_num(), nVars), letlens(all_kar.row_num(), nVars);
	words = -1; letlens = -1;
	Matrix templ, all_templ;
	for(ulong i=0; i<nCurves; ++i)
	{
		kar = all_kar.GetColumns(i);
		Run(find_word, nVars, res.GetBuffer());
		//Start(pop.GetBuffer(), nVars, true);
		//do {
		//	RastriginsFcn(nVars, popSize, pop.GetBuffer(), score.GetBuffer());
		//} while(GetNextPop(score.GetBuffer(), pop.GetBuffer(), &popSize));

		templ = kar2templ(res);
		all_templ |= templ;
		for(ulong j=0; j<word.size(); ++j) {
			words(i, j) = word[j];
			letlens(i, j) = letlen[j];
		}
	}

	ofstream f("templ.txt", ios::out | ios::trunc);
	all_templ.Print(f);
	f.clear(); f.close();
	f.open("words.txt", ios::out | ios::trunc);
	words.Print(f);
	f.clear(); f.close();
	f.open("letlens.txt", ios::out | ios::trunc);
	letlens.Print(f);

	clock_t finish = clock();
	std::cout << "Time elapsed: " << (double)(finish - start)/CLOCKS_PER_SEC << " seconds" << endl;
}
