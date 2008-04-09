#include "ga.h"
#include "prg.h"
#include "nn_addon.h"

#include <time.h>
#include <set>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <iostream>

#define NW 13

using namespace prg;
using namespace std;
using namespace GA;
using hybrid_adapt::ha_round;
using hybrid_adapt::my_sprintf;

typedef ga::indMatrix indMatrix;

/*
void DumpM(Matrix* m, char* pFname = NULL)
{
	if(pFname) {
		ofstream fd(pFname, ios::out | ios::trunc);
		m->Print(fd);
	}
	else m->Print(cout);
}
*/

const char* ga_except::explain_error(int code)
{
	switch(code) {
		default:
			return alg_except::explain_error(code);
	}
}

void ga::_print_err(const char* pErr)
{
	if(!errFile_.is_open()) {
		errFile_.open(opt_.errFname.c_str(), ios::trunc | ios::out);
		cerr.rdbuf(errFile_.rdbuf());
	}
	time_t cur_time = time(NULL);
	string sTime = ctime(&cur_time);
	string::size_type pos;
	if((pos = sTime.rfind('\n')) != string::npos)
		sTime.erase(pos);
	cerr << sTime << ": " << pErr << endl;
	cout << pErr << endl;
	state_.lastError = pErr;
}

void ga::set_def_opt()
{
	opt_.set_def_opt();
	GenLen_ = 0;
	state_.nStatus = Idle;
}

ga::ga(void)
	: apAddon_(), opt_(this)
{
	set_def_opt();
}

ga::~ga(void) {}

/*
void ga::ReadAddonOptions()
{
	for(ulong i=0; i<apAddon_.size(); ++i)
		apAddon_[i]->ReadOptions();
}
*/

Matrix ga::ScalingSimplest(const Matrix& scores)
{
	//for selection that is based just on comparision
	Matrix expect;
	if(opt_.minimizing)
		expect <<= -scores + scores.Max();
	else
		expect = scores;

	return expect;
}

Matrix ga::ScalingPropMean(const Matrix& scores)
{
	/*
	% Rotate scores around their mean because we are minimizing so lower scores
	% must yield a higher expectation.
	scores = 2 * mean(scores) - scores;

	% Negative expectations are simply not allowed so here we make sure that
	% doesn't happen by sliding the whole vector up until everything is
	% non-negative.
	m = min(scores);
	if(m < 0)
		scores = scores - m;
	end

	% Normalize: expectation should sum to one.
	expectation = nParents * scores ./ sum(scores);
	*/

	//Matrix expect = -scores + 2*scores.Mean();
	//double m = expect.Min();
	//if(m < 0) expect -= m;
	//expect /= expect.Sum();

	Matrix expect;
	double bound = 2*scores.Mean();
	if(opt_.minimizing)
		expect <<= -scores + bound;
	else
		expect <<= scores - (scores.Max() - bound);
	double m = expect.Min();
	if(m < 0) expect -= m;

	return expect/expect.Sum();
}

Matrix ga::ScalingProp(const Matrix& scores)
{
	//double dSum = scores.Abs().Sum();
	//Matrix expect = -scores/dSum + 1;
	//expect /= expect.Sum();

	//double bound;
	Matrix expect;
	if(opt_.minimizing)
		expect <<= - scores + scores.Max();
	else
		expect <<= scores - scores.Min();

	return expect/expect.Sum();
}

Matrix ga::ScalingPropInv(const Matrix& scores)
{
	// minimization: expect = 1/(1 + scores - min(scores))
	// maximization: expect = 1/(1 + max(scores) - scores)

	Matrix expect;
	if(!opt_.minimizing)
		expect <<= - scores + scores.Max();
	else
		expect <<= scores - scores.Min();
	expect += 1;
	transform(expect.begin(), expect.end(), expect.begin(), bind1st(divides<double>(), 1));

	return expect/expect.Sum();
}

Matrix ga::ScalingPropTime(const Matrix& scores)
{
	Matrix expect;
	if(opt_.minimizing) {
		state_.ffsc_factor = opt_.ffscParam * scores.Max() + (1 - opt_.ffscParam) * state_.ffsc_factor;
		expect <<= -scores + state_.ffsc_factor;
	}
	else {
		state_.ffsc_factor = opt_.ffscParam * scores.Min() + (1 - opt_.ffscParam) * state_.ffsc_factor;
		expect <<= scores - state_.ffsc_factor;
	}
	replace_if(expect.begin(), expect.end(), bind2nd(less<double>(), 0), 0);

	return expect/expect.Sum();
}

Matrix ga::ScalingPropSigma(const Matrix& scores)
{
	Matrix expect;
	double m = scores.Mean(), q = scores.Std();
	if(opt_.minimizing)
		expect <<= -scores + (m + opt_.ffscParam * q);
	else
		expect <<= scores - (m - opt_.ffscParam * q);
	replace_if(expect.begin(), expect.end(), bind2nd(less<double>(), 0), 0);

	return expect/expect.Sum();
}

Matrix ga::ScalingRank(const Matrix& scores)
{
	Matrix tmp;
	tmp = scores;
	ulong scSize = tmp.size();
	double alfa = 2 - opt_.ffscParam;
	indMatrix mInd = tmp.RawSort();
	if(opt_.minimizing) reverse(mInd.begin(), mInd.end());

	Matrix expect(scSize, 1);
	for(ulong i=0; i<scSize; ++i) {
		expect[mInd[i]] = (alfa + (double)i*(opt_.ffscParam - alfa)/(scSize - 1))/(double)scSize;
	}
	//expect /= expect.Sum();

	return expect;
}

Matrix ga::ScalingRankExp(const Matrix& scores)
{
	Matrix tmp;
	tmp = scores;
	ulong scSize = tmp.size();
	//double alfa = 2 - opt_.ffscParam;
	indMatrix mInd = tmp.RawSort();
	if(opt_.minimizing) reverse(mInd.begin(), mInd.end());

	Matrix expect(scSize, 1);
	for(ulong i=0; i<scSize; ++i) {
		expect[mInd[i]] = 1 - exp(-(double)i);
	}
	//expect /= expect.Sum();

	return expect/expect.Sum();
}

Matrix ga::ScalingRankSqr(const Matrix& scores)
{
	/*
	[unused,i] = sort(scores);

	expectation = zeros(size(scores));
	expectation(i) = 1 ./ ((1:length(scores))  .^ 0.5);

	expectation = nParents * expectation ./ sum(expectation);
	*/

	Matrix tmp;
	tmp = scores;
	ulong scSize = tmp.size();
	indMatrix mInd = tmp.RawSort();

	Matrix expect(scSize, 1);
	for(ulong i=0; i<scSize; ++i) {
		expect[mInd[i]] = 1/sqrt((double)(i + 1));
	}
	expect /= expect.Sum();

	return expect;
}

indMatrix ga::SelectionRoulette(Matrix& expect, ulong nParents)
{
	/*
	%SELECTIONROULETTE Choose parents using roulette wheel.
	%   PARENTS = SELECTIONROULETTE(EXPECTATION,NPARENTS,OPTIONS) chooses
	%   PARENTS using EXPECTATION and number of parents NPARENTS. On each
	%   of the NPARENTS trials, every parent has a probability of being selected
	%   that is proportional to their expectation.
	%
	wheel = cumsum(expectation) / nParents;

	parents = zeros(1,nParents);
	for i = 1:nParents
		r = rand;
		for j = 1:length(wheel)
			if(r < wheel(j))
				parents(i) = j;
				break;
			end
		end
	end
*/

	//Matrix wheel = expect.CumSum();

	indMatrix res(1, nParents);
	double r, sum;
	ulong winner;
	for(ind_iterator pos(res.begin()); pos != res.end(); ++pos) {
		r = prg::rand01();
		winner = 0;
		sum = expect[0];
		while(sum < r) {
			++winner;
			sum += expect[winner];
		}
		*pos = winner;
	}

	return res;
}

indMatrix ga::SelectionStochUnif(Matrix& expect, ulong nParents)
{
	//MPtr pmWheel = new Matrix(expect);
	//double dSum = 0;
	//for(ulong i=1;pmWheel->GetSize();++i) {
	//	dSum += (*pmWheel)[i];
	//	(*pmWheel)[i] = dSum/nParents;
	//}

	indMatrix res(1, nParents);
	double dStepSize = 1./nParents;
	double dPosition = dStepSize*prg::rand01();
	/*
	Matrix wheel = expect.CumSum();
	ulong lowest = 0;

	for(ind_iterator pos(res.begin()); pos != res.end(); ++pos) {
		*pos = 0;
		sum += expect[0];
		for(ulong j=lowest; j<wheel.size(); ++j) {
			if(dPosition < wheel[j]) {
				*pos = j;
				lowest = j;
				break;
			}
		}
		dPosition += dStepSize;
	}
	*/

	//new realization
	double sum = 0;
	ulong res_ind = 0;
	for(ulong i = 0; i < expect.size(); ++i) {
		sum += expect[i];
		while(dPosition < sum && res_ind < nParents) {
			res[res_ind++] = i;
			dPosition += dStepSize;
		}
	}

	return res;
}

indMatrix ga::SelectionOnce(Matrix& expect, ulong nParents)
{
	if(expect.size() != nParents)
		throw ga_except::alg_except(SizesMismatch, "SelectionOnce: number of new parents required don't match the population size");
	Matrix mexpect;
	mexpect = expect;

	indMatrix res(1, nParents), roulInd;
	double dSum;
	for(ind_iterator pos(res.begin()); pos != res.end(); ++pos) {
		roulInd <<= SelectionRoulette(mexpect, 1);
		*pos = roulInd[0];
		mexpect[roulInd[0]] = 0;

		dSum = mexpect.Sum();
		if(dSum) mexpect /= dSum;
		else break;
	}
	return res;
}

indMatrix ga::SelectionSort(Matrix& expect, ulong nParents)
{
	if(expect.size() != nParents)
		throw ga_except::alg_except(SizesMismatch, "SelectionSort: number of new parents required don't match the population size");

	indMatrix res(1, nParents);
	indMatrix mInd = expect.RawSort();
	indMatrix::r_iterator p_ind(mInd.begin());
	for(ind_iterator p_res(res.begin() + min(nParents, expect.size()) - 1); p_res >= res.begin(); --p_res) {
		*p_res = *p_ind;
		++p_ind;
	}
	return res;
}

indMatrix ga::SelectionUniform(Matrix& expect, ulong nParents)
{
	/*
	%SELECTIONUNIFORM Choose parents at random.
	%   PARENTS = SELECTIONUNIFORM(EXPECTATION,NPARENTS,OPTIONS) chooses
	%   PARENTS randomly using the EXPECTATION and number of parents NPARENTS.
	%
	%   Parent selection is NOT a function of performance. This selection function
	%   is useful for debugging your own custom selection, or for comparison. It is
	%   not useful for actual evolution of high performing individuals.
	%

	% nParents random numbers
	parents = rand(1,nParents);

	% integers on the interval [1, populationSize]
	parents = ceil(parents * length(expectation));
*/

	indMatrix res(1, nParents);
	ulong popSize = expect.size();
	//generate(res.begin(), res.end(), bind2nd(mem_fun_ref<ulong, ulong>(&prg::randIntUB), popSize));
	for(ind_iterator pos = res.begin(); pos != res.end(); ++pos)
		*pos = prg::randIntUB(popSize);

	return res;
}

indMatrix ga::SelectionTournament(Matrix& expect, ulong nParents)
{
	indMatrix res(1, nParents);
	ulong popSize = expect.size();
	const ulong ts = min(opt_.nTournSize, expect.size());

	ulong winner, player;
	double max_expect;
	//ul_vec players; players.reserve(ts);
	set< ulong > players;
	pair< set< ulong >::iterator, bool > new_player;
	for(ulong i = 0; i < nParents; ++i) {
		//select first player
		players.clear();
		//players.push_back(prg::randIntUB(popSize));
		new_player = players.insert(prg::randIntUB(popSize));
		winner = *new_player.first;
		max_expect = expect[winner];
		//select other players
		for(ulong j = 1; j < ts; ++j) {
			//generate unique index
			while(!(new_player = players.insert(prg::randIntUB(popSize))).second) {};
			//do {
			//	player = prg::randIntUB(popSize);
			//} while(find(players.begin(), players.end(), player) != players.end());
			//players.push_back(player);
			if(expect[*new_player.first] > max_expect) {
				winner = *new_player.first;
				max_expect = expect[winner];
			}
		}
		res[i] = winner;
	}

	//third realization - too slow
	/*
	ulong winner;
	double max_expect;
	//setup initial players ind
	ul_vec players(expect.size());
	for(ulong i = 0; i < players.size(); ++i)
		players[i] = i;
	ul_vec::iterator pl_beg(players.begin()), pl_end(players.end());

	for(ulong i=0; i<nParents; ++i) {
		//shuffle indices
		random_shuffle(pl_beg, pl_end, prg::randIntUB);

		//select first player
		winner = players[0];
		max_expect = expect[winner];
		//test other players
		for(ulong j = 1; j < ts; ++j) {
			if(expect[players[j]] > max_expect) {
				winner = players[j];
				max_expect = expect[winner];
			}
		}
		res[i] = winner;
	}
	*/

	return res;
}

Matrix ga::CrossoverOnePoint(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();
	prg::prepare_streams(1);
	prg::switch_stream(1);

	Matrix xoverKids; xoverKids.reserve(nKids*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2, tmp;
	ulong nPos;
	//ulong ind1, ind2;
	for(ulong i=0; i<nKids; ++i) {
		parent1 <<= pop.GetRows(parents[index++]);
		parent2 <<= pop.GetRows(parents[index++]);

		nPos = prg::randIntUB(GenomeLength);
		parent1.SetColumns(parent2.GetColumns(nPos, GenomeLength - nPos), nPos, GenomeLength - nPos);
		xoverKids &= parent1;
	}
	prg::switch_stream(0);
	return xoverKids;
}

Matrix ga::CrossoverTwoPoint(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();
	prg::prepare_streams(2);

	Matrix xoverKids; xoverKids.reserve(nKids*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2;
	ulong nPos1, nPos2, nSwap;
	for(ulong i=0; i<nKids; ++i) {
		parent1 <<= pop.GetRows(parents[index++]);
		parent2 <<= pop.GetRows(parents[index++]);

		prg::switch_stream(1);
		nPos1 = prg::randIntUB(GenomeLength);
		prg::switch_stream(2);
		while((nPos2 = prg::randIntUB(GenomeLength)) == nPos1) {};
		if(nPos2 < nPos1) {
			nSwap = nPos2;
			nPos2 = nPos1;
			nPos1 = nSwap;
		}
		parent1.SetColumns(parent2.GetColumns(nPos1, nPos2 - nPos1 + 1), nPos1, nPos2 - nPos1 + 1);
		xoverKids &= parent1;
	}
	prg::switch_stream(0);
	return xoverKids;
}

Matrix ga::CrossoverUniform(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();
	prg::prepare_streams(GenomeLength);

	Matrix xoverKids; xoverKids.reserve(nKids*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2;
	for(ulong i=0; i<nKids; ++i) {
		parent1 <<= pop.GetRows(parents[index++]);
		parent2 <<= pop.GetRows(parents[index++]);

		for(ulong j=0; j<GenomeLength; ++j) {
			prg::switch_stream(j + 1);
			if(prg::rand01() >= 0.5) parent1[j] = parent2[j];
		}
		xoverKids &= parent1;
	}
	prg::switch_stream(0);
	return xoverKids;
}

Matrix ga::CrossoverHeuristic(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();

	Matrix xoverKids; xoverKids.reserve(nKids*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2, child;
	ulong ind1, ind2;
	for(ulong i=0; i<nKids; ++i) {
		ind1 = parents[index++];
		ind2 = parents[index++];
		parent1 <<= pop.GetRows(ind1);
		parent2 <<= pop.GetRows(ind2);

		if(scores[ind1] < scores[ind2])
			child <<= parent2 + (parent1 - parent2)*opt_.xoverHeuRatio;
		else
			child <<= parent1 + (parent2 - parent1)*opt_.xoverHeuRatio;
		xoverKids &= child;
	}
	return xoverKids;
}

Matrix ga::CrossoverFlat(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();
	prg::prepare_streams(GenomeLength);

	Matrix xoverKids; xoverKids.reserve(nKids*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2;
	for(ulong i=0; i<nKids; ++i) {
		parent1 <<= pop.GetRows(parents[index++]);
		parent2 <<= pop.GetRows(parents[index++]);

		for(ulong j=0; j<GenomeLength; ++j) {
			prg::switch_stream(j + 1);
			parent1[j] += prg::rand01()*(parent2[j] - parent1[j]);
		}
		xoverKids &= parent1;
	}
	prg::switch_stream(0);
	return xoverKids;
}

Matrix ga::CrossoverBLX(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();
	prg::prepare_streams(GenomeLength);

	Matrix xoverKids; xoverKids.reserve(nKids*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2;
	double rng_min, rng_max, delta;
	for(ulong i=0; i<nKids; ++i) {
		parent1 <<= pop.GetRows(parents[index++]);
		parent2 <<= pop.GetRows(parents[index++]);

		for(ulong j=0; j<GenomeLength; ++j) {
			if(parent1[j] < parent2[j]) {
				rng_min = parent1[j];
				rng_max = parent2[j];
			}
			else {
				rng_min = parent2[j];
				rng_max = parent1[j];
			}
			delta = rng_max - rng_min;
			prg::switch_stream(j + 1);
			parent1[j] = rng_min - delta*(opt_.xoverBLXAlpha - prg::rand01()*(1 + 2*opt_.xoverBLXAlpha));
		}
		xoverKids &= parent1;
	}
	prg::switch_stream(0);
	return xoverKids;
}

Matrix ga::CrossoverArithmetic(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nHalfKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();
	//prg::prepare_streams(1); prg::switch_stream(1);

	Matrix xoverKids; xoverKids.reserve((nHalfKids << 1)*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2, child;
	for(ulong i=0; i<nHalfKids; ++i) {
		parent1 <<= pop.GetRows(parents[index++]);
		parent2 <<= pop.GetRows(parents[index++]);

		//if(prg::rand01() >= 0.5)
		//	parent1 <<= parent1*opt_.xoverArithmeticRatio + parent2*(1 - opt_.xoverArithmeticRatio);
		//else
		//	parent1 <<= parent2*opt_.xoverArithmeticRatio + parent1*(1 - opt_.xoverArithmeticRatio);
		child <<= parent1*opt_.xoverArithmeticRatio + parent2*(1 - opt_.xoverArithmeticRatio);
		xoverKids &= child;
		child <<= parent2*opt_.xoverArithmeticRatio + parent1*(1 - opt_.xoverArithmeticRatio);
		xoverKids &= child;
	}
	//prg::switch_stream(0);
	return xoverKids;
}

Matrix ga::CrossoverSBX(const Matrix& pop, const Matrix& scores, const indMatrix& parents)
{
	ulong nHalfKids = parents.size() >> 1;
	ulong GenomeLength = pop.col_num();
	prg::prepare_streams(GenomeLength);

	Matrix xoverKids; xoverKids.reserve((nHalfKids << 1)*GenomeLength);
	ulong index = 0;

	Matrix parent1, parent2, child(1, GenomeLength);
	double beta, spread = 1/(opt_.xoverSBXParam + 1);
	for(ulong i=0; i<nHalfKids; ++i) {
		parent1 <<= pop.GetRows(parents[index++]);
		parent2 <<= pop.GetRows(parents[index++]);

		for(ulong j=0; j<GenomeLength; ++j) {
			//test variable-to-variable probability!
			prg::switch_stream(0);
			if(prg::rand01() <= 0.5) {
				prg::switch_stream(j + 1);
				beta = prg::rand01();
				if(beta <= 0.5)
					beta = pow(2*beta, spread);
				else
					beta = pow(0.5/(1 - beta), spread);
					//beta = pow(2*(1 - beta), spread);
				child[j] = 0.5*((1 - beta)*parent1[j] + (1 + beta)*parent2[j]);
				parent1[j] = 0.5*((1 + beta)*parent1[j] + (1 - beta)*parent2[j]);
			}
			else child[j] = parent2[j];
		}
		xoverKids &= child;
		xoverKids &= parent1;
	}
	prg::switch_stream(0);
	return xoverKids;
}

ulong ga::_getXoverParentsCount(ulong xoverKidsCount)
{
	if(opt_.crossoverT == Arithmetic || opt_.crossoverT == SBX)
		return xoverKidsCount + (xoverKidsCount & 1);
	else
		return xoverKidsCount << 1;
}

void ga::MutateUniform(double& gene, ulong pos, const Matrix& range)
{
	gene = range(0, pos) + prg::rand01()*(range(1, pos) - range(0, pos));
}

void ga::MutateNormal(double& gene, ulong pos, const Matrix& range)
{
	double r = prg::randn(0, opt_.mutNormLawSigma2);
	if(r > 0) gene += r * (range(1, pos) - gene);
	else gene += r * (gene - range(0, pos));
	//gene = range(0, pos) + prg::randn(0.5, opt_.mutNormLawSigma2)*(range(1, pos) - range(0, pos));
}

double ga::_nonUniformMutDelta(ulong t, double y)
{
	return y*(1 - pow(prg::rand01(), pow(1 - (double)t/opt_.generations, opt_.mutNonUniformParam)));
}

void ga::MutateNonUniform(double& gene, ulong pos, const Matrix& range)
{
	if(prg::rand01() < 0.5)
		gene += _nonUniformMutDelta(state_.nGen, range(1, pos) - gene);
	else
		gene -= _nonUniformMutDelta(state_.nGen, gene - range(0, pos));
}

void ga::Mutation(Matrix& pop, const Matrix& range)
{
	//ulong nKids = parents.size();
	ulong GenomeLength = pop.col_num();
	prg::prepare_streams(GenomeLength);

	//Matrix mutKids(nKids, GenomeLength);

	//Matrix child;
	//vector<double> mutPoints(GenomeLength);
	//double dK;
	for(ulong i=0; i<pop.row_num(); ++i) {
		//mutKids.SetRows(pop.GetRows(parents[i]), i);
		//generate(mutPoints.begin(), mutPoints.end(), prg::rand01);
		for(ulong j=0; j<GenomeLength; ++j) {
			prg::switch_stream(j + 1);
			if(prg::rand01() < opt_.mutProb) {
				if(!opt_.useBitString) {
					//prg::switch_stream(0);
					(this->*_pMutationFcn)(pop(i, j), j, range);
				}
				else pop(i, j) = 1 - pop(i, j);
			}
		}
	}
	prg::switch_stream(0);
	//return mutKids;
}

Matrix ga::CreationUniform()
{
	//get genome length
	//ulong GenomeLength = opt_.initRange.col_num();
	prg::prepare_streams(GenLen_);

	Matrix initPop(opt_.popSize, GenLen_);
	for(ulong i=0; i<opt_.popSize; ++i) {
		for(ulong j=0; j<GenLen_; ++j) {
			prg::switch_stream(j + 1);
			if(opt_.useBitString)
				initPop(i, j) = prg::rand01() < 0.5 ? 0 : 1;
			else
				initPop(i, j) = opt_.initRange(0, j) + prg::rand01()*(opt_.initRange(1, j) - opt_.initRange(0, j));
		}
	}
	prg::switch_stream(0);

	//if(opt_.useBitString) initPop = Convert2BitPop(initPop);
	return initPop;
}

void ga::Migrate(Matrix& pop, Matrix& score, ul_vec& elite_ind)
{
	if(opt_.subpopT != Horizontal || state_.nGen % opt_.migInterval != 0) return;

	vector<indMatrix> vm_srt_i;
	ulong nMigrators;
	ulong dir_ind[2];
	ulong dir_cnt = 1;
	if(opt_.migDirection == MigrateBoth) dir_cnt = 2;

	//Matrix main_score;
	//if(opt_.subpopT == Vertical)
	//	main_score = score.GetColumns(score.col_num() - 1);
	//else main_score.NewExtern(score);

	//first pass
	ulong h_offs = 0;
	Matrix mig_ind;
	for(ulong i=0; i < opt_.hspSize.size(); ++i) {
		switch(opt_.migPolicy) {
			case WorstBest:
				vm_srt_i.push_back(state_.mainScore.GetRows(h_offs, opt_.hspSize[i]).RawSort() + h_offs);
				break;
			default:
			case RandomRandom:
				vm_srt_i.push_back(indMatrix(1, opt_.hspSize[i]));
				for(ulong j=0; j < opt_.hspSize[i]; ++j)
					vm_srt_i[i][j] = j + h_offs;
				break;
		}
		h_offs += opt_.hspSize[i];
	}

	//second pass - do migration
	pointer_to_unary_function<ulong, ulong> rnd_f = ptr_fun(prg::randIntUB);
	ul_vec::iterator el_beg(elite_ind.begin()), el_end(elite_ind.end());
	indMatrix::r_iterator p_src, p_dst;
	for(ulong i = 0; i < opt_.hspSize.size(); ++i) {
		//set dst subpop indexes
		dir_ind[0] = (i + 1) % opt_.hspSize.size();
		if(dir_cnt == 2) {
			//backward index
			if(i == 0) dir_ind[1] = opt_.hspSize.size() - 1;
			else dir_ind[1] = (i - 1) % opt_.hspSize.size();
		}

		for(ulong j = 0; j < dir_cnt; ++j) {
			//how many will migrate?
			nMigrators = min(opt_.hspSize[i], opt_.hspSize[dir_ind[j]]);
			nMigrators = min<ulong>(nMigrators - opt_.eliteCount, ha_round(nMigrators*opt_.migFraction));
			//migrate
			p_src = vm_srt_i[i].begin();
			if(opt_.migPolicy == RandomRandom) {
				p_dst = vm_srt_i[dir_ind[j]].begin();
				std::random_shuffle(p_src, vm_srt_i[i].end(), rnd_f);
				std::random_shuffle(p_dst, vm_srt_i[dir_ind[j]].end(), rnd_f);
			}
			else p_dst = vm_srt_i[dir_ind[j]].end() - nMigrators;

			for(ulong k = 0; k < nMigrators; ++k) {
				//elite won't be rplaced
				while(find(el_beg, el_end, (ulong)*p_dst) != el_end)
					++p_dst;

				pop.SetRows(pop.GetRows(*p_src), *p_dst);
				score.SetRows(score.GetRows(*p_src), *p_dst);
				++p_src; ++p_dst;
			}
		}
	}
}

/*
Matrix ga::ScalingCall(const Matrix& scores, ulong nParents)
{
	switch(opt_.scalingT) {
		case Proportional:
			return ScalingProp(scores, nParents);
		case ProportionalClassic:
			return ScalingPropClassic(scores, nParents);
		default:
		case Rank:
			return ScalingRank(scores, nParents);
	}
}

Matrix ga::SelectionCall(const Matrix& expect, ulong nParents)
{
	switch(opt_.selectionT) {
		case Roulette:
			return SelectionRoulette(expect, nParents);
		default:
		case StochasticUniform:
			return SelectionStochUnif(expect, nParents);
		case Tournament:
			return SelectionTournament(expect, nParents);
		case UniformSelection:
			return SelectionUniform(expect, nParents);
		case Once:
			return SelectionOnce(expect, nParents);
		case Sort:
			return SelectionSort(expect, nParents);
	}
}

Matrix ga::CrossoverCall(const Matrix& pop, const Matrix& scores, const Matrix& parents)
{
	switch(opt_.crossoverT) {
		default:
		case Heuristic:
			return CrossoverHeuristic(pop, scores, parents);
		case OnePoint:
			return CrossoverOnePoint(pop, scores, parents);
		case TwoPoint:
			return CrossoverTwoPoint(pop, scores, parents);
	}
}

Matrix ga::MutationCall(const Matrix& pop, const Matrix& scores, const Matrix& parents)
{
	return Mutation(pop, scores, parents, opt_.initRange);
}

Matrix ga::CreationCall(void)
{
	switch(opt_.creationT) {
		default:
		case UniformCreation:
			return CreationUniform();
	}
}
*/

void ga::_sepRepeats(Matrix& source, const ulMatrix& rep_ind, Matrix* pRepeats, ul_vec* pPrevReps)
{
	//repeats.NewMatrix(0, 0);
	if(rep_ind.row_num() == 0) return;

	ulMatrix mInd = rep_ind.GetColumns(0);
	indMatrix scInd = mInd.RawSort();
	for(long i=mInd.size() - 1; i>=0; --i) {
		if(pRepeats) *pRepeats &= source.GetRows(mInd[i]);
		if(pPrevReps) pPrevReps->push_back(rep_ind(scInd[i], 1));
		source.DelRows(mInd[i]);
	}
}

void ga::PushGeneration(const Matrix& thisPop, const Matrix& thisScore, const ulMatrix& rep_ind)
{
	if(!opt_.globalSearch || !opt_.calcUnique) return;
	ulMatrix this_rep = rep_ind.GetColumns(0).Sort();
	ulr_iterator p_beg(this_rep.begin()), p_end(this_rep.end());
	for(ulong i=0; i<thisPop.row_num(); ++i) {
		if(!binary_search(p_beg, p_end, i)) {
			state_.stackPop &= thisPop.GetRows(i);
			state_.stackScore &= thisScore.GetRows(i);
		}
	}
	//test
	//cout << state_.stackPop.get_container()->capacity();
}

Matrix ga::StepGA(const Matrix& thisPop, const Matrix& thisScore, const Matrix& mutRange,
				  ul_vec& elite_ind, ul_vec& addon_ind, ulong nAddonInd, ulong nNewKids)
{
	Matrix nextPop;
	ulong nGenLen = thisPop.col_num();
	ulong popSize = thisPop.row_num();
	//reserve mem
	nextPop.reserve(popSize*nGenLen);

	//sort scores
	Matrix sortedSc;
	sortedSc = thisScore;
	indMatrix k = sortedSc.RawSort();

	//elite kids
	ulong nEliteKids = opt_.eliteCount;
	elite_ind.clear();
	elite_ind.reserve(nEliteKids);
	for(ulong i=0; i<nEliteKids; ++i) {
		nextPop &= thisPop.GetRows(k[i]);
		elite_ind.push_back(i);
	}

	//addon kids
	addon_ind.clear();
	addon_ind.reserve(opt_.addonCount);
	if(opt_.h_scheme != ClearGA && opt_.addonCount != 0) {
		Matrix h_addons, real_pop, predict, h_chrom;
		string a_name;
		//double dPredict;
		spAddonObj pAddon;

		try {
			//for(ulong i=0; i<opt_.addonCount; ++i) {
			if(opt_.useBitString) real_pop <<= InterpretBitPop(thisPop);
			else real_pop <<= thisPop;

			nAddonInd = min<ulong>(nAddonInd, apAddon_.size() - 1);
			pAddon = apAddon_[nAddonInd];
			a_name = pAddon->GetName();
			predict <<= pAddon->GetAddon(real_pop, thisScore, *this, h_addons);
			for(ulong j = 0; j < h_addons.row_num(); ++j) {
				if(predict[j] >= ERR_VAL) {
					_print_err("WRN GA: Error occured during addon creation! This individual won't be added");
					continue;
				}
				h_chrom <<= h_addons.GetRows(j);
				if(h_chrom == h_chrom) {		//check fo NaN's
					cout << "GA: " << pAddon->GetName() << " addon " << j << " prediction = " << predict[j] << endl;
					if(opt_.useBitString) h_chrom <<= Convert2BitPop(h_chrom);
					nextPop &= h_chrom;
					addon_ind.push_back(nextPop.row_num() - 1);
				}
				else
					_print_err("WRN GA: addon contains NaN numbers! Won't be added");
			}
			//}
		}
		catch(alg_except& ex) {
			_print_err(ex.what());
		}
		catch(exception& ex) {
			_print_err(ex.what());
		}
		catch(...) {
			_print_err("ERR: Unknown run-time error occured during addons creation. No individuals added");
		};
	}

	//count parents & kids
	if(nNewKids == 0) nNewKids = opt_.popSize;
	else nNewKids = max(nextPop.row_num(), nNewKids);
	ulong nXoverKids = ha_round(opt_.xoverFraction*(nNewKids - nextPop.row_num()));
	ulong nXoverParents = _getXoverParentsCount(nXoverKids);
	ulong nRestKids = nNewKids - nextPop.row_num() - nXoverKids;
	ulong nParents = nXoverParents + nRestKids;

	//standart ga operators
	//scaling
	Matrix expect = (this->*_pScalingFcn)(thisScore);
	//selection
	indMatrix parents = (this->*_pSelectionFcn)(expect, nParents);
	//shuffle parents
	//random_shuffle(parents.begin(), parents.end());
	pointer_to_unary_function<ulong, ulong> rnd_f = ptr_fun(prg::randIntUB);
	std::random_shuffle(parents.begin(), parents.end(), rnd_f);

	//collect not touched by crossover parents
	Matrix newKids; newKids.reserve(nRestKids*nGenLen);
	for(ulong i = 0; i < nRestKids; ++i)
		newKids &= thisPop.GetRows(parents[i]);
	parents.DelColumns(0, nRestKids);

	//crossover operator
	Matrix xoverKids = (this->*_pCrossoverFcn)(thisPop, thisScore, parents);
	if(xoverKids.row_num() > nXoverKids)
		xoverKids.DelRows(nXoverKids - 1, xoverKids.row_num() - nXoverKids);

	//mutation operator
	if(opt_.mutSpace == PrevPopRest)
		Mutation(newKids, mutRange);
	else if(opt_.mutSpace == CrossoverKids)
		Mutation(xoverKids, mutRange);

	newKids &= xoverKids;
	if(opt_.mutSpace == WholePopulation)
		Mutation(newKids, mutRange);

	nextPop &= newKids;

	//state_.bestGens = eliteKids;
	return nextPop;
}

Matrix ga::HSPStepGA(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind)
{
	//Matrix newPop, rep_ind, bestGens, bestScore;
	Matrix newPop;
	//reserve mem
	newPop.reserve(thisPop.size());
	ul_vec sp_ei, sp_ai;
	ulong h_offs = 0;
	//clear indicies
	elite_ind.clear();
	addon_ind.clear();
	for(ulMatrix::r_iterator p_hsp(opt_.hspSize.begin()); p_hsp < opt_.hspSize.end(); ++p_hsp) {
		newPop &= StepGA(thisPop.GetRows(h_offs, *p_hsp), thisScore.GetRows(h_offs, *p_hsp), opt_.initRange,
			sp_ei, sp_ai, p_hsp - opt_.hspSize.begin(), *p_hsp);
		//correct subpop elite indicies
		transform(sp_ei.begin(), sp_ei.end(), sp_ei.begin(), bind2nd(plus<ulong>(), h_offs));
		elite_ind.insert(elite_ind.end(), sp_ei.begin(), sp_ei.end());
		//correct subpop addon indicies
		transform(sp_ai.begin(), sp_ai.end(), sp_ai.begin(), bind2nd(plus<ulong>(), h_offs));
		addon_ind.insert(addon_ind.end(), sp_ai.begin(), sp_ai.end());
		//bestGens &= state_.bestGens;
		//bestScore &= state_.bestScore;

		//if(rep_ind.row_num() > 0) {
		//	if(h_offs > 0) {
		//		col_iterator p_beg(rep_ind), p_end(rep_ind);
		//		p_beg = rep_ind.begin(); p_end = rep_ind.begin() + 1;
		//		transform(p_beg, p_end, p_beg, bind2nd(plus<double>(), h_offs));
		//	}
		//	reps &= rep_ind;
		//}

		h_offs += *p_hsp;
	}

	//sort elite indicies
	Matrix elScore(elite_ind.size(), 1);
	for(ulong i=0; i<elite_ind.size(); ++i)
		elScore[i] = thisScore[elite_ind[i]];
	indMatrix mInd = elScore.RawSort();
	ul_vec new_ei(elite_ind.size());
	for(ulong i=0; i<new_ei.size(); ++i)
		new_ei[i] = elite_ind[mInd[i]];
	elite_ind = new_ei;
	//rep_ind = bestScore.RawSort();
	//state_.bestScore = bestScore;
	//state_.bestGens.NewMatrix(0, 0);
	//for(r_iterator p_ind(rep_ind.begin()); p_ind != rep_ind.end(); ++p_ind)
	//	state_.bestGens &= bestGens.GetRows(*p_ind);

	return newPop;
}

Matrix ga::VSPStepGA(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind)
{
	Matrix subpop, part, cur_part, main_score, newPop, expect;
	indMatrix parents, srt_ind;
	ul_vec sp_ei, sp_ai;
	ulong h_offs = 0, v_offs = 0;
	vector<Matrix> vm_subpop, vm_subscore, vm_mainss;

	elite_ind.clear();
	addon_ind.clear();
	//reserve mem
	newPop.reserve(thisPop.size());
	//split population into subpops
	for(ulong i = 0; i < opt_.vspSize.size(); ++i) {
		vm_subpop.push_back(thisPop.GetRows(h_offs, opt_.hspSize[i]).GetColumns(v_offs, opt_.vspSize[i]));
		vm_subscore.push_back(thisScore.GetRows(h_offs, opt_.hspSize[i]).GetColumns(i));
		//vm_subpop.push_back(thisPop.GetColumns(v_offs, opt_.vspSize[i]));
		//vm_subscore.push_back(thisScore.GetColumns(i));
		vm_mainss.push_back(state_.mainScore.GetRows(h_offs, opt_.hspSize[i]));
		h_offs += opt_.hspSize[i];
		v_offs += opt_.vspSize[i];
	}

	//main_score = thisScore.GetColumns(thisScore.col_num() - 1);
	h_offs = 0; v_offs = 0;
	pointer_to_unary_function<ulong, ulong> rnd_f = ptr_fun(prg::randIntUB);
	for(ulong i = 0; i < vm_subpop.size(); ++i) {
		//manually find best
		srt_ind = state_.mainScore.GetRows(h_offs, opt_.hspSize[i]).RawSort();
		for(ulong j=0; j < opt_.eliteCount; ++j) {
			elite_ind.push_back(j + h_offs);
			newPop &= thisPop.GetRows(srt_ind[j] + h_offs);
		}

		cur_part = StepGA(vm_subpop[i], vm_subscore[i], opt_.initRange.GetColumns(v_offs, opt_.vspSize[i]),
			sp_ei, sp_ai, i, opt_.hspSize[i]);
		//remove elite kids
		cur_part.DelRows(0, sp_ei.size());
		//correct addon indicies
		transform(sp_ai.begin(), sp_ai.end(), sp_ai.begin(), bind2nd(plus<ulong>(), h_offs));
		addon_ind.insert(addon_ind.end(), sp_ai.begin(), sp_ai.end());

		//append other parts;
		subpop.NewMatrix(0, 0);
		for(ulong j = 0; j < vm_subpop.size(); ++j) {
			if(j == i)
				subpop |= cur_part;
			else {
				expect = (this->*_pScalingFcn)(vm_subscore[j]);
				parents = (this->*_pSelectionFcn)(expect, cur_part.row_num());
				std::random_shuffle(parents.begin(), parents.end(), rnd_f);
				part.NewMatrix(0, 0);
				part.reserve(cur_part.row_num()*vm_subpop[j].col_num());
				for(ind_iterator pos(parents.begin()); pos != parents.end(); ++pos)
					part &= vm_subpop[j].GetRows(*pos);
				subpop |= part;
			}
		}
		newPop &= subpop;
		h_offs += opt_.hspSize[i];
		v_offs += opt_.vspSize[i];
	}

	//sort elite indicies
	Matrix elScore(elite_ind.size(), 1);
	for(ulong i=0; i<elite_ind.size(); ++i)
		elScore[i] = state_.mainScore[elite_ind[i]];
	srt_ind = elScore.RawSort();
	ul_vec new_ei(elite_ind.size());
	for(ulong i=0; i<new_ei.size(); ++i)
		new_ei[i] = elite_ind[srt_ind[i]];
	elite_ind = new_ei;

	return newPop;
}

Matrix ga::VSPStepGAalt(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind)
{
	Matrix subpop, old_subpop, cur_row, newPop, replacer;
	ul_vec sp_ei, sp_ai, free_ind_t, free_ind;
	ul_vec::iterator beg_ai, end_ai;

	elite_ind.clear();
	addon_ind.clear();

	int save = opt_.h_scheme;
	if(opt_.sepAddonForEachVSP)
		//disable hybrid scheme
		opt_.h_scheme = ClearGA;
	//big GA
	newPop <<= StepGA(thisPop, state_.mainScore, opt_.initRange, elite_ind, addon_ind);
	if(opt_.sepAddonForEachVSP)
		//restore gybrid scheme
		opt_.h_scheme = save;
	else
		//disable hybrid scheme
		opt_.h_scheme = ClearGA;

	//prepare template of indexes that can be replaced
	for(ulong i = 0; i < newPop.row_num(); ++i)
		if(find(elite_ind.begin(), elite_ind.end(), i) == elite_ind.end())
			free_ind_t.push_back(i);

	//ul_vec::iterator beg_ei(elite_ind.begin()), end_ei(elite_ind.end());

	//debug
	//ofstream vspf_src("vsp_src.txt", ios::out | ios::trunc);
	//ofstream vspf("vsp.txt", ios::out | ios::trunc);
	//ofstream vspf_dst("vsp_dst.txt", ios::out | ios::trunc);
	//
	prg::prepare_streams(opt_.vspSize.size());
	ulong nKids;
	ulong v_offs = 0, ind, rep_ind, pool_ind;
	for(ulong i = 0; i < opt_.vspSize.size(); ++i) {
		nKids = min<ulong>(newPop.row_num()*opt_.vspFract[i], newPop.row_num() - elite_ind.size());
		if(nKids == 0) {
			v_offs += opt_.vspSize[i];
			continue;
		}
		//debug
		//subpop <<= thisPop.GetColumns(v_offs, opt_.vspSize[i]) | thisScore.GetColumns(i);
		//subpop.Print(vspf_src) << endl;
		//
		subpop <<= StepGA(thisPop.GetColumns(v_offs, opt_.vspSize[i]), thisScore.GetColumns(i),
			opt_.initRange.GetColumns(v_offs, opt_.vspSize[i]), sp_ei, sp_ai, i, nKids);
		//old_subpop <<= newPop.GetColumns(v_offs, opt_.vspSize[i]);
		//debug
		//subpop.Print(vspf) << endl;
		//
		//check for repeats

		//we start from clear template
		free_ind = free_ind_t;
		beg_ai = sp_ai.begin(); end_ai = sp_ai.end();
		prg::switch_stream(i + 1);
		for(ulong j = 0; j < subpop.row_num(); ++j) {
			replacer <<= subpop.GetRows(j);
			//choose random index from free indexes pool
			pool_ind = prg::randIntUB(free_ind.size());
			ind = free_ind[pool_ind];
			//do a replace for this index
			cur_row <<= newPop.GetRows(ind);
			cur_row.SetColumns(replacer, v_offs, replacer.col_num());
			newPop.SetRows(cur_row, ind);
			//remove used index from pool
			free_ind.erase(free_ind.begin() + pool_ind);

			if(opt_.sepAddonForEachVSP) {
				//check if it is an addon
				if(find(beg_ai, end_ai, j) != end_ai)
					addon_ind.push_back(ind);
			}
		}
		//debug
		//subpop <<= newPop.GetColumns(v_offs, opt_.vspSize[i]);
		//subpop.Print(vspf_dst) << endl;
		//

		//h_offs += opt_.hspSize[i];
		v_offs += opt_.vspSize[i];
	}

	prg::switch_stream(0);
	if(!opt_.sepAddonForEachVSP)
		//restore gybrid scheme
		opt_.h_scheme = save;

	return newPop;
}

ulMatrix ga::FindRepeats(const Matrix where, const Matrix& what)
{
	ulMatrix reps, rep_row(1, 2);
	//if(opt_.globalSearch) where.NewExtern(state_.stackPop);
	//else where.NewExtern(state_.lastPop);
	ulong rep_ind, n = where.row_num();
	for(ulong i=0; i<what.row_num(); ++i) {
		rep_ind = where.RRowInd(what.GetRows(i));
		if(rep_ind < n) {
			rep_row[0] = i;
			rep_row[1] = rep_ind;
			reps &= rep_row;
		}
	}

	return reps;
}

ulMatrix ga::EnsureUnique(Matrix& nextPop, const Matrix& thisPop, const Matrix& thisScore, const ul_vec& elite_ind)
{
	ulong nUnique;
	ulMatrix reps;
	if(!opt_.calcUnique) return reps;

	if(opt_.globalSearch)
		reps = FindRepeats(state_.stackPop, nextPop);
	else
		reps = FindRepeats(thisPop, nextPop);
	nUnique = nextPop.row_num() - reps.row_num();

	if(nUnique < opt_.minUnique) {
		//add new generation :)
		Matrix addPop;
		ulMatrix addReps;
		ul_vec addEliteInd, addAddonInd;
		ulong save_ec = opt_.eliteCount, nCnt, cur_ind, rep_ind;
		int save_gscheme = opt_.h_scheme;
		ul_vec::const_iterator p_beg(elite_ind.begin()), p_end(elite_ind.end());

		opt_.eliteCount = 0;
		opt_.h_scheme = ClearGA;
		while(nUnique < opt_.minUnique && reps.row_num() > elite_ind.size()) {
			addPop = StepGACall(thisPop, thisScore, addEliteInd, addAddonInd);
			//get repeats for new generation
			if(opt_.globalSearch)
				addReps = FindRepeats(state_.stackPop & nextPop, addPop);
			else
				addReps = FindRepeats(thisPop & nextPop, addPop);
			_sepRepeats(addPop, addReps);
			//how many we can add?
			nCnt = min(min(opt_.minUnique - nUnique, reps.row_num() - elite_ind.size()), addPop.row_num());
			//replace row in nextPop
			//rep_ind = reps.row_num() - 1;
			while(nCnt > 0) {
				do {
					rep_ind = prg::randIntUB(reps.row_num());
				} while(find(p_beg, p_end, reps(rep_ind, 0)) != p_end);

				cur_ind = prg::randIntUB(addPop.row_num());
				nextPop.SetRows(addPop.GetRows(cur_ind), reps(rep_ind, 0));
				addPop.DelRows(cur_ind);
				reps.DelRows(rep_ind);
				++nUnique; --nCnt;
				//--rep_ind;
			}
		}
		opt_.eliteCount = save_ec;
		opt_.h_scheme = save_gscheme;
	}

	return reps;
}

Matrix ga::StepGACall(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind)
{
	switch(opt_.subpopT) {
		default:
		case NoSubpops:
			return StepGA(thisPop, thisScore, opt_.initRange, elite_ind, addon_ind);
		case Horizontal:
			return HSPStepGA(thisPop, thisScore, elite_ind, addon_ind);
		case Vertical:
			return VSPStepGAalt(thisPop, thisScore, elite_ind, addon_ind);
	}
}

Matrix ga::_restoreScore(const Matrix& newScore, const ulMatrix& reps)
{
	Matrix prevScore;
	if(opt_.globalSearch) prevScore <<= state_.stackScore;
	else prevScore <<= state_.lastScore;
	Matrix score(reps.row_num() + newScore.row_num(), prevScore.col_num());
	ulMatrix mInd = reps.GetColumns(0);
	indMatrix scInd = mInd.RawSort();
	ulong rsz = mInd.size(), ri = 0, nri = 0;
	for(ulong i = 0; i < score.size(); ++i) {
		if(ri < rsz && mInd[ri] == i)
			score.SetRows(prevScore.GetRows(reps(scInd[ri++], 1)), i);
		else
			score.SetRows(newScore.GetRows(nri++), i);
	}
	return score;
}

Matrix ga::FitnessFcnCall(const Matrix& pop, const ulMatrix& reps)
{
	//ulong popSize, genLen, allSize;
	//allSize = pop.GetSize(&popSize, &genLen);
	Matrix newScore, real_pop;
	real_pop = pop;
	if(opt_.calcUnique && reps.size() > 0) {
		_sepRepeats(real_pop, reps);
		//real_pop = pop.GetRows(reps.size(), pop.row_num() - reps.size());
	}

	if(real_pop.row_num() > 0) {
		if(opt_.useBitString) real_pop <<= InterpretBitPop(real_pop);

		if(opt_.subpopT != Vertical)
			newScore.NewMatrix(real_pop.row_num(), 1);
		else
			newScore.NewMatrix(real_pop.row_num(), opt_.vspSize.size() + 1);
		//Matrix chrom;
		if(opt_.vectorized)
			(*_pFitFunCallback)(real_pop.col_num(), real_pop.row_num(), &real_pop[0], newScore.GetBuffer());
		else {
			for(ulong i=0; i<real_pop.row_num(); ++i)
				(*_pFitFunCallback)(real_pop.col_num(), 1, &real_pop(i, 0), &newScore(i, 0));
		}
	}

	if(opt_.calcUnique && reps.size() > 0)
		newScore = _restoreScore(newScore, reps);
	return newScore;
}

void ga::prepare2run(int genomeLength, bool bReadOptFromIni)
{
	if(state_.nStatus != Idle && state_.nStatus != FinishError)
		throw ga_except("ERR: Previous GA must be finished before you start a new one!");
	state_.nStatus = Working;

	//debug
	//throw ga_except("Test exception in _prepare2run");

	if(bReadOptFromIni)
		ReadOptions();
	SetOptions();

	if(!opt_.useBitString)	GenLen_ = genomeLength;
	else GenLen_ = genomeLength*opt_.bitsPerVar;
	FixInitRange(genomeLength);

	//set pointers to functions
	switch(opt_.scalingT) {
		case Proportional:
			_pScalingFcn = &ga::ScalingProp;
			break;
		case ProportionalMean:
			_pScalingFcn = &ga::ScalingPropMean;
			break;
		case ProportionalInv:
			_pScalingFcn = &ga::ScalingPropInv;
			break;
		case ProportionalTimeScaled:
			_pScalingFcn = &ga::ScalingPropTime;
			//state_.ffsc_factor =
			break;
		case ProportionalSigmaScaled:
			_pScalingFcn = &ga::ScalingPropSigma;
			break;
		case Rank:
			_pScalingFcn = &ga::ScalingRank;
			break;
		default:
		case RankSqr:
			_pScalingFcn = &ga::ScalingRankSqr;
			break;
		case RankExp:
			_pScalingFcn = &ga::ScalingRankSqr;
			break;
	}

	switch(opt_.selectionT) {
		case Roulette:
			_pSelectionFcn = &ga::SelectionRoulette;
			break;
		default:
		case StochasticUniform:
			_pSelectionFcn = &ga::SelectionStochUnif;
			break;
		case Tournament:
			_pScalingFcn = &ga::ScalingSimplest;
			_pSelectionFcn = &ga::SelectionTournament;
			break;
		case UniformSelection:
			_pScalingFcn = &ga::ScalingSimplest;
			_pSelectionFcn = &ga::SelectionUniform;
			break;
		case Once:
			_pSelectionFcn = &ga::SelectionOnce;
			break;
		case Sort:
			_pSelectionFcn = &ga::SelectionSort;
			break;
	}

	switch(opt_.crossoverT) {
		default:
		case Heuristic:
			_pCrossoverFcn = &ga::CrossoverHeuristic;
			break;
		case OnePoint:
			_pCrossoverFcn = &ga::CrossoverOnePoint;
			break;
		case TwoPoint:
			_pCrossoverFcn = &ga::CrossoverTwoPoint;
			break;
		case UniformCrossover:
			_pCrossoverFcn = &ga::CrossoverUniform;
			break;
		case Flat:
			_pCrossoverFcn = &ga::CrossoverFlat;
			break;
		case Arithmetic:
			_pCrossoverFcn = &ga::CrossoverArithmetic;
			break;
		case BLX:
			_pCrossoverFcn = &ga::CrossoverBLX;
			break;
		case SBX:
			_pCrossoverFcn = &ga::CrossoverSBX;
			break;
	}

	//_pMutationFcn = &ga::Mutation;
	switch(opt_.mutationT) {
		default:
		case UniformMutation:
			_pMutationFcn = &ga::MutateUniform;
			break;
		case NonUniformMutation:
			_pMutationFcn = &ga::MutateNonUniform;
			break;
		case NormalMutation:
			_pMutationFcn = &ga::MutateNormal;
			break;
	}

	if(opt_.creationT != Manual) _pCreationFcn = &ga::CreationUniform;

	//switch(opt_.subpopT) {
	//	default:
	//	case NoSubpops:
	//		_pStepGAFcn = &ga::StepGA;
	//		break;
	//	case Horizontal:
	//		_pStepGAFcn = &ga::HSPStepGA;
	//		break;
	//	case Vertical:
	//		_pStepGAFcn = &ga::VSPStepGA;
	//		break;
	//}

	//addons initialization
	Matrix subRng;
	ulong v_offs = 0;
	for(ulong i=0; i<apAddon_.size(); ++i) {
		//if(bReadOptFromIni) apAddon_[i]->ReadOptions();
		if(opt_.subpopT == Vertical && opt_.sepAddonForEachVSP) {
			subRng = opt_.initRange.GetColumns(v_offs, opt_.vspSize[i]);
			v_offs += opt_.vspSize[i];
			apAddon_[i]->Init(opt_.addonCount, opt_.vspSize[i], subRng);
		}
		else
			apAddon_[i]->Init(opt_.addonCount, genomeLength, opt_.initRange);
	}
	//reserve mem for subpops best & addons fitness fcn
	state_.best_sp.NewMatrix(1, opt_.vspSize.size());
	state_.addons_ff.NewMatrix(1, opt_.addonCount);
	//state_.addons_ff.NewMatrix(1, apAddon_.size());

	//reserve memory for statistics
	stat_.clear();
	stat_.reserve(opt_.generations);

	//seed random generator
	prg::init();

	state_.tStart = time(NULL);
	state_.nGen = 0;
	state_.nStallGen = 0;
	state_.nChromCount = 0;
	//state_.nlsCount = opt_.popSize;
	//create initial population
	if(opt_.creationT != Manual) state_.lastPop = (this->*_pCreationFcn)();

	//reserve memory to speedup search
	if(opt_.calcUnique && opt_.globalSearch) {
		state_.stackPop.reserve(opt_.popSize*GenLen_*opt_.generations);
		if(opt_.subpopT != Vertical)
			state_.stackScore.reserve(opt_.popSize*opt_.generations);
		else
			state_.stackScore.reserve(opt_.popSize*opt_.generations*(opt_.vspSize.size() + 1));
	}
}

void ga::_filterParents(Matrix& thisPop, Matrix& thisScore, ulong parents_num)
{
	//apply selection to preserve mu parents
	Matrix expect = (this->*_pScalingFcn)(thisScore);
	indMatrix parents = (this->*_pSelectionFcn)(expect, parents_num).Sort();
	random_shuffle(parents.begin(), parents.end());
	Matrix newPop(parents_num, thisPop.col_num()), newScore(parents_num, thisScore.col_num());
	for(ulong i=0; i < parents.size(); ++i) {
		newPop.SetRows(thisPop.GetRows(parents[i]), i);
		newScore.SetRows(thisScore.GetRows(parents[i]), i);
	}
	thisPop <<= newPop;
	thisScore <<= newScore;
}

Matrix ga::Run(FitnessFcnCallback FitFcn, int genomeLength, bool bReadOptFromIni)
{
	try {
		//if(pOpt) memcpy(&opt_, pOpt, sizeof(gaOptions));
		//SetOptions(pOpt);
		prepare2run(genomeLength, bReadOptFromIni);

		Matrix nextPop, nextScore;
		state_.rep_ind.NewMatrix(0, 0);
		//compute fitness
		_pFitFunCallback = FitFcn;
		state_.lastScore <<= FitnessFcnCall(state_.lastPop, state_.rep_ind);
		state_.mainScore <<= state_.lastScore.GetColumns(state_.lastScore.col_num() - 1);
		state_.nChromCount = state_.lastPop.row_num();
		state_.last_min = state_.mainScore.Min();
		//state_.lastPop = pop;
		//state_.lastScore = score;
		InformWorld();
		++state_.nGen;
		if(opt_.timeLimit > 0) {
			if(time(NULL) - state_.tStart >= opt_.timeLimit)
				state_.nStatus = FinishTimeLim;
		}

		for(; state_.nGen < opt_.generations && state_.nStatus == Working; ++state_.nGen)
		{
			//dBest = score.Min();

			//save history
			PushGeneration(state_.lastPop, state_.lastScore, state_.rep_ind);
			//migration
			Migrate(state_.lastPop, state_.lastScore, state_.elite_ind);
			//next step
			//nextPop = (this->*_pStepGAFcn)(state_.lastPop, state_.lastScore, state_.elite_ind);
			nextPop <<= StepGACall(state_.lastPop, state_.lastScore, state_.elite_ind, state_.addon_ind);
			state_.rep_ind <<= EnsureUnique(nextPop, state_.lastPop, state_.lastScore, state_.elite_ind);
			nextScore <<= FitnessFcnCall(nextPop, state_.rep_ind);

			if(opt_.subpopT == Vertical)
				state_.mainScore <<= nextScore.GetColumns(state_.lastScore.col_num() - 1);
			else
				state_.mainScore <<= nextScore;

			if(opt_.calcUnique)
				state_.nChromCount += nextScore.row_num() - state_.rep_ind.row_num();
			else
				state_.nChromCount += nextScore.row_num();


			//gybrid scheme - debug version
			//if(opt_.h_scheme != ClearGA) {
			//	Matrix g_chrom, g_score;
			//	vector<ulong> g_rep;
			//	double dPredict = apAddon_->GetAddon(state_.lastPop, state_.lastScore, *this, g_chrom, g_rep);
			//	cout << "NN prediction = " << dPredict << endl;
			//	g_score = FitnessFcnCall(g_chrom, state_.lastScore, g_rep);
			//	cout << "Real NN score = " << g_score[0] << endl;
			//	bool bGood = false;
			//	ulong nInd;
			//	for(ulong i=0; i<nextScore.size(); ++i) {
			//		if(nextScore[i] >= g_score[0]) {
			//			bGood = true;
			//			nInd = i;
			//			break;
			//		}
			//	}
			//	if(bGood) {
			//		cout << "NN in population!" << endl;
			//		nextPop.SetRows(g_chrom, nInd);
			//		nextScore[nInd] = g_score[0];
			//	}
			//}

			switch(opt_.scheme) {
				case MuPlusLambda:
					state_.lastPop <<= nextPop & state_.lastPop;
					state_.lastScore <<= nextScore & state_.lastScore;
					_filterParents(state_.lastPop, state_.lastScore, opt_.popSize);
					break;
				default:
					state_.lastPop <<= nextPop;
					state_.lastScore <<= nextScore;
					break;
			}

			double curMin = state_.mainScore.Min();
			if(curMin == state_.last_min)
				++state_.nStallGen;
			else state_.nStallGen = 0;
			state_.last_min = curMin;

			////find best addon
			//if(state_.addon_ind.size() > 0) {
			//	ul_vec::iterator pos(state_.addon_ind.begin());
			//	state_.best_addon = state_.mainScore[*pos];
			//	for(++pos; pos != state_.addon_ind.end(); ++pos) {
			//		if(state_.mainScore[*pos] < state_.best_addon)
			//			state_.best_addon = state_.mainScore[*pos];
			//	}
			//}
			//else state_.best_addon = 0;

			InformWorld();

			if(state_.nStallGen == opt_.stallGenLimit) {
				state_.nStatus = FinishStallGenLim;
				//break;
			}
			else if(opt_.timeLimit > 0 && time(NULL) - state_.tStart >= opt_.timeLimit) {
				state_.nStatus = FinishTimeLim;
			}
			else if(opt_.useFitLimit && curMin <= opt_.fitLimit) {
				state_.nStatus = FinishFitLim;
			}
		}

		if(state_.nStatus == Working) state_.nStatus = FinishGenLim;
		//Matrix res = state_.bestGens.GetRows(0);
	}
	catch(alg_except ex) {
		state_.nStatus = FinishError;
		//state_.lastError = ex.what();
		_print_err(ex.what());
		FinishGA();
	}
	return FinishGA();
}

void ga::Start(double* pInitPop, int genomeLength, bool bReadOptFromIni)
{
	try {
		prepare2run(genomeLength, bReadOptFromIni);

		if(opt_.subpopT != Vertical)
			state_.lastScore.NewMatrix(opt_.popSize, 1);
		else
			state_.lastScore.NewMatrix(opt_.popSize, opt_.vspSize.size() + 1);
		state_.rep_ind.NewMatrix(0, 0);

		Matrix real_pop;
		if(opt_.creationT == Manual) {
			real_pop.NewMatrix(opt_.popSize, genomeLength, pInitPop);
			if(opt_.useBitString)
				state_.lastPop <<= Convert2BitPop(real_pop);
			else state_.lastPop <<= real_pop;
		}
		else {
			//copy initial population
			if(opt_.useBitString) real_pop <<= InterpretBitPop(state_.lastPop);
			else real_pop <<= state_.lastPop;
			memcpy(pInitPop, real_pop.GetBuffer(), real_pop.raw_size());
		}
	}
	catch(alg_except& ex) {
		state_.nStatus = FinishError;
		//state_.lastError = ex.what();
		_print_err(ex.what());
		FinishGA();
	}
}

bool ga::NextPop(double* pPrevScore, double* pNextPop, unsigned long* pPopSize)
{
	bool bRes = false;
	try {
		if(state_.nStatus != Working) {
			if(state_.nStatus != Idle) FinishGA(pNextPop, pPrevScore);
			throw ga_except("Object isn't in working state! Call Start before NextPop!");
		}

		Matrix newScore(state_.lastScore.row_num() - state_.rep_ind.row_num(), state_.lastScore.col_num(), pPrevScore);
		if(opt_.calcUnique && state_.rep_ind.row_num() > 0) {
			newScore <<= _restoreScore(newScore, state_.rep_ind);
			//Matrix repScore(state_.rep_ind.size(), 1);
			//for(ulong i=0; i<state_.rep_ind.size(); ++i)
			//	repScore[i] = state_.lastScore[state_.rep_ind[i]];
			//state_.lastScore = repScore & newScore;
		}
		//else state_.lastScore.SetBuffer(pPrevScore);
		if(opt_.scheme == MuPlusLambda && state_.nGen > 0) {
			state_.lastScore &= newScore;
			_filterParents(state_.lastPop, state_.lastScore, opt_.popSize);
		}
		else state_.lastScore <<= newScore;

		if(opt_.subpopT == Vertical)
			state_.mainScore <<= state_.lastScore.GetColumns(state_.lastScore.col_num() - 1);
		else
			state_.mainScore <<= state_.lastScore;

		if(opt_.calcUnique)
			state_.nChromCount += state_.lastPop.row_num() - state_.rep_ind.row_num();
		else
			state_.nChromCount += state_.lastPop.row_num();

		//find current best
		double curMin = state_.mainScore.Min();
		if(state_.nGen > 0 && curMin == state_.last_min)
			++state_.nStallGen;
		else state_.nStallGen = 0;
		state_.last_min = curMin;

		////find best addon
		//if(state_.addon_ind.size() > 0) {
		//	ul_vec::iterator pos(state_.addon_ind.begin());
		//	state_.best_addon = state_.mainScore[*pos];
		//	for(++pos; pos != state_.addon_ind.end(); ++pos) {
		//		if(state_.mainScore[*pos] < state_.best_addon)
		//			state_.best_addon = state_.mainScore[*pos];
		//	}
		//}
		//else state_.best_addon = 0;

		InformWorld();

		if(state_.nStallGen == opt_.stallGenLimit) {
			throw (int)FinishStallGenLim;
			//state_.nStatus = FinishStallGenLim;
			//FinishGA(pNextPop, pPrevScore);
			//return false;
		}
		++state_.nGen;
		if(state_.nGen == opt_.generations) {
			throw (int)FinishGenLim;
			//state_.nStatus = FinishGenLim;
			//FinishGA(pNextPop, pPrevScore);
			//return false;
		}
		if(opt_.timeLimit > 0) {
			if(time(NULL) - state_.tStart >= opt_.timeLimit) {
				throw (int)FinishTimeLim;
				//state_.nStatus = FinishTimeLim;
				//FinishGA(pNextPop, pPrevScore);
				//return false;
			}
		}

		//save history
		PushGeneration(state_.lastPop, state_.lastScore, state_.rep_ind);
		//migration
		Migrate(state_.lastPop, state_.lastScore, state_.elite_ind);
		//next step
		//Matrix nextPop = (this->*_pStepGAFcn)(state_.lastPop, state_.lastScore, state_.elite_ind);
		Matrix nextPop = StepGACall(state_.lastPop, state_.lastScore, state_.elite_ind, state_.addon_ind);
		state_.rep_ind <<= EnsureUnique(nextPop, state_.lastPop, state_.lastScore, state_.elite_ind);
		//for testing
		//Matrix ri = FindRepeats(state_.lastPop, nextPop);
		//state_.nlsCount += nextPop.row_num() - ri.row_num();
		//opt_.globalSearch = save;

		//copy next population
		Matrix real_pop;
		real_pop = nextPop;
		if(opt_.calcUnique)
			_sepRepeats(real_pop, state_.rep_ind);
			//real_pop = state_.lastPop.GetRows(state_.rep_ind.size(), *pPopSize);
		if(opt_.useBitString) real_pop <<= InterpretBitPop(real_pop);
		*pPopSize = real_pop.row_num();
		memcpy(pNextPop, real_pop.GetBuffer(), real_pop.raw_size());
		*pPrevScore = state_.last_min;

		//ga main scheme
		switch(opt_.scheme) {
			case MuPlusLambda:
				state_.lastPop &= nextPop;
				break;
			default:
				state_.lastPop <<= nextPop;
				break;
		}

		bRes = true;
	}
	catch(int status) {
		//normal finish
		state_.nStatus = status;
		FinishGA(pNextPop, pPrevScore);
	}
	catch(alg_except& ex) {
		//error
		state_.nStatus = FinishError;
		//state_.lastError = ex.what();
		_print_err(ex.what());
		FinishGA();
	}
	return bRes;
}

void ga::InformWorld(void)
{
	if(opt_.subpopT == Vertical) {
		//calc subpops best
		Matrix vspScore;
		ulong ind = 0;
		for(ulong i = 0; i < opt_.vspSize.size(); ++i) {
			vspScore <<= state_.lastScore.GetColumns(i);
			state_.best_sp[i] = vspScore.Min();
			//if(opt_.h_scheme != ClearGA)
			//	for(ulong j = 0; j < opt_.addonCount; ++j) {
			//		state_.addons_ff[ind] = vspScore[state_.addon_ind[ind]];
			//		++ind;
			//	}
		}
	}
	if(opt_.h_scheme != ClearGA) {
		//calc addons ff
		if(state_.addons_ff.size() != state_.addon_ind.size())
			state_.addons_ff.Resize(1, state_.addon_ind.size());
		if(opt_.sepAddonForEachVSP) {
			Matrix score = state_.mainScore;
			for(ulong i = 0; i < state_.addon_ind.size(); ++i) {
				if(opt_.subpopT == Vertical && i % opt_.addonCount == 0)
					score <<= state_.lastScore.GetColumns(i/opt_.addonCount);
				state_.addons_ff[i] = score[state_.addon_ind[i]];
			}
		}
		else {
			for(ulong i = 0; i < state_.addon_ind.size(); ++i)
				state_.addons_ff[i] = state_.mainScore[state_.addon_ind[i]];
		}
	}
	//show iteration info
	ostringstream osinfo;
	OutpIterRes(osinfo);
	//info = osinfo.str();

	cout << osinfo.str();
	//OutpIterRes(cout);
	if(!logFile_.is_open())
		logFile_.open(opt_.logFname.c_str(), ios::out | opt_.openMode);
	logFile_ << osinfo.str();
	logFile_.flush();
	//OutpIterRes(_logFile);

	if(opt_.logEveryPop) {
		if(!histFile_.is_open()) histFile_.open(opt_.histFname.c_str(), ios::out | opt_.openMode);
		Matrix real_pop;
		//real_pop = state_.lastPop;
		if(opt_.useBitString) real_pop = InterpretBitPop(state_.lastPop);
		else real_pop = state_.lastPop;
		for(ulong i=0; i<real_pop.row_num(); ++i) {
			real_pop.GetRows(i).Print(histFile_, false); //<< setw(Matrix::num_width) << state_.mainScore[i] << endl;
			state_.lastScore.GetRows(i).Print(histFile_);
		}
		histFile_ << endl;
		//_histFile.flush();
	}
}

std::ostream& ga::OutpIterRes(std::ostream& outs)
{
	if(state_.nGen == 0) {
		outs << setw(NW) << "Generation" << ' ' << setw(NW) << "f-count" << ' ' << setw(NW) << "best f(x)"
			<< ' ' << setw(NW) << "Mean" << ' ' << setw(NW) << "StallGen";
		ostringstream sfmt;
		if(opt_.subpopT == Vertical) {
			//vertical subpops best
			for(ulong i=0; i < opt_.vspSize.size(); ++i) {
				sfmt.str("");
				sfmt << "VSP" << i << " best";
				outs << ' ' << setw(NW) << sfmt.str();
			}
		}
		if(opt_.h_scheme != ClearGA) {
			for(ulong i=0; i < opt_.addonCount; ++i) {
				sfmt.str("");
				sfmt << "NN" << i << " best";
				outs << ' ' << setw(NW) << sfmt.str();
			}
		}
		outs << endl;
	}
	outs << setw(NW) << state_.nGen + 1 << ' ' << setw(NW) << state_.nChromCount << ' ' << setw(NW) << state_.last_min;
	//calc score mean
	double mean_ff = 0;
	if(opt_.excludeErrors) {
		ulong cnt = 0;
		for(r_iterator pos = state_.mainScore.begin(), end = state_.mainScore.end(); pos != end; ++pos)
			if(*pos < ERR_VAL) {
				mean_ff += *pos; ++cnt;
			}
		mean_ff /= (double)cnt;
	}
	else
		mean_ff = state_.mainScore.Mean();
	outs << ' ' << setw(NW) << mean_ff << ' ' << setw(NW) << state_.nStallGen;

	//push back statistics
	stat_.add_record(state_.nChromCount, state_.last_min, mean_ff, state_.nStallGen);

	if(opt_.subpopT == Vertical) {
		//vertical subpops best
		for(r_iterator pos(state_.best_sp.begin()); pos != state_.best_sp.end(); ++pos)
			outs << ' ' << setw(NW) << *pos;
	}
	if(opt_.h_scheme != ClearGA) {
		for(r_iterator pos(state_.addons_ff.begin()); pos != state_.addons_ff.end(); ++pos)
			outs << ' ' << setw(NW) << *pos;
	}
	outs << endl;
	return outs;
}

std::ostream& ga::OutpFinishInfo(std::ostream& outs, const Matrix& bestChrom, double bestScore)
{
	if(state_.nStatus != FinishError) outs << "Finished! ";
	switch(state_.nStatus) {
		case FinishGenLim:
			outs << "Generations limit reached.";
			break;
		case FinishStallGenLim:
			outs << "Stall generations limit reached.";
			break;
		case FinishUserStop:
			outs << "User stopped.";
			break;
		case FinishTimeLim:
			outs << "Time limit reached.";
			break;
		case FinishFitLim:
			outs << "Best fitness value limit reached.";
			break;
		case FinishError:
			outs << "Execution was breaked by error.";
			break;
	}
	if(state_.nStatus != FinishError) {
		outs << endl << "Best population found: " << endl;
		//state_.bestGens.GetRows(0).Print(outs);
		bestChrom.Print(outs);
		outs << "It's fitness value: " << bestScore << endl;
	}
	//else outs << endl << state_.lastError << endl;
	return outs;
}

Matrix ga::FinishGA(double* pBestPop, double* pBestScore)
{
	if(state_.nStatus != FinishError) {
		//if(state_.nStallGen > 0 && opt_.eliteCount > 0) {
		//	_bestChrom = state_.bestGens.GetRows(0);
		//	_bestScore = state_.bestScore[0];
		//}
		//else {
			indMatrix mInd = state_.mainScore.RawSort();
			bestChrom_ = state_.lastPop.GetRows(mInd[0]);
			bestScore_ = state_.mainScore[0];
		//}
		if(opt_.useBitString) bestChrom_ = InterpretBitPop(bestChrom_);
		if(pBestPop) memcpy(pBestPop, bestChrom_.GetBuffer(), bestChrom_.raw_size());
		if(pBestScore) *pBestScore = bestScore_;
	}

	OutpFinishInfo(cout, bestChrom_, bestScore_);
	if(!logFile_.is_open()) logFile_.open(opt_.logFname.c_str(), ios::out | opt_.openMode);
	OutpFinishInfo(logFile_, bestChrom_, bestScore_);
	//_logFile.flush();
	logFile_.close();
	if(histFile_.is_open()) {
		//_histFile.flush();
		histFile_.close();
	}
	if(errFile_.is_open()) errFile_.close();

	//delete all addons
	apAddon_.clear();
	//set idle status if no error
	if(state_.nStatus != FinishError) state_.nStatus = Idle;

	return bestChrom_;
}

void ga::Stop(void)
{
	if(state_.nStatus != Idle) state_.nStatus = FinishUserStop;
}

Matrix Read2rows(istream& is)
{
	Matrix res;
	string sWord;
	Matrix val(1, 1);
	Matrix row(1, 1);
	int nPos = 0;
	while(is >> sWord) {
		val[0] = atof(sWord.c_str());
		if(nPos == 0) row[0] = val[0];
		else row = row | val;
		++nPos;
		if(res.row_num() == 1 && nPos == res.col_num()) break;
		if(is.peek() == is.widen('\n')) {
			res = row;
			row.NewMatrix(1, 1);
			nPos = 0;
			is >> ignoreLine;
		}
	}
	res = res & row;
	return res;
}

bool ga::SetAddonOptions(const void* pOpt, ulong addon_num)
{
	if(pOpt && addon_num < apAddon_.size()) {
		apAddon_[addon_num]->SetOptions(pOpt);
		return true;
	}
	else return false;
}

void* ga::GetAddonOptions(ulong addon_num)
{
	if(addon_num < apAddon_.size())
		return apAddon_[addon_num]->GetOptions();
	else return NULL;
}

Matrix ga::InterpretBitPop(const Matrix& bit_pop)
{
	if(opt_.bitsPerVar <= 0) return bit_pop;

	ulong real_gl = bit_pop.col_num()/opt_.bitsPerVar;
	Matrix res(bit_pop.row_num(), real_gl);

	double dPart, dSum;
	ulong offset;
	for(ulong r=0; r<bit_pop.row_num(); ++r) //population rows
	{
		offset = 0;
		for(ulong i=0; i<real_gl; ++i)	//result columns
		{
			dPart = 0.5;
			dSum = 0;
			for(ulong j=offset; j<offset + opt_.bitsPerVar; ++j)	//bit_pop columns
			{
				if(bit_pop(r, j) > 0) dSum += dPart;
				dPart /= 2;
			}
			res(r, i) = opt_.initRange(0, i) + dSum*(opt_.initRange(1, i) - opt_.initRange(0, i));
			offset += opt_.bitsPerVar;
		}
	}

	return res;
}

Matrix ga::Convert2BitPop(const Matrix& pop)
{
	if(opt_.bitsPerVar <= 0) return pop;
	Matrix bit_pop(pop.row_num(), pop.col_num()*opt_.bitsPerVar);

	double dPart, dVal;
	ulong offset;
	for(ulong r=0; r<pop.row_num(); ++r) //population rows
	{
		offset = 0;
		for(ulong i=0; i<pop.col_num(); ++i)	//population columns
		{
			dVal = (pop(r, i) - opt_.initRange(0, i))/(opt_.initRange(1, i) - opt_.initRange(0, i));
			dPart = 0.5;
			for(ulong j=offset; j<offset + opt_.bitsPerVar; ++j)	//bit_pop columns
			{
				if(dVal > dPart) bit_pop(r, j) = 1;
				else bit_pop(r, j) = 0;
				dVal -= dPart;
				dPart /= 2;
			}
			offset += opt_.bitsPerVar;
		}
	}

	return bit_pop;
}

void ga::AddonsCreation()
{
	if(apAddon_.size() > 0) return;
	nn_addon* new_addon;
	string::size_type pos;
	string sName;
	//for(ulong i = 0; i < opt_.addonCount; ++i) {
		switch(opt_.h_scheme) {
			case UseNN:
				//char sNum[34];
				nnAddonOptions* pOpt;
				if(opt_.subpopT == NoSubpops || !opt_.sepAddonForEachVSP) {
					//my_sprintf(sName, "nna%i", i+1);
					new_addon = new nn_addon;
					apAddon_.push_back(smart_ptr<ga_addon>(new_addon));
					//add addon to options chain
					opt_.add_embopt(new_addon->opt_.get_iopt_ptr(), false);
				}
				else {
					for(ulong j=0; ; ++j) {
						if((opt_.subpopT == Horizontal && j >= opt_.hspSize.size()) || (opt_.subpopT == Vertical && j >= opt_.vspSize.size()))
							break;
						if(opt_.subpopT == Horizontal)
							my_sprintf(sName, "nna_hsp%i", j);
						else my_sprintf(sName, "nna_vsp%i", j);
						new_addon = new nn_addon(sName.c_str());
						apAddon_.push_back(smart_ptr<ga_addon>(new_addon));
						//add addon to options chain
						opt_.add_embopt(new_addon->opt_.get_iopt_ptr(), false);

						//for(int j = 1; ; ++j) {
						//	sName = new_addon->GetFname(j);
						//	if(sName.empty()) break;
						//	//if(sName.length() > 0) {
						//
						//	if((pos = sName.rfind('.')) != string::npos)
						//		sName.insert(pos, sNum);
						//	else sName += sNum;
						//	new_addon->SetFname(j, sName.c_str());
						//	//}
						//}
					}
				}
				break;
		}
	//}
}

ga_addon* const ga::GetAddonObject(ulong addon_num) const
{
	if(addon_num >= apAddon_.size()) return NULL;
	else return apAddon_[addon_num].get();
}

void ga::ValidateOptions()
{
	if(opt_.popSize < 1) {
		throw ga_except("ERR: Wrong population size specified - abort");
		//_print_err("Wrong population size specified - set to 60");
		//opt_.popSize = 60;
	}
	ulong popS = opt_.popSize;
	if(opt_.useBitString && opt_.bitsPerVar < 1) {
		_print_err("WRN: Wrong bits per variable specified - set to 10");
		opt_.bitsPerVar = 10;
	}

	if(opt_.subpopT == Vertical) {
		if(opt_.vspSize.size() == 0 || opt_.vspSize.Sum() == 0) {
			_print_err("WRN: Wrong vertical subpops sizes specified - subpopulations disabled");
			opt_.subpopT = NoSubpops;
		}
		else {
			if(opt_.useBitString) opt_.vspSize *= opt_.bitsPerVar;
			//correct horizontal subpop num
			if(opt_.hspSize.size() == 0) {
				opt_.hspSize.NewMatrix(1, opt_.vspSize.size());
				opt_.hspSize = 0;
				ulong nSum = opt_.vspSize.Sum();
				ulong offs = 0;
				for(ulong i = 0; i < opt_.vspSize.size() - 1; ++i) {
					opt_.hspSize[i] = ha_round(opt_.vspSize[i]*popS/nSum);
					offs += opt_.hspSize[i];
				}
				opt_.hspSize[opt_.hspSize.size() - 1] = popS - offs;
				_print_err("WRN: No horizontal subpops sizes specified (for vertical subpops) - auto corrected");
			}
			//else if(opt_.hspSize.size() != opt_.vspSize.size()) {
			//	_print_err("WRN: Wrong horizontal subpops sizes specified (for vertical subpops) - subpopulations disabled");
			//	opt_.subpopT = NoSubpops;
			//}
			//new algorithm - needs opt_.vspFract
			if(opt_.vspFract.size() == 0) {
				_print_err("WRN! Vertical subpops fractions not set. Set all = 0.8");
				opt_.vspFract.Resize(1, opt_.vspSize.size());
				opt_.vspFract = 0.8;
			}
			else if(opt_.vspFract.size() != opt_.vspSize.size()) {
				_print_err("WRN! Wrong vertical subpops fractions specified - subpopulations disabled");
				opt_.subpopT = NoSubpops;
			}
		}
	}
	else if(opt_.subpopT == Horizontal) {
		if(opt_.hspSize.size() == 0 || opt_.hspSize.Sum() == 0) {
			_print_err("WRN: Wrong horizontal subpops sizes specified - subpopulations disabled");
			opt_.subpopT = NoSubpops;
		}
		else
			popS = opt_.hspSize.Min();
	}
	if(opt_.subpopT == NoSubpops) {
		opt_.hspSize.NewMatrix(0, 0);
		opt_.vspSize.NewMatrix(0, 0);
	}

	if(opt_.eliteCount < 0 || opt_.eliteCount >= popS) {
		//_print_err("WRN: Elite count exeeds possible bounds - corrected");
		opt_.eliteCount = max<uint>(0, min<uint>(2, popS - 1));
	}
	if(opt_.h_scheme != ClearGA) {
		//_print_err("WRN: Addon childs count exeeds possible bounds - corrected");
		opt_.addonCount = max<uint>(0, min<uint>(opt_.addonCount, popS - opt_.eliteCount - 1));
		//if(popS - opt_.eliteCount - 1 > 0)
		//	opt_.addonCount = 1;
		//else opt_.addonCount = 0;
	}
	if(opt_.calcUnique) {
		//_print_err("WRN: Unique childs count exeeds possible bounds - corrected");
		opt_.minUnique = max<uint>(0, min<uint>(opt_.minUnique, opt_.popSize - opt_.eliteCount - 1));
	}

	if(opt_.generations == 0) {
		_print_err("WRN: Wrong generations count specified - set to 100");
		opt_.generations = 100;
	}
	opt_.mutProb = min(1., max(0., opt_.mutProb));
	opt_.xoverFraction = min(1., max(0., opt_.xoverFraction));
	//if(opt_.xoverHeuRatio < 0) opt_.xoverHeuRatio = 1.2;
	opt_.mutNormLawSigma2 = max(0., opt_.mutNormLawSigma2);
	if(opt_.migInterval == 0) {
		_print_err("WRN: Wrong migration interval specified - corrected");
		opt_.migInterval = 20;
	}
	opt_.migFraction = min(1., max(0., opt_.migFraction));
	if(opt_.scalingT == ProportionalTimeScaled && (opt_.ffscParam > 1 || opt_.ffscParam < 0)) {
		_print_err("WRN: Invalid scaling parameter specified for time scaling - set to 0.9");
		opt_.ffscParam = 0.1;
	}
	else if(opt_.scalingT == ProportionalSigmaScaled && (opt_.ffscParam < 0)) {
		_print_err("WRN: Invalid scaling parameter specified for sigma scaling - set to 2");
		opt_.ffscParam = 2;
	}
	else if(opt_.scalingT == Proportional && (opt_.ffscParam <= 0)) {
		_print_err("WRN: Bad scaling parameter specified for proportional scaling - set to 2");
		opt_.ffscParam = 2;
	}

	//FixInitRange();
}

void ga::FixInitRange(ulong genomeLength)
{
	//debug - dump existing range
	_print_err("MSG: Dump existing initial range");
	opt_.initRange.Print(cerr);

	if(opt_.initRange.row_num() == 2 && opt_.initRange.col_num() == genomeLength) return;
	double dLower, dUpper;
	if(opt_.initRange.row_num() != 2) {
		_print_err("WRN: Wrong initial range rows count - set range to [0, 1] for all variables");
		dLower = 0;
		dUpper = 1;
	}
	else if(opt_.initRange.col_num() != genomeLength) {
		_print_err("WRN: Wrong initial range columns count - corrected");
		dLower = opt_.initRange[0];
		dUpper = opt_.initRange(1, 0);
	}
	opt_.initRange.NewMatrix(2, genomeLength);
	Matrix row(1, genomeLength);
	row = dLower;
	opt_.initRange.SetRows(row, 0);
	row = dUpper;
	opt_.initRange.SetRows(row, 1);

	//debug - dump corrected range
	_print_err("MSG: Dump corrected initial range");
	opt_.initRange.Print(cerr);
}

void ga::ReadOptions(const char* pFName)
{
	opt_.ReadOptions(pFName);
	//AfterOptionsSet();
	////TODO: refactor this code
	//for(ulong i = 0; i < apAddon_.size(); ++i) {
	//	apAddon_[i]->ReadOptions();
	//}
	////opt_.read_embopt(pFName);
}

void ga::SetOptions(const gaOptions* pOpt)
{
	opt_.SetOptions(pOpt);
}

void ga::AfterOptionsSet()
{
	try {
		ValidateOptions();
		AddonsCreation();
	}
	catch(alg_except& ex) {
		_print_err(ex.what());
	}
}

