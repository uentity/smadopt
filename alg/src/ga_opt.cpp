#include "ga.h"
#include "prg.h"

#include <iostream>
#include <fstream>

using namespace GA;
using namespace std;

ga_opt::ga_opt(ga* self) : self_(self)
{
	//BeforeEmbOptRead = ga_opt::BeforeEmbOptRead_def;
}
//copy constructor
ga_opt::ga_opt(const ga_opt& opt) : wrapper_opt(opt), self_(opt.self_)
{
}
//constructor by interface
ga_opt::ga_opt(const wrapper_opt& opt) : wrapper_opt(opt)
{
	const ga_opt* p_go = dynamic_cast< const ga_opt* >(opt.get_iopt_ptr());
	if(p_go)
		self_ = p_go->self_;
	else throw ga_except("Invalid ga_opt initialization with wrong object");
}

void ga_opt::AfterOptionsSet()
{
	if(self_) self_->AfterOptionsSet();
}

void ga_opt::set_def_opt(bool create_defs)
{
	scheme = MuLambda;
	h_scheme = ClearGA;

	scalingT = RankSqr;
	selectionT = StochasticUniform;
	creationT = UniformCreation;
	crossoverT = Heuristic;
	mutationT = UniformMutation;
	mutSpace = PrevPopRest;
	subpopT = NoSubpops;

	//misc options
	eliteCount = 2;
	addonCount = 1;
	generations = 100;
	stallGenLimit = -1;
	useFitLimit = false;
	fitLimit = 0;
	logEveryPop = false;
	useBitString = false;
	bitsPerVar = 10;
	timeLimit = -1;
	calcUnique = false;
	globalSearch = false;
	minUnique = 0;
	sepAddonForEachVSP = false;

	//scaling parameters
	nTournSize = 4;
	minimizing = true;
	ffscParam = 0;

	//crossover parameters
	xoverFraction = 0.8;
	xoverHeuRatio = 1.2;
	xoverBLXAlpha = 0.5;
	xoverArithmeticRatio = 0.5;
	xoverSBXParam = 3;

	//mutation parameters
	mutProb = 0.01;
	mutNormLawSigma2 = 0.25;
	popSize = 60;
	mutNonUniformParam = 5;

	migPolicy = RandomRandom;
	migDirection = MigrateForward;
	migInterval = 20;
	migFraction = 0.2;

	initRange.NewMatrix(2, 1);
	initRange[0] = 0; initRange[1] = 1;

	iniFname_ = "ga.ini";
	logFname = "ga_log.txt";
	errFname = "ga_err.txt";
	histFname = "ga_hist.txt";
	openMode = ios::trunc;
}

/*
void ga_opt::ReadOptions(const char* pFname)
{
	thisOpt::ReadOptions(pFname);

	SetOptions();

	//ReadAddonOptions();
	//for(ulong i=0; i<_apAddon.size(); ++i)
	//	_apAddon[i]->ReadOptions();
}
*/

bool ga_opt::process_option(std::istream& inif, std::string& word)
{
	string sOpts = " GAScheme ScalingType SelectionType CrossoverType MutationType CreationType";
	sOpts += " PopulationSize EliteCount CrossoverFraction CrossoverHeuRatio MutationProb Generations StallGenLimit";
	sOpts += " Vectorized LogEveryPop PopInitRange MutationNormalLawSigma2 TimeLimit UseBitString BitsPerVariable HybridScheme";
	sOpts += " CalcUniqueOnly MinUnique SubPopType HSubPopSizes VSubPopSizes MigrationDirection MigrationInterval MigrationFraction";
	sOpts += " GlobalSearch VSubPopFractions CrossoverArithmeticRatio CrossoverBLXAlpha MutationSpace MutNonUniformParam";
	sOpts += " CrossoverSBXParam MigrationPolicy TournamentSize Minimizing ScalingParam SeparateAddonForEachVSP AddonCount";
	const string sScaling = " Proportional ProportionalMean ProportionalInv ProportionalTimeScaled ProportionalSigmaScaled Rank RankSqr RankExp";
	const string sSelection = " StochasticUniform Roulette UniformSelection Tournament Once Sort";
	const string sXover = " Heuristic OnePoint TwoPoint UniformCrossover Flat Arithmetic BLX SBX";
	const string sMut = " UniformMutation NormalMutation NonUniformMutation";
	const string sMutSpace = " WholePopulation CrossoverKids PrevPopRest";
	const string sCreation = " UniformCreation Manual";
	const string sScheme = " MuLambda MuPlusLambda";
	const string sGybridScheme = " ClearGA UseNN";
	const string sSubpop = " NotUsed Horizontal Vertical";
	const string sMigDir = " Forward Both";
	const string sMigPolicy = " WorstBest RandomRandom";

	string sTmp;
	std::istringstream is;
	bool need_skipline;
	int nPos;
	need_skipline = true;
	if((nPos = word_pos(sOpts, word)) > 0) {
		switch(nPos) {
			case 1:		//GASheme
				inif >> word;
				if((nPos = word_pos(sScheme, word)) > 0) scheme = nPos;
				break;
			case 2:		//ScalingType
				inif >> word;
				if((nPos = word_pos(sScaling, word)) > 0) scalingT = nPos;
				break;
			case 3:		//SelectionType
				inif >> word;
				if((nPos = word_pos(sSelection, word)) > 0) selectionT = nPos;
				break;
			case 4:		//CrossoverType
				inif >> word;
				if((nPos = word_pos(sXover, word)) > 0) crossoverT = nPos;
				break;
			case 5:		//MutationType
				inif >> word;
				if((nPos = word_pos(sMut, word)) > 0) mutationT = nPos;
				break;
			case 6:		//CreationType
				inif >> word;
				if((nPos = word_pos(sCreation, word)) > 0) creationT = nPos;
				break;
			case 7:		//PopulationSize
				inif >> popSize;
				break;
			case 8:
				inif >> eliteCount;
				break;
			case 9:
				inif >> xoverFraction;
				break;
			case 10:
				inif >> xoverHeuRatio;
				break;
			case 11:
				inif >> mutProb;
				break;
			case 12:
				inif >> generations;
				break;
			case 13:
				inif >> stallGenLimit;
				break;
			case 14:
				inif >> vectorized;
				break;
			case 15:
				inif >> logEveryPop;
				break;
			case 16:	//Population range
				inif >> ignoreLine;
				//_initRange = Read2rows(inif);
				initRange = Matrix::Read(inif, 2);
				need_skipline = false;
				break;
			case 17:	//mutNormalLawSigma2
				inif >> mutNormLawSigma2;
				break;
			case 18:	//TimeLimit
				inif >> timeLimit;
				break;
			case 19:	//UseBitString
				inif >> useBitString;
				break;
			case 20:	//BitsPerVariable
				inif >> bitsPerVar;
				break;
			case 21:		//HybridScheme
				inif >> word;
				if((nPos = word_pos(sGybridScheme, word)) > 0) h_scheme = nPos;
				break;
			case 22:	//CalcUniqueOnly
				inif >> calcUnique;
				break;
			case 23:	//MinUnique
				inif >> minUnique;
				break;
			case 24:		//SubPopType
				inif >> word;
				if((nPos = word_pos(sSubpop, word)) > 0) subpopT = nPos;
				break;
			case 25:	//HSubPopSizes
				hspSize = Matrix::Read(inif, 1);
				need_skipline = false;
				break;
			case 26:	//VSubPopSizes
				vspSize = Matrix::Read(inif, 1);
				need_skipline = false;
				break;
			case 27:		//MigrationDirection
				inif >> word;
				if((nPos = word_pos(sMigDir, word)) > 0) migDirection = nPos;
				break;
			case 28:	//MinInterval
				inif >> migInterval;
				break;
			case 29:	//MinInterval
				inif >> migFraction;
				break;
			case 30:	//GlobalSearch
				inif >> globalSearch;
				break;
			case 31:	//VSubPopFractions
				vspFract = Matrix::Read(inif, 1);
				need_skipline = false;
				//inif >> word;
				//inif >> word;
				break;
			case 32:
				inif >> xoverArithmeticRatio;
				break;
			case 33:
				inif >> xoverBLXAlpha;
				break;
			case 34:		//mutSpace
				inif >> word;
				if((nPos = word_pos(sMutSpace, word)) > 0) mutSpace = nPos;
				break;
			case 35:
				inif >> mutNonUniformParam;
				break;
			case 36:
				inif >> xoverSBXParam;
				break;
			case 37:		//Migration policy
				inif >> word;
				if((nPos = word_pos(sMigPolicy, word)) > 0) migPolicy = nPos;
				break;
			case 38:
				inif >> nTournSize;
				break;
			case 39:
				inif >> minimizing;
				break;
			case 40:
				inif >> ffscParam;
				break;
			case 41:
				inif >> sepAddonForEachVSP;
				break;
			case 42:
				inif >> addonCount;
				break;
		}	//main options
	}
	return need_skipline;
}

bool ga_opt::set_wrapper_opt(const_iopt_ref iopt)
{
	return wrapper_opt::der_set_wrapper_opt<ga_opt>(iopt);
	//ga_opt* p_src = dynamic_cast<ga_opt*>(iopt.get_wrapper_opt());
	//if(p_src) {
	//	*this = *p_src;
	//	AfterOptionsSet();
	//	return true;
	//}
	//else
	//	return wrapper_opt::set_wrapper_opt(iopt);

		//ga_opt& opt = *(ga_opt*)iopt.get_wrapper_opt();
		//logFname = opt.logFname;
		//errFname = opt.errFname;
		//histFname = opt.histFname;
		//openMode = opt.openMode;

		//initRange = opt.initRange;
		//hspSize = opt.hspSize; vspSize = opt.vspSize;
		//vspFract = opt.vspFract;
}

/*
void ga_opt::set_def_opt(bool create_defs)
{
	opt_.scheme = MuLambda;
	opt_.h_scheme = ClearGA;

	opt_.scalingT = RankSqr;
	opt_.selectionT = StochasticUniform;
	opt_.creationT = UniformCreation;
	opt_.crossoverT = Heuristic;
	opt_.mutationT = UniformMutation;
	opt_.mutSpace = PrevPopRest;
	opt_.subpopT = NoSubpops;

	//misc options
	opt_.eliteCount = 2;
	opt_.addonCount = 1;
	opt_.generations = 100;
	opt_.stallGenLimit = -1;
	opt_.useFitLimit = false;
	opt_.fitLimit = 0;
	opt_.logEveryPop = false;
	opt_.useBitString = false;
	opt_.bitsPerVar = 10;
	opt_.timeLimit = -1;
	opt_.calcUnique = false;
	opt_.globalSearch = false;
	opt_.minUnique = 0;
	opt_.sepAddonForEachVSP = false;

	//scaling parameters
	opt_.nTournSize = 4;
	opt_.minimizing = true;
	opt_.ffscParam = 0;

	//crossover parameters
	opt_.xoverFraction = 0.8;
	opt_.xoverHeuRatio = 1.2;
	opt_.xoverBLXAlpha = 0.5;
	opt_.xoverArithmeticRatio = 0.5;
	opt_.xoverSBXParam = 3;

	//mutation parameters
	opt_.mutProb = 0.01;
	opt_.mutNormLawSigma2 = 0.25;
	opt_.popSize = 60;
	opt_.mutNonUniformParam = 5;

	opt_.migPolicy = RandomRandom;
	opt_.migDirection = MigrateForward;
	opt_.migInterval = 20;
	opt_.migFraction = 0.2;

	_initRange.NewMatrix(2, 1);
	_initRange[0] = 0; _initRange[1] = 1;

	iniFname_ = "ga.ini";
	_logFname = "ga_log.txt";
	_errFname = "ga_err.txt";
	_histFname = "ga_hist.txt";
	_openMode = ios::trunc;
}

bool ga_opt::process_option(std::istream& inif, std::string& word)
{
	string sOpts = " GAScheme ScalingType SelectionType CrossoverType MutationType CreationType";
	sOpts += " PopulationSize EliteCount CrossoverFraction CrossoverHeuRatio MutationProb Generations StallGenLimit";
	sOpts += " Vectorized LogEveryPop PopInitRange MutationNormalLawSigma2 TimeLimit UseBitString BitsPerVariable HybridScheme";
	sOpts += " CalcUniqueOnly MinUnique SubPopType HSubPopSizes VSubPopSizes MigrationDirection MigrationInterval MigrationFraction";
	sOpts += " GlobalSearch VSubPopFractions CrossoverArithmeticRatio CrossoverBLXAlpha MutationSpace MutNonUniformParam";
	sOpts += " CrossoverSBXParam MigrationPolicy TournamentSize Minimizing ScalingParam SeparateAddonForEachVSP AddonCount";
	const string sScaling = " Proportional ProportionalMean ProportionalInv ProportionalTimeScaled ProportionalSigmaScaled Rank RankSqr RankExp";
	const string sSelection = " StochasticUniform Roulette UniformSelection Tournament Once Sort";
	const string sXover = " Heuristic OnePoint TwoPoint UniformCrossover Flat Arithmetic BLX SBX";
	const string sMut = " UniformMutation NormalMutation NonUniformMutation";
	const string sMutSpace = " WholePopulation CrossoverKids PrevPopRest";
	const string sCreation = " UniformCreation Manual";
	const string sScheme = " MuLambda MuPlusLambda";
	const string sGybridScheme = " ClearGA UseNN";
	const string sSubpop = " NotUsed Horizontal Vertical";
	const string sMigDir = " Forward Both";
	const string sMigPolicy = " WorstBest RandomRandom";

	string sTmp;
	std::istringstream is;
	bool need_skipline;
	int nPos;
	need_skipline = true;
	if((nPos = word_pos(sOpts, word)) > 0) {
		switch(nPos) {
			case 1:		//GASheme
				inif >> word;
				if((nPos = word_pos(sScheme, word)) > 0) opt_.scheme = nPos;
				break;
			case 2:		//ScalingType
				inif >> word;
				if((nPos = word_pos(sScaling, word)) > 0) opt_.scalingT = nPos;
				break;
			case 3:		//SelectionType
				inif >> word;
				if((nPos = word_pos(sSelection, word)) > 0) opt_.selectionT = nPos;
				break;
			case 4:		//CrossoverType
				inif >> word;
				if((nPos = word_pos(sXover, word)) > 0) opt_.crossoverT = nPos;
				break;
			case 5:		//MutationType
				inif >> word;
				if((nPos = word_pos(sMut, word)) > 0) opt_.mutationT = nPos;
				break;
			case 6:		//CreationType
				inif >> word;
				if((nPos = word_pos(sCreation, word)) > 0) opt_.creationT = nPos;
				break;
			case 7:		//PopulationSize
				inif >> opt_.popSize;
				break;
			case 8:
				inif >> opt_.eliteCount;
				break;
			case 9:
				inif >> opt_.xoverFraction;
				break;
			case 10:
				inif >> opt_.xoverHeuRatio;
				break;
			case 11:
				inif >> opt_.mutProb;
				break;
			case 12:
				inif >> opt_.generations;
				break;
			case 13:
				inif >> opt_.stallGenLimit;
				break;
			case 14:
				inif >> opt_.vectorized;
				break;
			case 15:
				inif >> opt_.logEveryPop;
				break;
			case 16:	//Population range
				inif >> ignoreLine;
				//_initRange = Read2rows(inif);
				_initRange = Matrix::Read(inif, 2);
				need_skipline = false;
				break;
			case 17:	//mutNormalLawSigma2
				inif >> opt_.mutNormLawSigma2;
				break;
			case 18:	//TimeLimit
				inif >> opt_.timeLimit;
				break;
			case 19:	//UseBitString
				inif >> opt_.useBitString;
				break;
			case 20:	//BitsPerVariable
				inif >> opt_.bitsPerVar;
				break;
			case 21:		//HybridScheme
				inif >> word;
				if((nPos = word_pos(sGybridScheme, word)) > 0) opt_.h_scheme = nPos;
				break;
			case 22:	//CalcUniqueOnly
				inif >> opt_.calcUnique;
				break;
			case 23:	//MinUnique
				inif >> opt_.minUnique;
				break;
			case 24:		//SubPopType
				inif >> word;
				if((nPos = word_pos(sSubpop, word)) > 0) opt_.subpopT = nPos;
				break;
			case 25:	//HSubPopSizes
				_hspSize = Matrix::Read(inif, 1);
				need_skipline = false;
				break;
			case 26:	//VSubPopSizes
				_vspSize = Matrix::Read(inif, 1);
				need_skipline = false;
				break;
			case 27:		//MigrationDirection
				inif >> word;
				if((nPos = word_pos(sMigDir, word)) > 0) opt_.migDirection = nPos;
				break;
			case 28:	//MinInterval
				inif >> opt_.migInterval;
				break;
			case 29:	//MinInterval
				inif >> opt_.migFraction;
				break;
			case 30:	//GlobalSearch
				inif >> opt_.globalSearch;
				break;
			case 31:	//VSubPopFractions
				_vspFract = Matrix::Read(inif, 1);
				need_skipline = false;
				//inif >> word;
				//inif >> word;
				break;
			case 32:
				inif >> opt_.xoverArithmeticRatio;
				break;
			case 33:
				inif >> opt_.xoverBLXAlpha;
				break;
			case 34:		//mutSpace
				inif >> word;
				if((nPos = word_pos(sMutSpace, word)) > 0) opt_.mutSpace = nPos;
				break;
			case 35:
				inif >> opt_.mutNonUniformParam;
				break;
			case 36:
				inif >> opt_.xoverSBXParam;
				break;
			case 37:		//Migration policy
				inif >> word;
				if((nPos = word_pos(sMigPolicy, word)) > 0) opt_.migPolicy = nPos;
				break;
			case 38:
				inif >> opt_.nTournSize;
				break;
			case 39:
				inif >> opt_.minimizing;
				break;
			case 40:
				inif >> opt_.ffscParam;
				break;
			case 41:
				inif >> opt_.sepAddonForEachVSP;
				break;
			case 42:
				inif >> opt_.addonCount;
				break;
		}	//main options
	}
	return need_skipline;
}

void ga_opt::set_wrapper_opt(const_iopt_ref iopt)
{
	wrapper_opt::set_wrapper_opt(iopt);
	//set_inner_opt(opt);

	ga_opt& opt = *(ga_opt*)iopt.get_wrapper_opt();
	_logFname = opt._logFname;
	_errFname = opt._errFname;
	_histFname = opt._histFname;
	_openMode = opt._openMode;

	_initRange = opt._initRange;
	_hspSize = opt._hspSize; _vspSize = opt._vspSize;
	_vspFract = opt._vspFract;
}
*/
