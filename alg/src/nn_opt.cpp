#include "nn_common.h"

using namespace std;
using namespace NN;
using namespace KM;

bool mnn_opt::process_option(std::istream& inif, std::string& word)
{
	string sOpts = " LearningRate MomentumConst Goal MinGrad Batch Adaptive Saturate NormInput ShowPeriod";
	sOpts += " MaxCycles Threshold01 Threshold11 WeightsInitRange WeightsInitFun PerfomanceFun LearningFun";
	sOpts += " TansigA TansigB TansigE LogsigA LogsigE RPDeltInc RPDeltDec RPDelta0 RPDeltaMax";
	string sInitFun = " Random NW";
	string sPerfFun = " SSE MSE";
	string sLearnFun = " BackProp ResilientBP GHA";

	int nPos;
	if((nPos = word_pos(sOpts, word)) > 0) {
		switch(nPos) {
			case 1:		//Nu
				inif >> nu;
				break;
			case 2:		//Mu
				inif >> mu;
				break;
			case 3:		//Goal
				inif >> goal;
				break;
			case 4:		//limit
				inif >> limit;
				break;
			case 5:		//Batch
				inif >> batch;
				break;
			case 6:		//Adaptive
				inif >> adaptive;
				break;
			case 7:
				inif >> saturate;
				break;
			case 8:
				inif >> normInp;
				break;
			case 9:
				inif >> showPeriod;
				break;
			case 10:
				inif >> maxCycles;
				break;
			case 11:
				inif >> thresh01;
				break;
			case 12:
				inif >> thresh11;
				break;
			case 13:
				inif >> wiRange;
				break;
			case 14:		//initFun
				inif >> word;
				if((nPos = word_pos(sInitFun, word)) > 0) initFun = nPos;
				break;
			case 15:		//perfFun
				inif >> word;
				if((nPos = word_pos(sPerfFun, word)) > 0) perfFun = nPos;
				break;
			case 16:		//learnFun
				inif >> word;
				if((nPos = word_pos(sLearnFun, word)) > 0) learnFun = nPos;
				break;
			case 17:
				inif >> tansig_a;
				break;
			case 18:
				inif >> tansig_b;
				break;
			case 19:
				inif >> tansig_e;
				break;
			case 20:
				inif >> logsig_a;
				break;
			case 21:
				inif >> logsig_e;
				break;
			case 22:
				inif >> rp_delt_inc;
				break;
			case 23:
				inif >> rp_delt_dec;
				break;
			case 24:
				inif >> rp_delta0;
				break;
			case 25:
				inif >> rp_deltamax;
				break;
		}	//main options
	}
	return true;
}

namespace NN {

template< class opt_type >
bool process_objnet_opt(opt_type& opt, std::istream& inif, std::string& word)
{
	string sOpts = " LearningRate MomentumConst Goal Epsilon Batch Adaptive Saturate ShowPeriod";
	sOpts += " MaxCycles WeightsInitRange WeightsInitFun PerfomanceFun LearningFun";
	sOpts += " TansigA TansigB TansigE LogsigA LogsigE RPDeltInc RPDeltDec RPDelta0 RPDeltaMax";
	sOpts += " GoalCheckFun NNPatience NNPatienceCycles ValidationFraction UseLSQ";
	string sInitFun = " Random NW";
	string sPerfFun = " SSE MSE";
	string sLearnFun = " BackProp ResilientBP ResilientBPplus QuickProp Hebbian GHA APEX";
	string sGCFun = " NoCheck Patience TestValidation";

	ulong nPos;
	if((nPos = opt.word_pos(sOpts, word)) > 0) {
		switch(nPos) {
			case 1:		//Nu
				inif >> opt.nu;
				break;
			case 2:		//Mu
				inif >> opt.mu;
				break;
			case 3:		//Goal
				inif >> opt.goal;
				break;
			case 4:		//epsilon
				inif >> opt.epsilon;
				break;
			case 5:		//Batch
				inif >> opt.batch;
				break;
			case 6:		//Adaptive
				inif >> opt.adaptive;
				break;
			case 7:
				inif >> opt.saturate;
				break;
			case 8:
				inif >> opt.showPeriod;
				break;
			case 9:
				inif >> opt.maxCycles;
				break;
			case 10:
				inif >> opt.wiRange;
				break;
			case 11:		//initFun
				inif >> word;
				if((nPos = opt.word_pos(sInitFun, word)) > 0) opt.initFun = nPos;
				break;
			case 12:		//perfFun
				inif >> word;
				if((nPos = opt.word_pos(sPerfFun, word)) > 0) opt.perfFun = nPos;
				break;
			case 13:		//learnFun
				inif >> word;
				if((nPos = opt.word_pos(sLearnFun, word)) > 0) opt.learnFun = nPos;
				break;
			case 14:
				inif >> opt.tansig_a;
				break;
			case 15:
				inif >> opt.tansig_b;
				break;
			case 16:
				inif >> opt.tansig_e;
				break;
			case 17:
				inif >> opt.logsig_a;
				break;
			case 18:
				inif >> opt.logsig_e;
				break;
			case 19:
				inif >> opt.rp_delt_inc;
				break;
			case 20:
				inif >> opt.rp_delt_dec;
				break;
			case 21:
				inif >> opt.rp_delta0;
				break;
			case 22:
				inif >> opt.rp_deltamax;
				break;
			case 23:		//goal_checkFun
				inif >> word;
				if((nPos = opt.word_pos(sGCFun, word)) > 0) opt.goal_checkFun = nPos;
				break;
			case 24:
				inif >> opt.patience;
				break;
			case 25:
				inif >> opt.patience_cycles;
				break;
			case 26:
				inif >> opt.validation_fract;
				break;
			case 27:
				inif >> opt.use_lsq;
				break;
				/*
				case 25:		//RBNECPolicy
				inif >> word;
				if((nPos = word_pos(sKmPol, word)) > 0) rbnec_policy = nPos;
				break;
				*/
		}	//main options
	}
	return true;
}

}	//namespace NN

//template< > _CLASS_DECLSPEC
//bool nn_opt< objnet >::process_option(std::istream& inif, std::string& word)
bool nn_opt::process_option(std::istream& inif, std::string& word)
{
	return process_objnet_opt(*this, inif, word);
}

//template< > _CLASS_DECLSPEC
//bool nn_opt< rbn >::process_option(std::istream& inif, std::string& word)
bool rbn_opt::process_option(std::istream& inif, std::string& word)
{
	string sECPol = " DoNothing Drop Singleton";
	string sRBfunc = " Gauss RevGauss Multiquad RevMultiquad";
	int nPos;
	if(word == "RBNEmptyCentPolicy") {
		inif >> word;
		if((nPos = word_pos(sECPol, word)) > 0) rbnec_policy_ = nPos;
	}
	else if(word == "RBLActFun") {
		inif >> word;
		switch (nPos = word_pos(sRBfunc, word)) {
			default:
			case 1: gft_ = radbas;
				break;
			case 2: gft_ = revradbas;
				break;
			case 3: gft_ = multiquad;
				break;
			case 4: gft_ = revmultiquad;
				break;
		}
	}
	else if(word == "RBLPatience")
		inif >> rbl_patience_;
	else if(word == "RBNIOLinked")
		inif >> io_linked_;
	else if(word == "RBNNeuronIncreaseMult")
		inif >> neur_incr_mult_;
	else if(word == "RBNNeuronIncreaseStep")
		inif >> neur_incr_step_;
	//else if(word == "UseLSQ")
	//	inif >> use_lsq;
	else return process_objnet_opt(*this, inif, word);
	return true;
}

//template< > _CLASS_DECLSPEC
//bool nn_opt< ccn >::process_option(std::istream& inif, std::string& word)
bool ccn_opt::process_option(std::istream& inif, std::string& word)
{
	string sLearnType = " MaxCor BackProp FullyBackProp";
	int nPos;
	if(word == "MaxFalmanLayerLearnCycles")
		inif >> maxFLLcycles_;
	else if(word == "FalmanLayerCandidatesCount")
		inif >> fl_candidates_;
	else if(word == "MaxFalmanLayers")
		inif >> maxFL_;
	else if(word == "FalmanLayerInsertBetween")
		inif >> insert_between_;
	else if(word == "GrowVirtualLayer")
		inif >> grow_vlayer_;
	else if(word == "FalmanLayerLearnType") {
		inif >> word;
		if((nPos = word_pos(sLearnType, word)) > 0) learnType = nPos + 20;
	}
	else if(word == "LSQPatienceCycles") {
		inif >> lsq_patience_cycles_;
	}
	else if(word == "FalmanLayerCandidatesSurvive") {
		inif >> fl_candidates_survive_;
	}
	else return process_objnet_opt(*this, inif, word);
	return true;
}

//-------------------------------------set_def_opt-----------------------------------------------------
void mnn_opt::set_def_opt(bool create_defs)
{
	nu = 0.5; mu = 0.05; goal = 0.01;
	limit = 0.0000001;
	batch = false; adaptive = false;
	saturate = false;
	showPeriod = 10;
	maxCycles = 5000;
	normInp = false; noise = 0;
	thresh01 = 0.7; thresh11 = 0.1;
	initFun = if_random;
	perfFun = sse;
	wiRange = 0.01;
	logsig_a = 1;
	tansig_a = 1.7159; tansig_b = 2./3.;
	tansig_e = 0.7159; logsig_e = 0.1;
	rp_delt_dec = 0.5; rp_delt_inc = 1.2;
	rp_delta0 = 0.07; rp_deltamax = 50;
	learnFun = BP;
	useSimpleRP = true;

	flags_ = MLP;
	iniFname_ = "nn.ini";
	errFname_ = "nn_err.txt";
}

namespace NN {

template< class opt_type >
void set_objnet_defs(opt_type& opt, bool create_defs)
{
	opt.nu = 0.5; opt.mu = 0.05; opt.goal = 0.01;
	opt.epsilon = 0.0000001;
	opt.batch = false; opt.adaptive = false;
	opt.saturate = false;
	opt.showPeriod = 10;
	opt.maxCycles = 5000;
	opt.initFun = if_random;
	opt.perfFun = sse;
	opt.wiRange = 0.01;
	opt.logsig_a = 1; opt.logsig_e = 0.1;
	opt.tansig_a = 1.7159; opt.tansig_b = 2./3.;
	opt.tansig_e = 0.7159;
	opt.patience_cycles = 100;
	opt.patience = 0.001;
	opt.validation_fract = 0.2;
	opt.use_lsq = false;

	opt.learnFun = BP;
	opt.learnType = backprop;
	opt.goal_checkFun = no_check;

	opt.rp_delt_dec = 0.5; opt.rp_delt_inc = 1.2;
	opt.rp_delta0 = 0.07; opt.rp_deltamax = 50;
	opt.useSimpleRP = false;

	opt.qp_lambda = 0.0001;
	opt.qp_alfamax = 1.75;
	opt.useSimpleQP = true;

	//rbnec_policy = kmeans::singleton;

	//use biases by default
	opt.use_biases_ = true;

	opt.iniFname_ = "nn.ini";
	opt.errFname_ = "nn_err.txt";

	if(create_defs)
		opt.template create_def_embobj< km_opt >("km_opt");
}

}	//namespace NN

//template< > _CLASS_DECLSPEC
//void nn_opt< objnet >::set_def_opt(bool create_defs)
void nn_opt::set_def_opt(bool create_defs)
{
	set_objnet_defs(*this, create_defs);
}

//template< > _CLASS_DECLSPEC
//void nn_opt< rbn >::set_def_opt(bool create_defs)
void rbn_opt::set_def_opt(bool create_defs)
{
	set_objnet_defs(*this, create_defs);
	rbnec_policy_ = KM::singleton;
	gft_ = DEF_GFT; rbl_patience_ = 0.01; io_linked_ = false;
	neur_incr_mult_ = 1.2; neur_incr_step_ = 5;
}

//template< > _CLASS_DECLSPEC
//void nn_opt< ccn >::set_def_opt(bool create_defs)
void ccn_opt::set_def_opt(bool create_defs)
{
	set_objnet_defs(*this, create_defs);
	maxFLLcycles_ = 2500; fl_candidates_ = 10;
	insert_between_ = false; grow_vlayer_ = true;
	maxFL_ = 500; learnType = ccn_maxcor;
	lsq_patience_cycles_ = 5;
	fl_candidates_survive_ = 1;
}

//------------------------------------------------nn_opt-----------------------------------------
//template< class nn_type >
//void nn_opt< nn_type >::set_embopt_def(iopt_ref emb_opt)
void nn_opt::set_embopt_def(iopt_ref emb_opt)
{
	//emb_opt.set_def_opt();
	if(string(emb_opt.get_opt_type()) == "km_opt") {
		//set default kmeans options
		emb_opt.iniFname_ = iniFname_;
		((km_opt*)emb_opt.get_wrapper_opt())->emptyc_pol = singleton;
		emb_opt.update_embopt_defs();
	}
}

//------------------------------------------------set_wrapper_opt---------------------------------
bool mnn_opt::set_wrapper_opt(const_iopt_ref iopt)
{
	return wrapper_opt::der_set_wrapper_opt<mnn_opt>(iopt);
}

//template< class nn_type >
//bool nn_opt< nn_type >::set_wrapper_opt(const_iopt_ref iopt)
bool nn_opt::set_wrapper_opt(const_iopt_ref iopt)
{
	return wrapper_opt::der_set_wrapper_opt< nn_opt >(iopt);
}

bool rbn_opt::set_wrapper_opt(const_iopt_ref iopt)
{
	return wrapper_opt::der_set_wrapper_opt< rbn_opt >(iopt);
}

bool ccn_opt::set_wrapper_opt(const_iopt_ref iopt)
{
	return wrapper_opt::der_set_wrapper_opt< ccn_opt >(iopt);
}

