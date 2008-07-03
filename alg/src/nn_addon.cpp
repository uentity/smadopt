#include "nn_addon.h"
#include "objnet.h"
#include "prg.h"
#include "m_algorithm.h"

#include <algorithm>
#include <sstream>
#include <cmath>
#include <functional>
#include <iostream>

using namespace std;
using namespace NN;
using namespace GA;

nn_addon* pnna;
objnet* pnn;

/*
void DumpM(const Matrix& m, const char* pFname = NULL)
{
	if(pFname) {
		ofstream fd(pFname, ios::out | ios::trunc);
		m.Print(fd);
	}
	else m.Print(cout);
}
*/

void print_nn_state(const nnState& state, const char* preambula = NULL)
{
	if(state.status != learning && state.status != ccn_maxcor && state.status != ccn_bp) {
		if(preambula)
			cout << preambula << ' ';
		switch(state.status) {
			case learned:
				cout << "learning successfull - goal reached!";
				break;
			case stop_maxcycle:
				cout << "learning stopped - max cycles reached";
				break;
			case stop_palsy:
				cout << "learning stopped - palsy";
				break;
			case stop_patience:
				cout << "learning stopped - no significant performance improve";
				break;
			case stop_test_validation:
				cout << "learning stopped - error on validation set doesn't improve";
				break;
		}
		cout << endl;
	}
}

void print_winner(int aft)
{
	cout << "The winner is: ";
	switch(aft) {
		case logsig : cout << "logsig"; break;
		case tansig : cout << "tansig"; break;
		case purelin : cout << "purelin"; break;
		case poslin : cout << "poslin"; break;
		case radbas : cout << "radbas"; break;
		case expws : cout << "expws"; break;
		case revradbas : cout << "revradbas"; break;
		case multiquad : cout << "multiquad"; break;
	}
	cout << " neuron" << endl;
}

bool PCAInformer(ulong uCycle, double perf, void* pNet)
{
	MNet* pNN = (MNet*)pNet;
	cout << "cycle " << uCycle << " error " << perf << ", goal " << pNN->opt_.goal << ", nu " << pNN->opt_.nu << endl;
	print_nn_state(pNN->state());
	return true;
}

bool MNetInformer(ulong uCycle, double perf, void* pNet)
{
	MNet* pNN = (MNet*)pNet;
	cout << "cycle " << uCycle << ", error " << perf << ", goal " << pNN->opt_.goal << endl;
	print_nn_state(pNN->state());
	return true;
}

bool StandardInformer(ulong uCycle, double perf, void* pNet)
{
	mlp* pNN = (mlp*)pNet;
	cout << "cycle " << uCycle << ", error " << perf << ", goal " << pNN->opt_.goal << endl;
	print_nn_state(pNN->state());

	//char c = cin.peek();
	//if(c == 's') return false;
	return true;
}

bool CcnInformer(ulong uCycle, double perf, void* pNet)
{
	ccn* pNN = (ccn*)pNet;
	nnState mainState = pNN->get_mainState();
	nnState tstate = pNN->state();
	falman_layer* p_l = pNN->get_cur_flayer();
	if(mainState.status == learning) {
		if(tstate.cycle == 1) {
			/*
			if(pNN->flayers_num() > 0) {
				p_l = &pNN->get_flayer(pNN->flayers_num() - 1);
				print_winner(p_l->aft()[p_l->get_winner_ind()]);
			}
			*/
			cout << "Output layer learning started" << endl;
		}
		cout << "cycle " << uCycle << ", error " << perf << ", goal " << pNN->opt_.goal << endl;
		print_nn_state(tstate, "Output layer:");
	}
	else {
		if(tstate.cycle == 1)
			cout << "Error still high - new falman layer #" << pNN->flayers_num() << " learning started" << endl;
		cout << "cycle " << uCycle << ", goal " << perf << endl;
		print_nn_state(tstate, "Falman layer:");
		if(mainState.status == ccn_maxcor && tstate.status != learning)
			print_winner(p_l->aft()[p_l->get_winner_ind()]);
	}
	//else if(mainState.status == bp_falman_learning) {
	//	if(tstate.cycle == 1)
	//		cout << "Output with last falman layers learning started" << endl;
	//	cout << "cycle " << uCycle << ", error " << perf << ", goal " << pNN->opt_.goal << endl;
	//	print_nn_state(tstate, "Output layer:");
	//}
	print_nn_state(mainState, "CCN");
	if(mainState.status != learning && mainState.status != pNN->opt_.learnType)
		cout << "We end up with " << pNN->flayers_num() << " falman layers" << endl;

	ifstream f("stop.txt");
	if(f.rdbuf()->sgetc() == '1') return false;
	else return true;
}

namespace GA {
void NNFitnessFcn(int nVars, int nPopSize, double* pPop, double* pScore)
{
	Matrix pop(nPopSize, nVars, pPop);
	pop = !pop;
	Matrix score(nPopSize, 1);
	Matrix chrom, ch_sc, diff;
	Matrix mean = pop.vMean();
	bool bBad;
	//bounds penalty coefs
	Matrix bpc = pnna->ga_.opt_.initRange.GetRows(1) - pnna->ga_.opt_.initRange.GetRows(0);
	bpc *= 0.1;
	transform(bpc.begin(), bpc.end(), bpc.begin(), bind1st(divides<double>(), 1));
	//calc scores
	for(ulong i=0; i<score.size(); ++i) {
		chrom = pop.GetColumns(i);
		bBad = false;
		//score[i] = 0;
		/*
		for(ulong j=0; j<chrom.size(); ++j) {
			if(chrom[j] < pnna->_ga.opt_.initRange(0, j) || chrom[j] > pnna->_ga.opt_.initRange(1, j)) {
				score[i] = 1;
				bBad = true;
				break;
			}
		}
		*/
		if(pnna->opt_.normInp)
			chrom /= pnna->state_.max_ch_r;
		if(pnna->opt_.usePCA)
			chrom <<= pnna->netPCA_->Sim(chrom - pnna->state_.inpMean);

		if(pnna->opt_.netType == matrix_nn)
			ch_sc = pnna->net_.Sim(chrom);
		else
			ch_sc = pnn->sim(chrom);

		//if(bBad || pnna->opt_.pred_ratio <= 0) continue;
		if(pnna->opt_.pred_ratio > 0 && ch_sc[0] < pnna->state_.n_cur_min)
			ch_sc[0] += pow((pnna->state_.n_cur_min - ch_sc[0])/pnna->state_.pred_r, 4);

		score[i] = ch_sc[0];

		//check bounds
		if(1 == 1) {
			chrom <<= !chrom;
			diff <<= chrom - pnna->ga_.opt_.initRange.GetRows(0);
			replace_if(diff.begin(), diff.end(), bind2nd(greater<double>(), 0), 0);
			//multiply by penalty coefs
			diff *= bpc;
			score[i] += abs(score[i])*diff.Mul(diff).Sum();
			//score[i] -= abs(score[i])*diff.Sum();

			diff <<= pnna->ga_.opt_.initRange.GetRows(1) - chrom;
			replace_if(diff.begin(), diff.end(), bind2nd(greater<double>(), 0), 0);
			//multiply by penalty coefs
			diff *= bpc;
			score[i] += abs(score[i])*diff.Mul(diff).Sum();
			//score[i] -= abs(score[i])*diff.Sum();
		}
	}
	//score = pnna->GetRealData(score);
	memcpy(pScore, score.GetBuffer(), score.raw_size());
}
}	//end of namespace GA

//------------------------------------------nn_addon implementation--------------------------------------------
nn_addon::nn_addon() : opt_(this)
{
	set_def_opt();
}

nn_addon::nn_addon(const char* psName) : opt_(this)
{
	set_def_opt();
	opt_.name = psName;
}

void nn_addon::set_def_opt()
{
	opt_.set_def_opt(false);
	//opt_.AfterOptionsSet = &nn_addon::AfterOptionsSet;
	//create objnet
	//_create_onet();

	//fill options chain
	opt_.add_embopt(ga_.opt_.get_iopt_ptr());
	opt_.add_embopt(net_.opt_.get_iopt_ptr());
	opt_.add_embopt(state_.km.opt_.get_iopt_ptr());
	//add defaul objnet opt
	//opt_.create_def_embobj<nn_opt>("nn_opt");

	//set default function pointers
	_pInitFcn = &nn_addon::InitStd;
	_pAddonFcn = &nn_addon::_GetOneAddonStd;
}

/*
void nn_addon::_construct()
{
	//fill options chain
	add_embopt(_ga.get_iopt_ptr());
	add_embopt(_net.get_iopt_ptr());
	add_embopt(_state.km.get_iopt_ptr());
}
*/

nn_addon::~nn_addon(void)
{
}

const char* nn_addon::GetName() const
{
	return opt_.name.c_str();
}

void nn_addon::SetName(const char* psName) {
	//ga_addon::SetName(psName);
	opt_.name = psName;

	string s;
	string::size_type pos;
	for(uint i=0; !(s = GetFname(i)).empty(); ++i) {
		//don't change ini filenames
		if(i == IniFname || i == iga_IniFname) continue;
		s.insert(0, " ");
		s.insert(0, opt_.name);
		SetFname(i, s.c_str());

		//if((pos = s.rfind('.')) != string::npos) {
		//	s.insert(pos, opt_.name);
		//	s.insert(pos, "_");
		//}
		//else {
		//	s += '_';
		//	s += opt_.name;
		//}
	}
}

void* nn_addon::GetOptions()
{
	return opt_.GetOptions();
}

void nn_addon::SetOptions(const void* pOpt)
{
	opt_.SetOptions(pOpt);
	AfterOptionsSet();
}

void nn_addon::ReadOptions(const char* pFName)
{
	opt_.ReadOptions(pFName);
	AfterOptionsSet();
	opt_.read_embopt(pFName);
}

void* nn_addon::GetObject(int obj_id) const {
	switch(obj_id) {
		case nnaNeuralNet:
			return (void*)&net_;
		case nnaInnerGA:
			return (void*)&ga_;
		default:
			return NULL;
	}
}

const char* nn_addon::GetFname(int fname) const
{
	switch(fname) {
		case IniFname:
			return opt_.iniFname_.c_str();
		case NN_ErrFname:
			return net_.opt_.errFname_.c_str();
		case iga_IniFname:
			return ga_.opt_.iniFname_.c_str();
		case iga_LogFname:
			return ga_.opt_.logFname.c_str();
		case iga_HistFname:
			return ga_.opt_.histFname.c_str();
		case iga_ErrFname:
			return ga_.opt_.errFname.c_str();
	}
	return "";
}

bool nn_addon::SetFname(int fname, const char* pNewFname)
{
	if(!pNewFname) return false;
	switch(fname) {
		case IniFname:
			opt_.iniFname_ = pNewFname;
			return true;
		case NN_ErrFname:
			net_.opt_.errFname_ = pNewFname;
			return true;
		case iga_IniFname:
			ga_.opt_.iniFname_ = pNewFname;
			return true;
		case iga_LogFname:
			ga_.opt_.logFname = pNewFname;
			return true;
		case iga_HistFname:
			ga_.opt_.histFname = pNewFname;
			return true;
		case iga_ErrFname:
			ga_.opt_.iniFname_ = pNewFname;
			return true;
	}
	return false;
}

objnet* nn_addon::onet_fab()
{
	switch(opt_.netType) {
		default:
			return NULL;
		case mlp_nn:
			return new mlp;
		case rb_nn:
			return new rbn;
		case ccn_nn:
			return new ccn;
		case pca_nn:
			return new pcan;
	}
}

void nn_addon::_create_onet(ulong nets_count)
{
	//create object
	if(_onet.size() > 0) return;
	objnet* p_net;
	for(ulong i = 0; i < nets_count; ++i) {
		p_net = onet_fab();
		if(p_net) {
			_onet.push_back(sp_onet(p_net));
			//add objnet to options chain
			opt_.add_embopt(p_net->opt_);
		}
	}
}

void nn_addon::AfterOptionsSet()
{
	//set function pointers
	switch(opt_.addon_scheme) {
		default:
		case FitFcnAsTarget:
			_pInitFcn = &nn_addon::InitStd;
			_pAddonFcn = &nn_addon::_GetOneAddonStd;
			break;
		case FitFcnAsInput:
			_pInitFcn = &nn_addon::InitAlt;
			_pAddonFcn = &nn_addon::_GetOneAddonAlt;
			break;
		case PCAPredict:
			_pInitFcn = &nn_addon::InitPCA;
			_pAddonFcn = &nn_addon::_GetOneAddonPCA;
			break;
	}

	//create default object net that we will use
	switch(opt_.netType) {
		case pca_nn:
		case mlp_nn:
			opt_.create_def_embobj<nn_opt>("nn_opt");
			break;
		case rb_nn:
			opt_.create_def_embobj<rbn_opt>("nn_opt");
			break;
		case ccn_nn:
			opt_.create_def_embobj<ccn_opt>("nn_opt");
			break;
	}
}

bool nn_addon::Init(ulong addon_count, ulong chromLength, const Matrix& searchRange)
{
	return (this->*_pInitFcn)(addon_count, chromLength, searchRange);
}

bool nn_addon::InitStd(ulong addon_count, ulong chromLength, const Matrix& searchRange)
{
	//if(ReadOptFromIni) ReadOptions();
	//else SetOptions(&opt_);
	//if(opt_.layers_num == 0 || opt_.pNeurons == NULL || opt_.pLayerTypes == NULL) {
	//	_setDefNNSize();
	//}

	_create_onet(addon_count);

	//if(ReadOptFromIni) {
	//	if(opt_.netType != matrix_nn)
	//		_onet->ReadOptions();
	//	else
	//		_net.ReadOptions();
	//}

	if(opt_.netType == matrix_nn) net_.SetLayersNum(opt_.layers_.size());

	if(opt_.usePCA) {
		netPCA_.reset(new MNet);
		netPCA_->opt_.initFun = if_random;
		netPCA_->opt_.wiRange = 0.01;
		netPCA_->opt_.learnFun = GHA;
		if(netPCA_->opt_.flags_ & useBiases) netPCA_->opt_.flags_ -= useBiases;
		if(netPCA_->opt_.flags_ & useLateral) netPCA_->opt_.flags_ -= useLateral;
		netPCA_->opt_.nu = 0.0001;
		netPCA_->opt_.goal = 0.01;
		netPCA_->opt_.maxCycles = 50000;
		netPCA_->opt_.showPeriod = 100;
		netPCA_->opt_.adaptive = true;

		netPCA_->SetInputSize(chromLength);
		opt_.PC_num = min(opt_.PC_num, chromLength);
		if(opt_.PC_num > 0) {
			netPCA_->AddLayer(opt_.PC_num, purelin);
			chromLength = opt_.PC_num;
			//_net.SetInputSize(opt_.PC_num);
		}
		else {
			netPCA_->AddLayer(chromLength, purelin);
			//_net.SetInputSize(chromLength);
		}
		state_.init_PCA = true;
	}

	state_.init_nn = false;
	if(opt_.netType == matrix_nn) {
		//set input size
		net_.SetInputSize(chromLength);
		//set layers
		for(ulong i=0; i<opt_.layers_.size(); ++i)
			net_.SetLayer(i, opt_.layers_[i], opt_.lTypes_[i]);
		state_.init_nn = true;
	}
	else {
		for(onet_iterator p_net = _onet.begin(); p_net != _onet.end(); ++p_net) {
			(*p_net)->set_input_size(chromLength);
			switch(opt_.netType) {
				case mlp_nn:
					for(ulong i=0; i<opt_.layers_.size(); ++i)
						((mlp*)(*p_net).get())->add_layer(opt_.layers_[i], opt_.lTypes_[i]);
					state_.init_nn = true;
					break;
				case rb_nn:
					((rbn*)(*p_net).get())->set_output_layer(opt_.layers_[opt_.layers_.size() - 1]);
					state_.init_nn = true;
					break;
				case ccn_nn:
					((ccn*)(*p_net).get())->set_output_layer(opt_.layers_[opt_.layers_.size() - 1], opt_.lTypes_[opt_.lTypes_.size() - 1]);
					state_.init_nn = true;
					break;
			}
		}
	}

	if(opt_.is_ffRestricted) {
		state_.norm_min = opt_.inf_ff;
		state_.norm_max = opt_.sup_ff;
	}

	//set init range
	ga_.opt_.initRange = searchRange;

	state_.km.opt_.emptyc_pol = opt_.kmfec_policy;

	//save minimum NN learning goal from nn_opt
	state_.min_goal = _onet[0].get()->opt_.goal;

	return true;
}

bool nn_addon::InitAlt(ulong addon_count, ulong chromLength, const Matrix& searchRange)
{
	//if(ReadOptFromIni) ReadOptions();
	//else SetOptions(&opt_);
	//if(opt_.layers_num == 0 || opt_.pNeurons == NULL || opt_.pLayerTypes == NULL) {
	//	_setDefNNSize();
	//}

	_create_onet(addon_count);

	//if(ReadOptFromIni) {
	//	if(opt_.netType != matrix_nn)
	//		_onet->ReadOptions();
	//	else
	//		_net.ReadOptions();
	//}

	net_.SetLayersNum(opt_.layers_.size());

	if(opt_.usePCA) {
		netPCA_.reset(new MNet);
		netPCA_->opt_.initFun = if_random;
		netPCA_->opt_.wiRange = 0.01;
		netPCA_->opt_.learnFun = GHA;
		if(netPCA_->opt_.flags_ & useBiases) netPCA_->opt_.flags_ -= useBiases;
		if(netPCA_->opt_.flags_ & useLateral) netPCA_->opt_.flags_ -= useLateral;
		netPCA_->opt_.nu = 0.0001;
		netPCA_->opt_.goal = 0.01;
		netPCA_->opt_.maxCycles = 50000;
		netPCA_->opt_.showPeriod = 100;

		netPCA_->SetInputSize(1);
		opt_.PC_num = 1;
		netPCA_->AddLayer(1, purelin);
		state_.init_PCA = true;
	}

	state_.init_nn = false;
	if(opt_.netType == matrix_nn) {
		net_.SetInputSize(1);
		for(ulong i=0; i < opt_.layers_.size(); ++i)
			net_.SetLayer(i, opt_.layers_[i], opt_.lTypes_[i]);
		state_.init_nn = true;
	}
	else {
		for(onet_iterator p_net = _onet.begin(); p_net != _onet.end(); ++p_net) {
			(*p_net)->set_input_size(1);
			switch(opt_.netType) {
				case mlp_nn:
					for(ulong i=0; i < opt_.layers_.size(); ++i)
						((mlp*)(*p_net).get())->add_layer(opt_.layers_[i], opt_.lTypes_[i]);
					state_.init_nn = true;
					break;
				case rb_nn:
					((rbn*)(*p_net).get())->set_output_layer(chromLength);
					state_.init_nn = true;
					break;
				case ccn_nn:
					((ccn*)(*p_net).get())->set_output_layer(chromLength, opt_.lTypes_[opt_.lTypes_.size() - 1]);
					break;
			}
		}
	}

	if(opt_.is_ffRestricted) {
		state_.norm_min = opt_.inf_ff;
		state_.norm_max = opt_.sup_ff;
	}

	//set init range
	ga_.opt_.initRange = searchRange;

	//save minimum NN learning goal from nn_opt
	state_.min_goal = _onet[0].get()->opt_.goal;

	return true;
}

bool nn_addon::InitPCA(ulong addon_count, ulong chromLength, const Matrix& searchRange)
{
	//if(ReadOptFromIni) ReadOptions();
	//else SetOptions(&opt_);

	netPCA_.reset(new MNet);
	netPCA_->opt_.initFun = if_random;
	netPCA_->opt_.wiRange = 0.01;
	netPCA_->opt_.learnFun = GHA;
	if(netPCA_->opt_.flags_ & useBiases) netPCA_->opt_.flags_ -= useBiases;
	if(netPCA_->opt_.flags_ & useLateral) netPCA_->opt_.flags_ -= useLateral;
	netPCA_->opt_.nu = 0.0001;
	netPCA_->opt_.goal = 0.01;
	netPCA_->opt_.maxCycles = 50000;
	netPCA_->opt_.showPeriod = 100;

	netPCA_->SetInputSize(chromLength);
	opt_.PC_num = min(opt_.PC_num, chromLength);
	if(opt_.PC_num > 0)
		netPCA_->AddLayer(opt_.PC_num, purelin);
	else
		netPCA_->AddLayer(chromLength, purelin);
	state_.init_PCA = true;

	return true;
}

int nn_addon::_learn_network(const Matrix& input, const Matrix& targets, ulong net_ind)
{
#ifdef VERBOSE
	//dump learn data
	DumpMatrix(input, "inputs.txt");
	DumpMatrix(targets, "targets.txt");
#endif

	if(targets.size() == 0)
		return NN::error;
	int ret_state;
	try {
		//Make transposed local copy
		Matrix inp = !input;
		Matrix tar = !targets;
		if(opt_.usePCA) {
			cout << opt_.name << ": PCA NN learning" << endl;
			//subtract mean
			state_.inpMean = inp.vMean(true);
			inp <<= inp.SubMean(state_.inpMean, true);
			//debug
			ofstream popf("x.txt", ios::out | ios::ate);
			//pcaf << "Original population" << endl;
			inp.Print(popf);

			//_net.LinPCA(!inp, _state.netPCA, LearnNNInformer);
			netPCA_->PCALearn(inp, state_.init_PCA, MNetInformer);

			inp <<= netPCA_->Sim(inp);
			state_.init_PCA = opt_.initPCAEveryIter;

			//debug test
			ofstream pcaf("nn_pca.txt", ios::out | ios::ate);
			//pcaf << "PCA Network weights" << endl;
			netPCA_->weights(0).Print(pcaf);
			//w.Print(pcaf);

			//Matrix tmp = _state.netPCA->Sim(!input);
			//tmp = _state.netPCA->ReverseSim(tmp);
			//popf << "Restored pop" << endl;
			//(!tmp).Print(popf);
		}

		cout << opt_.name << ": NN" << net_ind << " learning" << endl;
		if(opt_.netType == matrix_nn) {
			net_._inp_range <<= inp.minmax(true);
			if(opt_.goalQuota > 0)
				net_.opt_.goal = max(_calc_goalQuota(tar), state_.min_goal);
			ret_state = net_.BPLearn(inp, tar, state_.init_nn, MNetInformer);
		}
		else {
			objnet* p_net = _onet[net_ind].get();
			p_net->opt_.inp_range_ <<= inp.minmax(true);
			if(opt_.goalQuota > 0)
				p_net->opt_.goal = max(_calc_goalQuota(tar), state_.min_goal);

			if(opt_.netType == rb_nn) {
				switch(opt_.rbn_learn_type) {
					case rbn_exact:
						((rbn*)p_net)->set_rb_layer_exact(inp);
						break;
					case rbn_random:
						((rbn*)p_net)->set_rb_layer_random(inp, opt_.rbn_cmult);
						break;
					default:
					case rbn_kmeans_bp:
						if(opt_.samples_filter != GA::best_filter)
							//((rbn*)p_net)->set_rb_layer_kmeans(inp, tar, opt_.rbn_cmult, &_state.km.get_centers());
							((rbn*)p_net)->set_rb_layer_drops(inp, tar, state_.learn_cl_cent, opt_.rbn_cmult);
						break;
					case rbn_fully_bp:
						if(opt_.samples_filter != GA::best_filter)
							((rbn*)p_net)->set_rb_layer(inp, opt_.layers_[opt_.layers_.size() - 2]);
						break;
				}
			}

			pLearnInformer pInformer;
			switch(opt_.netType) {
				default:
				case mlp_nn:
					pInformer = StandardInformer;
					break;
				case ccn_nn:
					if(state_.init_nn && opt_.samples_filter != GA::best_filter)
						((ccn*)p_net)->add_rb_layer(inp, tar, state_.learn_cl_cent, opt_.rbn_cmult);
					pInformer = CcnInformer;
					break;
			}
			ret_state = p_net->learn(inp, tar, state_.init_nn, pInformer);
		}
		if(ret_state != NN::learned)
			cerr << opt_.name << ": WRN: NN" << net_ind << " state after learning = " << ret_state << endl;

#ifdef VERBOSE
		//test learned network
		Matrix tres;
		if(opt_.netType == matrix_nn)
			tres <<= net_.Sim(inp);
		else
			tres <<= _onet[net_ind]->sim(inp);
		tres <<= (!tar | GetRealData(!tres));
		DumpMatrix(tres, "tres.txt");

		//if(inp.row_num() <= 2) {
			_build_surf(net_ind, input, targets);
			char c;
			cin >> c;
		//}
#endif

		state_.init_nn = opt_.initNetEveryIter;
	}
	catch(nn_except ex) {
		cerr << opt_.name << ": ERR: NN" << net_ind << endl;
		cerr << ex.what() << endl;
		ret_state = NN::error;
	}
	return ret_state;
}

void nn_addon::_fillup_storage(Matrix& p, Matrix& s)
{
	//collect unique samples for learning
	Matrix row;
	double min_dist, cur_dist;
	bool add_chrom;
	for(ulong i = p.row_num() - 1; i < p.row_num(); --i) {
		//clear samples "not in range"
		if(s[i] >= ERR_VAL || (opt_.is_ffRestricted && (s[i] > opt_.sup_ff || s[i] < opt_.inf_ff))) {
			s.DelRows(i);
			p.DelRows(i);
			continue;
		}

		add_chrom = false;
		row <<= p.GetRows(i);
		if(learn_.row_num() == 0)
			add_chrom = true;
		else if(learn_.RRowInd(row) >= learn_.row_num()) {
			//find minimun distance
			min_dist = sqrt((learn_.GetRows(0) - row).norm2());
			for(ulong j = 1; j < learn_.row_num(); ++j) {
				cur_dist = sqrt((learn_.GetRows(j) - row).norm2());
				if(cur_dist < min_dist) min_dist = cur_dist;
			}
			//if distance > tolerance - add to learn samples
			if(min_dist > opt_.tol) add_chrom = true;
		}

		if(add_chrom) {
			learn_ &= row;
			tar_ &= s.GetRows(i);
		}
	}

	//identify best individual
	if(tar_.row_num() > 0) {
		state_.ind_best = tar_.ElementInd(tar_.Min());
		state_.tar_mean = tar_.Mean();
		state_.tar_std = tar_.Std();
	}
	else
		throw ga_except("Empty learning data storage! Nothing to do");

	/*
	if(tar.size() ==0 || abs(_state.tar_std) < 0.000001) {
		//very low variance in data - skip all stuff
		throw ga_except("Very low variance in learning data! Nothing to do");
	}
	*/

#ifdef VERBOSE
	DumpMatrix(learn_, "p.txt");
	DumpMatrix(tar_, "s.txt");
#endif

	//remove very big samples
	//for(long i = _tar.size() - 1; i>=0; --i) {
	//	if(_tar[i] > m + q) {
	//		_tar.DelRows(i);
	//		_learn.DelRows(i);
	//	}
	//
}

void nn_addon::_get_learnData(Matrix& p, Matrix& s, Matrix& learn, Matrix& targets)
{
	//select clustering engine depending on corresponding option
	struct call_kmeans_filter {
		static Matrix go(nn_addon& nna, const Matrix& p, const Matrix& f, Matrix& lp, Matrix& lf) {
			if(nna.opt_.clustEngine == ce_kmeans)
				return nna._kmeans_filter(nna.state_.km, p, f, lp, lf);
			else
				return nna._kmeans_filter(nna.state_.da, p, f, lp, lf);
		}
	};

	//define learning data
	if(opt_.bestCount > 0) { //&& opt_.bestCount < _tar.size()) {
		switch(opt_.samples_filter) {
			default:
			case best_filter:
				ga_.opt_.initRange = _best_filter(learn_, tar_, learn, targets);
				break;
			case kmeans_filter:
				ga_.opt_.initRange = call_kmeans_filter::go(*this, learn_, tar_, learn, targets);
				break;
			case best_km_filter:
				Matrix b_l, b_t;
				_best_filter(learn_, tar_, b_l, b_t);
				ga_.opt_.initRange = call_kmeans_filter::go(*this, b_l, b_t, learn, targets);
				break;
		}
	}
	else if(opt_.bestCount == 0) {
		learn <<= p;
		targets <<= s;
	}
	else {
		learn = learn_;
		targets = tar_;
	}
}

double nn_addon::_calc_goalQuota(const Matrix& targets)
{
	Matrix tar2 = targets.Mul(targets);
	double r, min_r;
	min_r = tar2.GetColumns(0).Sum();
	for(ulong i=1; i<tar2.col_num(); ++i) {
		r = tar2.GetColumns(i).Sum();
		if(r < min_r) min_r = r;
	}
	min_r *= opt_.goalQuota*opt_.goalQuota;
	if(net_.opt_.perfFun == NN::sse) min_r *= targets.col_num();
	return min_r;
}

Matrix nn_addon::_best_filter(const Matrix& p, const Matrix& f, Matrix& lp, Matrix& lf)
{
	lf = f;
	indMatrix mInd = lf.RawSort();
	lp.NewMatrix(min<ulong>(opt_.bestCount, p.row_num()), p.col_num());
	for(ulong i = 0; i < lp.row_num(); ++i)
		lp.SetRows(p.GetRows(mInd[i]), i);
	if(lp.row_num() < lf.row_num())
		lf.DelRows(lp.row_num(), lf.row_num() - lp.row_num());

	if(opt_.search_samples_num > 0)
		return lp.GetRows(0, opt_.search_samples_num).minmax();
	else
		return lp.minmax();
}

template< class clusterizer >
Matrix nn_addon::_kmeans_filter(clusterizer& cengine, const Matrix& p, const Matrix& f, Matrix& lp, Matrix& lf)
{
	//select which function to call depending on clustering engine
	struct find_clusters {
		static void go(KM::kmeans& cengine, const Matrix& p, const Matrix& f, double mult, ulong maxiter) {
			cengine.find_clusters_f(p, f, mult, maxiter, NULL, false);
			//cengine.drops_hetero_map(p, f, mult, maxiter);
			//cengine.drops_hetero_simple(p, f, mult, 200);
		}
		static void go(DA::determ_annealing& cengine, const Matrix& p, const Matrix& f, double mult, ulong maxiter) {
			cengine.find_clusters(p, f, mult, maxiter);
		}
	};

	//do a clusterization of learning data
	find_clusters::go(cengine, p, f, max< ulong >(p.row_num() * opt_.kmf_cmult, 1), 200);

	//cengine.drops_hetero_simple(p, f, opt_.kmf_cmult, 200);
	//cengine.find_clusters_f(p, f, max< ulong >(p.row_num() * opt_.kmf_cmult, 1), 200, NULL, false);

	//cengine.find_clusters(p, max<ulong>(p.row_num() * opt_.kmf_cmult, 1), 200, false, NULL, true);
	//cengine.opt_.use_prev_cent = true;

	Matrix c = cengine.get_centers();
	ulMatrix ind = cengine.get_ind();

	//find best solution
	ulong best_ind = f.min_ind();
	Matrix cur_b = p.GetRows(best_ind);
#ifdef VERBOSE
	DumpMatrix(cur_b | f.GetRows(best_ind), "best_sol.txt");
#endif

	//calc distances from best solution to all cluster centers
	Matrix dist(1, c.row_num());
	for(ulong i = 0; i < c.row_num(); ++i)
		dist[i] = (c.GetRows(i) - cur_b).norm2();

	//sort distances
	ulMatrix ci = dist.RawSort();

	//get affiliation
	const KM::kmeans::vvul& aff = cengine.get_aff();
	//collect learning samples, & search for closest centers and more
	lp.clear();	lf.clear();
	lp.reserve(opt_.bestCount*p.col_num()); lf.reserve(opt_.bestCount*f.row_num());
	ulong search_bound = 0;
	Matrix close_cl;
	bool learn_built = false, search_built = false;
	ulong c_ind = 0;
	for(; c_ind < c.row_num() && !learn_built; ++c_ind) {
		if(opt_.learn_clust_num > 0 && c_ind >= opt_.learn_clust_num) break;
		if(opt_.search_clust_num > 0 && c_ind >= opt_.search_clust_num) search_built = true;
		close_cl &= c.GetRows(ci[c_ind]);
		const ulong* cur_aff = (ulong*)&aff[ci[c_ind]][0];
		for(ulong j = 0; j < aff[ci[c_ind]].size(); ++j) {
			lp &= p.GetRows(cur_aff[j]); lf &= f.GetRows(cur_aff[j]);
			if(lp.row_num() >= (ulong)opt_.bestCount) learn_built = true;
			if(!search_built) ++search_bound;
		}
		/*
		for(ulong j = 0; j < ind.size() && !learn_built; ++j) {
			if(ind[j] == ci[c_ind]) {
				lp &= p.GetRows(j); lf &= f.GetRows(j);
				if(lp.row_num() >= opt_.bestCount) learn_built = true;
				if(!search_built) ++search_bound;
			}
		}
		*/
	}
	//update closest centers - add centers within learning points square
	Matrix lpmm = lp.minmax();
	Matrix lpmin = lpmm.GetRows(0), lpmax = lpmm.GetRows(1);
	Matrix cur_c;
	state_.learn_cl_cent.clear();
	for(; c_ind < c.row_num(); ++c_ind) {
		cur_c <<= c.GetRows(ci[c_ind]);
		if(cur_c >= lpmin && cur_c <= lpmax)
			close_cl &= cur_c;
		else
			state_.learn_cl_cent &= cur_c;
	}
	//insert closest cluster centers in the beginning
	state_.learn_cl_cent <<= close_cl & state_.learn_cl_cent;

#ifdef VERBOSE
	DumpMatrix(close_cl, "c.txt");
#endif

	if(opt_.search_clust_num == 0 && opt_.search_samples_num > 0)
		search_bound = opt_.search_samples_num;

	if(search_bound < lp.row_num()) {
#ifdef VERBOSE
		DumpMatrix(lp.GetRows(0, search_bound), "ss.txt");
#endif
		return lp.GetRows(0, search_bound).minmax();
	}
	else {
#ifdef VERBOSE
		DumpMatrix(lp, "ss.txt");
#endif
		return lp.minmax();
	}
}

double nn_addon::_GetOneAddonStd(const ga& caller, Matrix& p, Matrix& s, Matrix& new_chrom, ulong net_ind)
{
	//define learning data
	Matrix learn, tar;
	_get_learnData(p, s, learn, tar);

	//data normalizing
	state_.cur_min = tar.Min();
	state_.cur_max = tar.Max();
	if(!opt_.is_ffRestricted) {
		state_.norm_min = state_.cur_min;
		state_.norm_max = state_.cur_max;
	}

	if(opt_.normType == LogNorm) {
		state_.lnorm_max = log(state_.norm_max);
		state_.lnorm_min = log(state_.norm_min);
	}
	else if(opt_.normType == LogsigNorm) {
		state_.a = -log(1/opt_.maxtar - 1)/max(abs(state_.norm_max), abs(state_.norm_min));
		state_.lsnorm_max = 1/(1 + exp(-state_.a * state_.norm_max));
		state_.lsnorm_min = 1/(1 + exp(-state_.a * state_.norm_min));
	}

	Matrix mm(1, 2);
	mm[0] = state_.cur_min;
	mm[1] = state_.cur_max;
	mm = GetNormalizedData(mm);
	state_.n_cur_min = mm[0];
	state_.n_cur_max = mm[1];
	if(opt_.pred_ratio > 0) {
		state_.pred_r = abs(tar_[state_.ind_best])*opt_.pred_ratio;
		//_state.pred_r = abs(_state.n_cur_max - _state.n_cur_min)*opt_.pred_ratio;
		//_state.pred_r *= (1 - caller._state.nGen/caller.opt_.generations);
	}

	Matrix ntar = GetNormalizedData(tar);
	if(opt_.normInp) learn <<= NormalizePop(learn);

	//learn neural network
	if(_learn_network(learn, ntar, net_ind) == NN::error)
		return ERR_VAL;

	//_ga.opt_.initRange <<= learn.minmax();
	//ga optimization on learned NN
	if(ga_.opt_.scheme != ClearGA) {
		cerr << opt_.name << ": WRN: Inner GA scheme must be ClearGA! Corrected" << endl;
		ga_.opt_.scheme = ClearGA;
	}
	if(caller.state_.nGen > 1) ga_.opt_.openMode = std::ios::app;

	//_ga.opt_.initRange = caller.opt_.initRange;

	//debug rough method
	//_ga.opt_.initRange.NewMatrix(0, 0);
	//_ga.opt_.initRange &= _learn.GetRows(_state.ind_best) - 0.5*(1 - caller._state.nGen/caller.opt_.generations);
	//_ga.opt_.initRange &= _learn.GetRows(_state.ind_best) + 0.5*(1 - caller._state.nGen/caller.opt_.generations);

	//smart method - collect best 10% individuals
	//ulong points_num = 10; //_tar.size()*0.1;
	//Matrix points; points.reserve(points_num*_learn.col_num());
	//Matrix srt_tar = _tar; Matrix mInd = srt_tar.RawSort();
	//for(ulong i=0; i<points_num; ++i)
	//	points &= _learn.GetRows(mInd[i]);
	////now determine search region as minmax of selected points
	//_ga.opt_.initRange = points.minmax();

	ga_.opt_.useFitLimit = false;
	ga_.opt_.fitLimit = state_.n_cur_min - state_.pred_r;
	cout << opt_.name << ": inner GA starts" << endl;
	pnna = this;
	pnn = _onet[net_ind].get();
	new_chrom &= ga_.Run(NNFitnessFcn, p.col_num());
	//new_chrom <<= _ga.Run(NNFitnessFcn, pop.col_num());

	return ga_.bestScore_;
}

double nn_addon::_GetOneAddonAlt(const ga& caller, Matrix& p, Matrix& s, Matrix& new_chrom, ulong net_ind)
{
	Matrix chrom_mm = GetChromMM(p);

	//define learning data
	Matrix learn, tar;
	_get_learnData(p, s, tar, learn);

	//data normalization
	if(opt_.normInp)
		tar <<= NormalizePop(tar);

	chrom_mm = GetChromMM(tar);
	state_.cur_min = chrom_mm.GetColumns(0).Min();
	state_.cur_max = chrom_mm.GetColumns(1).Max();
	if(!opt_.is_ffRestricted) {
		state_.norm_min = state_.cur_min;
		state_.norm_max = state_.cur_max;
	}

	if(opt_.normType == LogNorm) {
		state_.lnorm_max = log(state_.norm_max);
		state_.lnorm_min = log(state_.norm_min);
	}
	else if(opt_.normType == LogsigNorm) {
		state_.a = -log(1/0.9 - 1)/max(abs(state_.norm_max), abs(state_.norm_min));
		state_.lsnorm_max = 1/(1 + exp(-state_.a * state_.norm_max));
		state_.lsnorm_min = 1/(1 + exp(-state_.a * state_.norm_min));
	}

	Matrix mm(1, 2);
	mm[0] = state_.cur_min;
	mm[1] = state_.cur_max;
	mm = GetNormalizedData(mm);
	state_.n_cur_min = mm[0];
	state_.n_cur_max = mm[1];
	//if(opt_.pred_ratio > 0) _state.pred_r = abs(_state.n_cur_max - _state.n_cur_min)*opt_.pred_ratio;

	Matrix ntar = GetNormalizedData(tar);

	//if(opt_.normInp)
	//	learn <<= NormalizePop(learn);

	//get normalized lengths
	//Matrix ntar_len = GetNormalizedData(tar_len);
	//now normalize population
	//Matrix ntar(tar.row_num(), tar.col_num());
	//for(ulong i=0; i<tar.row_num(); ++i)
	//	ntar.SetRows(tar.GetRows(i)*ntar_len[i], i);

	//but really need to do targets normalizing if last layer not linear
	//Matrix ntar = tar;
	//if(*_net._LTypes.rbegin() != NN::purelin)
	//	ntar <<= NormalizePop(tar);

	//learn neural network
	if(_learn_network(learn, ntar, net_ind) == NN::error) {
		new_chrom &= learn_.GetRows(state_.ind_best);
		return ERR_VAL;
	}

	//no ga optimizitaion - directly get parameters vector
	Matrix new_best(1, 1);
	new_best[0] = tar_[state_.ind_best] - tar_[state_.ind_best]*opt_.pred_ratio;
	//if(opt_.normInp) new_best[0] /= _state.max_ch_r;
	if(opt_.usePCA)
		new_best = netPCA_->Sim(new_best - state_.inpMean);
	new_chrom &= GetRealData(!net_.Sim(new_best));
	if(opt_.normInp) new_chrom *= state_.max_ch_r;

	//now normalize new best
	//ntar_len.NewMatrix(1, 1);
	//ntar_len = GetChromMaxs(new_chrom);
	//GetNormalizedData(ntar_len);
	//new_chrom *= ntar_len[0];

	//if(*_net._LTypes.rbegin() != NN::purelin)
	//	new_chrom *= _state.max_ch_r;

	return new_best[0];
}

double nn_addon::_GetOneAddonPCA(const ga& caller, Matrix& p, Matrix& s, Matrix& new_chrom, ulong net_ind)
{
	//define learning data
	Matrix learn, tar;
	_get_learnData(p, s, learn, tar);
	learn <<= !learn;
	//tar <<= !tar;

	cout << opt_.name << ": PCA NN learning" << endl;
	//subtract mean
	state_.inpMean <<= learn.vMean(true);
	learn <<= learn.SubMean(state_.inpMean, true);
	//debug
	ofstream popf("x.txt", ios::out | ios::ate);
	//pcaf << "Original population" << endl;
	learn.Print(popf);

	//_net.LinPCA(!input, _state.netPCA, LearnNNInformer);
	netPCA_->PCALearn(learn, state_.init_PCA, MNetInformer);

	new_chrom &= netPCA_->weights(0).GetRows(0)*opt_.pred_ratio + learn_.GetRows(state_.ind_best);

	return tar_[state_.ind_best];
}

Matrix nn_addon::GetAddon(const Matrix& pop, const Matrix& score, const GA::ga& caller, Matrix& new_chrom)
{
	Matrix res(_onet.size(), 1);
	res = ERR_VAL;

	try {
		new_chrom = 0;

		//make local copy for modifying
		Matrix p, s;
		p = pop; s = score;

		_fillup_storage(p, s);

		//get addons from neural networks
		for(ulong i = 0; i < _onet.size(); ++i) {
			res[i] = (this->*_pAddonFcn)(caller, p, s, new_chrom, i);
		}
	}
	catch (...) {
		ga_.Stop();
		throw;
	}

	return res;
}

void nn_addon::BuildApproximation(const Matrix& samples, const Matrix& want_resp)
{
	//make local copy for modifying
	Matrix p, s;
	p = samples; s = want_resp;
	tar_.clear(); learn_.clear();

	_fillup_storage(p, s);

	//define learning data
	Matrix learn, tar;
	opt_.bestCount = p.row_num();
	_get_learnData(p, s, learn, tar);
	/*
	_learn = p; _tar = s;
	learn <<= p; tar <<= s;
	_state.ind_best = _tar.ElementInd(_tar.Min());
	_state.tar_mean = _tar.Mean();
	_state.tar_std = _tar.Std();
	if(opt_.netType == rb_nn) {
		opt_.bestCount = learn.row_num();
		_ga.opt_.initRange = _kmeans_filter(_learn, _tar, learn, tar);
	}
	*/

	//data normalizing
	state_.cur_min = tar.Min();
	state_.cur_max = tar.Max();
	if(!opt_.is_ffRestricted) {
		state_.norm_min = state_.cur_min;
		state_.norm_max = state_.cur_max;
	}

	if(opt_.normType == LogNorm) {
		state_.lnorm_max = log(state_.norm_max);
		state_.lnorm_min = log(state_.norm_min);
	}
	else if(opt_.normType == LogsigNorm) {
		state_.a = -log(1/opt_.maxtar - 1)/max(abs(state_.norm_max), abs(state_.norm_min));
		state_.lsnorm_max = 1/(1 + exp(-state_.a * state_.norm_max));
		state_.lsnorm_min = 1/(1 + exp(-state_.a * state_.norm_min));
	}

	Matrix mm(1, 2);
	mm[0] = state_.cur_min;
	mm[1] = state_.cur_max;
	mm = GetNormalizedData(mm);
	state_.n_cur_min = mm[0];
	state_.n_cur_max = mm[1];
	if(opt_.pred_ratio > 0) {
		state_.pred_r = abs(tar_[state_.ind_best])*opt_.pred_ratio;
		//_state.pred_r = abs(_state.n_cur_max - _state.n_cur_min)*opt_.pred_ratio;
		//_state.pred_r *= (1 - caller._state.nGen/caller.opt_.generations);
	}

	Matrix ntar = GetNormalizedData(tar);
	if(opt_.normInp) learn <<= NormalizePop(learn);

	//learn neural network
	_learn_network(learn, ntar);
}

Matrix nn_addon::Sim(const Matrix& samples, ulong net_ind)
{
	if(opt_.netType == matrix_nn)
		return net_.Sim(samples);
	else
		return _onet[net_ind]->sim(samples);
}

const Matrix& nn_addon::GetClusterCenters() const
{
	return state_.km.get_centers();
}

Matrix nn_addon::GetNormalizedData(Matrix& data)
{
	Matrix res;
	//Matrix ldata;
	switch(opt_.normType) {
		default:
		case UnchangedData:
			res = data;
			break;
		case LinearNorm:
			res = (data - state_.norm_min)*(opt_.maxtar - opt_.mintar)/(state_.norm_max - state_.norm_min) +
				opt_.mintar;
			break;
		case LogNorm:
			res = data;
			transform(res.begin(), res.end(), res.begin(), ptr_fun<double, double>(std::log));
			res = (res - state_.lnorm_min)*(opt_.maxtar - opt_.mintar)/(state_.lnorm_max - state_.lnorm_min) +
				opt_.mintar;
			break;
		case LogsigNorm:
			res.NewMatrix(data.row_num(), data.col_num());
			Matrix::r_iterator p_data(data.begin());
			for(Matrix::r_iterator p_res = res.begin(); p_res != res.end(); ++p_res) {
				*p_res = 1.0/(1.0 + exp(-state_.a * (*p_data)));
				//*p_res = (*p_res - _state.lsnorm_min)*(opt_.maxtar - opt_.mintar)/
				//	(_state.lsnorm_max - _state.lsnorm_min) + opt_.mintar;
				++p_data;
			}
	}
	return res;
}

Matrix nn_addon::GetRealData(const Matrix& ndata)
{
	Matrix res;
	//Matrix ldata;
	switch(opt_.normType) {
		default:
		case UnchangedData:
			res = ndata;
			break;
		case LinearNorm:
			res = (ndata - opt_.mintar)*(state_.norm_max - state_.norm_min)/(opt_.maxtar - opt_.mintar) +
				state_.norm_min;
			break;
		case LogNorm:
			res = (ndata - opt_.mintar)*(state_.lnorm_max - state_.lnorm_min)/(opt_.maxtar - opt_.mintar) +
				state_.lnorm_min;
			for(ulong i=0; i<res.size(); ++i)
				res[i] = pow(10.0, res[i]);
			//for_each(res.begin(), res.end(), bind1st(pow, 10.0));
			break;
		case LogsigNorm:
			//er = -log(1/x - 1)/a;
			res.NewMatrix(ndata.row_num(), ndata.col_num());
			Matrix::cr_iterator p_ndata(ndata.begin());
			for(Matrix::r_iterator p_res = res.begin(); p_res != res.end(); ++p_res) {
				//*p_res = (*p_ndata - opt_.mintar)*(_state.lsnorm_max - _state.lsnorm_min)/
				//	 (opt_.maxtar - opt_.mintar) + _state.lsnorm_min;
				*p_res = -log(1/(*p_res) - 1.)/state_.a;
				++p_ndata;
			}
			break;
	}
	return res;
}

Matrix nn_addon::GetChromLengths(const Matrix& pop)
{
	Matrix res(pop.row_num(), 1);
	Matrix pop2 = pop.Mul(pop);
	for(ulong i=0; i<pop.row_num(); ++i)
		res[i] = sqrt(pop2.GetRows(i).Sum());
	return res;
}

Matrix nn_addon::GetChromMM(const Matrix& pop)
{
	Matrix res(pop.row_num(), 2);
	for(ulong i=0; i<pop.row_num(); ++i) {
		res(i, 0) = pop.GetRows(i).Min();
		res(i, 1) = pop.GetRows(i).Max();
	}
	return res;
}

Matrix nn_addon::NormalizePop(const Matrix& pop)
{
	Matrix pop2 = pop.Mul(pop);
	state_.max_ch_r = 0;
	double r;
	for(ulong i=0; i<pop.row_num(); ++i) {
		r = pop2.GetRows(i).Sum();
		if(r > state_.max_ch_r) state_.max_ch_r = r;
	}
	state_.max_ch_r = sqrt(state_.max_ch_r);

	Matrix npop = pop / state_.max_ch_r;
	return npop;
}

void GA::nn_addon::_build_surf(ulong net_ind, const Matrix& input, const Matrix& targets)
{
	/*
	Matrix x(1, 100), y(100, 1);
	x = abs(_ga.opt_.initRange(1, 0) - _ga.opt_.initRange(0, 0))/(x.size() - 1);
	x[0] = 0; x <<= !(!x).CumSum(); x += _ga.opt_.initRange(0, 0);
	y = abs(_ga.opt_.initRange(1, 1) - _ga.opt_.initRange(0, 1))/(y.size() - 1);
	y[0] = 0; y <<= y.CumSum(); y += _ga.opt_.initRange(0, 1);

	Matrix yrow(1, 100);
	Matrix surf;
	for(ulong i = 0; i < y.row_num(); ++i) {
		yrow = y[i];
		if(opt_.netType == matrix_nn)
			surf &= _net.Sim(x & yrow);
		else
			surf &= _onet[net_ind]->sim(x & yrow);
	}l
	DumpMatrix(x, "x.txt");
	DumpMatrix(y, "y.txt");
	DumpMatrix(surf, "z.txt");
	*/

	Matrix points(ga_.opt_.initRange.col_num(), 10000);
	generate(points.begin(), points.end(), prg::rand01);
	const ulong pnum = points.col_num();
	//range matrix
	Matrix range = input.minmax(false);
	//lower bound
	Matrix a = range.GetRows(0);
	//difference between bounds
	Matrix scale = range.GetRows(1) - a;
	//start moving points
	Matrix::r_iterator pos = points.begin();
	Matrix::r_iterator p_a = a.begin();
	Matrix::r_iterator p_sc = scale.begin();
	for(ulong i = 0; i < points.row_num(); ++i, ++p_a, ++p_sc)
	{
		//x = x*(b - a)
		transform(pos, pos + pnum, pos, bind2nd(multiplies<double>(), *p_sc));
		pos = transform(pos, pos + pnum, pos, bind2nd(plus<double>(), *p_a));
		//pos += points.col_num();
	}
	DumpMatrix(points, "y.txt");

	Matrix surf;
	if(opt_.netType == matrix_nn)
		surf <<= net_.Sim(points);
	else
		surf <<= _onet[net_ind]->sim(points);
	DumpMatrix(surf, "z.txt");
}
