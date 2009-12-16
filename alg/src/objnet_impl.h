#include "objnet.h"
#include "m_algorithm.h"
// text tables
//#include "text_table.h"

#include <time.h>
#include <sstream>
//#include <iosfwd>
#include <iostream>

// defaul record width for formatted output
#define NW 13

//using namespace NN;
namespace NN {

using namespace std;
using namespace hybrid_adapt;

std::string decode_nn_type(int type) {
	switch(type) {
		case mlp_nn:
			return "Multi-Layered Perceptron";
		case cc_nn:
			return "Cascade-Correlation NN";
		case rb_nn:
			return "Radial Basis NN";
		case pca_nn:
			return "Principal Component Analysis NN";
		default:
			return "Unknown NN";
	}
}

std::string decode_layer_type(int type) {
	switch(type) {
		case common_nnl:
			return "Std layer";
		case bp_nnl:
			return "Backprop layer";
		case rb_nnl:
			return "RBN layer";
		case falman_nnl:
			return "Fahlman layer";
		case jump_nnl:
			return "Std layer with jump AF";
		default:
			return "Unknown layer";
	}
}

std::string decode_neuron_type(int af) {
	switch(af) {
		case logsig:
			return "logsig";
		case tansig:
			return "tansig";
		case radbas:
			return "radbas";
		case purelin:
			return "purelin";
		case poslin:
			return "poslin";
		case expws:
			return "expws";
		case multiquad:
			return "multiquad";
		case revradbas:
			return "revradbas";
		case revmultiquad:
			return "revmultiquad";
		default:
			return "unknown";
	}
}

text_table decode_neuron_type(const iMatrix& af, bool summarize) {
	iMatrix af_(1, af.size(), af.GetBuffer());
	if(summarize)
		af_ = af.Sort(less< int >());
	else af_ = af;
	// ensure af_ is a row
	// af_.Resize(1, af_.size());

	// go throw matrix
	text_table tt;
	tt << "- 0 _ 0 -" << tt_endr();
	ulong n;
	for(ulong i = 0; i < af_.size(); ++i) {
		n = 1;
		if(summarize) {
			while(i < af_.size() - 1 && af_[i + 1] == af_[i]) {
				++n; ++i;
			}
		}
		tt << n << "&" << decode_neuron_type(static_cast< ActFun >(af_[i])) << tt_endr();
	}
	return tt;
}

std::string decode_nn_state(int status) {
	switch(status) {
		case learned:
			return "learning successfull - goal reached!";
		case stop_maxcycle:
			return "learning stopped - max cycles reached";
		case stop_palsy:
			return "learning stopped - palsy";
		case stop_patience:
			return "learning stopped - no significant performance improve";
		case stop_test_validation:
			return "learning stopped - error on validation set doesn't improve";
		default:
			return "";
	}
}

//-------------------------------objnet implementation-------------------------------------------
void objnet::_print_err(const char* pErr)
{
	if(!errFile_.is_open()) {
		errFile_.open(opt_.errFname_.c_str(), ios::trunc | ios::out);
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

objnet::objnet() :
	input_(*this),
	opt_holder_(new nn_opt), opt_(*opt_holder_)
{
	opt_.set_def_opt();
	state_.status = not_learned;
}

objnet::objnet(nn_opt* opt) :
	input_(*this),
	opt_holder_(opt), opt_(*opt_holder_)
{
	opt_.set_def_opt();
	state_.status = not_learned;
}

void objnet::set_input_size(ulong inp_size, bitMatrix *const pConMat)
{
	input_.init(inp_size, purelin);
	if(layers_.size() > 0) {
		if(pConMat)
			layers_[0].set_links(create_ptr_mat(input_.neurons_), *pConMat);
		else
			layers_[0].set_links(create_ptr_mat(input_.neurons_));
	}
}

void objnet::set_input(const Matrix& input)
{
	if(input.col_num() != 1 || input.row_num() != inp_size()) {
		_print_err(nn_except::explain_error(SizesMismatch));
		throw nn_except(SizesMismatch);
	}
	input_.axons_ = input;
}

//template<class layer_type>
sp_layer objnet::add_layer(ulong neurons_count, int af_type, int layer_type, ulong where_ind) {
	//check if no input size specified
	if(layers_num() == 0 && inp_size() == 0) {
		_print_err(nn_except::explain_error(NoInputSize));
		throw nn_except(NoInputSize);
	}
	// create given layer
	sp_layer p_l;
	switch(layer_type) {
		default:
		case common_nnl:
			p_l = new layer(*this, neurons_count, af_type);
			break;
		case bp_nnl:
			p_l = new bp_layer(*this, neurons_count, af_type);
			break;
		case jump_nnl:
			p_l = new layer_jump(*this, neurons_count);
			break;
		case rb_nnl:
			p_l = new rb_layer(*this, neurons_count);
			break;
		case falman_nnl:
			p_l = new falman_layer(*this, neurons_count);
			break;
	}
	// add layer to network
	if(where_ind < layers_.size())
		layers_.insert(p_l, where_ind, false);
	else
		layers_.push_back(p_l, false);

	return p_l;
	//return (layer_type&)*p_l;
}

//template< class layer_type >
sp_layer objnet::add_layer(ulong neurons_count, const iMatrix& af_mat, int layer_type, ulong where_ind) {
	//check if no input size specified
	if(layers_num() == 0 && inp_size() == 0) {
		_print_err(nn_except::explain_error(NoInputSize));
		throw nn_except(NoInputSize);
	}
	// create given layer
	sp_layer p_l;
	switch(layer_type) {
		case common_nnl:
			p_l = new layer(*this, af_mat);
			break;
		case bp_nnl:
			p_l = new bp_layer(*this, af_mat);
			break;
		case falman_nnl:
			p_l = new falman_layer(*this, neurons_count);
			break;
		default:
			throw nn_except(string(string("Layer of type '") + decode_layer_type(layer_type) +
					"' doesn't support AF matrix").c_str());
	}
	// add layer to network
	if(where_ind < layers_.size())
		layers_.insert(p_l, where_ind, false);
	else
		layers_.push_back(p_l, false);

	return p_l;
	//return (layer_type&)*p_l;
}

void objnet::propagate()
{
	for(l_iterator p_l = layers_.begin(); p_l != layers_.end(); ++p_l)
		p_l->propagate();
}

void objnet::init_weights(const Matrix& inputs)
{
	for(l_iterator p_l = layers_.begin(); p_l != layers_.end(); ++p_l)
		p_l->init_weights(inputs);
}

void objnet::prepare2learn()
{
	for(l_iterator p_l = layers_.begin(); p_l != layers_.end(); ++p_l)
		p_l->prepare2learn();
	//no local grad backprop for first layer
	layers_.begin()->backprop_lg_ = false;
}

bool objnet::calc_grad(const Matrix& target)
{
	//propagate stage moved to epoch level
	//set_input(input);
	//propagate();

	//position to the last layer
	l_iterator p_l = layers_.end() - 1;
	//calc error
	p_l->Goal_ = target - p_l->axons_;
	state_.perf += p_l->Goal_.Mul(p_l->Goal_).Sum();

	//if no gradient backprop from last layer - exit
	//if(!p_l->backprop_lg_) return false;

	//zero errors in preceding layers
	for(--p_l; p_l >= layers_.begin(); --p_l)
		p_l->Goal_ = 0;

	//calc gradient
	bool palsy = true;
	for(p_l = layers_.end() - 1; p_l >= layers_.begin(); --p_l) {
		palsy &= p_l->calc_grad();
		if(!p_l->backprop_lg_) break;
	}

	return palsy;
}

void objnet::bp_epoch(const Matrix& inputs, const Matrix& targets)
{
	//create patterns present order
	vector<ulong> order(inputs.col_num());
	vector<ulong>::iterator p_order;
	for(ulong i = 0; i < order.size(); ++i)
		order[i] = i;
	//shuffle order
	if(!opt_.batch) random_shuffle(order.begin(), order.end());

	state_.perf = 0;
	bool palsy = true;
	for(p_order = order.begin(); p_order != order.end() && state_.status != stop_palsy; ++p_order)
	{
		set_input(inputs.GetColumns(*p_order));
		propagate();

		palsy &= calc_grad(targets.GetColumns(*p_order));
		bp_after_grad();

		if(!opt_.batch) update_epoch();
	}

	bp_batch_correct_grad(inputs, targets);

	if(palsy) state_.status = stop_palsy;
}

void objnet::bp_batch_correct_grad(const Matrix& inputs, const Matrix& targets) {
	if(opt_.batch) {
		for(l_iterator p_l = layers_.begin(); p_l != layers_.end(); ++p_l) {
			if(opt_.learnFun != R_BP) {
				//normalize gradient
				p_l->BG_ /= inputs.col_num();
				for(n_iterator p_n = p_l->neurons_.begin(); p_n != p_l->neurons_.end(); ++p_n)
					p_n->grad_ /= inputs.col_num();
			}
			else if(state_.cycle == 0) {
				//old grad = grad
				p_l->OBG_ = p_l->BG_;
				for(n_iterator p_n = p_l->neurons_.begin(); p_n != p_l->neurons_.end(); ++p_n)
					p_n->prevg_ = p_n->grad_;
			}
		}
	}
}

void objnet::lsq_epoch(const Matrix& inputs, const Matrix& targets)
{
	l_iterator ol = layers_.begin() + (layers_num() - 1);
	n_iterator p_n;

	bool lsq_applicable = true;
	if(ol->aft() == (int)purelin) {
		//check if lsq is applicable - number of samples must be > number of inputs of each neuron in last layer
		for(p_n = ol->neurons().begin(); p_n != ol->neurons().end(); ++p_n)
			if(p_n->inputs_.size() + opt_.use_biases_ > inputs.col_num()) {
				lsq_applicable = false;
				break;
			}
	}
	else
		lsq_applicable = false;
	//if non-applicable - switch to bp_epoch
	if(!lsq_applicable) {
		bp_epoch(inputs, targets);
		return;
	}

	layer* gl;
	if(layers_.size() > 1)
		gl = &layers_[layers_num() - 2];
	else
		gl = &input_;
	const MatrixPtr o_axons = gl->out();

	//setup G matrices
	vector<Matrix> G(ol->size());
	vector<Matrix>::iterator p_G;
	p_n = ol->neurons().begin();
	for(ulong i = 0; i < ol->neurons().size(); ++i)
		//+ 1 for bias input
		G[i].NewMatrix(inputs.col_num(), p_n->inputs_.size() + opt_.use_biases_, 1);

	//calculate G matrices - main cycle
	ulong cnt;
	state_.perf = 0;
	for(ulong i = 0; i < inputs.col_num(); ++i) {
		//propagate sample
		set_input(inputs.GetColumns(i));
		propagate();

		//now calculate corresponding G matrix
		p_G = G.begin(); cnt = 0;
		for(p_n = ol->neurons().begin(); p_n != ol->neurons().end(); ++p_n) {
			ulong j = 0;
			for(; j < p_n->inputs_.size(); ++j)
				(*p_G)(i, j) = p_n->inputs_[j].axon_;
			if(opt_.use_biases_) (*p_G)(i, j) = 1;
			++p_G; ++cnt;
		}

		// do standard gradient calculation for preceding layers
		calc_grad(targets.GetColumns(i));
		bp_after_grad();
	}

	bp_batch_correct_grad(inputs, targets);

	//update weights in last layer
	Matrix piG, full_weights;
	p_G = G.begin();
	p_n = ol->neurons().begin();
	for(ulong i = 0; i < ol->neurons().size(); ++i) {
		//calc pseudo inverse of G
		pseudo_inv(*p_G, piG);
		//calc weights + bias
		full_weights <<= piG * (!targets.GetRows(i));
		p_n->weights_ <<= full_weights.GetRows(0, p_n->weights_.size());
		ol->B_[i] = full_weights[full_weights.size() - 1];
		// zero gradient to prevent later wheights modification
		p_n->grad_ = 0;
		++p_n; ++p_G;
	}
	ol->BG_ = 0;
}

void objnet::std_update_epoch()
{
	if(state_.status == learning) {
		for(l_iterator p_l = layers_.begin(); p_l != layers_.end(); ++p_l)
			p_l->update_epoch();
			//(p_l->*_pUpdateFun)(true);
	}
}

void objnet::prep_learn_valid_sets(const Matrix& input, const Matrix& targets, smart_ptr< const Matrix >& test_inp,
		smart_ptr< const Matrix >& test_tar, Matrix& real_input, Matrix& real_targets)
{
	if(opt_.goal_checkFun == test_validation) {
		if(!test_inp) {
			// extract randomly validation set from learning set
			ulong val_size = ha_round(input.col_num() * opt_.validation_fract);
			ulong ls_size = input.col_num() - val_size;
			real_input.Resize(input.row_num(), ls_size);
			real_targets.Resize(targets.row_num(), ls_size);
			Matrix* xv_inp = new Matrix(input.row_num(), val_size);
			Matrix* xv_tar = new Matrix(targets.row_num(), val_size);

			// generate rand permutation
			vector< ulong > rp = prg::rand_perm(input.col_num());
			for(ulong i = 0; i < input.col_num(); ++i) {
				if(i < val_size) {
					xv_inp->SetColumns(input.GetColumns(rp[i]), i);
					xv_tar->SetColumns(targets.GetColumns(rp[i]), i);
				}
				else {
					real_input.SetColumns(input.GetColumns(rp[i]), i);
					real_targets.SetColumns(targets.GetColumns(rp[i]), i);
				}
			}
			test_tar = xv_tar;
			test_inp = xv_inp;
		}
		else {
			real_input <<= input;
			real_targets <<= targets;
		}
		// do some initialization
		check_early_stop(state_, *test_inp, *test_tar);
	}
	else {
		real_input <<= input;
		real_targets <<= targets;
	}
}

int objnet::common_learn(const Matrix& inputs, const Matrix& targets, bool initialize, pLearnInformer pProc,
		smart_ptr< const Matrix > test_inp, smart_ptr< const Matrix > test_tar)
{
	try {
#ifdef VERBOSE
		DumpMatrix(inputs, "inputs.txt");
		DumpMatrix(targets, "targets.txt");
#endif
		if(state_.status == learning)
			throw nn_except(NN_Busy, "This network is already in learning state");
		state_.status = learning;
		state_.cycle = 0;
		state_.patience_counter = 0;
		state_.lastPerf = 0;
		state_.nu = opt_.nu;
		//if(opt_.showPeriod == 0) opt_.showPeriod = opt_.maxCycles;

		//prepare for learning
		prepare2learn();

		Matrix real_inputs;
		Matrix real_targets;
		// prepare learn & validation sets
		prep_learn_valid_sets(inputs, targets, test_inp, test_tar, real_inputs, real_targets);

		//init weights if needed
		if(initialize) init_weights(real_inputs);

		//main learn cycle
		bool can_use_informer = pProc && (opt_.showPeriod != 0);
		while(state_.status == learning) {
			if(state_.cycle > 0) state_.lastPerf = state_.perf;

			//epoch main data processing function
			learn_epoch(real_inputs, real_targets);

			//calc final performance
			if(opt_.perfFun == mse)
				state_.perf /= real_inputs.col_num();

			//update cycles counter
			++state_.cycle;

			//check if goal is reached and other stopping criteria is met
			is_goal_reached();

			//make additional checks
			if(state_.status == learning) {
				if(opt_.goal_checkFun & patience)
					check_patience(state_, opt_.patience, opt_.patience_cycles);
				if(opt_.goal_checkFun & test_validation)
					check_early_stop(state_, *test_inp, *test_tar);
			}

			// update weights
			update_epoch();

			//if(++state_.cycle == opt_.maxCycles && state_.status == learning)
			//	state_.status = stop_maxcycle;

			//call informer
			if(can_use_informer && (state_.status != learning || (state_.cycle % opt_.showPeriod == 1)) &&
				!pProc(state_.cycle, state_.perf, (void*)this))
			{
				state_.status = stop_breaked;
				//break;
			}
		}
	}
	//errors handling
	catch(alg_except& ex) {
		state_.status = error;
		_print_err(ex.what());
		throw;
	}
	catch(exception ex) {
		state_.status = error;
		_print_err(ex.what());
		throw nn_except(ex.what());
	}
	catch(...) {
		state_.status = error;
		_print_err("Unknown run-time exception thrown");
		throw nn_except(-1, state_.lastError.c_str());
	}

	return state_.status;
}

template< class goal_action >
int objnet::_check_patience(nnState& state, double patience, ulong patience_cycles, int patience_status)
{
	//delta = -(state.perf - state.perfMean) if we are minimizing and should be positive
	//to reset patience counter
	double delta;
	goal_action::assign(delta, state.perf - state.perfMean);

	if(state.cycle == 1) {
		state.patience_counter = 0;
		state.perfMean = state.perf;
	}
	else if(delta >= abs(patience * state.perfMean)) {
		state.patience_counter = 0;
		state.perfMean = state.perf;
	}
	else if(++state.patience_counter == patience_cycles)
		state.status = patience_status;
	return state.status;

	/*
	//calc runing mean
	if(state_.cycle == 0) state_.perfMean = state_.perf;
	else {
		double prevMean = state_.perfMean;
		state_.perfMean = (state_.perfMean + state_.perf)/2;
		if(abs(prevMean - state_.perfMean) < opt_.epsilon)
			++state_.no_improve_counter;
		else
			state_.no_improve_counter = 0;
	}
	if(state_.no_improve_counter >= opt_.maxNoImproveCycles)
		state_.status = stop_no_improve;
	*/
}

int objnet::check_patience(nnState& state, double patience, ulong patience_cycles, int patience_status) {
	return _check_patience< anti_grad >(state, patience, patience_cycles, patience_status);
}

ulong objnet::check_early_stop(nnState& state, const Matrix& test_set, const Matrix& test_tar)
{
	//static nnState test_state;

	if(state.cycle == 0) {
		xvalid_state_.status = learning;
		xvalid_state_.perf = state_.perf;
		return 0;
	}

	//calc error on validation set
	double val_err = 0;
	layer& outp = layers_[layers_num() - 1];
	Matrix cur_err;
	for(ulong i = 0; i < test_set.col_num(); ++i) {
		set_input(test_set.GetColumns(i));
		propagate();
		cur_err <<= test_tar.GetColumns(i) - outp.out();
		val_err += cur_err.norm2();
	}
	// DEBUG
	//cout << "check_early_stop: last test set err = " << xvalid_state_.perfMean << "; current err = " << val_err << endl;

	//check stop conditions with patience
	xvalid_state_.perf = val_err;
	xvalid_state_.cycle = state.cycle;
	check_patience(xvalid_state_, 0.001, 5, stop_test_validation);
	//cout << "check_early_stop: patience_counter = " << xvalid_state_.patience_counter << endl;
	if(xvalid_state_.status == stop_test_validation)
		state.status = xvalid_state_.status;
	return xvalid_state_.patience_counter;

	//simple check
//	if(state.cycle == 1)
//		xvalid_state_.perfMean = val_err;
//	else {
//		double delta;
//		anti_grad::assign(delta, val_err - xvalid_state_.perfMean);
//		if(delta < 0) state.status = stop_test_validation;
//		xvalid_state_.perfMean = val_err;
//	}
}

Matrix objnet::sim(const Matrix& inp)
{
	Matrix mOutp;
	if(inp.row_num() == inp_size()) {
		mOutp.NewMatrix(layers_[layers_num() - 1].size(), inp.col_num());
		for(ulong i=0; i<inp.col_num(); ++i) {
			set_input(inp.GetColumns(i));
			propagate();
			mOutp.SetColumns(layers_[layers_num() - 1].axons_, i);
		}
	}
	return mOutp;
}

text_table objnet::detailed_info(int level) const {
	text_table tt;
	tt.fmt().sep_cols = true;
	tt.fmt().align = 2;

	// level 0 - brief info about NN layers num
	if(level == 0) {
		// show type
		tt << tt_begh() << "- 30 - 0 -" << tt_endrh();
		tt << "Network type: &" << decode_nn_type(nn_type()) << tt_endr();
		tt << "Layers number: &" << layers_num() << tt_endrh();
	}
	else if(level > 0) {
		// deep level of information about network
		// display table with detailed information about layers
		tt << tt_begh() << "| 0 | 0 | 0 | 0 |" << tt_endrh();
		tt << "Layer # & Layer type & Neurons num & Neuron types" << tt_endrh();
		TMatrix< string > neur_info;
		for(ulong i = 0; i < layers_.size(); ++i) {
			// decode layer's info about neuron types
			neur_info = decode_neuron_type(layers_[i].aft(), level > 2 ? false : true).content();
			for(ulong j = 0; j < neur_info.row_num(); ++j) {
				if(j == 0)
					tt << i;
				tt << "&" << decode_layer_type(layers_[i].layer_type()) << "&" << neur_info(j, 0) << "&" << neur_info(j, 1) << tt_endr();
			}
			tt << "\\hline" << tt_endr();
			//for(ulong j = 0; j < layers_[i].size(); ++j)
		}
	}
	return tt;
}

std::string objnet::status_info(int level) const {
	ostringstream os;
	// common status
	os << "cycle " << state_.cycle << ", error " << state_.perf << ", goal " << opt_.goal;
	// print cross-validation info
	if(opt_.goal_checkFun & test_validation)
		os << "; " << xvalid_info();
	os << endl;
	if(state_.status != learning)
		os << decode_nn_state(state_.status) << endl;
	return os.str();
}

std::string objnet::xvalid_info(int /*  level 1 */) const {
	ostringstream os;
	//if(opt_.goal_checkFun & test_validation) {
		os << "xvalidation: last err = " << xvalid_state_.perfMean << ", current err = " << xvalid_state_.perf;
		os << ", patience counter = " << xvalid_state_.patience_counter;
	//}
	return os.str();
}

}	//end of namespace NN

