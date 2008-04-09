#include "objnet.h"
#include "m_algorithm.h"

#include <time.h>
#include <sstream>
//#include <iosfwd>
#include <iostream>

//using namespace NN;
namespace NN {

using namespace std;
using namespace hybrid_adapt;

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

template<class layer_type>
layer_type& objnet::add_layer(ulong neurons_count, int af_type, ulong where_ind)
{
	//check if no input size specified
	if(layers_num() == 0 && inp_size() == 0) {
		_print_err(nn_except::explain_error(NoInputSize));
		throw nn_except(NoInputSize);
	}
	smart_ptr<layer> p_l(new layer_type(*this, neurons_count, af_type));
	if(where_ind < layers_.size())
		layers_.insert(p_l, where_ind, false);
	else
		layers_.push_back(p_l, false);
	return (layer_type&)*p_l;
}

/*
template<>
layer& objnet::add_layer<falman_layer>(ulong neurons_count, int af_type, ulong where_ind)
{
	//check if no input size specified
	if(layers_num() == 0 && inp_size() == 0) {
		_print_err(nn_except::explain_error(NoInputSize));
		throw nn_except(NoInputSize);
	}
	falman_layer l(*this, neurons_count);
	if(where_ind < layers_.size()) {
		layers_.insert(l, where_ind, false);
		return layers_[where_ind];
	}
	else {
		layers_.push_back(l, false);
		return layers_[layers_.size() - 1];
	}
}
*/

template<class layer_type>
layer_type& objnet::add_layer(ulong neurons_count, const iMatrix& af_mat, ulong where_ind)
{
	//check if no input size specified
	if(layers_num() == 0 && inp_size() == 0) {
		_print_err(nn_except::explain_error(NoInputSize));
		throw nn_except(NoInputSize);
	}
	smart_ptr<layer> p_l(new layer_type(*this, af_mat));
	if(where_ind < layers_.size())
		layers_.insert(p_l, where_ind, false);
	else
		layers_.push_back(p_l, false);
	return (layer_type&)*p_l;
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

		if(!opt_.batch) _update_epoch();
	}

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

	if(palsy) state_.status = stop_palsy;
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

		//first do standard gradient calculation for preceding layers
		calc_grad(targets.GetColumns(i));
	}

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
		++p_n; ++p_G;
	}

	//debug lsq_patience
	//state_.perf = 9.1;


	//estimate SSE
	//state_.perf = 0;
	//for(ulong i = 0; i < inputs.col_num(); ++i) {
	//	set_input(inputs.GetColumns(i));
	//	propagate();
	//	ol->Goal_ = targets.GetColumns(i) - o_axons;
	//	state_.perf += ol->Goal_.Mul(ol->Goal_).Sum();
	//}

	//if error is high - try backprop
	//double save = opt_.patience;
	//opt_.patience = rbl_patience_;
	/*
	while(state_.status != learned) {
		common_learn(inputs, targets, false, pProc);
		break;
		//if(state_.status == learned) break;
		//add new neuron
		neuron& n = gl->add_neuron(gl->gft_, create_ptr_mat(input_.neurons()));

		//generate(n.weights_.begin(), n.weights_.end(), prg::rand01);
		//n.weights_ -= 0.5; n.weights_ *= 2*opt_.wiRange;
		//collect errors
		Matrix err(inputs.col_num(), 1), er_col;
		for(ulong i = 0; i < inputs.col_num(); ++i) {
			set_input(inputs.GetColumns(i));
			propagate();
			er_col <<= targets.GetColumns(i) - o_axons;
			err[i] = er_col.Mul(er_col).Sum();
		}
		ulong worst = err.ElementInd(err.Max());
		n.weights_ = inputs.GetColumns(worst);

		if(gl->gft_ == radbas)
			gl->B_[gl->size() - 1] = 0;
		else
			gl->B_[gl->size() - 1] = 1;
	}
	//opt_.patience = save;

	ol->backprop_lg_ = true;
	*/

	/*
	if(state_.perf > opt_.epsilon) {
		//dump weights
		layer* l = &layers_[0];
		Matrix weights;
		for(n_iterator p_n = l->neurons().begin(); p_n != l->neurons().end(); ++p_n)
			weights |= p_n->weights_;
		DumpMatrix(weights, "weights.txt");
		DumpMatrix(l->B_, "biases.txt");

		o_axons <<= l->out();
		for(ulong i = 0; i < inputs.col_num(); ++i) {
			set_input(inputs.GetColumns(i));
			l->propagate();
			G.SetRows(!o_axons, i);
		}
		DumpMatrix(G, "G.txt", 50);

		//DumpMatrix(G, "G.txt", 50);
		Matrix U, V, E;
		svd(G, U, E, V);
		DumpMatrix(U, "U.txt", 50);
		DumpMatrix(V, "V.txt", 50);
		DumpMatrix(E, "E.txt", 50);

		Matrix::r_iterator p_e(E.begin());
		for(ulong i = 0; i < E.row_num(); ++i) {
			if(abs(*p_e) > 0.0000000001) *p_e = 1 / *p_e;
			p_e += E.col_num() + 1;
		}
		DumpMatrix(E, "invE.txt", 50);
		U <<= !U;
		DumpMatrix(U, "Ut.txt", 50);
		DumpMatrix(piG, "piG.txt", 50);

		throw nn_except("Big error!");
	}

	return state_.status;
	*/
}

void objnet::_update_epoch()
{
	if(state_.status == learning) {
		for(l_iterator p_l = layers_.begin(); p_l != layers_.end(); ++p_l)
			p_l->update_epoch();
			//(p_l->*_pUpdateFun)(true);
	}
}

int objnet::common_learn(const Matrix& inputs, const Matrix& targets, bool initialize, pLearnInformer pProc)
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
		//init weights if needed
		if(initialize) init_weights(inputs);
		//if using early stopping - do initialization
		if(opt_.goal_checkFun == test_validation)
			check_early_stop(inputs, targets);

		//main learn cycle
		while(state_.status == learning)
		{
			if(state_.cycle > 0) state_.lastPerf = state_.perf;

			//epoch main data processing function
			learn_epoch(inputs, targets);

			//calc final performance
			if(opt_.perfFun == mse)
				state_.perf /= inputs.col_num();

			//update cycles counter
			++state_.cycle;

			//check if goal is reached and other stopping criteria is met
			is_goal_reached();

			//make additional checks
			if(state_.status == learning) {
				switch(opt_.goal_checkFun) {
					case patience:
						check_patience(state_, opt_.patience, opt_.patience_cycles);
						break;
					case test_validation:
						check_early_stop(inputs, targets);
						break;
				}
			}

			//update weights
			update_epoch();

			//if(++state_.cycle == opt_.maxCycles && state_.status == learning)
			//	state_.status = stop_maxcycle;

			//call informer
			if(opt_.showPeriod != 0 && (state_.cycle % opt_.showPeriod == 1 || state_.status != learning) &&
				pProc && !pProc(state_.cycle, state_.perf, (void*)this))
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

void objnet::check_early_stop(const Matrix& inputs, const Matrix& targets)
{
	static Matrix test_set;
	static Matrix test_tar;

	if(state_.cycle == 0) {
		//initialization phase - extract validation set from learning
		//remove constness from learning set
		Matrix& inp_uc = const_cast< Matrix& >(inputs);
		Matrix& tar_uc = const_cast< Matrix& >(targets);
		//extract validation set
		ulong val_size = ha_round(inputs.col_num()*opt_.validation_fract);
		test_set.Resize(val_size, inputs.row_num());
		test_tar.Resize(val_size, targets.row_num());
		ulong val_ind;
		for(ulong i = 0; i < val_size; ++i) {
			val_ind = prg::randIntUB(inputs.col_num());
			test_set.SetColumns(inputs.GetColumns(val_ind), i);
			test_set.DelColumns(val_ind);
			test_tar.SetColumns(test_tar.GetColumns(val_ind), i);
			test_tar.DelColumns(val_ind);
		}
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
	//check stop conditions with patience
	state_.perf = val_err;
	check_patience(state_, opt_.patience, opt_.patience_cycles, stop_test_validation);

	//if(state_.cycle == 1) {
	//	state_.patience_counter = 0;
	//	state_.perfMean = val_err;
	//}
	//else if(state_.perfMean < val_err) {
	//	state_.patience_counter = 0;
	//	state_.perfMean = state_.perf;
	//}
	//else if(++state_.patience_counter == opt_.patience_cycles)
	//	state_.status = stop_test_validation;
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

}	//end of namespace NN
