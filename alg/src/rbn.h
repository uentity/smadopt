#include "objnet.h"

using namespace std;
using namespace NN;
using namespace KM;

//--------------------------------RBF network implementation---------------------------------------------------
//--------------------------------RBF layer------------------
void rb_layer::init_weights(const Matrix& inputs)
{
	return init_weights_radbas< true >(inputs);
	//return init_weights_random();
}

void rb_layer::calc_isotropic_sigma()
{
	if(gft_ != radbas && gft_ != revradbas) return;
	//procedure works correctly if neurons >= 2
	if(neurons_.size() < 2) return;
	//calc maximum distance between centers
	double max_dist = 0, cur_dist;
	Matrix diff;
	n_iterator p_n = neurons_.begin(), end = neurons_.end(), p_n1;
	for(; p_n != end; ++p_n) {
		for(p_n1 = p_n + 1; p_n1 != end; ++p_n1) {
			diff <<= p_n->weights_ - p_n1->weights_;
			if((cur_dist = diff.Mul(diff).Sum()) > max_dist)
				max_dist = cur_dist;
		}
	}
	//set all biases = sigma^-1
	B_ = sqrt((double)neurons_.size())/sqrt(max_dist);
}

void rb_layer::calc_varbased_sigma(const KM::kmeans& km)
{
	if(gft_ != radbas && gft_ != revradbas) return;
	//first calc isotropic sigma
	calc_isotropic_sigma();
	//assume that neurons with weights from kmeans are placed at the beginning
	double q;
	const kmeans::vvul aff = km.get_aff();
	const Matrix& norms = km.get_norms();
	for(ulong i = 0; i < aff.size(); ++i) {
		//calc variance-based bias
		if(aff.size() > 0) {
			q = 0;
			for(ulong j = 0; j < aff[i].size(); ++j)
				q += norms[aff[i][j]]*norms[aff[i][j]];
			q = sqrt((double)aff.size()/q);
		}
		else q = 1;
		B_[i] = q;
	}
}

void rb_layer::construct_exact(const Matrix& inputs)
{
	init(inputs.col_num(), gft_);
	//set weights
	n_iterator p_n = neurons_.begin();
	for(ulong i = 0; i < inputs.col_num(); ++i) {
		p_n->weights_ = inputs.GetColumns(i);
		++p_n;
	}
	//calc sigma
	calc_isotropic_sigma();
}

void rb_layer::construct_random(const Matrix& inputs, double rate)
{
	//calc neurons num
	ulong nn = ha_round(rate*inputs.col_num());
	init(nn, gft_);

	//calc random indices
	vector<ulong> ind(nn);
	for(ulong i = 0; i < nn; ++i)
		ind[i] = i;
	random_shuffle(ind.begin(), ind.end(), prg::randIntUB);

	//set weights
	n_iterator p_n = neurons_.begin();
	for(ulong i = 0; i < nn; ++i) {
		p_n->weights_ = inputs.GetColumns(ind[i]);
		++p_n;
	}
	//calc sigma
	calc_isotropic_sigma();
}

void rb_layer::construct_kmeans(const Matrix& inputs, const Matrix& targets, double rate, const Matrix* pCent)
{
	//calc neurons num
	ulong nn = ha_round(rate*inputs.col_num());

	//clusterisation
	km_.opt_.nu = 0.001;
	km_.opt_.emptyc_pol = ((rbn&)net_).opt_.rbnec_policy_;
	km_.find_clusters_f(!inputs, !targets, nn, 100, pCent, true);
	//km_.find_clusters(!inputs, nn, 100, false, pCent, true);
	//km_.opt_.use_prev_cent = true;

	//construct layer
	const Matrix& c = km_.get_centers();
	nn = c.row_num();
	init(nn, gft_);
	//set weights
	n_iterator p_n = neurons_.begin();
	for(ulong i = 0; i < nn; ++i) {
		p_n->weights_ = !c.GetRows(i);
		++p_n;
	}
	//calc sigma
	//calc_varbased_sigma(km_);
	calc_isotropic_sigma();
}

void rb_layer::construct_kmeans_p(const Matrix& inputs, const Matrix& targets, double rate, const Matrix* pCent)
{
	if(net_.layers_num() < 2)
		construct_kmeans(inputs, targets, rate, pCent);

	//calc neurons num
	ulong nn = static_cast<ulong>(ha_round(rate*inputs.col_num()));
	if(nn <= neurons_.size()) return;

	//clusterisation
	km_.opt_.nu = 0.001;
	km_.opt_.emptyc_pol = ((rbn&)net_).opt_.rbnec_policy_;
	km_.find_clusters_f(!inputs, !targets, nn, 100, pCent, true);
	//km_.find_clusters(!inputs, nn, 100, false, pCent, true);
	//km_.opt_.use_prev_cent = true;

	//construct layer
	Matrix c = !km_.get_centers();
	nn = c.col_num();
	//calc distances
	Matrix dist(1, c.col_num());
	Matrix cur_c, diff;
	for(ulong i = 0; i < c.col_num(); ++i) {
		cur_c <<= c.GetColumns(i);
		dist[i] = 0;
		for(n_iterator p_n = neurons_.begin(); p_n != neurons_.end(); ++p_n) {
			diff <<= p_n->weights_ - c;
			dist[i] += diff.norm2();
		}
	}
	//sort distances
	Matrix::indMatrix ind = dist.RawSort();
	ind.Reverse();
	nn -= neurons_.size();
	//neurMatrixPtr inp = create_ptr_mat(net_.input_.neurons());
	const Matrix& norms = km_.get_norms();
	double q;
	for(ulong i = 0; i < nn; ++i) {
		neuron& newn = add_neuron(gft_);
		newn.weights_ = c.GetColumns(ind[i]);
		//calc variance-based bias
		const ul_vec& aff = km_.get_aff()[ind[i]];
		if(aff.size() > 0) {
			q = 0;
			for(ulong j = 0; j < aff.size(); ++j)
				q += norms[aff[j]]*norms[aff[j]];
			q = sqrt((double)aff.size()/q);
		}
		else q = 1;
		B_[neurons_.size() - 1] = q;
	}

	//calc sigma
	//calc_varbased_sigma(km_.get_ind(), km_.get_norms());
	//calc_isotropic_sigma();
}

void rb_layer::construct_drops(const Matrix& inputs, const Matrix& targets, const Matrix& centers, double stock_mult)
{
	//ensure stock_mult >= 0
	stock_mult = max(stock_mult, 0.);
	//construct layer
	//const Matrix& c = km.get_centers();
	ulong nn = static_cast<ulong>(ceil(centers.row_num() * stock_mult));
	init(nn, gft_, inputs.row_num());
	//first init weights randomly
	init_weights_radbas<false>(inputs);
	//set weights from centers
	n_iterator p_n = neurons_.begin();
	for(ulong i = 0; i < centers.row_num(); ++i) {
		p_n->weights_ = !centers.GetRows(i);
		++p_n;
	}
	//calc sigma
	//calc_varbased_sigma(km);
	calc_isotropic_sigma();
}

//--------------------------------RBF network------------------
rbn::rbn() :
	objnet(new rbn_opt), opt_((rbn_opt&)*opt_holder_)
{
}

rb_layer& rbn::add_rb_layer()
{
	if(inp_size() == 0)
		throw nn_except(NoInputSize);
	if(layers_num() < 2) {
		layers_.insert(smart_ptr<layer>(new rb_layer(*this, opt_.gft_)), 0, false);
	}
	rb_layer& l = (rb_layer&)*layers_.begin();
	return l;
}

void rbn::set_rb_layer(const Matrix& inputs, ulong neurons_count)
{
	rb_layer& l = add_rb_layer();
	l.init(neurons_count, opt_.gft_);
	l.set_links(create_ptr_mat(input_.neurons()));
	l.init_weights(inputs);
	if(layers_num() > 1)
		layers_[1].set_links(create_ptr_mat(l.neurons_));

	opt_.learnType = backprop;
}

void rbn::set_rb_layer_exact(const Matrix& inputs)
{
	rb_layer& l = add_rb_layer();
	l.construct_exact(inputs);
	l.set_links(create_ptr_mat(input_.neurons()));
	if(layers_num() > 1)
		layers_[1].set_links(create_ptr_mat(l.neurons_));

	opt_.learnType = rbn_exact;
}

void rbn::set_rb_layer_random(const Matrix& inputs, double rate)
{
	rb_layer& l = add_rb_layer();
	l.construct_random(inputs, rate);
	l.set_links(create_ptr_mat(input_.neurons()));
	if(layers_num() > 1)
		layers_[1].set_links(create_ptr_mat(l.neurons_));

	opt_.learnType = rbn_random;
}

void rbn::set_rb_layer_kmeans(const Matrix& inputs, const Matrix& targets, double rate, const Matrix* pCent)
{
	rb_layer& l = add_rb_layer();
	//add kmeans in new layer to options chain
	opt_.add_embopt(l.km_.opt_);

	l.construct_kmeans(inputs, targets, rate, pCent);
	l.set_links(create_ptr_mat(input_.neurons()));
	if(layers_num() > 1)
		layers_[1].set_links(create_ptr_mat(l.neurons_));

	opt_.learnType = rbn_kmeans;
}

void rbn::set_rb_layer_drops(const Matrix& inputs, const Matrix& targets, const Matrix& centers, double stock_mult)
{
	rb_layer& l = add_rb_layer();
	//add kmeans in new layer to options chain
	opt_.add_embopt(l.km_.opt_);

	l.construct_drops(inputs, targets, centers, stock_mult);
	l.set_links(create_ptr_mat(input_.neurons()));
	if(layers_num() > 1)
		layers_[1].set_links(create_ptr_mat(l.neurons_));

	//opt_.learnType = rbn_kmeans;
	opt_.learnType = rbn_neuron_adding;
}

void rbn::set_output_layer(ulong neurons_count)
{
	layer& l = objnet::add_layer<bp_layer>(neurons_count, purelin);
	if(opt_.io_linked_)
		l.set_links(create_ptr_mat(input_.neurons()));
	if(layers_num() > 1)
		l.set_links(create_ptr_mat(layers_[layers_num() - 2].neurons()));
}

void rbn::prepare2learn() {
	objnet::prepare2learn();

	//these learning types always use lsq
	if(opt_.learnType == rbn_exact || opt_.learnType == rbn_random) {
		opt_.use_lsq = true;
		//disable local gradient backprop from last layer
		layers_[layers_num() - 1].backprop_lg_ = false;
	}

	if(opt_.use_lsq)
		//disable updates in last layer
		layers_[layers_num() - 1]._pUpdateFun = &layer::empty_update;

	if(opt_.learnType == rbn_exact || opt_.learnType == rbn_random) {
		if(opt_.learnFun == R_BP) {
			//weights to last layer probably very big after svd calculation
			//so make initial changes small
			layer& l = layers_[0];
			double mult = log(1.00001);
			for(n_iterator p_n = l.neurons().begin(); p_n != l.neurons().end(); ++p_n) {
				p_n->deltas_ *= mult/l.size();
			}
			l.BD_ *= mult/l.BD_.size();
		}
		//layers_[layers_num() - 1].backprop_lg_ = false;
	}
}

void rbn::_neuron_adding_learn(const Matrix& inputs, const Matrix& targets, pLearnInformer pProc)
{
	Matrix ps_er(1, inputs.col_num()), cur_er;
	Matrix::indMatrix er_ind;
	layer& outl = layers_[layers_.size() - 1];
	layer& rbl = layers_[0];
	ulong neurons_added = 0;
	ulong max_neurons = 0;
	if(opt_.neur_incr_mult_ > 0)
		max_neurons = ha_round(rbl.neurons().size() * opt_.neur_incr_mult_);
	neurMatrixPtr input_ptr = create_ptr_mat(input_.neurons());
	do {
		//start backprop learning
		common_learn(inputs, targets, false, pProc);

		//check if max neurons already added
		if(rbl.neurons().size() >= max_neurons) break;

		//check state
		if(state_.status != learned && state_.status != stop_palsy) {
			//if error is still high - add new neurons
			//make additional simulation first to determine per-sample errors
			for(ulong i = 0; i < inputs.col_num(); ++i) {
				set_input(inputs.GetColumns(i));
				propagate();
				//calc error
				cur_er <<= targets.GetColumns(i) - outl.out();
				cur_er *= cur_er;
				ps_er[i] = cur_er.Sum();
			}
			//sort errors
			er_ind <<= ps_er.RawSort();
			//add neurons in place of samples with highest errors
			ulong added_cnt = 0;
			double q = rbl.B_.Mean();
			for(ulong i = er_ind.size() - 1; i < er_ind.size(); --i) {
				neuron& n = rbl.add_neuron(opt_.gft_, input_ptr);
				n.weights_ = inputs.GetColumns(er_ind[i]);
				//random weights initialization
				//generate(n.weights_.begin(), n.weights_.end(), prg::rand01);
				//n.weights_ -= 0.5; n.weights_ *= opt_.wiRange;

				rbl.B_[rbl.B_.size() - 1] = q;
				if(++added_cnt >= opt_.neur_incr_step_) break;
			}
			neurons_added += added_cnt;
		}
	} while(state_.status != learned);
}

int rbn::learn(const Matrix& inputs, const Matrix& targets, bool initialize, pLearnInformer pProc)
{
	if(layers_num() < 1) throw nn_except("Radial basis layer must be initialized before learning can start");
	layer& ol = layers_[layers_num() - 1];
	switch(opt_.learnType) {
		case rbn_random:
		case rbn_exact:
			//lsq_learn(inputs, targets, pProc);
			common_learn(inputs, targets, false, pProc);
			break;
		case rbn_neuron_adding:
			layers_[layers_num() - 1].init_weights(inputs);
			//set output's layer weight = 1 - DEBUG!
			//for(n_iterator p_n = ol.neurons_.begin(); p_n != ol.neurons_.end(); ++p_n)
			//	p_n->weights_ = 0.01;
			_neuron_adding_learn(inputs, targets, pProc);
			break;
		case rbn_kmeans:
			//set output's layer weight = 1 - DEBUG!
			//for(n_iterator p_n = ol.neurons_.begin(); p_n != ol.neurons_.end(); ++p_n)
			//	p_n->weights_ = 1;
			//common_learn(inputs, targets, false, pProc);
			//break;
		case backprop:
		default:
			layers_[layers_num() - 1].init_weights(inputs);
			//_svd_learn(inputs, targets, pProc);
			common_learn(inputs, targets, false, pProc);
			break;
	}
	return state_.status;
}

void rbn::update_epoch()
{
	//do not update in case of lsq learning
	if(opt_.learnType != rbn_exact)
		objnet::update_epoch();
}

void rbn::learn_epoch(const Matrix& inputs, const Matrix& targets)
{
	if(opt_.use_lsq)
		objnet::lsq_epoch(inputs, targets);
	else
		objnet::learn_epoch(inputs, targets);
}

void rbn::is_goal_reached()
{
	if(opt_.learnType == rbn_exact)
		state_.status = learned;
	else
		objnet::is_goal_reached();
}
