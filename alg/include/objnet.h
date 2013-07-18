#ifndef _OBJNET_H
#define _OBJNET_H

#include "nn_common.h"
#include "kmeans.h"
#include "determ_annealing.h"
#include "text_table.h"

namespace NN {
	class _CLASS_DECLSPEC neuron_base
	{
	protected:
		//constructors - prevent creating neuron_base objects
		neuron_base();
		neuron_base(const neuron_base& n);

	public:
		//virtual destructor
		~neuron_base();
	};

	class _CLASS_DECLSPEC neuron
	{
		friend class layer;
		friend class layer_jump;
		friend class falman_layer;
		friend class rb_layer;
		friend class objnet;
		friend class ccn;
		friend class pcan;
		friend class rbn;

	public:
		//typedef TMatrix<neuron> neurMatrix;
		typedef TMatrix<neuron, val_ptr_buffer> neurMatrixPtr;
		//typedef neurMatrix::r_iterator n_iterator;
		typedef neurMatrixPtr::r_iterator np_iterator;

	private:
		//MatrixPtr inputs_;
		neurMatrixPtr inputs_;

		Matrix weights_, deltas_;
		Matrix grad_, prevg_;
		//additional info for learning
		//Matrix _a;

		double bias_, error_;
		//+1 signal for bias input
		//const static double s_bias1;
		//int _af_type;
		//int _upd_rule;

		void (neuron::*state_fcn)();
		void (neuron::*grad_fcn)(double);
		void (neuron::*act_fcn)(const new_nnOptions&);
		void (neuron::*deriv_fcn)(const new_nnOptions&);

		//state calculation functions
		void weighted_sum_sf();
		void eucl_dist_sf();

		//gradient calculation functions - if backprop_er = true - backpropagate local grad
		//for weighted-sum state function
		void grad_ws(double mult);
		void grad_ws_blg(double mult);
		void grad_ed(double mult);
		void grad_ed_blg(double mult);
		//template<bool backprop_er>
		//void calc_grad_ws(double mult);
		//for euclidian distance (radial) state function
		//template<bool backprop_er>
		//void calc_grad_ed(double mult);

		//activation functions
		void act_logsig(const new_nnOptions& opt);
		void act_tansig(const new_nnOptions& opt);
		void act_gauss(const new_nnOptions& opt);
		void act_revgauss(const new_nnOptions& opt);
		void act_purelin(const new_nnOptions& opt);
		void act_poslin(const new_nnOptions& opt);
		void act_expws(const new_nnOptions& opt);
		void act_multiquad(const new_nnOptions& opt);
		void act_revmultiquad(const new_nnOptions& opt);

		//activation function derivatives
		void d_logsig(const new_nnOptions& opt);
		void d_tansig(const new_nnOptions& opt);
		void d_gauss(const new_nnOptions& opt);
		void d_revgauss(const new_nnOptions& opt);
		void d_purelin(const new_nnOptions& opt);
		void d_poslin(const new_nnOptions& opt);
		void d_expws(const new_nnOptions& opt);
		void d_multiquad(const new_nnOptions& opt);
		void d_revmultiquad(const new_nnOptions& opt);

	public:
		double state_;
		double axon_;

		//API here
		neuron(ulong inp_count = 0, int aft = tansig, bool backprop_lg = true);
		neuron(const neurMatrixPtr& synapses, int aft = tansig, bool backprop_lg = true);
		//copy constructor - reference semantic
		neuron(const neuron& n);
		~neuron() {};

		//copy semantics
		neuron& operator =(const neuron& n);

		void init(ulong inp_count, int aft, bool backprop_lg);

		//void inline calc_grad_ws(bool backprop_er = true, double mult = 1);
		//void inline calc_grad_ed(bool backprop_er = true, double mult = 1);
		void calc_state();
		void calc_grad(double mult);
		void activate(const new_nnOptions& opt);
		void calc_deriv(const new_nnOptions& opt);

		void add_synapse(neuron *const p_n);
		void rem_synapse(neuron *const p_n);
		void set_synapses(const neurMatrixPtr& synapses);
		void add_synapses(const neurMatrixPtr& synapses, const Matrix* p_weights = NULL);
		void rem_synapses(const neurMatrixPtr& synapses);
	};

	//typedef neuron::neurMatrix neurMatrix;
	typedef neuron::neurMatrixPtr neurMatrixPtr;
	//typedef neuron::n_iterator n_iterator;
	typedef neuron::np_iterator np_iterator;
	typedef TMatrix<neuron, val_sp_buffer> neurMatrixSP;
	typedef neurMatrixSP::r_iterator n_iterator;
	typedef neurMatrixSP::cr_iterator cn_iterator;

//------------------------------------layer classes declaration-------------------------------------------------
	class objnet;

	class _CLASS_DECLSPEC layer
	{
		friend class objnet;
		friend class pcan;
		friend class rbn;
		friend class ccn;

		class layer_impl;

	protected:

		neurMatrixSP neurons_;
		iMatrix aft_;
		MatrixPtr axons_, states_;

		//biases support
		Matrix B_, BD_, BG_, OBG_;
		//additional info for learning
		//Matrix _A;

		//const new_nnOptions& opt_;
		objnet& net_;

		void (layer::*_pUpdateFun)(bool);
		void (layer::*_pRP_alg)(Matrix&, Matrix&, Matrix&, Matrix&);
		void (layer::*_pQP_alg)(Matrix&, Matrix&, Matrix&, Matrix&);

		Matrix active_af_region();
		//virtual void activate();
		virtual void deriv_af();

		void init_weights_random();
		void init_weights_nw();

		template<bool uniformly>
		void init_weights_radbas(const Matrix& inputs);

		void _construct_layer(ulong neurons_count);
		void _construct_axons();
		//void _construct_grad(bool prev_also = false);

		template<class goal_action>
		void _prepare2learn();
		//default is to minimize error - go in anti-grad direction
		virtual void prepare2learn() {
			_prepare2learn< anti_grad >();
		}

		virtual bool calc_grad();

		template<class goal_action>
		bool _bp_gradless_update(bool backprop_er = true);

		template<class goal_action>
		void _rp_original(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights);

		template<class goal_action>
		void _rp_simple(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights);

		template<class goal_action>
		void _rbp_update(bool zero_grad = true);

		template<class goal_action>
		void _rp_plus(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights);

		template<class goal_action>
		void _rp_plus_update(bool zero_grad = true);

		template<class goal_action>
		void _bp_update(bool zero_grad = true);

		template<class goal_action>
		void _qp_original(Matrix& W, Matrix& D, Matrix& G, Matrix& OG);

		template<class goal_action>
		void _qp_simple(Matrix& W, Matrix& D, Matrix& G, Matrix& OG);

		template<class goal_action>
		void _qp_modified(Matrix& W, Matrix& D, Matrix& G, Matrix& OG);

		template<class goal_action>
		void _qp_update(bool zero_grad = true);

		//don't really update weights - for lsq
		void empty_update(bool zero_grad = true);

		virtual void update_epoch(bool zero_grad = true) {
			(this->*_pUpdateFun)(zero_grad);
		}

	public:
		MatrixPtr Goal_;
		bool backprop_lg_;

		layer(objnet& net, ulong neurons_count = 0, int af_type = logsig);
		layer(objnet& net, const iMatrix& act_fun);
		//copy constructor
		layer(const layer& l);
		virtual ~layer() {};

		//copy semantics
		layer& operator =(const layer& l);

		//layer's output
		const MatrixPtr& out() const {
			return axons_;
		}
		//neurons access
		neurMatrixSP& neurons() {
			return neurons_;
		}
		//act fun access
		const iMatrix& aft() const {
			return aft_;
		}

		ulong size() const {
			return neurons_.size();
		}

		void set_af(int af_type) {
			aft_ = af_type;
		}

		void init(ulong neurons_count, int af_type = logsig, ulong inp_count = 0, const iMatrix* p_af_mat = NULL);
		void set_links(const neurMatrixPtr& inputs);
		void set_links(const neurMatrixPtr& inputs, const bitMatrix& con_mat);
		void add_links(const neurMatrixPtr& inputs, const Matrix* p_weights = NULL, ulong neur_ind = -1);
		void rem_links(const neurMatrixPtr& inputs, ulong neur_ind = -1);

		neuron& add_neuron(int af_type, ulong inp_num = 0);
		neuron& add_neuron(int af_type, const neurMatrixPtr& inputs);
		void rem_neuron(ulong ind);

		virtual void propagate();
		virtual void init_weights(const Matrix& inputs);

		// get layer type
		virtual layer_types layer_type() const {
			return common_nnl;
		}
	};

	typedef smart_ptr< layer > sp_layer;

	// layer with jump AF
	class layer_jump : public layer {
		class layer_jump_impl;

		bool calc_grad();
		//void deriv_af();
		void update_epoch();
		void prepare2learn();

	public:
		layer_jump(objnet& net, ulong neurons_count = 0)
			: layer(net, neurons_count, jump)
		{}

		//void propagate();
		void init_weights(const Matrix& inputs);

		// get layer type
		layer_types layer_type() const {
			return jump_nnl;
		}
	};

	//typedef TMatrix<layer> layerMatrix;
	//typedef layerMatrix::r_iterator l_iterator;
	typedef TMatrix<layer, val_sp_buffer> layerMatrixSP;
	typedef layerMatrixSP::r_iterator l_iterator;
	typedef layerMatrixSP::cr_iterator cl_iterator;

//-------------------------------------objnet class declaration-------------------------------------------------
	class _CLASS_DECLSPEC objnet
	{
		friend class layer;
		friend class rb_layer;

		std::ofstream errFile_;

	protected:
		layerMatrixSP layers_;
		layer input_;
		MatrixPtr output_;
		nnState state_;
		nnState xvalid_state_;

		virtual void set_def_opt(bool create_defs = true) {
			opt_.set_def_opt(create_defs);
			state_.status = not_learned;
		}
		inline void _print_err(const char* pErr);
		//void _construct_biases();
		//void _construct_grad(bool prev_also = false);

		// default prepare is to minimize error - go in anti-grad direction
		virtual void prepare2learn();
		
		virtual bool calc_grad(const Matrix& target);
		// this function will be called for each pattern after gradient calculation
		virtual void bp_after_grad() {};

		// standart back propagation learning epoch
		void bp_epoch(const Matrix& inputs, const Matrix& targets);
		// do some needed in batch mode grad correction
		// called inside bp_epoch
		void bp_batch_correct_grad(const Matrix& inputs, const Matrix& targets);
		// least-squares learn function for linear neurons in last layer
		void lsq_epoch(const Matrix& inputs, const Matrix& targets);

		// default learning epoch is bp_epoch
		virtual void learn_epoch(const Matrix& inputs, const Matrix& targets) {
			bp_epoch(inputs, targets);
		}

		//standart back propagation update after learn
		void std_update_epoch();
		//default update is bp_update_epoch
		virtual void update_epoch() {
			if(opt_.batch) std_update_epoch();
		}

		//checks whether learning is successful or not
		void is_goal_minimized() {
			if(state_.perf < opt_.goal)
				state_.status = learned;
			else if(state_.cycle == opt_.maxCycles)
				state_.status = stop_maxcycle;
		}

		virtual void is_goal_reached() {
			is_goal_minimized();
		}

		//checks if no improvement in goal is happening for a long time
		template< class goal_action >
		static int _check_patience(nnState& state, double patience, ulong patience_cycles,
			int patience_status = stop_patience);
		//default case if we are minimizing error
		virtual int check_patience(nnState& state, double patience, ulong patience_cycles,
			int patience_status = stop_patience);

		//checks whether error on test samples rises
		ulong check_early_stop(nnState& state, const Matrix& inputs, const Matrix& targets);

		//common learn function
		int common_learn(const Matrix& inputs, const Matrix& targets, bool initialize = true, pLearnInformer pProc = NULL,
				smart_ptr< const Matrix > test_inp = NULL, smart_ptr< const Matrix > test_tar = NULL);

		smart_ptr< nn_opt > opt_holder_;
		//constructor for overriding options from derived classes
		objnet(nn_opt* opt);

	public:
		nn_opt& opt_;

		objnet();
		virtual ~objnet() {};
		// operator = for making copies
		objnet& operator=(const objnet& rhs);

		//access to input layer
		layer& get_input() {
			return input_;
		}
		ulong inp_size() const {
			return input_.size();
		}
		ulong layers_num() const {
			return layers_.size();
		}
		//access to specific layer
		layer& get_layer(ulong layer_ind) {
			return layers_[std::min<ulong>(layer_ind, layers_.size() - 1)];
		}

		layer& output() {
			return get_layer(layers_num() - 1);
		}

		nnState state() const {
			return state_;
		}

		void set_input_size(ulong inp_size, bitMatrix *const pConMat = NULL);
		void set_input(const Matrix& input);

		//template< class layer_type >
		sp_layer add_layer(ulong neurons_count, int af_type = logsig, int layer_type = bp_nnl, ulong where_ind = -1);
		//template< class layer_type >
		sp_layer add_layer(ulong neurons_count, const iMatrix& af_mat, int layer_type = bp_nnl, ulong where_ind = -1);
		
		virtual void propagate();
		virtual void init_weights(const Matrix& inputs);

		// function to prepare proper learning & validation sets
		void prep_learn_valid_sets(const Matrix& input, const Matrix& targets, smart_ptr< const Matrix >& test_inp,
				smart_ptr< const Matrix >& test_tar, Matrix& real_input, Matrix& real_targets);

		virtual int learn(const Matrix& input, const Matrix& targets, bool initialize = true, pLearnInformer pProc = NULL,
				smart_ptr< const Matrix > test_inp = NULL, smart_ptr< const Matrix > test_tar = 0) = 0;

		// return NN type
		virtual nn_types nn_type() const = 0;

		// return NN detailed info (structure details)
		virtual text_table detailed_info(int level = 1) const;
		// return current network status desccription
		virtual std::string status_info(int level = 1) const;
		// return cross-validation status info
		virtual std::string xvalid_info(int level = 1) const;

		Matrix sim(const Matrix& inp);
	};

//---------------------------------Multi Layer Perceptron declaration--------------------------------------------

	//-----------------------------Standart back propagation layer declaration-----------------------------------
	class _CLASS_DECLSPEC bp_layer : public layer
	{
	public:
		bp_layer(objnet& net, ulong neurons_count = 0, int af_type = logsig)
			: layer(net, neurons_count, af_type)
		{}

		bp_layer(objnet& net, const iMatrix& act_fun)
			: layer(net, act_fun)
		{}

		//copy constructor
		bp_layer(const layer& l)
			: layer(l)
		{}

		bp_layer& operator=(const layer& rhs) {
			layer::operator=(rhs);
			return *this;
		}
	};

	class _CLASS_DECLSPEC mlp : public objnet
	{
	public:
		mlp() { set_def_opt(false); }
		mlp(ulong layers_num, ulong inp_size, ...);
		~mlp() {};

		bp_layer& add_layer(ulong neurons_count, int af_type = logsig);
		sp_layer add_layer_ex(ulong neurons_count, int af_type, int layer_type);

		bool set_layer(ulong layer_ind, ulong neurons_count, int af_type);
		bool set_layer_type(ulong layer_ind, int af_type);
		int learn(const Matrix& input, const Matrix& targets, bool initialize = true, pLearnInformer pProc = NULL, 
				smart_ptr< const Matrix > test_inp = NULL, smart_ptr< const Matrix > test_tar = NULL)
		{
			return common_learn(input, targets, initialize, pProc, test_inp, test_tar);
		}

		nn_types nn_type() const { return mlp_nn; }
	};

//--------------------------------Falman cascade correlation network---------------------------------------------

	//----------------------------Falman layer declaration-------------------------------------------------------
	class _CLASS_DECLSPEC falman_layer : public layer
	{
		friend class ccn;

		enum {
			no_cache = 0,
			collecting = 1,
			use_cache = 2
		};

		struct cache_prop;
		smart_ptr< cache_prop > cp_;

		//best performing neurons indexes
		ulMatrix winner_ind_;

		void _construct_aft();

		//cache mode control
		void cache_mode_on();
		void cache_mode_off();
		//void cache_mode_pause();
		//void cache_mode_unpause();

		//update rules specifications
		//void prepare2learn() {
		//	_prepare2learn<follow_grad>();
		//}

	public:
		falman_layer(objnet& net, ulong candidates_count = 5);
		falman_layer(objnet& net, const iMatrix& act_fun);
		falman_layer(const layer& l);

		//copy constructor
		falman_layer(const falman_layer& l);

		const iMatrix& aft() {
			return layer::aft();
		}

		void propagate();

		inline void delete_losers(ulong survivals);

		ulong get_winner_ind() {
			return winner_ind_.max_ind();
		}

		layer_types layer_type() const {
			return falman_nnl;
		}
	};

	//--------------------------------------CCN declaration----------------------------------------------------
	class _CLASS_DECLSPEC ccn : public objnet {
		class ccn_impl;

	public:
		ccn_opt& opt_;

		ccn();
		~ccn() {};

		//void set_def_opt(bool create_defs = true);
		//number of falman layers
		ulong flayers_num() const {
			return flayers_.size();
		}
		//access to specific falman layer
		falman_layer& get_flayer(ulong layer_ind) {
			return flayers_[std::min<ulong>(layer_ind, flayers_.size() - 1)];
		}
		//current learning falman layer
		falman_layer* get_cur_flayer() {
			return cur_fl_;
		}

		//add falman layer with radial basis neurons based on pre-clustering
		void add_rb_layer(const Matrix& inputs, const Matrix& targets, const Matrix& centers, double stock_mult);

		nnState get_mainState() const {
			return mainState_;
		}
		//modified propagation function
		void propagate();
		//construct output layer
		void set_output_layer(ulong neurons_count, int af_type = logsig, int layer_type = bp_nnl);
		//resets network t only one output layer
		void reset();
		//complex learning function - implies network constructing
		int learn(const Matrix& inputs, const Matrix& targets, bool initialize = true, pLearnInformer pProc = NULL,
				smart_ptr< const Matrix > test_inp = NULL, smart_ptr< const Matrix > test_tar = NULL);

		// nn_type
		nn_types nn_type() const { return cc_nn; }

		// NN information function
		text_table detailed_info(int layer = 1) const;
		std::string status_info(int level = 1) const;

	private:
		typedef TMatrix<falman_layer, val_sp_buffer> flMatrix;
		typedef flMatrix::r_iterator fl_iterator;

		class ccn_impl;

		flMatrix flayers_;
		falman_layer* cur_fl_;
		nnState mainState_;
		//flag-indicator that first falman layer is pre-created from rb_layer before learning
		bool rbfl_;

		//void (ccn::*_pLearnEpoch)(const Matrix&, const Matrix&);
		//void (ccn::*_pGoalFun)();
		//void (ccn::*_pUpdateEpochFun)();

		falman_layer& add_falman_layer(ulong candidates_count = MIN_CAND_COUNT);
		//void learn_layer(falman_layer& l, Matrix& inputs, Matrix& targets);

		//gradient calculation for falman layer
		void falman_epoch(const Matrix& inputs, const Matrix& targets);
		//backprop falman layer learning
		void bp_after_grad();

		//override learn epoch
		void learn_epoch(const Matrix& inputs, const Matrix& targets);

		//custom learn stop function
		void is_goal_reached();

		//custom prepare function
		void prepare2learn();
		//custom patience check function
		int check_patience(nnState& state, double patience, ulong patience_cycles,
			int patience_status = stop_patience);

		void update_epoch();

		bool delete_losers(const Matrix& inputs, const Matrix& targets);

		void _init_fl_radbas(const Matrix& inputs);
		void _calc_winner_maxw();
		void _calc_winner_corr(const Matrix& inputs, const Matrix& targets);

		//bool process_option(std::istream& inif, std::string& word);
	};

	//----------------------------------PCA network-----------------------------------------------------------
	class _CLASS_DECLSPEC pcan : public objnet
	{
		//void set_def_opt();
		void prepare2learn();
		void learn_epoch(const Matrix& inputs, const Matrix& targets);

		//empty update epoch
		void update_epoch() {};

		void gha_step();

	public:
		//constructors
		pcan() {
			opt_.nu = 0.0001;
			opt_.adaptive = true;
		};
		pcan(ulong input_size, ulong prin_comp_num = 0);

		void set_output_layer(ulong prin_comp_num = 0);
		//learn function
		int learn(const Matrix& inputs, const Matrix& targets, bool initialize = true, pLearnInformer pProc = NULL,
				smart_ptr< const Matrix > test_inp = NULL, smart_ptr< const Matrix > test_tar = NULL);

		// nn_type
		nn_types nn_type() const { return pca_nn; }
	};

	//--------------------------------------RBF network----------------------------------------------------
	class _CLASS_DECLSPEC rb_layer : public layer
	{
		friend class rbn;

		//Green function type
		int gft_;

		void calc_isotropic_sigma();

		template< class clusterizer >
		void calc_varbased_sigma(const clusterizer& cengine);

	public:
		KM::kmeans km_;

		rb_layer(objnet& net, int gft = DEF_GFT)
			:layer(net), gft_(gft)
		{}

		rb_layer(objnet& net, ulong neurons_count, int gft = DEF_GFT)
			:layer(net, neurons_count, gft), gft_(gft)
		{}

		//copy constructor
		rb_layer(const rb_layer& l)
			:layer(l), gft_(l.gft_)
		{}

		//exact rbf layer construction - neurons num equal to learning samples num
		void construct_exact(const Matrix& inputs);
		//random rbf layer construction - varying neurons num
		void construct_random(const Matrix& inputs, double rate);
		//construct layer with k-means clustering algorithm
		void construct_kmeans(const Matrix& inputs, const Matrix& targets, double rate, const Matrix* pCent = NULL);
		void construct_kmeans_p(const Matrix& inputs, const Matrix& targets, double rate, const Matrix* pCent = NULL);
		//construct layer fully based on weights given (drops method)
		void construct_drops(const Matrix& inputs, const Matrix& targets, const Matrix& centers, double stock_mult);

		void construct_da(const DA::determ_annealing& da, const Matrix& inputs, const Matrix& targets,
			const Matrix& centers);


		void init_weights(const Matrix& inputs);

		layer_types layer_type() const {
			return rb_nnl;
		}
	};

	class _CLASS_DECLSPEC rbn : public objnet
	{
		//bool process_option(std::istream& inif, std::string& word);

		rb_layer& add_rb_layer();
		void prepare2learn();
		void _neuron_adding_learn(const Matrix& inputs, const Matrix& targets, pLearnInformer pProc,
				smart_ptr< const Matrix > test_inp, smart_ptr< const Matrix > test_tar);

		//override std learning epoch
		void learn_epoch(const Matrix& inputs, const Matrix& targets);
		//override std update epoch
		void update_epoch();
		//custom goal check function
		void is_goal_reached();

	public:
		rbn_opt& opt_;

		//constructors
		rbn();

		void set_rb_layer(const Matrix& inputs, ulong neurons_count);
		void set_rb_layer_exact(const Matrix& inputs);
		void set_rb_layer_random(const Matrix& inputs, double rate = 0.8);
		void set_rb_layer_kmeans(const Matrix& inputs, const Matrix& targets, double rate = 0.8,
			const Matrix* pCent = NULL);
		void set_rb_layer_drops(const Matrix& inputs, const Matrix& targets, const Matrix& centers,
			double stock_mult = 1.2);

		void set_output_layer(ulong neurons_count);

		//learning function for rbf networks
		int learn(const Matrix& inputs, const Matrix& targets, bool initialize = true, pLearnInformer pProc = NULL,
				smart_ptr< const Matrix > test_inp = NULL, smart_ptr< const Matrix > test_tar = NULL);

		// nn_type
		nn_types nn_type() const { return rb_nn; }
	};
}

#endif	//_OBJNET_H

