#ifndef _NN_COMMON_H
#define _NN_COMMON_H

#ifdef _MSC_VER
#pragma warning(disable: 4251)
#endif

#include "common.h"
#include "matrix.h"
#include "alg_except.h"
#include "alg_opt.h"
#include "kmeans_common.h"

typedef bool (*pLearnInformer)(ulong uCycle, double dSSE, void* pNet);
typedef ulong (*pNewLAProc)(int nStatus, ulong uCycle, double dSSE, void* pNet);

namespace NN {

//	#define VERBOSE
	#define HORIZONTAL 1
	#define VERTICAL 0
	#define E_EXP 2.7182818284591
	#define MIN_CAND_COUNT 5
	#define DEF_GFT revradbas

	typedef Matrix::r_iterator r_iterator;
	typedef Matrix::cr_iterator cr_iterator;
	typedef MatrixPtr::r_iterator mp_iterator;

	//forward declarations
	class neuron; class layer;
	class bp_layer; class rb_layer;
	class falman_layer;
	class MNet; class objnet;
	class mlp; class pcan;
	class rbn; class ccn;

	enum nn_flags {
		useBiases = 1,
		useGradient = 2,
		useLateral = 4
	};

	enum NN_Type {
		MLP = useBiases,
		LinGHA = 0,
		LinAPEX = useLateral
	};

	enum StateFun {
		weighted_sum = 0,
		eucl_dist = 1
	};

	enum ActFun {
		logsig = 0,
		tansig = 1,
		radbas = 2,
		purelin = 3,
		poslin = 4,
		expws = 5,
		multiquad = 6,
		revradbas = 7,
		revmultiquad = 8
	};

	enum InitFun {
		if_random = 1,
		nw = 2
	};

	enum LearnFun {
		BP = 1,
		R_BP = 2,
		R_BP_PLUS = 3,
		QP = 4,
		hebbian = 5,
		GHA = 6,
		APEX = 7
	};

	enum LearnType {
		//standart for most networks
		backprop = 10,
		//radial basis networks learning
		rbn_exact = 11,
		rbn_random = 12,
		//rbn_fully_bp = 13,
		rbn_kmeans = 14,
		rbn_neuron_adding = 15,
		//ccn learning
		ccn_maxcor = 21,
		ccn_bp = 22,
		ccn_fully_bp = 23
	};

	enum GoalCheckFun {
		no_check = 1,
		patience = 2,
		test_validation = 4
	};

	enum PerformanceFun {
		sse = 1,
		mse = 2
	};

	enum NN_State {
		not_learned = 0,
		learned = 1,
		error = -1,
		learning = 2,
		stop_palsy = 3,
		stop_breaked = 4,
		stop_maxcycle = 5,
		stop_patience = 6,
		stop_test_validation = 7
	};

	enum NN_Error {
		InvalidLayer = 2,
		NoInputSize = 3,
		SizesMismatch = 4,
		NN_Busy = 5,
	};

	struct nnState {
		double nu, perf, lastPerf;
		int status;
		ulong cycle, patience_counter;
		double perfMean;
		Matrix validate_inp;
		Matrix validate_tar;
		std::string lastError;
	};

//-----------------------------------------------------------------------------------------------------------
	struct anti_grad {
		template<class Tx, class Ty>
		static void update(Tx& x, const Ty& delta) {
			x -= delta;
		}

		template<class Tx, class Ty>
		static void assign(Tx& x, const Ty& delta) {
			x = -delta;
		}

		//template<class Tx, class Ty>
		//static bool is_goal_reached(const Tx& cur_perf, const Ty& goal) {
		//	if(cur_perf <= goal) return true;
		//	else return false;
		//}
	};

	struct follow_grad {
		template<class Tx, class Ty>
		static void update(Tx& x, const Ty& delta) {
			x += delta;
		}

		template<class Tx, class Ty>
		static void assign(Tx& x, const Ty& delta) {
			x = delta;
		}

		//template<class Tx, class Ty>
		//static bool is_goal_reached(const Tx& cur_perf, const Ty& goal) {
		//	if(cur_perf >= goal) return true;
		//	else return false;
		//}
	};
//-----------------------------------------------------------------------------------------------------------

	class _CLASS_DECLSPEC nn_except : public alg_except
	{
		//static std::string s_buf;
	public:
		nn_except()
			: alg_except()
		{
		}
		nn_except(const alg_except& ex)
			: alg_except(ex)
		{
		}
		nn_except(int code, const char* what)
			: alg_except(code, what)
		{
		}
		nn_except(int code)
		{
			_code = code;
			_what = explain_error(code);
		}
		nn_except(const char* what)
			: alg_except(what)
		{
		}
		~nn_except() {};

		static const char* explain_error(int code);
	};

//-------------------------Options classes-----------------------------------------------------

	//options template
	template< class net > struct _CLASS_DECLSPEC nnOptions {
	private:
		nnOptions() {};
	};

	//old options
	template<>
	struct _CLASS_DECLSPEC nnOptions< MNet > {
		double nu, mu, goal, limit;
		bool batch, adaptive, saturate, normInp;
		ulong showPeriod, maxCycles;
		double noise, thresh01, thresh11;
		double wiRange;
		int initFun, perfFun, learnFun;
		double tansig_a, tansig_b, tansig_e, logsig_a, logsig_e;
		double rp_delt_inc, rp_delt_dec, rp_delta0, rp_deltamax;
		bool useSimpleRP;
	};

	//new options
	template<>
	struct _CLASS_DECLSPEC nnOptions< objnet > {
		int initFun, perfFun, learnFun, learnType, goal_checkFun;
		bool batch, adaptive, saturate, useSimpleRP, useSimpleQP;
		ulong showPeriod, maxCycles, patience_cycles;
		double nu, mu, goal;
		double epsilon;
		double wiRange;
		double tansig_a, tansig_b, tansig_e, logsig_a, logsig_e;
		double rp_delt_inc, rp_delt_dec, rp_delta0, rp_deltamax;
		double qp_lambda, qp_alfamax;
		double patience, validation_fract;
		bool use_lsq;
	};
	typedef nnOptions< objnet > new_nnOptions;

	//ccn options
	template<>
	struct _CLASS_DECLSPEC nnOptions< ccn > : nnOptions< objnet >
	{
		//additional options for ccn
		ulong maxFLLcycles_, fl_candidates_, maxFL_;
		bool insert_between_, grow_vlayer_;
	};

	//rbn options
	template<>
	struct _CLASS_DECLSPEC nnOptions< rbn > : nnOptions< objnet >
	{
		//additional options for rbn
		int rbnec_policy_;
		double rbl_patience_;
		bool io_linked_;
		//Green function type
		int gft_;
		double neur_incr_mult_;
		ulong neur_incr_step_;
	};

	/*
	template<class net>
	class _CLASS_DECLSPEC tnn_opt_base : public alg_options< nnOptions<net> >
	{
		bool process_option(std::istream& inif, std::string& word);

	public:
		typedef tnn_opt<net> this_nn_opt;

		tnn_opt() { set_def_opt(); }
		tnn_opt(const this_nn_opt& opt) : wrapper_opt(opt) {}

		const char* get_opt_type() const; //{ return opt_.get_opt_type(); }

		void set_def_opt(bool create_defs = true);
	};

	template<>
	const char* tnn_opt<MNet>::get_opt_type() const {
		return "mnn_opt";
	}
	typedef tnn_opt_base<MNet> mnn_opt;

	template<>
	class _CLASS_DECLSPEC tnn_opt<objnet> : public alg_options< nnOptions<objnet> >
	{
	protected:
		virtual bool process_option(std::istream& inif, std::string& word);

	public:
		typedef tnn_opt<objnet> this_nn_opt;

		tnn_opt() { set_def_opt(); }
		tnn_opt(const this_nn_opt& opt) : wrapper_opt(opt) {}

		const char* get_opt_type() const { return "nn_opt"; }

		void set_def_opt(bool create_defs = true);
		void set_embopt_def(const_iopt_ref emb_opt);
	};
	typedef tnn_opt<objnet> nn_opt;

	template<>
	class _CLASS_DECLSPEC tnn_opt<rbn> : public nn_opt
	{
		bool process_option(std::istream& inif, std::string& word);

	public:
		typedef tnn_opt<rbn> this_nn_opt;

		tnn_opt() { set_def_opt(); }
		tnn_opt(const this_nn_opt& opt) : nn_opt(opt) {}

		const char* get_opt_type() const { return "rbn_opt"; }

		void set_def_opt(bool create_defs = true);
	};
	typedef tnn_opt<rbn> rbn_opt;

	template<>
	class _CLASS_DECLSPEC tnn_opt<ccn> : public tnn_opt<objnet>
	{
		bool process_option(std::istream& inif, std::string& word);

	public:
		typedef tnn_opt<ccn> this_nn_opt;

		tnn_opt() { set_def_opt(); }
		tnn_opt(const this_nn_opt& opt) : nn_opt(opt) {}

		const char* get_opt_type() const { return "ccn_opt"; }

		void set_def_opt(bool create_defs = true);
		int test(void);
	};
	typedef tnn_opt<rbn> ccn_opt;
	*/

	//NN options class
	class _CLASS_DECLSPEC mnn_opt : public alg_options< nnOptions<MNet> >
	{
		friend class MNet;

		bool process_option(std::istream& inif, std::string& word);

	public:
		//Matrix inp_range_;
		int flags_;
		std::string errFname_;

		mnn_opt() { set_def_opt(); };
		//copy constructor
		mnn_opt(const mnn_opt& opt) : wrapper_opt(opt) {}

		const char* get_opt_type() const { return "mnn_opt"; }

		void set_def_opt(bool create_defs = true);
		bool set_wrapper_opt(const_iopt_ref iopt);
	};

	//template< class nn_type >
	//class _CLASS_DECLSPEC nn_opt : public alg_options< nnOptions< nn_type > >
	class _CLASS_DECLSPEC nn_opt : public alg_options< nnOptions< objnet > >
	{
		friend class layer;
		friend class objnet;
		friend class ccn;
		friend class rbn;

		template< class opt_type > friend bool process_objnet_opt(opt_type& opt, std::istream& inif, std::string& word);
		template< class opt_type > friend void set_objnet_defs(opt_type& opt, bool create_defs);
//
//		//friend templates
//#ifdef _MSC_VER
//		friend class nn_opt;
//#else
//		template< class N > friend class nn_opt< N >;
//#endif
//
//		typedef nnOptions< nn_type > opt_type;

		bool use_biases_;

	protected:
		virtual bool process_option(std::istream& inif, std::string& word);

	public:
		Matrix inp_range_;
		std::string errFname_;

		nn_opt() {};
		//copy constructor
		nn_opt(const nn_opt& opt) : wrapper_opt(opt) {
			set_wrapper_opt(opt);
		}
		//template copy constructor
		//template< class r_nn_type >
		//nn_opt(const nn_opt< r_nn_type >& opt) : wrapper_opt(opt) {
		//	set_wrapper_opt(opt);
		//}

		//virtual destructor
		virtual ~nn_opt() {};

		const char* get_opt_type() const { return "nn_opt"; }

		virtual void set_def_opt(bool create_defs = true);
		virtual bool set_wrapper_opt(const_iopt_ref iopt);

		void set_embopt_def(iopt_ref emb_opt);
	};

	//typedef nn_opt< objnet > objnet_opt;
	//typedef nn_opt< rbn > rbn_opt;
	//typedef nn_opt< ccn > ccn_opt;

	//template< > class _CLASS_DECLSPEC nn_opt< rbn > : public nn_opt< objnet >
	//{
	//	bool process_option(std::istream& inif, std::string& word);

	//public:
	//	void set_def_opt(bool create_defs = true);
	//	bool set_wrapper_opt(const_iopt_ref iopt);
	//};
	//template< > class _CLASS_DECLSPEC nn_opt< ccn > : public nn_opt< objnet >
	//{
	//	bool process_option(std::istream& inif, std::string& word);

	//public:
	//	void set_def_opt(bool create_defs = true);
	//	bool set_wrapper_opt(const_iopt_ref iopt);
	//};


	class _CLASS_DECLSPEC rbn_opt : public nn_opt
	{
		bool process_option(std::istream& inif, std::string& word);

	public:
		//options for rbn
		int rbnec_policy_;
		double rbl_patience_;
		bool io_linked_;
		//Green function type
		int gft_;
		double neur_incr_mult_;
		ulong neur_incr_step_;

		rbn_opt() {};
		//copy constructor
		rbn_opt(const rbn_opt& opt) : nn_opt(opt) {}

		//const char* get_opt_type() const { return "rbn_opt"; }

		void set_def_opt(bool create_defs = true);
		bool set_wrapper_opt(const_iopt_ref iopt);
	};

	class _CLASS_DECLSPEC ccn_opt : public nn_opt
	{
		bool process_option(std::istream& inif, std::string& word);

	public:
		ulong maxFLLcycles_, fl_candidates_, maxFL_;
		bool insert_between_, grow_vlayer_;
		ulong lsq_patience_cycles_;
		ulong fl_candidates_survive_;

		ccn_opt() {};
		ccn_opt(const ccn_opt& opt)	: nn_opt(opt) {}

		//const char* get_opt_type() const { return "ccn_opt"; }

		void set_def_opt(bool create_defs = true);
		bool set_wrapper_opt(const_iopt_ref iopt);
	};
}

#endif	//_NN_COMMON_H
