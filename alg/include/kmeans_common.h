#ifndef _KMEANS_COMMON_H
#define _KMEANS_COMMON_H

#include "common.h"
#include "alg_opt.h"
#include "matrix.h"
#include "ga_common.h"

namespace KM {
	//forward declarations
	class kmeans;

	enum norm_type {
		eucl_l2 = 1
	};

	enum seed_type {
		sample = 1,
		uniform = 2
	};

	enum emptyc_policy {
		do_nothing = 1,
		drop = 2,
		singleton = 3
	};

	struct kmOptions {
		double nu, init_v, alfa, patience;
		ulong patience_cycles;
		int norm_t, seed_t, emptyc_pol;
		//bool use_prev_cent;

		GA::gaOptions* pGAopt;
		//int scalingT, selectionT;
		//bool minimizing;
		//double ffscParam;
	};

	class km_opt : public alg_options<kmOptions>
	{
		bool process_option(std::istream& inif, std::string& word);

	protected:
		void set_data_opt(const data_opt* pOpt);

		void set_embopt_def(iopt_ref emb_opt);

	public:

		km_opt() {};
		//copy constructor
		km_opt(const km_opt& opt) : wrapper_opt(opt) {
			//set_wrapper_opt(opt);
		}
		//constructor by interface
		km_opt(const Ialg_options& opt) : wrapper_opt(opt) {
			//set_wrapper_opt(opt);
		}
		//virtual destructor
		virtual ~km_opt() {};		

		const char* get_opt_type() const { return "km_opt"; }

		virtual void set_def_opt(bool create_defs = true);

		void* GetOptions();
	};
}

#endif	// _KMEANS_COMMON_H
