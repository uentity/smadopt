#ifndef _NN_ADDON_COMMON_H
#define _NN_ADDON_COMMON_H

#include "common.h"
#include "ga_common.h"
#include "nn_common.h"
#include "kmeans_common.h"

namespace GA {

	#define VERBOSE
	#define TOL 1e-10

	//forward declarations
	class nn_addon;

	enum networkType {
		matrix_nn = 1,
		mlp_nn = 2,
		rb_nn = 3,
		ccn_nn = 4,
		pca_nn = 5
	};

	enum normType {
		UnchangedData = 1,
		LinearNorm = 2,
		LogNorm = 3,
		LogsigNorm = 4
	};

	enum objType {
		nnaNeuralNet = 1,
		nnaObjNet = 2,
		nnaInnerGA = 3,
		nnaInnerKM = 4
	};

	enum nnaFnames {
		NN_IniFname = 2,
		NN_ErrFname = 3,
		iga_IniFname = 4,
		iga_LogFname = 5,
		iga_HistFname = 6,
		iga_ErrFname = 7
	};

	enum addon_scheme {
		FitFcnAsTarget = 1,
		FitFcnAsInput = 2,
		PCAPredict = 3
	};

	enum sf_Type {
		best_filter = 1,
		kmeans_filter = 2,
		best_km_filter = 3
	};

	enum rbn_learnType {
		rbn_exact = 1,
		rbn_fully_bp = 2,
		rbn_kmeans_bp = 3
	};

	enum nnaClustEngine {
		ce_kmeans = 1,
		ce_DA = 2
	};

	struct nnAddonOptions {
		int netType, normType;
		double maxtar, mintar, pred_ratio;
		double inf_ff, sup_ff;
		long bestCount;
		bool is_ffRestricted, usePCA, normInp, initNetEveryIter, initPCAEveryIter;
		ulong PC_num;

		NN::nnOptions<NN::MNet>* pNNetOpt;
		NN::new_nnOptions* pNewNNetOpt;
		ulong layers_num;
		ulong* pNeurons;
		int* pLayerTypes;

		gaOptions* pInnerGAOpt;

		int addon_scheme, samples_filter, kmfec_policy, rbn_learn_type;
		double goalQuota, tol, kmf_cmult, rbn_cmult;
		ulong search_samples_num, search_clust_num, learn_clust_num;
		KM::kmOptions* pKMopt;
		int clustEngine;
	};

	class nna_opt : public alg_options<nnAddonOptions>
	{
		friend class nn_addon;
		nn_addon* self_;

		bool process_option(std::istream& inif, std::string& word);
		//we will read embded options manually
		//bool BeforeEmbOptRead() { return false; }

	protected:

		void set_data_opt(const data_opt* pOpt);

		void _setDefNNSize();
		void set_embopt_def(iopt_ref iopt);
		virtual void AfterOptionsSet();

	public:
		std::string name;
		std::vector<ulong> layers_;
		std::vector<int> lTypes_;

		nna_opt(nn_addon* self = NULL);
		//copy constructor
		nna_opt(const nna_opt& opt);
		//constructor by interface
		nna_opt(const Ialg_options& opt);
		//~nna_opt() {};

		const char* get_opt_type() const { return "nna_opt"; }

		void set_def_opt(bool create_defs = true);

		bool set_wrapper_opt(const_iopt_ref iopt);

		void* GetOptions();
		//void SetOptions(const void* pOpt);
		void ReadOptions(const char* pFname = NULL);
	};
}

#endif	// _NN_ADDON_COMMON_H
