#ifndef _NN_ADDON_H
#define _NN_ADDON_H

#include "nn_addon_common.h"
#include "kmeans.h"
#include "ga.h"
#include "mnet.h"
#include "determ_annealing.h"

namespace GA {

	class nn_addon : public ga_addon
	{
		friend class nna_opt;
		friend void NNFitnessFcn(int nVars, int nPopSize, double* pPop, double* pScore);

		typedef Matrix::indMatrix indMatrix;
		typedef smart_ptr<NN::objnet> sp_onet;
		typedef std::vector<sp_onet> vsp_onet;
		typedef vsp_onet::iterator onet_iterator;

		struct nn_state {
			double cur_min, cur_max;		//current min & max over learn data
			double n_cur_min, n_cur_max;	//normalized current min & max
			double norm_min, norm_max;		//min & max for normalization
			double lnorm_min, lnorm_max;	//log of norm_min & norm_max
			double lsnorm_min, lsnorm_max;	//logsig of norm_min & norm_max
			double log_zero;				//log of zero
			double a;
			double pred_r;					//max prediction radius
			bool init_nn, init_PCA;
			double max_ch_r;
			Matrix inpMean;
			ulong ind_best;
			double tar_mean, tar_std;
			Matrix learn_cl_cent;

			KM::kmeans km;
			DA::determ_annealing da;
		};

		Matrix learn_;
		Matrix tar_;

		GA::ga ga_;
		NN::MNet net_;
		vsp_onet _onet;
		std::auto_ptr<NN::MNet> netPCA_;

		nn_state state_;

		bool (nn_addon::*_pInitFcn)(ulong, ulong, const Matrix&);
		double (nn_addon::*_pAddonFcn)(const GA::ga&, Matrix&, Matrix&, Matrix&, ulong);

		//fill storage with duplicates filter
		void _fillup_storage(Matrix& p, Matrix& s);

		//filter functions for selecting learning samples
		template< class clusterizer >
		Matrix _kmeans_filter(clusterizer& cengine, const Matrix& p, const Matrix& f, Matrix& lp, Matrix& lf);

		Matrix _best_filter(const Matrix& p, const Matrix& f, Matrix& lp, Matrix& lf);

		void _get_learnData(Matrix& p, Matrix& s, Matrix& learn, Matrix& targets);
		int _learn_network(const Matrix& input, const Matrix& targets, ulong net_ind = 0);
		void _build_surf(ulong net_ind, const Matrix&, const Matrix&);

		double _calc_goalQuota(const Matrix& targets);
		NN::objnet* onet_fab();
		void _create_onet(ulong nets_count);

		//initialization finctions
		bool InitStd(ulong addon_count, ulong chromLength, const Matrix& searchRange);
		bool InitAlt(ulong addon_count, ulong chromLength, const Matrix& searchRange);
		bool InitPCA(ulong addon_count, ulong chromLength, const Matrix& searchRange);

		//main processing function
		double _GetOneAddonStd(const ga& caller, Matrix& p, Matrix& s, Matrix& new_chrom, ulong net_ind);
		double _GetOneAddonAlt(const ga& caller, Matrix& p, Matrix& s, Matrix& new_chrom, ulong net_ind);
		double _GetOneAddonPCA(const ga& caller, Matrix& p, Matrix& s, Matrix& new_chrom, ulong net_ind);

		Matrix GetNormalizedData(Matrix& data);
		Matrix GetRealData(const Matrix& ndata);
		Matrix GetChromLengths(const Matrix& pop);
		Matrix GetChromMM(const Matrix& pop);
		Matrix NormalizePop(const Matrix& pop);

		//delegate from nna_opt
		void AfterOptionsSet();

	public:

		nna_opt opt_;
		//std::string iniFname_;

		//normal constructors
		nn_addon();
		nn_addon(const char* psName);

		//nn_addon() { _construct(); }
		//nn_addon(const char* psName) : nna_opt(psName) { _construct(); };
		~nn_addon();

		const char* GetName() const;
		void SetName(const char* psName);

		void set_def_opt();
		void* GetOptions();
		void SetOptions(const void* pOpt);
		void ReadOptions(const char* pFName = NULL);

		void* GetObject(int obj_id) const;

		const char* GetFname(int fname) const;
		bool SetFname(int fname, const char* pNewFname = NULL);

		bool Init(ulong addon_count, ulong chromLength, const Matrix& searchRange);

		Matrix GetAddon(const Matrix& pop, const Matrix& score, const GA::ga& caller, Matrix& new_chrom);
		//double GetAddonStd(const Matrix& pop, const Matrix& score, const GA::ga& caller, Matrix& new_chrom);
		//double GetAddonAlt(const Matrix& pop, const Matrix& score, const GA::ga& caller, Matrix& new_chrom);
		//double GetAddonPCA(const Matrix& pop, const Matrix& score, const GA::ga& caller, Matrix& new_chrom);

		void BuildApproximation(const Matrix& samples, const Matrix& want_resp);
		Matrix Sim(const Matrix& samples, ulong net_ind = 0);
		const Matrix& GetClusterCenters() const;

		//void NNFitnessFcn(int nVars, int nPopSize, double* pPop, double* pScore);
		//bool LearnNNInformer(ulong uCycle, double dSSE, void* pNet);
	};
}

#endif	//_NN_ADDON_H
