#ifndef _GA_H
#define _GA_H

#include "ga_common.h"
#include <fstream>

namespace GA {

	//MAIN GA
	class _CLASS_DECLSPEC ga
	{
		friend class ga_opt;
		friend class nn_addon;

		typedef Matrix::r_iterator r_iterator;
		typedef std::vector<Matrix>::iterator vm_iterator;
		typedef Matrix::ind_iterator ind_iterator;
		typedef ulMatrix::r_iterator ulr_iterator;
	public:
		typedef Matrix::indMatrix indMatrix;
		typedef smart_ptr<ga_addon> spAddonObj;

	private:
		struct ga_state {
			//Matrix bestGens;
			//Matrix bestScore;
			Matrix lastPop;
			Matrix lastScore;
			ulMatrix rep_ind;
			Matrix stackPop;
			Matrix stackScore;
			Matrix mainScore;
			ul_vec elite_ind;
			ul_vec addon_ind;
			double last_min;
			Matrix best_sp;
			Matrix addons_ff;
			//double best_addon;

			int nStatus;
			ulong nGen;
			ulong nStallGen;
			clock_t tStart;
			ulong nChromCount, nlsCount;
			double ffsc_factor;
			std::string lastError;
		};

		//std::ifstream iniFile;
		std::ofstream logFile_;
		std::ofstream errFile_;
		std::ofstream histFile_;
		std::ofstream statFile_;

		int GenLen_;
		ga_state state_;
		ga_stat stat_;
		std::vector< spAddonObj > apAddon_;

		//operator pointers
		Matrix (ga::*_pScalingFcn)(const Matrix&);
		indMatrix (ga::*_pSelectionFcn)(Matrix&, ulong);
		Matrix (ga::*_pCrossoverFcn)(const Matrix&, const Matrix&, const indMatrix&);
		void (ga::*_pMutationFcn)(double&, ulong, const Matrix&);
		Matrix (ga::*_pCreationFcn)(void);
		Matrix (ga::*_pStepGAFcn)(const Matrix&, const Matrix&, ul_vec&);
		FitnessFcnCallback _pFitFunCallback;

		void MutateUniform(double& gene, ulong pos, const Matrix& range);
		void MutateNonUniform(double& gene, ulong pos, const Matrix& range);
		void MutateNormal(double& gene, ulong pos, const Matrix& range);

		void inline _print_err(const char* pErr);
		ulong inline _getXoverParentsCount(ulong xoverKidsCount);
		double inline _nonUniformMutDelta(ulong t, double y);
		inline void _sepRepeats(Matrix& source, const ulMatrix& rep_ind, Matrix* pRepeats = NULL,
			ul_vec* pPrevReps = NULL);
		inline void _filterParents(Matrix& thisPop, Matrix& thisScore, ulong parents_num);
		inline Matrix _restoreScore(const Matrix& newScore, const ulMatrix& reps);

		//inline Matrix ScalingCall(const Matrix& scores, ulong nParents);
		//inline Matrix SelectionCall(const Matrix& expect, ulong nParents);
		//inline Matrix CrossoverCall(const Matrix& pop, const Matrix& scores, const Matrix& parents);
		//inline Matrix MutationCall(const Matrix& pop, const Matrix& scores, const Matrix& parents);
		//inline Matrix CreationCall(void);
		inline Matrix FitnessFcnCall(const Matrix& pop, const ulMatrix& reps);
		inline void PushGeneration(const Matrix& thisPop, const Matrix& thisScore, const ulMatrix& rep_ind);

		Matrix StepGA(const Matrix& thisPop, const Matrix& thisScore, const Matrix& mutRange,
			ul_vec& elite_ind, ul_vec& addon_ind, ulong nAddonInd = 0, ulong nNewKids = 0);
		//Matrix SimpleStepGA(const Matrix& thisPop, const Matrix& thisScore, Matrix& reps);
		//horizontal subpop step GA
		Matrix HSPStepGA(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind);
		//Matrix SimpleHSPStepGA(const Matrix& thisPop, const Matrix& thisScore, Matrix& reps);
		//vertical subpop step GA
		Matrix VSPStepGA(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind);
		Matrix VSPStepGAalt(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind,
			ulong nNewKids = 0);

		ulMatrix FindRepeats(const Matrix where, const Matrix& what);
		ulMatrix EnsureUnique(Matrix& nextPop, const Matrix& thisPop, const Matrix& thisScore, const ul_vec& elite_ind);

		void InformWorld(void);
		std::ostream& OutpFinishInfo(std::ostream& outs, const Matrix& bestChrom, double bestScore);
		std::ostream& OutpIterRes(std::ostream& outs);
		Matrix FinishGA(double* pBestPop = NULL, double* pBestScore = NULL);

		inline void AddonsCreation();
		//void ReadAddonOptions();
		void FixInitRange(ulong genomeLength);
		void ValidateOptions();

		//delegates from ga_opt
		void AfterOptionsSet();

	public:
		ga_opt opt_;

		//gaOptions opt_;

		//Matrix _initRange;
		//Matrix _hspSize, _vspSize, _vspFract;

		//std::string iniFname_;

		Matrix bestChrom_;
		double bestScore_;

		ga(void);
		~ga(void);

		//override set default options
		void set_def_opt();

		//wrappers for ga operators
		Matrix ScalingCall(const Matrix& scores) {
			return (this->*_pScalingFcn)(scores);
		}
		indMatrix SelectionCall(Matrix& expect, ulong nParents) {
			return (this->*_pSelectionFcn)(expect, nParents);
		}
		Matrix CrossoverCall(const Matrix& pop, const Matrix& scores, const indMatrix& parents) {
			return (this->*_pCrossoverFcn)(pop, scores, parents);
		}
		Matrix CreationCall() {
			return (this->*_pCreationFcn)();
		}
		inline Matrix StepGACall(const Matrix& thisPop, const Matrix& thisScore, ul_vec& elite_ind, ul_vec& addon_ind,
			ulong nKids = 0);

		Matrix ScalingSimplest(const Matrix& scores);
		Matrix ScalingPropMean(const Matrix& scores);
		Matrix ScalingProp(const Matrix& scores);
		Matrix ScalingPropInv(const Matrix& scores);
		Matrix ScalingPropTime(const Matrix& scores);
		Matrix ScalingPropSigma(const Matrix& scores);
		Matrix ScalingRank(const Matrix& scores);
		Matrix ScalingRankSqr(const Matrix& scores);
		Matrix ScalingRankExp(const Matrix& scores);

		indMatrix SelectionStochUnif(Matrix& expect, ulong nParents);
		indMatrix SelectionRoulette(Matrix& expect, ulong nParents);
		indMatrix SelectionUniform(Matrix& expect, ulong nParents);
		indMatrix SelectionTournament(Matrix& expect, ulong nParents);
		indMatrix SelectionOnce(Matrix& expect, ulong nParents);
		indMatrix SelectionSort(Matrix& expect, ulong nParents);

		Matrix CrossoverOnePoint(const Matrix& pop, const Matrix& scores, const indMatrix& parents);
		Matrix CrossoverTwoPoint(const Matrix& pop, const Matrix& scores, const indMatrix& parents);
		Matrix CrossoverUniform(const Matrix& pop, const Matrix& scores, const indMatrix& parents);
		Matrix CrossoverHeuristic(const Matrix& pop, const Matrix& scores, const indMatrix& parents);
		Matrix CrossoverFlat(const Matrix& pop, const Matrix& scores, const indMatrix& parents);
		Matrix CrossoverBLX(const Matrix& pop, const Matrix& scores, const indMatrix& parents);
		Matrix CrossoverArithmetic(const Matrix& pop, const Matrix& scores, const indMatrix& parents);
		Matrix CrossoverSBX(const Matrix& pop, const Matrix& scores, const indMatrix& parents);

		void Mutation(Matrix& pop, const Matrix& range);

		Matrix CreationUniform();
		void Migrate(Matrix& pop, Matrix& score, ul_vec& elite_ind);

		void prepare2run(int genomeLength, bool bReadOptFromIni = false);
		Matrix Run(FitnessFcnCallback FitFcn, int genomeLength, bool bReadOptFromIni = false);
		void Start(double* pInitPop, int genomeLength, bool bReadOptFromIni = false);
		bool NextPop(double* pPrevScore, double* pNextPop, unsigned long* pPopSize);
		void Stop(void);

		void ReadOptions(const char* pFName = NULL);
		void SetOptions(const gaOptions* pOpt = NULL);
		bool SetAddonOptions(const void* pOpt, ulong addon_num = 0);
		void* GetAddonOptions(ulong addon_num = 0);

		Matrix InterpretBitPop(const Matrix& bit_pop);
		Matrix Convert2BitPop(const Matrix& pop);
		ga_addon* GetAddonObject(ulong addon_num);

		const ga_stat& GetStatistics() { return stat_; }
	};
}

#endif	// _GA_H
