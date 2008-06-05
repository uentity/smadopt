#ifndef _GA_COMMON_H
#define _GA_COMMON_H

#include "common.h"
#include "alg_except.h"
#include "alg_opt.h"
#include "matrix.h"

//forward declaration
namespace GA { class ga; }

namespace GA {
	#define ERR_VAL 1e16

	//forward declarations
	class ga;

	enum GA_Errors {
		SizesMismatch = 1
	};

	enum ScalingType {
		Proportional = 1,
		ProportionalMean = 2,
		ProportionalInv = 3,
		ProportionalTimeScaled = 4,
		ProportionalSigmaScaled = 5,
		Rank = 6,
		RankSqr = 7,
		RankExp = 8
	};

	enum SelectionType {
		StochasticUniform = 1,
		Roulette = 2,
		UniformSelection = 3,
		Tournament = 4,
		Once = 5,
		Sort = 6
	};

	enum CrossoverType {
		Heuristic = 1,
		OnePoint = 2,
		TwoPoint = 3,
		UniformCrossover = 4,
		Flat = 5,
		Arithmetic = 6,
		BLX = 7,
		SBX = 8
	};

	enum MutationType {
		UniformMutation = 1,
		NormalMutation = 2,
		NonUniformMutation = 3
	};

	enum MutationSpace {
		WholePopulation = 1,
		CrossoverKids = 2,
		PrevPopRest = 3
	};

	enum CreationType {
		UniformCreation = 1,
		Manual = 2
	};

	enum GAScheme {
		MuLambda = 1,
		MuPlusLambda = 2
	};

	enum GAStatus {
		Idle = 0,
		Working = 1,
		FinishGenLim = 2,
		FinishStallGenLim = 3,
		FinishTimeLim = 4,
		FinishFitLim = 5,
		FinishUserStop = 6,
		FinishError = -1
	};

	enum GAGybridScheme {
		ClearGA = 1,
		UseNN = 2
	};

	enum SubPopType {
		NoSubpops = 1,
		Horizontal = 2,
		Vertical = 3
	};

	enum MigrationPolicy {
		WorstBest = 1,
		RandomRandom = 2
	};

	enum MigrationDirection {
		MigrateForward = 1,
		MigrateBoth = 2
	};

	enum GAFnames {
		IniFname = 1,
		LogFname = 2,
		HistFname = 3,
		ErrFname = 4
	};

	//GA exceptions class
	class ga_except : public alg_except
	{
		//static std::string s_buf;
	public:
		ga_except()
			: alg_except()
		{
		}
		ga_except(const alg_except& ex)
			: alg_except(ex)
		{
		}
		ga_except(int code, const char* what)
			: alg_except(code, what)
		{
		}
		ga_except(int code)
		{
			_code = code;
			_what = explain_error(code);
		}
		ga_except(const char* what)
			: alg_except(what)
		{
		}
		~ga_except() {};

		static const char* explain_error(int code);
	};

	struct gaOptions {
	//template<>
	//class options<GA::ga> {
		int scheme;
		int h_scheme;
		int creationT;
		int scalingT;
		int selectionT;
		int crossoverT;
		int mutationT;
		int subpopT;

		//crossover parameters
		unsigned long nTournSize;
		double xoverFraction;
		double xoverHeuRatio;
		double xoverArithmeticRatio;
		double xoverBLXAlpha;
		double xoverSBXParam;
		//mutation parameters
		int mutSpace;
		double mutProb;
		double mutNormLawSigma2;
		double mutNonUniformParam;
		//migration parameters
		int migPolicy;
		int migDirection;
		unsigned long migInterval;
		double migFraction;

		unsigned long generations;
		int popSize;
		int eliteCount;
		long timeLimit;
		int stallGenLimit;
		double fitLimit;
		unsigned long addonCount;

		bool vectorized;
		bool useBitString;
		int bitsPerVar;
		bool logEveryPop;
		bool calcUnique;
		bool useFitLimit;
		unsigned long minUnique;
		bool globalSearch;
		bool minimizing;
		double ffscParam;
		bool sepAddonForEachVSP;
		bool excludeErrors;
	};

	class _CLASS_DECLSPEC ga_opt : public alg_options<gaOptions>
	{
		friend class ga;
		ga* self_;

		bool process_option(std::istream& inif, std::string& word);

		const char* get_opt_type() const { return "ga_opt"; }
		void AfterOptionsSet();
		//we will read embded options manually
		//bool BeforeEmbOptRead() { return false; }

	public:
		Matrix initRange, vspFract;
		ulMatrix hspSize, vspSize;

		std::string logFname;
		std::string errFname;
		std::string histFname;
		std::ios::openmode openMode;

		ga_opt(ga* self = NULL);
		//copy constructor
		ga_opt(const ga_opt& opt);
		//constructor by interface
		ga_opt(const wrapper_opt& opt);
		//virtual destructor
		//~ga_opt() {};

		void set_def_opt(bool create_defs = true);
		bool set_wrapper_opt(const_iopt_ref iopt);
	};

	//GA statistics
	class _CLASS_DECLSPEC ga_stat {
	public:
		ulMatrix chrom_cnt_;
		Matrix best_ff_, mean_ff_;
		ulMatrix stall_cnt_;
		clock_t startt_, curt_;
		bool timer_flushed_;

		ga_stat(ulong iterations = 0);

		void reserve(ulong iterations);
		void clear();
		ulong size() const;
		void add_record(ulong chrom_cnt, double best_ff, double mean_ff, ulong stall_g);

		void reset_timer();
		double sec_elapsed() const;

		//standart assignment will work fine
		//ga_stat& operator =(const ga_stat& s);
		const ga_stat& operator +=(const ga_stat& s);
		const ga_stat& operator /=(ulong cnt);
	};

	//GA addon abstract class
	class ga_addon
	{
	public:
		//ga_addon() : _name("ha") {};
		//ga_addon(const char* psName) : _name(psName) {};
		virtual ~ga_addon() {};

		virtual const char* GetName() const = 0; //{ return _name.c_str(); }
		virtual void SetName(const char* psName) = 0; //{ _name = psName; };

		virtual bool Init(ulong addon_count, ulong chromLength, const Matrix& searchRange) = 0;
		virtual Matrix GetAddon(const Matrix& pop, const Matrix& score, const ga& caller, Matrix& new_chrom) = 0;

		virtual void* GetOptions() = 0;
		virtual void SetOptions(const void* pOpt) = 0;
		virtual void ReadOptions(const char* pFName = NULL) = 0;

		virtual void* GetObject(int obj_id) const = 0;

		virtual const char* GetFname(int fname) const = 0;
		virtual bool SetFname(int fname, const char* pNewFname = NULL) = 0;
	};
}

typedef void (*FitnessFcnCallback)(int nVars, int nPopSize, double* pPopulation, double* pComputeScoreHere);

#endif	// _GA_COMMON_H
