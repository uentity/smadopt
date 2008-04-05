#ifndef _MNET_H
#define _MNET_H

#pragma warning(disable: 4251)

#include "nn_common.h"
#include "alg_opt.h"
//#include "matrix.h"
#include <fstream>

namespace NN {

	class _CLASS_DECLSPEC MNet
	{
	public:
		typedef std::vector<Matrix> vm_type;
		typedef vm_type::iterator vm_iterator;
		//typedef Matrix::r_iterator r_iterator;

	private:
		nnState _state;
		ulong _nLayers, _nInpSize;

		std::ofstream errFile_;

		//weights, biases & their deltas
		vm_type _W, _B, _D, _BD;
		//lateral connections
		vm_type _L, _LD;
		//gradient storage
		vm_type _G, _BG, _LG;

		inline void _print_err(const char* pErr);
		void _dealloc(void);
		//void _setDefOpt();
		void _constructBiases();
		void _constructLateral();
		void _constructGrad();

		inline void _rp_classic(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights);
		inline void _rp_simple(const Matrix& grad, const Matrix& old_grad, Matrix& deltas, Matrix& weights);

		Matrix _active_af_region(int ActFunType);
		void ActFunc(Matrix& m, int SigmoidType);
		void D_Sigmoid(Matrix& m, int SigmoidType);

		inline void InitWeights();
		void InitWeightsRandom();
		void InitNW();

		double CalcGrad(const Matrix& mInput, const Matrix& mTarget, const Matrix& mask);
		double CalcPerfomance(void);
		void BPCalcSpeed(ulong cur_layer, Matrix& Er);
		Matrix BPCalcError(vm_type& input, vm_type& targets, ulong nPattern);
		double BPLearnStep(const Matrix& mInput, const Matrix& mTarget, const Matrix& mask);
		void R_BPUpdate(vm_type& old_G, vm_type& old_BG);
		void BPUpdate();
		double GHALearnStep(const Matrix& input);
		double APEXLearnStep(const Matrix& input);

	public:
		mnn_opt opt_;
		//int _flags;

		//std::string iniFname_;
		//std::string _errFname;

		//Matrix mInp;
		//Matrix mOut;
		vm_type _Out;
		std::vector<int> _LTypes;
		Matrix _inp_range;

		MNet(void);
		//copy constructor
		MNet(const MNet& net);
		MNet(ulong nLayNum, ulong nInpSize, ...);
		~MNet(void);

		MNet& operator =(MNet& net);

		nnState state() {
			return _state;
		}
		Matrix weights(ulong layer)
		{
			if(layer >= _nLayers) {
				//_state.lastError = nn_except::explain_error(InvalidLayer);
				_print_err(nn_except::explain_error(InvalidLayer));
				throw nn_except(InvalidLayer);
			}
			return _W[layer];
		}
		ulong GetLayersNum(void);
		ulong GetNeuronsNum(ulong nLayer);
		bool VerifyNet(void);

		bool SetLayersNum(ulong nNum);
		bool SetLayerType(ulong nLayer, int nType);
		bool SetLayer(ulong nLayNum, ulong nNeurons, int nType = tansig);
		ulong AddLayer(ulong nNeurons, int nType = tansig);

		void SetInputSize(ulong nSize);
		bool SetInput(const Matrix& mInp);	
		void Normalize(ulong nLayer);
		void AddNoise(bool bSaveInput);
			
		void Propagate(void);
		void BackPropagate(void);
		void Update(ulong nNum = 1);
		bool IsConverged(void);

		int BPLearn(const Matrix& input, const Matrix& targets, bool init_weights = true, pLearnInformer pProc = NULL, const Matrix& mask = Matrix()) throw(alg_except);
		int PCALearn(const Matrix& input, bool init_weights = true, pLearnInformer pProc = NULL) throw(alg_except);
		Matrix Sim(const Matrix& mInp);
		Matrix ReverseSim(Matrix& mInp);
		bool SimBinary(char* pCode);

		//DWORD Save2File(LPCTSTR psFName, void* pSaveTo = NULL);
		//bool LoadFromFile(LPCTSTR psFName, void* pLoadFrom = NULL);
		void LinPCA(const Matrix& input, std::auto_ptr<MNet>& net, pLearnInformer pProc = NULL,
			ulong comp_num = 0, int learnType = GHA);
		//void ReadOptions(const char* pFName = NULL);
	};
}

#endif //_MNET_H
