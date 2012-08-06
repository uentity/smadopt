#ifndef _ALG_API_H
#define _ALG_API_H

#include "common.h"

typedef void (*FitnessFcnCallback)(int nVars, int nPopSize, double* pPopulation, double* pComputeScoreHere);

//--------------------Options manipulating
//pGAopts = &gaOptions!
_LIBAPI void SetGAOptions(const void* pGAopts);
//returns pointer to gaOptions for direct modifications. Use properly!
_LIBAPI void* GetGAOptions();
//Get & set misc filenames
_LIBAPI const char* GetGAFname(int ga_fname);
_LIBAPI bool SetGAFname(int ga_fname, const char* pNewFname = NULL);

_LIBAPI void* GetAddonOptions(unsigned long addon_num = 0);

_LIBAPI bool SetAddonOptions(const void* pOpt, unsigned long addon_num = 0);

//Get & set misc filenames for addons
_LIBAPI const char* GetAddonFname(int addon_fname, int addon_num = 0);
_LIBAPI bool SetAddonFname(int addon_fname, unsigned long addon_num = 0, const char* pNewFname = NULL);

//--------------------Main functions
_LIBAPI void* GetGAObject();

_LIBAPI void SetGAInitRange(double* pInitRange, int genomeLength);	//length(pInitRange) = 2*genomeLength!

_LIBAPI void Start(double* pInitPop, int genomeLength, bool bReadFromIni = false);

_LIBAPI bool GetNextPop(double* pPrevScore, double* pNextPop, unsigned long* pPopSize);

_LIBAPI void Stop();

_LIBAPI double Run(FitnessFcnCallback FitFcn, int genomeLength, double* pBestChrom, bool bReadFromIni = false);

_LIBAPI void ReadOptions(const char* psFileName = NULL);

_LIBAPI bool ReadAddonOptions(unsigned long addon_num = 0, const char* psFileName = NULL);

_LIBAPI void SetHSPSizes(ulong nSubpops, ulong* pSizes);

_LIBAPI void SetVSPSizes(ulong nSubpops, ulong* pSizes);
//new algorithm - vertical subpops fractions
_LIBAPI void SetVSPFractions(double* pFractions);

//--------------------Approximation only - for Oleg
_LIBAPI unsigned long BuildApproximation(unsigned long inp_num, unsigned long sampl_num,
								const double* samples, const double* want_resp);
_LIBAPI void Sim(unsigned long inp_num, unsigned long sampl_num, const double* samples, double* res);
_LIBAPI void GetClusterCenters(double* res);

//-------------------- NN building functions exposed - for David
/// @brief Create Multilayer Perceptron NN
///
/// @param layers_num - number of NN layers (including input layer)
/// @param neurons_num - array of neurons number in each NN layer (inlcuding input layer)
/// @param neuron_af - array of activation function types for each _hidden_ layer
///        (not including hidden layer!)
///
/// @return void
_LIBAPI void BuildMP(unsigned long layers_num, unsigned long* neurons_num,
	unsigned int* neuron_af);

/// @brief Train MP
///
/// @param sampl_num - number of training samples
/// @param samples - array of learning samples, arranged as matrix columns(!)
/// @param want_resp - array of learning targets, arranged as matrix columns(!)
///
/// @return Learning error
_LIBAPI double LearnMP(unsigned long sampl_num, const double* samples, const double* want_resp);

/// @brief Simulate network
///
/// @param sampl_num - number of test samples
/// @param samples - array of test samples, arranged as matrix columns(!)
/// @param res - buffer for storing simulation results
///
/// @return void
_LIBAPI void SimMP(unsigned long sampl_num, const double* samples, double* res);

#endif
