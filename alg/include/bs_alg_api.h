#ifndef BS_ALG_API_VNTNBE7G
#define BS_ALG_API_VNTNBE7G

#include <bs_common.h>
#include <conf.h>

namespace blue_sky { namespace smadopt {

//--------------------Main functions
//BS_API void* GetGAObject();

BS_API void SetGAInitRange(spv_float pInitRange, int genomeLength);	//length(pInitRange) = 2*genomeLength!

BS_API ulong Start(int genomeLength, spv_float pInitPop, bool bReadFromIni = false);

BS_API bool GetNextPop(spv_float pPrevScore, spv_float pNextPop, unsigned long* pPopSize);

BS_API void Stop();

//BS_API double Run(FitnessFcnCallback FitFcn, int genomeLength, double* pBestChrom, bool bReadFromIni = false);

BS_API void ReadOptions(const std::string& psFileName = "");

BS_API bool ReadAddonOptions(unsigned long addon_num = 0, const std::string& psFileName = "");

BS_API void SetHSPSizes(ulong nSubpops, spv_ulong pSizes);

BS_API void SetVSPSizes(ulong nSubpops, spv_ulong pSizes);
//new algorithm - vertical subpops fractions
BS_API void SetVSPFractions(spv_float pFractions);

// helper fucntion to obtain population size
BS_API ulong GetPopSize();
BS_API void SetPopSize(const ulong pop_size);

}} /* namespace blue_sky::smadopt */

#endif /* end of include guard: BS_ALG_API_VNTNBE7G */

