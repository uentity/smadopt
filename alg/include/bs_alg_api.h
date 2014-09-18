#ifndef BS_ALG_API_VNTNBE7G
#define BS_ALG_API_VNTNBE7G

#include <bs_common.h>
#include BS_FORCE_PLUGIN_IMPORT()
#include <conf.h>
#include BS_STOP_PLUGIN_IMPORT()

namespace blue_sky { namespace smadopt {

//--------------------Main functions
//BS_API void* GetGAObject();

BS_API_PLUGIN void SetGAInitRange(spv_float pInitRange, int genomeLength);	//length(pInitRange) = 2*genomeLength!

BS_API_PLUGIN ulong Start(int genomeLength, spv_float pInitPop, bool bReadFromIni = false);

BS_API_PLUGIN bool GetNextPop(spv_float pPrevScore, spv_float pNextPop, ulong* pPopSize);

BS_API_PLUGIN void Stop();

//BS_API double Run(FitnessFcnCallback FitFcn, int genomeLength, double* pBestChrom, bool bReadFromIni = false);

BS_API_PLUGIN void ReadOptions(const std::string& psFileName = "");

BS_API_PLUGIN bool ReadAddonOptions(ulong addon_num = 0, const std::string& psFileName = "");

BS_API_PLUGIN void SetHSPSizes(ulong nSubpops, spv_ulong pSizes);

BS_API_PLUGIN void SetVSPSizes(ulong nSubpops, spv_ulong pSizes);
//new algorithm - vertical subpops fractions
BS_API_PLUGIN void SetVSPFractions(spv_float pFractions);

// helper fucntion to obtain population size
BS_API_PLUGIN ulong GetPopSize();
BS_API_PLUGIN void SetPopSize(const ulong pop_size);

// gain access to internal GA population & score arrays
BS_API_PLUGIN void get_internal_population(spv_float pop, spv_float score);
BS_API_PLUGIN void set_internal_population(spv_float pop, spv_float score);

}} /* namespace blue_sky::smadopt */

#endif /* end of include guard: BS_ALG_API_VNTNBE7G */

