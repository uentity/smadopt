/// @file bs_alg_api.cpp
/// @brief Light wrappers around smadopt C API for making it BS-friendly
/// @author uentity
/// @version 1.0
/// @date 18.07.2013
/// @copyright Copyright (C) 
///            
///            This program is free software; you can redistribute it and/or
///            modify it under the terms of the GNU General Public License
///            as published by the Free Software Foundation; either version 2
///            of the License, or (at your option) any later version.
///            
///            This program is distributed in the hope that it will be useful,
///            but WITHOUT ANY WARRANTY; without even the implied warranty of
///            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///            GNU General Public License for more details.
///            
///            You should have received a copy of the GNU General Public License
///            along with this program; if not, write to the Free Software
///            Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
///            

#ifdef BLUE_SKY_COMPAT

#include "ga.h"
#include "bs_alg_api.h"
#include "alg_api.h"
//#include "bs_misc.h"

#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>

// forward declare binding functions
namespace GA { namespace python {
	void py_ga_common();
}}
#endif

namespace blue_sky {
namespace smadopt {

void SetGAInitRange(spv_float pInitRange, int genomeLength) {
	::SetGAInitRange(pInitRange.lock()->data(), genomeLength);
}

ulong Start(int genomeLength, spv_float pInitPop, bool bReadFromIni) {
	return ::Start(pInitPop.lock()->data(), genomeLength, bReadFromIni);
}

bool GetNextPop(spv_float pPrevScore, spv_float pNextPop, unsigned long* pPopSize) {
	return ::GetNextPop(pPrevScore.lock()->data(), pNextPop.lock()->data(), pPopSize);
}

void Stop() {
	::Stop();
}

void ReadOptions(const std::string& psFileName) {
//#ifdef UNIX
//	std::string fname = wstr2str(psFileName);
//#else
//	std::string fname = wstr2str(psFileName, "ru_RU.CP1251");
//#endif
	::ReadOptions(psFileName.c_str());
}

bool ReadAddonOptions(unsigned long addon_num, const std::string& psFileName) {
//#ifdef UNIX
//	std::string fname = wstr2str(psFileName);
//#else
//	std::string fname = wstr2str(psFileName, "ru_RU.CP1251");
//#endif
	return ::ReadAddonOptions(addon_num, psFileName.c_str());
}

void SetHSPSizes(ulong nSubpops, spv_ulong pSizes) {
	::SetHSPSizes(nSubpops, pSizes.lock()->data());
}

void SetVSPSizes(ulong nSubpops, spv_ulong pSizes) {
	::SetVSPSizes(nSubpops, pSizes.lock()->data());
}

void SetVSPFractions(spv_float pFractions) {
	::SetVSPFractions(pFractions.lock()->data());
}

ulong GetPopSize() {
	GA::ga* ga_obj = reinterpret_cast< GA::ga* >(GetGAObject());
	return ulong(ga_obj->opt_.popSize);
}

void SetPopSize(const ulong pop_size) {
	GA::ga* ga_obj = reinterpret_cast< GA::ga* >(GetGAObject());
	ga_obj->opt_.popSize = pop_size;
}

} /* namespace blue_sky::smadopt */

#ifdef BSPY_EXPORTING_PLUGIN

namespace python {
namespace bspy = boost::python;
// we need a wrapper for GetNextPop, because arrays are copied from Python,
// hence all GA calculations (new population, etc) will be lost
// solution: return tuple of input arguments
bspy::tuple GetNextPop_py(spv_float pPrevScore, spv_float pNextPop, unsigned long pPopSize) {
	bool res = smadopt::GetNextPop(pPrevScore, pNextPop, &pPopSize);
	return bspy::make_tuple(pNextPop, pPopSize, res);
}

GA::gaOptions* GetGAOptions_py() {
	return (GA::gaOptions*)::GetGAOptions();
}

BOOST_PYTHON_FUNCTION_OVERLOADS(start_overl, smadopt::Start, 2, 3);
BOOST_PYTHON_FUNCTION_OVERLOADS(read_opt_overl, smadopt::ReadOptions, 0, 1);
BOOST_PYTHON_FUNCTION_OVERLOADS(read_addonopt_overl, smadopt::ReadAddonOptions, 0, 2);

BLUE_SKY_INIT_PY_FUN
{
	::GA::python::py_ga_common();

	bspy::def("SetGAInitRange", &smadopt::SetGAInitRange);
	bspy::def("Start", &smadopt::Start, start_overl());
	bspy::def("GetNextPop", &GetNextPop_py);
	bspy::def("Stop", &smadopt::Stop);
	bspy::def("ReadOptions", &smadopt::ReadOptions, read_opt_overl());
	bspy::def("ReadAddonOptions", &smadopt::ReadAddonOptions, read_addonopt_overl());
	bspy::def("SetHSPSizes", &smadopt::SetHSPSizes);
	bspy::def("SetVSPSizes", &smadopt::SetVSPSizes);
	bspy::def("SetVSPFractions", &smadopt::SetVSPFractions);
	bspy::def("GetPopSize", &smadopt::GetPopSize);
	bspy::def("SetPopSize", &smadopt::SetPopSize);
	bspy::def("GetGAOptions", &GetGAOptions_py,
		bspy::return_value_policy< bspy::reference_existing_object >()
	);
}

} /* namespace python */
#endif

BLUE_SKY_PLUGIN_DESCRIPTOR_EXT("smadopt", "1.0.0", "Smart optimization and adaptation library",
	"Smart optimization and adaptation algorithms library", "smadopt"
)

BLUE_SKY_REGISTER_PLUGIN_FUN
{
	(void)bs_init;
	return true;
}

} /* namespace blue_sky */

#endif

