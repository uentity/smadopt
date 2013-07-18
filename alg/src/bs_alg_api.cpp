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

#ifdef BSPY_EXPORTING_PLUGIN
#include <boost/python.hpp>
#endif

#include "bs_alg_api.h"
#include "alg_api.h"
#include "bs_misc.h"

namespace blue_sky {
namespace smadopt {

void SetGAInitRange(spv_float pInitRange, int genomeLength) {
	::SetGAInitRange(pInitRange.lock()->data(), genomeLength);
}

void Start(spv_float pInitPop, int genomeLength, bool bReadFromIni) {
	::Start(pInitPop.lock()->data(), genomeLength, bReadFromIni);
}

bool GetNextPop(spv_float pPrevScore, spv_float pNextPop, unsigned long* pPopSize) {
	return ::GetNextPop(pPrevScore.lock()->data(), pNextPop.lock()->data(), pPopSize);
}

void Stop() {
	::Stop();
}

void ReadOptions(const std::wstring& psFileName) {
#ifdef UNIX
	std::string fname = wstr2str(psFileName);
#else
	std::string fname = wstr2str(psFileName, "ru_RU.CP1251");
#endif
	::ReadOptions(fname.c_str());
}

bool ReadAddonOptions(unsigned long addon_num, const std::wstring& psFileName) {
#ifdef UNIX
	std::string fname = wstr2str(psFileName);
#else
	std::string fname = wstr2str(psFileName, "ru_RU.CP1251");
#endif
	return ::ReadAddonOptions(addon_num, fname.c_str());
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

} /* namespace blue_sky::smadopt */

namespace python {
namespace bspy = boost::python;



BLUE_SKY_INIT_PY_FUN
{
	bspy::def("SetGAInitRange", &smadopt::SetGAInitRange);
	bspy::def("Start", &smadopt::Start);
	bspy::def("GetNextPop", &smadopt::GetNextPop);
	bspy::def("Stop", &smadopt::Stop);
	bspy::def("ReadOptions", &smadopt::ReadOptions);
	bspy::def("ReadAddonOptions", &smadopt::ReadAddonOptions);
	bspy::def("SetHSPSizes", &smadopt::SetHSPSizes);
	bspy::def("SetVSPSizes", &smadopt::SetVSPSizes);
	bspy::def("SetVSPFractions", &smadopt::SetVSPFractions);
}

} /* namespace python */

BLUE_SKY_PLUGIN_DESCRIPTOR_EXT("smadopt", "1.0.0", "Smart optimization and adaptation library",
	"Smart optimization and adaptation library", "smadopt"
)

} /* namespace blue_sky */

#endif

