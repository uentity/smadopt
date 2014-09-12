// Copyright (C) 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
// 

#include "ga_common.h"
//#include "bs_common.h"
#include <boost/python.hpp>

namespace GA { namespace python {
using namespace boost::python;

void py_ga_common() {
	// export all GA enums

	enum_< GA_Errors >("GA_Errors")
		.value("SizesMismatch", SizesMismatch);


	enum_< ScalingType >("ScalingType")
		.value("Proportional", Proportional)
		.value("ProportionalMean", ProportionalMean)
		.value("ProportionalInv", ProportionalInv)
		.value("ProportionalTimeScaled", ProportionalTimeScaled)
		.value("ProportionalSigmaScaled", ProportionalSigmaScaled)
		.value("Rank", Rank)
		.value("RankSqr", RankSqr)
		.value("RankExp", RankExp)
	;

	enum_< SelectionType >("SelectionType")
		.value("StochasticUniform", StochasticUniform)
		.value("Roulette", Roulette)
		.value("UniformSelection", UniformSelection)
		.value("Tournament", Tournament)
		.value("Once", Once)
		.value("Sort", Sort)
	;

	enum_< CrossoverType >("CrossoverType")
		.value("Heuristic", Heuristic)
		.value("OnePoint", OnePoint)
		.value("TwoPoint", TwoPoint)
		.value("UniformCrossover", UniformCrossover)
		.value("Flat", Flat)
		.value("Arithmetic", Arithmetic)
		.value("BLX", BLX)
		.value("SBX", SBX)
	;

	enum_< MutationType >("MutationType")
		.value("UniformMutation", UniformMutation)
		.value("NormalMutation", NormalMutation)
		.value("NonUniformMutation", NonUniformMutation)
	;

	enum_< MutationSpace >("MutationSpace")
		.value("WholePopulation", WholePopulation)
		.value("CrossoverKids", CrossoverKids)
		.value("PrevPopRest", PrevPopRest)
	;

	enum_< CreationType >("CreationType")
		.value("UniformCreation", UniformCreation)
		.value("Manual", Manual)
	;

	enum_< GAScheme >("GAScheme")
		.value("MuLambda", MuLambda)
		.value("MuPlusLambda", MuPlusLambda)
	;

	enum_< GAStatus >("GAStatus")
		.value("Idle", Idle)
		.value("Working", Working)
		.value("FinishGenLim", FinishGenLim)
		.value("FinishStallGenLim", FinishStallGenLim)
		.value("FinishTimeLim", FinishTimeLim)
		.value("FinishFitLim", FinishFitLim)
		.value("FinishUserStop", FinishUserStop)
		.value("FinishError", FinishError)
	;

	enum_< GAHybridScheme >("GAHybridScheme")
		.value("ClearGA", ClearGA)
		.value("UseNN", UseNN)
	;

	enum_< SubPopType >("SubPopType")
		.value("NoSubpops", NoSubpops)
		.value("Horizontal", Horizontal)
		.value("Vertical", Vertical)
	;

	enum_< MigrationPolicy >("MigrationPolicy")
		.value("WorstBest", WorstBest)
		.value("RandomRandom", RandomRandom)
	;

	enum_< MigrationDirection >("MigrationDirection")
		.value("MigrateForward", MigrateForward)
		.value("MigrateBoth", MigrateBoth)
	;

	enum_< GAFnames >("GAFnames")
		.value("IniFname", IniFname)
		.value("LogFname", LogFname)
		.value("HistFname", HistFname)
		.value("ErrFname", ErrFname)
	;

	class_< gaOptions >("gaOptions")
		.def_readwrite("scheme",     &gaOptions::scheme)
		.def_readwrite("h_scheme",   &gaOptions::h_scheme)
		.def_readwrite("creationT",  &gaOptions::creationT)
		.def_readwrite("scalingT",   &gaOptions::scalingT)
		.def_readwrite("selectionT", &gaOptions::selectionT)
		.def_readwrite("crossoverT", &gaOptions::crossoverT)
		.def_readwrite("mutationT",  &gaOptions::mutationT)
		.def_readwrite("subpopT",    &gaOptions::subpopT)

		//crossover parameters
		.def_readwrite("nTournSize",           &gaOptions::nTournSize)
		.def_readwrite("xoverFraction",        &gaOptions::xoverFraction)
		.def_readwrite("xoverHeuRatio",        &gaOptions::xoverHeuRatio)
		.def_readwrite("xoverArithmeticRatio", &gaOptions::xoverArithmeticRatio)
		.def_readwrite("xoverBLXAlpha",        &gaOptions::xoverBLXAlpha)
		.def_readwrite("xoverSBXParam",        &gaOptions::xoverSBXParam)

		//mutation parameters
		.def_readwrite("mutSpace",           &gaOptions::mutSpace)
		.def_readwrite("mutProb",            &gaOptions::mutProb)
		.def_readwrite("mutNormLawSigma2",   &gaOptions::mutNormLawSigma2)
		.def_readwrite("mutNonUniformParam", &gaOptions::mutNonUniformParam)

		//migration parameters
		.def_readwrite("migPolicy",    &gaOptions::migPolicy)
		.def_readwrite("migDirection", &gaOptions::migDirection)
		.def_readwrite("migInterval",  &gaOptions::migInterval)
		.def_readwrite("migFraction",  &gaOptions::migFraction)

		.def_readwrite("generations",        &gaOptions::generations)
		.def_readwrite("popSize",            &gaOptions::popSize)
		.def_readwrite("eliteCount",         &gaOptions::eliteCount)
		.def_readwrite("timeLimit",          &gaOptions::timeLimit)
		.def_readwrite("stallGenLimit",      &gaOptions::stallGenLimit)
		.def_readwrite("fitLimit",           &gaOptions::fitLimit)
		.def_readwrite("addonCount",         &gaOptions::addonCount)

		.def_readwrite("vectorized",         &gaOptions::vectorized)
		.def_readwrite("useBitString",       &gaOptions::useBitString)
		.def_readwrite("bitsPerVar",         &gaOptions::bitsPerVar)
		.def_readwrite("logEveryPop",        &gaOptions::logEveryPop)
		.def_readwrite("calcUnique",         &gaOptions::calcUnique)
		.def_readwrite("useFitLimit",        &gaOptions::useFitLimit)
		.def_readwrite("minUnique",          &gaOptions::minUnique)
		.def_readwrite("globalSearch",       &gaOptions::globalSearch)
		.def_readwrite("minimizing",         &gaOptions::minimizing)
		.def_readwrite("ffscParam",          &gaOptions::ffscParam)
		.def_readwrite("sepAddonForEachVSP", &gaOptions::sepAddonForEachVSP)
		.def_readwrite("excludeErrors",      &gaOptions::excludeErrors)
	;

}

}} /* namespace GA::python */

