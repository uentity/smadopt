// ga_win.cpp : Defines the entry point for the DLL application.
//

#include "alg_api.h"
#include "ga.h"
#include "nn_addon.h"

using namespace GA;

ga g_ga;
nn_addon g_nna;

void* GetGAObject()
{
	return (void*)&g_ga;
}

void SetGAOptions(const void* pGAopts)
{
	//TODO: this is only for PPP
	((gaOptions*)pGAopts)->addonCount = 1;

	g_ga.SetOptions((gaOptions*)pGAopts);
}

void* GetGAOptions()
{
	return g_ga.opt_.GetOptions();
}

bool SetAddonOptions(const void* pOpt, unsigned long addon_num)
{
	//DEBUG - remove this line
	//((gaOptions*)pOpt)->calcUnique = false;
	return g_ga.SetAddonOptions(pOpt, addon_num);
}

void* GetAddonOptions(unsigned long addon_num)
{
	return g_ga.GetAddonOptions(addon_num);
}

void SetGAInitRange(double* pInitRange, int genomeLength)
{
	g_ga.opt_.initRange.NewMatrix(2, genomeLength, pInitRange);
}

const char* GetGAFname(int ga_fname)
{
	switch(ga_fname) {
		case IniFname:
			return g_ga.opt_.iniFname_.c_str();
		case LogFname:
			return g_ga.opt_.logFname.c_str();
		case HistFname:
			return g_ga.opt_.histFname.c_str();
		case ErrFname:
			return g_ga.opt_.errFname.c_str();
	}
	return NULL;
}

const char* GetAddonFname(int addon_fname, int addon_num)
{
	const ga_addon* pAddon = g_ga.GetAddonObject(addon_num);
	if(pAddon) return pAddon->GetFname(addon_fname);
	else return NULL;
}

bool SetGAFname(int ga_fname, const char* pNewFname)
{
	if(!pNewFname) return false;
	switch(ga_fname) {
		case IniFname:
			g_ga.opt_.iniFname_ = pNewFname;
			return true;
		case LogFname:
			g_ga.opt_.logFname = pNewFname;
			return true;
		case HistFname:
			g_ga.opt_.histFname = pNewFname;
			return true;
		case ErrFname:
			g_ga.opt_.errFname = pNewFname;
			return true;
	}
	return false;
}

bool SetAddonFname(int addon_fname, unsigned long addon_num, const char* pNewFname)
{
	ga_addon* pAddon = g_ga.GetAddonObject(addon_num);
	if(pAddon) return pAddon->SetFname(addon_fname, pNewFname);
	else return false;
}

void Start(double* pInitPop, int genomeLength, bool bReadFromIni)
{
	g_ga.Start(pInitPop, genomeLength, bReadFromIni);
}

bool GetNextPop(double* pPrevScore, double* pNextPop, unsigned long* pPopSize)
{
	return g_ga.NextPop(pPrevScore, pNextPop, pPopSize);
}

void Stop()
{
	g_ga.Stop();
}

double Run(FitnessFcnCallback FitFcn, int genomeLength, double* pBestChrom, bool bReadFromIni)
{
	Matrix res = g_ga.Run(FitFcn, genomeLength, bReadFromIni);
	memcpy(pBestChrom, res.GetBuffer(), res.raw_size());
	return g_ga.bestScore_;
}

void ReadOptions(const char* psFileName)
{
	g_ga.ReadOptions(psFileName);
}

bool ReadAddonOptions(unsigned long addon_num, const char* psFileName)
{
	ga_addon* const pAddon = g_ga.GetAddonObject(addon_num);
	if(pAddon) {
		pAddon->ReadOptions(psFileName);
		return true;
	}
	else return false;
}

void SetHSPSizes(ulong nSubpops, ulong* pSizes)
{
	g_ga.opt_.hspSize.NewMatrix(1, nSubpops);
	ulong* pos = pSizes;
	for(ulong i=0; i<nSubpops; ++i) {
		g_ga.opt_.hspSize[i] = *pos;
		++pos;
	}
}

void SetVSPSizes(ulong nSubpops, ulong* pSizes)
{
	g_ga.opt_.vspSize.NewMatrix(1, nSubpops);
	ulong* pos = pSizes;
	for(ulong i=0; i<nSubpops; ++i) {
		g_ga.opt_.vspSize[i] = *pos;
		++pos;
	}
}

void SetVSPFractions(double* pFractions)
{
	g_ga.opt_.vspFract.Resize(1, g_ga.opt_.vspSize.size());
	g_ga.opt_.vspFract.SetBuffer(pFractions);
}

unsigned long BuildApproximation(ulong inp_num, ulong sampl_num, const double* pSamples, const double* pResp)
{
	Matrix samples(inp_num, sampl_num, pSamples), resp(1, sampl_num, pResp);
	g_nna.ReadOptions();
	g_nna.Init(1, inp_num, Matrix());
	g_nna.BuildApproximation(!samples, !resp);
	return g_nna.GetKMCenters().row_num();
}

void GetKMCenters(double* res)
{
	memcpy(res, g_nna.GetKMCenters().GetBuffer(), g_nna.GetKMCenters().raw_size());
}

void Sim(unsigned long inp_num, unsigned long sampl_num, const double* pSamples, double* res)
{
	Matrix samples(inp_num, sampl_num, pSamples);
	Matrix est = g_nna.Sim(samples);
	memcpy(res, est.GetBuffer(), est.raw_size());
}
