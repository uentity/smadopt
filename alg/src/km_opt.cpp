#include "kmeans_common.h"

using namespace std;
using namespace KM;
using namespace GA;

void km_opt::set_def_opt(bool create_defs)
{
	iniFname_ = "kmeans.ini";

	init_v = 1e-10; alfa = 0.9999;
	nu = 0.01;
	patience = 0.001;
	patience_cycles = 10;
	seed_t = uniform;
	norm_t = eucl_l2;
	emptyc_pol = singleton;
	//use_prev_cent = false;

	//if(create_defs)
	//	create_def_embobj<ga_opt>("ga_opt");
}

void km_opt::set_embopt_def(iopt_ref emb_opt)
{
	//emb_opt.set_def_opt();
	if(std::string(emb_opt.get_opt_type()) == "ga_opt") {
		ga_opt* pGA = (ga_opt*)emb_opt.get_wrapper_opt();
		pGAopt = pGA;
		pGA->iniFname_ = iniFname_;
		pGA->scalingT = Rank;
		pGA->selectionT = Tournament;
		pGA->nTournSize = 3;
		pGA->minimizing = true;
		pGA->ffscParam = 2;
		emb_opt.update_embopt_defs();
	}
}

/*
void km_opt::AfterEmbOptAdded(iopt_ref emb_opt)
{
	if(emb_opt.get_opt_type() == "ga_opt") {
		ga_opt* pGA = (ga_opt*)emb_opt.get_opt();
		pGAopt = &pGA->opt_;
		pGA->iniFname_ = iniFname_;
		pGA->scalingT = Rank;
		pGA->selectionT = StochasticUniform;
		pGA->minimizing = true;
		pGA->ffscParam = 2;
	}
}
*/

void* km_opt::GetOptions()
{
	iopt_ptr piOpt;
	if((piOpt = get_embopt("ga_opt")))
		pGAopt = (gaOptions*)piOpt->GetOptions();
	return (void*)this;
}

void km_opt::set_data_opt(const data_opt* pOpt)
{
	if(pOpt == NULL) return;
	wrapper_opt::set_data_opt(pOpt);

	iopt_ptr piOpt;
	if((piOpt = get_embopt("ga_opt")))
		piOpt->SetOptions(pOpt->pGAopt);
}

bool km_opt::process_option(std::istream& inif, std::string& word)
{
	const string sOpts = " KMNu KMInitV KMAlfa KMPatience KMPatienceCycles KMNormType KMSeedType KMEmptyCentPolicy";
	const string sNormT = " Euclidian";
	const string sSeedT = " Sample Uniform";
	const string sECPol = " DoNothing Drop Singleton";

	ulong nPos;
	if((nPos = word_pos(sOpts, word)) > 0) {
		switch(nPos) {
			case 1:		//Nu
				inif >> nu;
				break;
			case 2:
				inif >> init_v;
				break;
			case 3:
				inif >> alfa;
				break;
			case 4:
				inif >> patience;
				break;
			case 5:
				inif >> patience_cycles;
				break;
			case 6:		//NormT
				inif >> word;
				if((nPos = word_pos(sNormT, word)) > 0) norm_t = nPos;
				break;
			case 7:		//SeedT
				inif >> word;
				if((nPos = word_pos(sSeedT, word)) > 0) seed_t = nPos;
				break;
			case 8:		//NormT
				inif >> word;
				if((nPos = word_pos(sECPol, word)) > 0) emptyc_pol = nPos;
				break;
		}
	}

	return true;
}
