#include "nn_addon.h"

using namespace std;
using namespace GA;
using namespace NN;
using namespace KM;

nna_opt::nna_opt(nn_addon* self) : self_(self) {
};
//copy constructor
nna_opt::nna_opt(const nna_opt& opt) : wrapper_opt(opt), self_(opt.self_) {}
//constructor by interface
nna_opt::nna_opt(const Ialg_options& opt) : wrapper_opt(opt) {
	const nna_opt* p_go = dynamic_cast< const nna_opt* >(opt.get_iopt_ptr());
	if(p_go)
		self_ = p_go->self_;
	else throw ga_except("Invalid nna_opt initialization with wrong object");
}

void nna_opt::AfterOptionsSet()
{
	if(self_) self_->AfterOptionsSet();
}

void nna_opt::set_def_opt(bool create_defs)
{
	iniFname_ = "nn_addon.ini";
	name = "hnna";

	bestCount = 200;
	normType = LogsigNorm;
	maxtar = 0.95;
	mintar = 0.1;
	pred_ratio = 0.05;
	is_ffRestricted = false;
	inf_ff = -1;
	sup_ff = -1;

	normInp = false;
	initNetEveryIter = true;
	usePCA = false;
	initPCAEveryIter = false;
	PC_num = 0;

	netType = rb_nn;
	addon_scheme = FitFcnAsTarget;
	goalQuota = 0.01;
	tol = TOL;
	kmfec_policy = KM::singleton;
	samples_filter = kmeans_filter;
	search_clust_num = 3; search_samples_num = 0;
	learn_clust_num = 0;
	kmf_cmult = 0.2;
	rbn_cmult = 0.1;
	rbn_learn_type = rbn_kmeans_bp;
	clustEngine = ce_kmeans;

	//string a_name = GetName();

	if(create_defs) {
		create_def_embobj<ga_opt>("ga_opt");
		create_def_embobj<mnn_opt>("mnn_opt");
		create_def_embobj<nn_opt>("nn_opt");
		create_def_embobj<km_opt>("km_opt");
	}
	////set default inner ga options
	//_pdIGAo = new ga_opt;
	//_pdIGAo->set_def_opt();
	//pInnerGAOpt = &_pdIGAo->opt_;
	//_pdIGAo->generations = 200;
	////_pdIGAo->_initRange = caller._initRange;
	//_pdIGAo->stallGenLimit = 20;
	//_pdIGAo->mutProb = 0.1;
	//_pdIGAo->minUnique = 0;
	//_pdIGAo->_logFname = a_name + "_ga_nn_log.txt";
	//_pdIGAo->_histFname = a_name + "_ga_nn_hist.txt";
	//_pdIGAo->_errFname = a_name + "_ga_nn_err.txt";
	//_pdIGAo->iniFname_ = "nn_addon_ga.ini";
	//add_def_embopt(_pdIGAo);

	////set default matrix NN options
	//_pdMNNo = new mnn_opt;
	//_pdMNNo->set_def_opt();
	//_pdMNNo->goal = 0.01;
	//_pdMNNo->maxCycles = 200;
	//_pdMNNo->nu = 0.5;
	//_pdMNNo->mu = 0.1;
	//_pdMNNo->wiRange = 0.01;
	//_pdMNNo->adaptive = false;
	//_pdMNNo->normInp = false;
	//_pdMNNo->showPeriod = 1;
	//_pdMNNo->batch = true;
	//_pdMNNo->saturate = true;
	//_pdMNNo->learnFun = R_BP;
	//_pdMNNo->_errFname = a_name + "_" + _pdMNNo->_errFname;
	//_pdMNNo->iniFname_ = "nn.ini";
	//add_def_embopt(_pdMNNo);

	////set object NN options
	//_pdNNo = new nn_opt;
	//_pdNNo->set_def_opt();
	//_pdNNo->goal = 0.01;
	//_pdNNo->maxCycles = 200;
	//_pdNNo->nu = 0.5;
	//_pdNNo->mu = 0.1;
	//_pdNNo->wiRange = 0.01;
	//_pdNNo->adaptive = false;
	//_pdNNo->showPeriod = 1;
	//_pdNNo->batch = true;
	//_pdNNo->saturate = true;
	//_pdNNo->learnFun = R_BP;
	//_pdNNo->_errFname = a_name + "_" + _pdNNo->_errFname;
	//_pdNNo->iniFname_ = "nn.ini";
	//add_def_embopt(_pdNNo);

	//set default NN size
	_setDefNNSize();

	//set pointers
	//GetOptions();
}

bool nna_opt::set_wrapper_opt(const_iopt_ref iopt)
{
	if(wrapper_opt::set_wrapper_opt(iopt)) {

		nna_opt& opt = *(nna_opt*)iopt.get_wrapper_opt();
		//_pdIGAo = opt._pdIGAo;
		//_pdMNNo = opt._pdMNNo;
		//_pdNNo = opt._pdNNo;

		layers_ = opt.layers_;
		lTypes_ = opt.lTypes_;
		return true;
	}
	return false;
}

void nna_opt::set_embopt_def(iopt_ref emb_opt)
{
	//string a_name = GetName();

	//emb_opt.set_def_opt();
	if(std::string(emb_opt.get_opt_type()) == "ga_opt") {
		//set inner ga options
		ga_opt* pGAo = (ga_opt*)emb_opt.get_wrapper_opt();
		pInnerGAOpt = pGAo;
		pGAo->generations = 200;
		//pGAo->_initRange = caller._initRange;
		pGAo->stallGenLimit = 20;
		pGAo->mutProb = 0.1;
		pGAo->minUnique = 0;
		pGAo->logFname = name + "_ga_nn_log.txt";
		pGAo->histFname = name + "_ga_nn_hist.txt";
		pGAo->errFname = name + "_ga_nn_err.txt";
		pGAo->statFname = name + "_ga_nn_stat.txt";
		pGAo->iniFname_ = "nn_addon_ga.ini";
		emb_opt.update_embopt_defs();
	}
	else if(std::string(emb_opt.get_opt_type()) == "mnn_opt") {
		mnn_opt* pMNNo = (mnn_opt*)emb_opt.get_wrapper_opt();
		pMNNo->goal = 0.01;
		pMNNo->maxCycles = 200;
		pMNNo->nu = 0.5;
		pMNNo->mu = 0.1;
		pMNNo->wiRange = 0.01;
		pMNNo->adaptive = false;
		pMNNo->normInp = false;
		pMNNo->showPeriod = 1;
		pMNNo->batch = true;
		pMNNo->saturate = true;
		pMNNo->learnFun = R_BP;
		pMNNo->errFname_ = name + "_" + pMNNo->errFname_;
		pMNNo->iniFname_ = "nn.ini";
		emb_opt.update_embopt_defs();
	}
	else if(std::string(emb_opt.get_opt_type()) == "nn_opt") {
		nn_opt* pNNo = (nn_opt*)emb_opt.get_wrapper_opt();
		pNNo->set_def_opt();
		pNNo->goal = 0.01;
		pNNo->maxCycles = 200;
		pNNo->nu = 0.5;
		pNNo->mu = 0.1;
		pNNo->wiRange = 0.01;
		pNNo->adaptive = false;
		pNNo->showPeriod = 1;
		pNNo->batch = true;
		pNNo->saturate = true;
		pNNo->learnFun = R_BP;
		pNNo->errFname_ = name + "_" + pNNo->errFname_;
		pNNo->iniFname_ = "nn.ini";
		emb_opt.update_embopt_defs();
	}
	else if(std::string(emb_opt.get_opt_type()) == "km_opt") {
		km_opt& kmo = *(km_opt*)emb_opt.get_wrapper_opt();
		pKMopt = &kmo;
		//read options from the same file
		kmo.iniFname_ = iniFname_;
		kmo.emptyc_pol = singleton;
		emb_opt.update_embopt_defs();
	}
}

/*
void nna_opt::AfterEmbOptAdded(iopt_ref emb_opt)
{
	string a_name = GetName();

	if(emb_opt.get_opt_type() == "ga_opt") {
	//set inner ga options
		ga_opt& iga_opt = *(ga_opt*)emb_opt.get_opt();
		pInnerGAOpt = &iga_opt.opt_;
		iga_opt.generations = 200;
		//iga_opt._initRange = caller._initRange;
		iga_opt.stallGenLimit = 20;
		iga_opt.mutProb = 0.1;
		iga_opt.minUnique = 0;
		iga_opt._logFname = a_name + "_ga_nn_log.txt";
		iga_opt._histFname = a_name + "_ga_nn_hist.txt";
		iga_opt._errFname = a_name + "_ga_nn_err.txt";
		iga_opt.iniFname_ = "nn_addon_ga.ini";
	}
	else if(emb_opt.get_opt_type() == "mnn_opt") {
		//set matrix NN options
		mnn_opt& mnno = *(mnn_opt*)emb_opt.get_opt();
		pNNetOpt = &mnno.opt_;
		mnno.goal = 0.01;
		mnno.maxCycles = 200;
		mnno.nu = 0.5;
		mnno.mu = 0.1;
		mnno.wiRange = 0.01;
		mnno.adaptive = false;
		mnno.normInp = false;
		mnno.showPeriod = 1;
		mnno.batch = true;
		mnno.saturate = true;
		mnno.learnFun = R_BP;
		mnno._errFname = a_name + "_" + mnno._errFname;
	}
	else if(emb_opt.get_opt_type() == "nn_opt") {
		//set object NN options
		nn_opt& nno = *(nn_opt*)emb_opt.get_opt();
		pNewNNetOpt = &nno.opt_;
		nno.goal = 0.01;
		nno.maxCycles = 200;
		nno.nu = 0.5;
		nno.mu = 0.1;
		nno.wiRange = 0.01;
		nno.adaptive = false;
		nno.showPeriod = 1;
		nno.batch = true;
		nno.saturate = true;
		nno.learnFun = R_BP;
		nno._errFname = a_name + "_" + nno._errFname;
	}
	else if(emb_opt.get_opt_type() == "km_opt") {
		//no special options for kmeans
		pKMopt = (kmOptions*)emb_opt.GetOptions();
	}
}
*/

void nna_opt::set_data_opt(const data_opt* pOpt)
{
	if(pOpt == NULL) return;
	wrapper_opt::set_data_opt(pOpt);
	//if(pOpt != &opt_) opt_ = *pOpt;

	if(pOpt->layers_num > 0 && pOpt->pNeurons && pOpt->pLayerTypes) {
		if(pOpt->layers_num != layers_.size()) {
			layers_.resize(pOpt->layers_num);
			lTypes_.resize(pOpt->layers_num);
		}
		if(pOpt->pNeurons != &layers_[0])
			memcpy(&layers_[0], pOpt->pNeurons, layers_num*sizeof(ulong));
		if(pOpt->pLayerTypes != &lTypes_[0])
			memcpy(&lTypes_[0], pOpt->pLayerTypes, layers_num*sizeof(int));
	}

	iopt_ptr piOpt;
	//set inner GA options
	if((piOpt = get_embopt("ga_opt"))) piOpt->SetOptions(pOpt->pInnerGAOpt);
	//set matrix NN options
	if((piOpt = get_embopt("mnn_opt"))) piOpt->SetOptions(pOpt->pNNetOpt);
	//set NN options
	if((piOpt = get_embopt("nn_opt"))) piOpt->SetOptions(pOpt->pNewNNetOpt);
	//set kmeans options
	//TODO : enable back - disabled for old PPP
	//if(piOpt = get_embopt("km_opt")) piOpt->SetOptions(pOpt->pKMopt);

	//if(get_nn_opt() != NULL) get_nn_opt()->set_inner_opt(pOpt->pNewNNetOpt);
	//if(get_km_opt() != NULL) get_km_opt()->set_inner_opt(pOpt->pKMopt);

	//set structure pointers
	GetOptions();

	//call delegate
	//AfterOptionsSet();
}

void nna_opt::_setDefNNSize()
{
	layers_num = 3;
	layers_.resize(layers_num);
	layers_[0] = 20;
	layers_[1] = 20;
	layers_[2] = 1;
	lTypes_.resize(layers_num);
	lTypes_[0] = logsig;
	lTypes_[1] = logsig;
	lTypes_[2] = logsig;
}

void* nna_opt::GetOptions()
{
	iopt_ptr piOpt;
	//set GA options pointer
	pInnerGAOpt = NULL;
	if((piOpt = get_embopt("ga_opt")))
		pInnerGAOpt = (gaOptions*)piOpt->GetOptions();
	//set matrix NN options pointer
	pNNetOpt = NULL;
	if((piOpt = get_embopt("mnn_opt")))
		pNNetOpt = (nnOptions<MNet>*)piOpt->GetOptions();
	//set object NN options pointer
	pNewNNetOpt = NULL;
	if((piOpt = get_embopt("nn_opt")))
		pNewNNetOpt = (new_nnOptions*)piOpt->GetOptions();
	//set kmeans options pointer
	pKMopt = NULL;
	if((piOpt = get_embopt("km_opt")))
		pKMopt = (kmOptions*)piOpt->GetOptions();

	//pNNetOpt = (nnOptions*)get_embopt(1).GetOptions();
	//pNewNNetOpt = (new_nnOptions*)get_embopt(2).GetOptions();
	//pKMopt = (kmOptions*)get_embopt(3).GetOptions();

	pNeurons = &layers_[0];
	pLayerTypes = &lTypes_[0];
	return (void*)this;
}

bool nna_opt::process_option(std::istream& inif, std::string& word)
{
	string sOpts = " TarNormType MaxTar MinTar PredictionRatio InfFF SupFF BestCount";
	sOpts += " UsePCA NormalizeInput InitNetEveryIter InitPCAEveryIter PrinCompNum Layers AddonScheme GoalQuota IsFFRestricted";
	sOpts += " NetworkType SamplesFilter SearchSamplesNum SearchClusterNum LearnClusterNum KMFECPolicy";
	sOpts += " KMFCmult RBNCmult RBNLearnType ClusteringEngine";
	string sTarNorm = " UnchangedData LinearNorm LogNorm LogsigNorm";
	string sAddonScheme = " FitnessFcnAsTargets FitnessFcnAsInputs PCAPredict";
	string sNetType = " matrix_nn mlp_nn rb_nn ccn_nn";
	string sSamplesFlt = " BestFilter KmeansFilter BestKmeansFilter";
	string sKmPol = " DoNothing Drop Singleton";
	string sRBNLearnT = " Exact FullyBackProp KmeansBackProp";
	string clust_engine = " kmeans DA";

	string sTmp;
	Matrix tmp;
	int nPos;
	bool need_skipline = true;
	if((nPos = word_pos(sOpts, word)) > 0) {
		switch(nPos) {
			case 1:
				inif >> word;
				if((nPos = word_pos(sTarNorm, word)) > 0) normType = nPos;
				break;
			case 2:
				inif >> maxtar;
				break;
			case 3:
				inif >> mintar;
				break;
			case 4:
				inif >> pred_ratio;
				break;
			case 5:
				inif >> inf_ff;
				break;
			case 6:
				inif >> sup_ff;
				break;
			case 7:
				inif >> bestCount;
				break;
			case 8:
				inif >> usePCA;
				break;
			case 9:
				inif >> normInp;
				break;
			case 10:
				inif >> initNetEveryIter;
				break;
			case 11:
				inif >> initPCAEveryIter;
				break;
			case 12:
				inif >> PC_num;
				break;
			case 13:		//layers definition
				inif >> ignoreLine;
				tmp = Matrix::Read(inif, 2);
				//copy to inner vectors
				layers_num = tmp.col_num();
				layers_.resize(layers_num);
				lTypes_.resize(layers_num);
				for(ulong i=0; i<layers_num; ++i) {
					layers_[i] = tmp(0, i);
					lTypes_[i] = tmp(1, i);
				}
				need_skipline = false;
				break;
			case 14:
				inif >> word;
				if((nPos = word_pos(sAddonScheme, word)) > 0) addon_scheme = nPos;
				break;
			case 15:
				inif >> goalQuota;
				break;
			case 16:
				inif >> is_ffRestricted;
				break;
			case 17:
				inif >> word;
				if((nPos = word_pos(sNetType, word)) > 0) netType = nPos;
				break;
			case 18:		//SamplesFilter
				inif >> word;
				if((nPos = word_pos(sSamplesFlt, word)) > 0) samples_filter = nPos;
				break;
			case 19:
				inif >> search_samples_num;
				break;
			case 20:
				inif >> search_clust_num;
				break;
			case 21:
				inif >> learn_clust_num;
				break;
			case 22:
				inif >> word;
				if((nPos = word_pos(sKmPol, word)) > 0) kmfec_policy = nPos;
				break;
			case 23:
				inif >> kmf_cmult;
				break;
			case 24:
				inif >> rbn_cmult;
				break;
			case 25:
				inif >> word;
				if((nPos = word_pos(sRBNLearnT, word)) > 0) rbn_learn_type = nPos;
				break;
			case 26:
				inif >> word;
				if((nPos = word_pos(clust_engine, word)) > 0) clustEngine = nPos;
				break;
		}	//main options
	}
	return need_skipline;
}

void nna_opt::ReadOptions(const char* pFname)
{
	wrapper_opt::ReadOptions(pFname);
	//set pointers
	GetOptions();

	/*
	//read nn_addon options
	wrapper_opt::ReadOptions(pFname);

	//read inner GA options
	get_iga_opt()->ReadOptions();

	//read NN options
	get_mnn_opt()->ReadOptions();
	if(get_nn_opt() != NULL)
		get_nn_opt()->ReadOptions();

	//read kmeans options
	if(get_km_opt() != NULL)
		get_km_opt()->ReadOptions();

	//set pointers
	GetOptions();
	//process options
	SetOptions(&opt_);
	*/
}

