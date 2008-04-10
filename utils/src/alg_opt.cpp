#include "alg_opt.h"
#include <string.h>

Ialg_options::Ialg_options(bool is_def)
	: is_default_(is_def)
	//,AfterOptionsSet(AfterOptionsSet_def), BeforeEmbOptRead(BeforeEmbOptRead_def)
{
}

Ialg_options::Ialg_options(const_iopt_ref iopt)
	: is_default_(iopt.is_default_), embOpt_(iopt.embOpt_)
	//,AfterOptionsSet(iopt.AfterOptionsSet), BeforeEmbOptRead(iopt.BeforeEmbOptRead)
{
}

int Ialg_options::word_pos(const std::string& sSrc, const std::string& sWord)
{
	size_t nPos, nCur;
	nCur = 0;
	int nCnt = 0;
	if((nPos = sSrc.find(sWord)) != sSrc.npos) {
		while((nCur = sSrc.find(' ', nCur)) != sSrc.npos) {
			++nCnt;
			++nCur;
			if(nCur == nPos) break;
		}
	}
	return nCnt;
}

//reads inner options
void Ialg_options::read_options(const char* pFName)
{
	using namespace std;

	if(pFName && *pFName != 0) iniFname_ = pFName;

	ifstream inif(iniFname_.c_str());

	string sWord, sTmp;
	//std::istringstream is;
	while(inif >> sWord) {
		if(sWord.size() == 0 || sWord.find(';') == 0) {
			inif >> ignoreLine;
			continue;
		}
		inif >> sTmp;
		if(sTmp[0] != '=') {
			inif >> ignoreLine;
			continue;
		}

		if(process_option(inif, sWord))
			inif >> ignoreLine;
	}
}

void Ialg_options::apply_def_opt(const_iopt_ptr iopt)
{
	if(iopt == NULL || iopt->get_opt_type() != get_opt_type()) return;
	set_wrapper_opt(iopt->get_iopt_ref());
	//copy references to default objects
	for(vp_iopt::const_iterator src(iopt->embOpt_.begin()); src != iopt->embOpt_.end(); ++src) {
		if((*src)->is_default_) {
			add_def_embopt((*src)->get_iopt_ptr(), false);
			//apply defaults to embedded options
			for(vp_iopt::iterator dst(embOpt_.begin()); dst != embOpt_.end(); ++dst) {
				if(!(*dst)->is_default_ && (*dst)->get_opt_type() == (*src)->get_opt_type())
					(*dst)->apply_def_opt((*src)->get_iopt_ptr());
			}
		}
	}
}

void Ialg_options::ReadOptions(const char* pFName)
{
	read_options(pFName);
	//call delegate
	AfterOptionsSet();
	if(BeforeEmbOptRead()) read_embopt(pFName);
}

void Ialg_options::read_embopt(const char*)
{
	for(vp_iopt::iterator pos(embOpt_.begin()); pos != embOpt_.end(); ++pos)
		(*pos)->ReadOptions();
}

//embded options manipualtions
//access to embded options
ulong Ialg_options::get_emb_count() const {
	return static_cast<ulong>(embOpt_.size());
}
Ialg_options::iopt_ptr Ialg_options::get_embopt(ulong ind) const
{
	if(ind >= embOpt_.size()) return NULL;
	else return embOpt_[ind];
}

Ialg_options::iopt_ptr Ialg_options::get_embopt(const char* otype, bool get_def, ulong seq_num) const
{
	ulong _num = 0;
	//iopt_ptr pRet = NULL;
	for(vp_iopt::const_iterator pos(embOpt_.begin()); pos != embOpt_.end(); ++pos) {
		if(strcmp((*pos)->get_opt_type(), otype) == 0) {
			//pRet = pos->get_ptr();
			if(get_def) {
				if((*pos)->is_default_)
					return (*pos);
			}
			else if(_num++ == seq_num)
				return (*pos);
		}
	}
	return NULL;
}

//add default embedded options interface
void Ialg_options::add_def_embopt(iopt_ptr iop, bool apply_defaults)
{
	if(iop == NULL) return;
	iop->is_default_ = true;
	if(apply_defaults) set_embopt_def(iop->get_iopt_ref());
	if(embOpt_.size() > 0) {
		for(vp_iopt::iterator pos(embOpt_.begin()); pos != embOpt_.end(); ++pos) {
			if((*pos)->get_opt_type() == iop->get_opt_type() && (*pos)->is_default_) {
				embOpt_.erase(pos);
				break;
			}
		}
	}
	embOpt_.push_back(iop);
	//return iop->get_iopt_ref();
}

//add embedded options class
void Ialg_options::add_embopt(iopt_ptr iop, bool apply_defaults)
{
	if(iop == NULL) return;
	//check if that options already in the list
	if(find(embOpt_.begin(), embOpt_.end(), iop) == embOpt_.end()) {
		embOpt_.push_back(iop);
		//call delegate
		//if(OnEmbOptAdded(iop->get_iopt_ref())) {
		if(apply_defaults) {
			//search for default options
			iopt_ptr pdOpt = get_embopt(iop->get_opt_type(), true);
			//set default options
			if(pdOpt != NULL) iop->apply_def_opt(pdOpt);
			else set_embopt_def(iop->get_iopt_ref());
		}
	}
	//return iop->get_iopt_ref();
}

//delete embded options
ulong Ialg_options::rem_embopt(ulong ind) {
	embOpt_.erase(embOpt_.begin() + ind);
	return static_cast<ulong>(embOpt_.size());
}

ulong Ialg_options::rem_embopt(const_iopt_ptr iopt_ptr) {
	for(vp_iopt::iterator pos(embOpt_.begin()); pos != embOpt_.end(); ++pos) {
		if((*pos)->get_iopt_ptr() == iopt_ptr) {
			embOpt_.erase(pos);
			break;
		}
	}
	return static_cast<ulong>(embOpt_.size());
}

void Ialg_options::update_embopt_defs() {
	for(vp_iopt::iterator pos(embOpt_.begin()); pos != embOpt_.end(); ++pos) {
		if((*pos)->is_default_)
			set_embopt_def((*pos)->get_iopt_ref());
	}
}

/*
ulong Ialg_options::rem_emb_opt(const char* otype, ulong seq_num = 0) {
	ulong _num = 0;
	for(v_sp_iopt::iterator pos(embOpt_.begin()); pos != embOpt_.end(); ++pos) {
		if(strcmp((*pos)->get_opt_type(), otype) == 0) {
			if(_num == seq_num) {
				embOpt_.erase(pos);
				break;
			}
			++_num;
		}
	}
	return embOpt_.size();
}
*/
