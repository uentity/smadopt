#ifndef _ALG_OPT_H
#define _ALG_OPT_H

#include "common.h"
#include <fstream>
#include <algorithm>

//#define OPT_TEMPL template<class optT>

//data-storage class
template<class T> class options {};

//abstract interface for all options classes - most important functions
class _CLASS_DECLSPEC Ialg_options
{
public:
	typedef Ialg_options& iopt_ref;
	typedef Ialg_options* iopt_ptr;
	typedef const Ialg_options& const_iopt_ref;
	typedef const Ialg_options* const_iopt_ptr;
	typedef std::vector<iopt_ptr> vp_iopt;
	//typedef smart_ptr<Ialg_options> sp_iopt;
	//typedef std::vector<sp_iopt> v_sp_iopt;

	//typedef void (*AfterOptionsSet_fcn)();
	//typedef bool (*BeforeEmbOptRead_fcn)();

protected:
	bool is_default_;
	//array of pointers to embedded options
	vp_iopt embOpt_;

	//reads data options
	void read_options(const char* pFName = NULL);

	//commonly used function for reading
	static int word_pos(const std::string& sSrc, const std::string& sWord);

	//processes option during read
	virtual bool process_option(std::istream& inif, std::string& word) = 0;

	virtual void set_embopt_def(iopt_ref iopt) = 0;
	//delegates for reaction in derived classes
	//return whether to call ReadOptions for embedded options chain
	virtual bool BeforeEmbOptRead() = 0;
	virtual void AfterOptionsSet() = 0;

	//static bool BeforeEmbOptRead_def() { return true; }
	//static void AfterOptionsSet_def() {};


public:
	//filename where options are hold
	std::string iniFname_;
	//delegates
	//AfterOptionsSet_fcn AfterOptionsSet;
	//BeforeEmbOptRead_fcn BeforeEmbOptRead;

	//default constructor
	Ialg_options(bool is_def = false);
	Ialg_options(const_iopt_ref iopt);
	//virtual destructor
	virtual ~Ialg_options() {};

	//asks for options type in string form
	virtual const char* get_opt_type() const = 0;

	//sets default options
	virtual void set_def_opt(bool create_defs = true) = 0;
	//fully sets options
	virtual bool set_wrapper_opt(const_iopt_ref iopt) = 0;
	//get pointer to derived options class
	virtual const_iopt_ptr get_wrapper_opt() const = 0;

	//read options from conf file
	virtual void ReadOptions(const char* pFName = NULL);
	//sets options from void pointer
	virtual void SetOptions(const void* pOpt = NULL) = 0;
	//gets pointer to options structure as void pointer
	virtual void* GetOptions() = 0;

	//embded options manipualtions
	void apply_def_opt(const_iopt_ptr iopt);
	//access to embded options
	ulong get_emb_count() const;
	iopt_ptr get_embopt(ulong ind) const;
	iopt_ptr get_embopt(const char* otype, bool get_def = false, ulong seq_num = 0) const;
	//add default embedded options interface
	void add_def_embopt(iopt_ptr iop, bool apply_defaults = true);
	//add embedded options class
	void add_embopt(iopt_ptr iop, bool apply_defaults = true);
	//delete embded options
	ulong rem_embopt(ulong ind);
	ulong rem_embopt(const_iopt_ptr iopt_ptr);
	//read embded options from conf file
	void read_embopt(const char* pFName = NULL);

	void update_embopt_defs();

	//helper interface functions
	iopt_ref get_iopt_ref() {
		return (Ialg_options&)(*this);
	}
	iopt_ptr get_iopt_ptr() {
		return (Ialg_options*)this;
	}
	const_iopt_ref get_iopt_ref() const {
		return (Ialg_options&)(*this);
	}
	const_iopt_ptr get_iopt_ptr() const {
		return (Ialg_options*)this;
	}

	operator iopt_ref() { return get_iopt_ref(); }
	operator iopt_ptr() { return get_iopt_ptr(); }
	operator const_iopt_ref() const { return get_iopt_ref(); }
	operator const_iopt_ptr() const { return get_iopt_ptr(); }
};

template<class opt_class>
class alg_options : public opt_class, public Ialg_options
{
public:
	//typedef options<T> data_opt;
	typedef opt_class data_opt;
	typedef alg_options<data_opt> wrapper_opt;
	typedef std::vector< smart_ptr<Ialg_options> > vsp_deo;	//smart pointers to default embedded objects

protected:
	//array of smart pointers to default embedded objects
	vsp_deo md_embObj;

	virtual void set_embopt_def(iopt_ref) {};
	//delegates for reaction in derived classes
	virtual bool BeforeEmbOptRead() { return true; }
	virtual void AfterOptionsSet() {};

	//virtual bool OnEmbOptAdded(iopt_ref emb_opt) { return true; }

public:
	//inner options structure
	//data_opt opt_;

	//default constructor - allocate options & create reference
	alg_options() {};
	//constructor when reference to main options
	alg_options(data_opt& opt)
		: opt_class(opt) {};
	//copy constructor
	alg_options(const wrapper_opt& opt)
		: opt_class(opt), Ialg_options(opt) {};
	//constructor with interface reference
	alg_options(const_iopt_ref iopt) {
		set_wrapper_opt(iopt);
	}

	//virtual destructor
	virtual ~alg_options() {};

	//asks for options type in string form - static function in opt_class
	virtual const char* get_opt_type() const = 0;

	//sets opt_ structure
	virtual void set_data_opt(const data_opt* p_opt) {
		if(p_opt != NULL && p_opt != get_data_opt()) {
			data_opt::operator=(*p_opt);
			//set_inner_opt(*p_opt);
		}
		AfterOptionsSet();
	}

	const data_opt* get_data_opt() {
		return (data_opt*)this;
		//return &opt_;
	}

	virtual const_iopt_ptr get_wrapper_opt() const {
		return this->get_iopt_ptr();
	}
	virtual bool set_wrapper_opt(const_iopt_ref iopt) {
		if(iopt.get_opt_type() == get_opt_type()) {
			wrapper_opt* pOpt = (wrapper_opt*)iopt.get_wrapper_opt();
			if(pOpt != this)
				*this = *pOpt;
			AfterOptionsSet();
			return true;
		}
		return false;
		//_iniFname = pOpt->_iniFname;
		//set_data_opt(&pOpt->opt_);
	}

	//template for set_wrapper_opt in derived classes
	template<class T>
	bool der_set_wrapper_opt(const_iopt_ref iopt) {
		const T* p_src = dynamic_cast<const T*>(iopt.get_iopt_ptr());
		if(p_src) {
			*((T*)this) = *p_src;
			AfterOptionsSet();
			return true;
		}
		else return wrapper_opt::set_wrapper_opt(iopt);
	}

	//add default embedded options object
	template<class embT>
	void create_def_embobj(const char* stype, bool apply_defaults = true)
	{
		//search if such object already present
		if(get_embopt(stype, true) != NULL)
			return;
		//create object
		smart_ptr<Ialg_options> pEmb(new embT);
		//embT* pEmb = new embT;
		pEmb->set_def_opt();
		if(apply_defaults)
			set_embopt_def(pEmb->get_iopt_ref());
		//add it to storage
		md_embObj.push_back(pEmb);
		//add to options chain
		add_def_embopt(pEmb->get_iopt_ptr(), false);
	}

	//sets options structure from pointer
	virtual void SetOptions(const void* pOpt = NULL) {
		set_data_opt((data_opt*)pOpt);
	}
	//get options structure void pointer
	virtual void* GetOptions() {
		return (void*)(opt_class*)this;
		//return &opt_;
	}
};

#endif //_ALG_OPT
