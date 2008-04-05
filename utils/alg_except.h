#ifndef _ALG_EXCEPT_H
#define _ALG_EXCEPT_H

#include "common.h"
#include <ostream>

enum common_error_codes {
	NoError = 0,
	InvalidParameter = 1,
	UnknownError = -1
};

class _CLASS_DECLSPEC alg_except
{
protected:
	std::string _what;
	int _code;

	//static std::string s_errFname;

public:
	alg_except() 
		: _what(""), _code(0)
	{
	}
	alg_except(const alg_except& ex) 
		: _what(ex._what), _code(ex._code)
	{
	}
	alg_except(int code, const char* what)
		: _what(what), _code(code)
	{
	}
	alg_except(int code)
		: _what(explain_error(code)), _code(code)
	{
	}
	alg_except(const char* what)
		: _what(what), _code(-1)
	{
	}
	virtual ~alg_except() {
	}

	virtual const char* what() const throw() {
		return _what.c_str();
	}

	int code() const throw() {
		return _code;
	}

	static const char* explain_error(int code);

	virtual void print_err(std::ostream& err_strm, const char* what = NULL) const throw() {
		if(what) err_strm << what;
		else err_strm << _what;
	}
};

#endif
