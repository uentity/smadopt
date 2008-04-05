#include "alg_except.h"

using namespace std;

//string alg_except::s_errFname = "err.txt";

const char* alg_except::explain_error(int code) 
{
	switch(code) {
		case NoError:
			return "No Error";
		case InvalidParameter:
			return "Invalid parameter specified";
		default:
			return "Unknown error!";
	}
}
