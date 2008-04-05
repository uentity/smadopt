#include "nn_common.h"

using namespace std;
using namespace NN;

//string nn_except::s_buf;

const char* nn_except::explain_error(int code)
{
	switch(code) {
		case InvalidLayer:
			return "Invalid layer number specified";
		case NoInputSize:
			return "Input size not specified";
		case SizesMismatch:
			return "Specified and required data sizes mismatch";
		case NN_Busy:
			return "Network is busy";
		default:
			return alg_except::explain_error(code);
	}
}
