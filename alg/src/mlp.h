#include "objnet.h"
//#ifdef _UNIX
#include <stdarg.h>
//#endif

using namespace std;
using namespace NN;

//-------------------------------mlp implementation-------------------------------------------
//constructors
mlp::mlp(ulong layers_num, ulong inp_size, ...) :
	objnet()
{
	set_input_size(inp_size);
	va_list varlist;
	va_start(varlist, inp_size);
	for(ulong i=0; i<layers_num; ++i)
		add_layer(va_arg(varlist, ulong), va_arg(varlist, int));
	va_end(varlist);

	set_def_opt(false);
}

bp_layer& mlp::add_layer(ulong neurons_count, int af_type)
{
	try {
		if(layers_num() == 0 && inp_size() == 0) throw nn_except(NoInputSize);
		bp_layer& l = objnet::add_layer<bp_layer>(neurons_count, af_type);
		//fully connect to previous layer
		if(layers_num() > 1)
			l.set_links(create_ptr_mat(layers_[layers_num() - 2].neurons()));
		else
			l.set_links(create_ptr_mat(input_.neurons()));
		return l;
	}
	catch(alg_except& ex) {
		_print_err(ex.what());
		throw;
	}
}

bool mlp::set_layer(ulong layer_ind, ulong neurons_count, int af_type)
{
	bool res = false;
	try {
		if(layer_ind >= layers_num()) throw nn_except(InvalidLayer);
		if(neurons_count == 0) throw nn_except(InvalidParameter, "Neurons number must be > 0");

		if(layer_ind == 0 && inp_size() == 0) throw nn_except(NoInputSize);
		else
			layers_[layer_ind].init(neurons_count, af_type);
		if(layer_ind + 1 < layers_num())
			layers_[layer_ind + 1].set_links(create_ptr_mat(layers_[layer_ind].neurons()));

		res = true;
	}
	catch(alg_except& ex) {
		_print_err(ex.what());
		throw;
	}
	return res;
}

bool mlp::set_layer_type(ulong layer_ind, int af_type)
{
	if(layer_ind >= layers_num()) return false;
	layers_[layer_ind].set_af(af_type);
	return true;
}
