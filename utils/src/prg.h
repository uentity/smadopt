#ifndef _PRG_H
#define _PRG_H

#include "common.h"

namespace prg 
{
	enum prg_type {
		standart = 0,
		combined_lcg = 1,
		shift_reg = 2,
		xaoc = 3
	};
	const prg_type def_base_rg = combined_lcg;
	const prg_type def_new_rg = combined_lcg;

	//init whole prg subsystem
	_LIBAPI void init();
	//manipulating with prg streams
	_LIBAPI unsigned int create_stream(int rg_type = def_new_rg);
	_LIBAPI void init_stream(unsigned int nStream);
	_LIBAPI bool change_stream(unsigned int nStream, int new_rg_type);
	_LIBAPI bool switch_stream(unsigned int nStream);
	_LIBAPI void prepare_streams(unsigned int nStreams, int rg_type = def_new_rg);
	_LIBAPI unsigned int streams_count();

	//generate random numbers
	_LIBAPI double rand01();
	_LIBAPI double randn(double mu = 0, double q2 = 1);
	_LIBAPI long randInt(long a, long b);
	_LIBAPI unsigned long randIntUB(unsigned long upper_bound);

	// generate random permutation of sequence 0, 2, ..., n - 1
	_CLASS_DECLSPEC std::vector< ulong > rand_perm(ulong n);
	
	//prg classes
	class prg_store;
	class randgen;
	class shiftreg_rg;
	class xaoc_gen;
	class combinedLCG;
}

#endif	//_PRG_H
