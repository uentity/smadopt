#ifndef __GRAY_CODE_H__
#define __GRAY_CODE_H__

#include "matrix.h"

template< class src_mat_t, class dst_elem_t >
TMatrix< dst_elem_t > gray_code(const src_mat_t& src, ulong bits_num, const dst_elem_t level_0 = 0, const dst_elem_t level_1 = 1) {
	typedef TMatrix< dst_elem_t > dst_mat_t;
	dst_mat_t res(bits_num, src.col_num());
	dst_mat_t code(bits_num, 1);

	for(ulong i = 0; i < src.col_num(); ++i) {
		ulong mask = 1 << (bits_num - 1);
		ulong val = static_cast< ulong >(src[i]);
		// extract bits
		bool prev_bit = false;
		for(ulong j = 0; j < bits_num; ++j) {
			if(bool(val & mask) ^ prev_bit) {
				code[j] = level_1;
				prev_bit = true;
			}
			else {
				code[j] = level_0;
				prev_bit = false;
			}
			mask >>= 1;
		}
		// save to result
		res.SetColumns(code, i);
	}

	return res;
}

// everuthing below level_1 is accepted as 0
template< class src_mat_t, class dst_elem_t >
TMatrix< dst_elem_t > gray_decode(const src_mat_t& src, const typename src_mat_t::value_type level_1 = 1) {
	typedef TMatrix< dst_elem_t > dst_mat_t;
	dst_mat_t res(1, src.col_num()), code;
	ulong bits_num = src.row_num();

	for(ulong i = 0; i < src.col_num(); ++i) {
		code <<= src.GetColumns(i);
		ulong val = 0;
		ulong mask = 1 << (bits_num - 1);
		bool prev_bit = false, cur_bit;
		for(ulong j = 0; j < bits_num; ++j) {
			cur_bit = bool(code[j] >= level_1);
			if(cur_bit ^ prev_bit)
				val += mask;
			prev_bit = cur_bit;
			mask >>= 1;
		}
		res[i] = val;
	}

	return res;
}

#endif

