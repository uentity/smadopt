#if defined(_MSC_INLINE) || !defined(_MSC_VER)

#ifndef _MSC_INLINE
#include "matrix.h"
#endif

template<>
TMatrix<bool>::TMatrix(size_type rows, size_type cols, cbuf_pointer ptr) :
	data_(new buf_type(size_ = rows*cols))
{
	rows_ = cols_ = 0;
	if(size_ > 0) {
		rows_ = rows; cols_ = cols;
		alloc_ = INNER;
		if(ptr != NULL) {
			//copy - element by element for vector<bool>
			//std::copy(ptr, ptr + size_, buf_begin());
			for(buf_iterator pos = buf_begin(); pos != buf_end(); ++pos) {
				*pos = *ptr;
				++ptr;
			}
		}
	}
}

template<>
void TMatrix<bool>::NewMatrix(size_type rows, size_type cols, cbuf_pointer ptr)
{
	NewMatrix(rows, cols);
	if(ptr != NULL) {
		//copy - element by element for vector<bool>
		for(buf_iterator pos = buf_begin(); pos != buf_end(); ++pos) {
			*pos = *ptr;
			++ptr;
		}
	}
}

template<>
TMatrix<bool>::cbuf_pointer TMatrix<bool>::GetBuffer() const
{
	//always return NULL for vector<bool>
	return NULL;
}

template<>
TMatrix<bool>::buf_pointer TMatrix<bool>::GetBuffer()
{
	//always return NULL for vector<bool>
	return NULL;
}

template<>
void TMatrix<bool>::SetBuffer(cbuf_pointer pBuf)
{
	//copy - element by element for vector<bool>
	for(buf_iterator pos = buf_begin(); pos != buf_end(); ++pos) {
		*pos = *pBuf;
		++pBuf;
	}
}

#endif
