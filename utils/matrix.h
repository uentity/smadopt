#ifndef _MATRIX_H
#define _MATRIX_H

#include "mbuf_traits.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <sstream>
#include <iosfwd>
//#include <ostream>
#include <iomanip>
/*
#ifdef _DEBUG
#include <iostream>
#endif
*/

#define INNER    0      //   
#define EXTERN   1
#define DEF_WIDTH 14
//#define USE_ROWS_AR

#define TMATRIX(T, buf_traits) TMatrix< T, buf_traits >
#define TEMPLATE_PARAM template< class Tr, template< class > class r_buf_traits >
#define PARAM_MATRIX TMatrix< Tr, r_buf_traits >

#ifdef _WIN32
//disable some annoying warnings
#pragma warning(push)
#pragma warning(disable: 4251 4146 4018 4267 4244)
#endif

//matrix-level transform - <algorithm> extension
template<class matrix1, class unary_func>
void transform(matrix1& m, unary_func f)
{
	transform(m.begin(), m.end(), m.begin(), f);
}

template<class matrix1, class matrix2, class binary_func>
void transform(matrix1& m1, matrix2& m2, binary_func f)
{
	//result stored in m1
	transform(m1.begin(), m1.end(), m2.begin(), m1.begin(), f);
}

//-----------------------------------TMatrix template begin-------------------------------------------------
//template<typename Tl, typename Tr, typename l_buf_traits, typename r_buf_traits>
//struct assign_mat;

template<class T, template <class> class buf_traits_type = val_buffer>
class TMatrix 
#ifdef _WIN32
	: public CHeapManage< TMatrix<T, buf_traits_type> >
#endif
{
	friend struct assign_mat;

public:

	//----public typedefs section begin
	typedef buf_traits_type<T> buf_traits;
	//values typedefs
	typedef typename buf_traits::value_type value_type;
	typedef typename buf_traits::reference reference;
	typedef typename buf_traits::const_reference const_reference;
	typedef typename buf_traits::pointer pointer;
	typedef typename buf_traits::const_pointer const_pointer;
	typedef typename buf_traits::r_iterator r_iterator;
	typedef typename buf_traits::cr_iterator cr_iterator;

	//buffer elements typedefs
	typedef typename buf_traits::buf_type buf_type;
	typedef typename buf_traits::buf_value_type buf_value_type;
	typedef typename buf_traits::buf_reference buf_reference;
	typedef typename buf_traits::cbuf_reference cbuf_reference;
	typedef typename buf_traits::buf_pointer buf_pointer;
	typedef typename buf_traits::cbuf_pointer cbuf_pointer;
	typedef typename buf_traits::buf_iterator buf_iterator;
	typedef typename buf_traits::cbuf_iterator cbuf_iterator;

	typedef typename buf_traits::size_type size_type;
	typedef typename buf_traits::diff_type diff_type;

	typedef typename buf_traits::sp_buf sp_buf;
	typedef typename buf_traits::const_sp_buf const_sp_buf;

	//this & return matrices typedefs
	typedef TMatrix< T, buf_traits_type > this_t;
	typedef TMatrix< T, val_buffer > retMatrix;
	//indexes typedefs
	typedef TMatrix< size_type > indMatrix;
	typedef std::vector< size_type > ind_vec;
	typedef typename indMatrix::r_iterator ind_iterator;
	typedef typename ind_vec::iterator indv_iterator;
	typedef smart_ptr< ind_vec > sp_ind_vec;

private:
	typedef typename retMatrix::r_iterator retr_iterator;
	typedef typename retMatrix::cr_iterator cretr_iterator;

	//commonly used matrices
	typedef TMatrix< double, val_buffer > Matrix;
	typedef std::auto_ptr< Matrix > dMPtr;
	typedef typename std::vector< double >::iterator dr_iterator;

	//data members
	sp_buf _pData;

	int alloc_;
	size_type size_;
	size_type rows_;
	size_type cols_;

	template<class it>
	std::ostream& _print(std::ostream& outs, bool delimRows, int num_width) const
	{
		if(num_width == 0) num_width = DEF_WIDTH;
		it pos(_pData->begin());
		for(size_type i=0; i<rows_; ++i) {
			for(size_type j=0; j<cols_; ++j) {
				outs.width(num_width);
				outs << *pos << ' ';
				++pos;
			}
			if(delimRows) outs << std::endl;
		}
		return outs;
	}


public:

	//--------------------------column iterator for matrix begin
	template< class it_type = r_iterator, class matrix_t = this_t >
	class column_iterator
		: public std::iterator< std::random_access_iterator_tag, typename it_type::value_type, typename it_type::difference_type, 
		  typename it_type::pointer, typename it_type::reference >
	{
		it_type pos_;
		sp_buf data_;
		size_type cols_;

	public:
		bool cycled_;

		explicit column_iterator(matrix_t& m, bool cycled = false) 
			: pos_(m.begin()), data_(m._pData), cols_(m.cols_), cycled_(cycled)
		{}

		reference operator *() const {
			return *pos_;
		}

		pointer operator ->() const {
			return pos_.operator ->();
		}

		reference operator [](size_type ind) const {
			return pos_[ind];
		}

		column_iterator& operator =(const it_type& it) {
			pos_ = it;
			return *this;
		}

		//standard operator= is fine

		operator it_type() {
			return pos_;
		}

		column_iterator& operator ++() {
			pos_ += cols_;
			if(pos_ - data_->end() >= 0) {
				pos_ -= data_->size();
				if(!cycled_) {
					++pos_;
					if(pos_ - data_->begin() >= cols_) 
						pos_ = data_->end();
					//if(pos_ - data_->buf_begin() >= cols_) pos_ -= cols_;
				}
			}
			return *this;
		}

		column_iterator operator ++(int) {
			column_iterator tmp(*this);
			this->operator++();
			return tmp;
		}

		column_iterator& operator +=(size_type offset) {
			//new version
			pos_ += offset*cols_;
			if(pos_ - data_->end() >= 0) {
				if(cycled_)
					pos_ = data_->begin() + (pos_ - data_->end()) % data_->size();
				else {
					pos_ += 1 - data_->size();
					if(pos_ - data_->begin() >= cols_) pos_ = data_->end();
				}
			}

			//for(size_type i=0; i<offset; ++i)
			//	++tmp;
			return *this;
		}

		column_iterator operator +(size_type offset) const {
			column_iterator tmp(*this);
			//for(size_type i = 0; i < offset; ++i)
			//	++tmp;
			tmp.operator+=(offset);
			return tmp;
			//return (tmp + offset);
		}

		column_iterator& operator --() {
			pos_ -= cols_;
			if(data_->begin() - pos_ > 0) {
				pos_ += data_->size();
				if(!cycled_) {
					--pos_;
					if(data_->end() - pos_ > cols_) pos_ = data_->begin();
					//if(data_->buf_end() - pos_ > cols_) pos_ += cols_;
				}
			}
			return *this;
		}

		column_iterator operator --(int) {
			column_iterator tmp(*this);
			operator --();
			return tmp;
		}

		column_iterator& operator -=(size_type offset) {
			//new version
			pos_ -= offset*cols_;
			if(data_->begin() - pos_ > 0) {
				if(cycled_)
					pos_ = data_->end() - (data_->begin() - pos_)%data_->size();
				else {
					pos_ += data_->size() - 1;
					if(data_->end() - pos_ > cols_) pos_ = data_->begin();
				}
			}

			//for(size_type i=0; i<offset; ++i)
			//	--tmp;
			return *this;
		}

		column_iterator operator -(size_type offset) const {
			column_iterator tmp(*this);
			return (tmp -= offset);
		}

		diff_type operator -(const it_type& it) const {
			return (pos_ - it);
		}
		diff_type operator -(const column_iterator& it) const {
			diff_type ret = (pos_ - it.pos_)/cols_;
			return (pos_ - it.pos_)*data_->size()/cols_ - ret*(data_->size() - 1);
		}

		bool operator !=(const column_iterator& it) const {
			return (pos_ != it.pos_);
		}
		bool operator !=(const it_type& it) const {
			return (pos_ != it);
		}

		bool operator ==(const it_type& it) const {
			return (pos_ == it);
		}
		bool operator ==(const column_iterator& it) const {
			return (pos_ == it.pos_);
		}

		bool operator <(const column_iterator& it) const {
			return (pos_ < it.pos_);
		}
		bool operator <(const it_type& it) const {
			return (pos_ < it);
		}
		bool operator >(const column_iterator& it) const {
			return (pos_ > it.pos_);
		}
		bool operator >(const it_type& it) const {
			return (pos_ > it);
		}
	};

	typedef column_iterator< r_iterator > col_iterator;
	typedef column_iterator< cr_iterator, const this_t > ccol_iterator;

	typedef column_iterator< buf_iterator > col_buf_iterator;
	typedef column_iterator< cbuf_iterator, const this_t > ccol_buf_iterator;
	typedef typename retMatrix::col_iterator retc_iterator;
	//typedef typename retMatrix::ccol_iterator cretc_iterator;

	//--------------------------column iterator for matrix end

	template<class X>
	struct my_abs {
		T operator ()(const X& op) {
			return (op < 0 ? -op : op);
		}
	};


	//ctors
	TMatrix() :
		_pData(new buf_type)
	{
		rows_ = cols_ = size_ = 0;
		alloc_ = INNER;
	}

	TMatrix(size_type rows, size_type cols, cbuf_pointer ptr = NULL) :
		_pData(new buf_type(size_ = rows*cols))
	{
		rows_ = cols_ = 0;
		alloc_ = INNER;
		if(size_ > 0) {
			rows_ = rows; cols_ = cols;
			if(ptr) 
				memcpy(GetBuffer(), ptr, raw_size());
		}
	}

	//conversion constructor
	TMatrix(const this_t& m) :
		_pData(m._pData)
	{
		rows_ = m.rows_; cols_ = m.cols_;
		size_ = m.size_;
		alloc_ = EXTERN;
	}

	//empty destructor
	~TMatrix() {};

	//create matrix with internal buffer and optionally fill it
	void NewMatrix(size_type rows, size_type cols, buf_value_type fill_val = 0)
	{
		size_ = rows*cols;
		if(alloc_ == EXTERN) {
			_pData = new buf_type(size_, fill_val);
			alloc_ = INNER;
		}
		else {
			_pData->clear();
			if(size_ > 0) _pData->resize(size_, fill_val);
		}

		if(size_ > 0) {
			rows_ = rows; cols_ = cols;
		}
		else {
			rows_ = cols_ = 0;
		}
	}

	void NewMatrix(size_type rows, size_type cols, cbuf_pointer ptr)
	{
		NewMatrix(rows, cols);
		if(ptr) 
			memcpy(GetBuffer(), ptr, raw_size());
	}

	//create matrix with external buffer (memory not freed on exit)
	/*
	void NewExtern(size_type rows, size_type cols, const_sp_buf& sp_buf)
	{
		size_ = rows*cols;
		if(size_ != sp_buf->size()) {
			size_ = size();
			return;
		}
		rows_ = rows; cols_ = cols;
		alloc_ = EXTERN;
		_pData = sp_buf;
	}
	*/

	void NewExtern(const this_t& m)	{
		size_ = m.size_;
		rows_ = m.rows_; cols_ = m.cols_;
		alloc_ = EXTERN;
		_pData = m._pData;

		//NewExtern(m.rows_, m.cols_, m._pData);
	}

	//NewExtern operator for matrix
	this_t& operator <<=(const this_t& m) {
		NewExtern(m);
		return *this;
	}

	//matrix element access in form m(i, j)
	reference operator ()(size_type row, size_type col)
	{
		return operator[](row*cols_ + col);
	}

	const_reference operator ()(size_type row, size_type col) const 
	{
		return operator[](row*cols_ + col);
	}

	//matrix element access in form m[i] (all elements in one long vector, row by row)
	reference operator [](size_type ind)
	{
		return buf_traits::val(_pData, ind);
	}

	const_reference operator [](size_type ind) const 
	{
		return buf_traits::val(_pData, ind);
	}

	//buffer elements access
	buf_reference at_buf(size_type ind)
	{
		return _pData->operator [](ind);
	}

	cbuf_reference at_buf(size_type ind) const
	{
		return _pData->operator [](ind);
	}

	//buffer element access in form m(i, j)
	buf_reference at_buf(size_type row, size_type col)
	{
		return _pData->operator [](row*cols_ + col);
	}

	cbuf_reference at_buf(size_type row, size_type col) const
	{
		return _pData->operator [](row*cols_ + col);
	}

	//implicit conversion to buf_type
	operator const buf_type&() const {
		return *_pData.p;
	}

	const buf_type* get_container() const {
		return _pData.p;
	}

	cr_iterator begin() const {
		return r_iterator(_pData->begin());
	}

	r_iterator begin() {
		return r_iterator(_pData->begin());
	}

	cr_iterator end() const {
		return r_iterator(_pData->end());
	}

	r_iterator end() {
		return r_iterator(_pData->end());
	}

	cbuf_iterator buf_begin() const {
		return _pData->begin();
	}

	buf_iterator buf_begin() {
		return _pData->begin();
	}

	cbuf_iterator buf_end() const {
		return _pData->end();
	}

	buf_iterator buf_end() {
		return _pData->end();
	}

	size_type row_num() const {
		return rows_;
	}
	size_type col_num() const {
		return cols_;
	}
	size_type size() const {
		return size_;
	}
	size_type raw_size() const {
		return sizeof(buf_value_type)*size_;
	}

	buf_pointer GetBuffer() {
		if(!_pData->empty())
			return &_pData->operator[](0);
		else return NULL;
	}

	cbuf_pointer GetBuffer() const {
		if(!_pData->empty())
			return &_pData->operator[](0);
		else return NULL;
	}

	void SetBuffer(cbuf_pointer pBuf) {
		if(size_ > 0)
			memcpy(GetBuffer(), pBuf, raw_size());
	}

	void reserve(size_type count) {
		_pData->reserve(count);
	}

	//reset matrix
	void clear() {
		_pData->clear();
		size_ = rows_ = cols_ = 0;
	}

	//matrices assignment operator - can assign matrices of different types
	TEMPLATE_PARAM
	this_t& operator =(const PARAM_MATRIX& m) {
		//return assign_mat::assign<T, Tr, buf_traits, r_buf_traits>(*this, m);
		if((void*)this != (void*)&m) {
			rows_ = m.row_num(); cols_ = m.col_num();
			if(size_ != m.size()) {
				size_ = rows_ * cols_;
				_pData->resize(size_);
			}
			std::copy(m.begin(), m.end(), begin());
		}
		return *this;
	}

	//buffers assignment operator
	this_t& operator ^=(const this_t& m) {
		if((void*)this != (void*)&m) {
			rows_ = m.rows_; cols_ = m.cols_;
			if(size_ != m.size_) {
				size_ = rows_ * cols_;
				_pData->resize(size_);
			}
			copy(m.buf_begin(), m.buf_end(), buf_begin());
		}
		return *this;
	}

	//assign fully same type
	this_t& operator =(const this_t& m) {
		return operator =<value_type, buf_traits_type>(m);
	}

	//sets all elements = dT
	this_t& operator =(value_type dT)
	{
		fill(begin(), end(), dT);
		return *this;
	}

	//comparision operators
	TEMPLATE_PARAM
	bool operator ==(const PARAM_MATRIX& m) const
	{
		if(m.row_num() != rows_ || m.col_num() != cols_) return false;
		else
			return equal(begin(), end(), m.begin());
	}

	TEMPLATE_PARAM
	bool operator !=(const PARAM_MATRIX& m) const {
		return !(*this == m);
	}

	TEMPLATE_PARAM
	bool operator > (const PARAM_MATRIX& m) const {
		if(size_ == m.size())
			return equal(begin(), end(), m.begin(), std::greater< value_type >());
		else return (size_ > m.size());
	}

	TEMPLATE_PARAM
	bool operator < (const PARAM_MATRIX& m) const {
		if(size_ == m.size())
			return equal(begin(), end(), m.begin(), std::less< value_type >());
		else return (size_ < m.size());
	}

	TEMPLATE_PARAM
	bool operator >= (const PARAM_MATRIX& m) const {
		if(size_ == m.size())
			return equal(begin(), end(), m.begin(), std::greater_equal< value_type >());
		else return (size_ < m.size());
	}

	TEMPLATE_PARAM
	bool operator <= (const PARAM_MATRIX& m) const {
		if(size_ == m.size())
			return equal(begin(), end(), m.begin(), std::less_equal< value_type >());
		else return (size_ < m.size());
	}

	//these operators are per-element for matrices with same size
	TEMPLATE_PARAM
	this_t& operator +=(const PARAM_MATRIX& m)
	{
		if(rows_==m.row_num() && cols_==m.col_num())
			std::transform(begin(), end(), m.begin(), begin(), std::plus<value_type>());
		return *this;
	}

	TEMPLATE_PARAM
	this_t& operator -=(const PARAM_MATRIX& m)
	{
		if(rows_==m.row_num() && cols_==m.col_num())
			std::transform(begin(), end(), m.begin(), begin(), std::minus<value_type>());
		return *this;
	}

	TEMPLATE_PARAM
	this_t& operator *=(const PARAM_MATRIX& m)
	{
		if(rows_==m.row_num() && cols_==m.col_num())
			std::transform(begin(), end(), m.begin(), begin(), std::multiplies<value_type>());
		return *this;
	}

	TEMPLATE_PARAM
	this_t& operator /=(const PARAM_MATRIX& m)
	{
		if(rows_==m.row_num() && cols_==m.col_num())
			std::transform(begin(), end(), m.begin(), begin(), std::divides<value_type>());
		return *this;
	}

	this_t& operator *=(value_type dMul) {
		std::transform(begin(), end(), begin(), bind2nd(std::multiplies<value_type>(), dMul));
		return *this;
	}

	this_t& operator /=(value_type dMul) {
		std::transform(begin(), end(), begin(), bind2nd(std::divides<value_type>(), dMul));
		return *this;
	}

	this_t& operator +=(value_type d) {
		std::transform(begin(), end(), begin(), bind2nd(std::plus<value_type>(), d));
		return *this;
	}

	this_t& operator -=(value_type d) {
		std::transform(begin(), end(), begin(), bind2nd(std::minus<value_type>(), d));
		return *this;
	}

	const retMatrix operator +(value_type d) const
	{
		retMatrix r(rows_, cols_);
		std::transform(begin(), end(), r.begin(), bind2nd(std::plus<value_type>(), d));
		return r;
	}

	const retMatrix operator -(value_type d) const
	{
		retMatrix r(rows_, cols_);
		std::transform(begin(), end(), r.begin(), bind2nd(std::minus<value_type>(), d));
		return r;
	}
	//per-element multiplication
	const retMatrix operator *(value_type dMul) const
	{
		retMatrix r(rows_, cols_);
		std::transform(begin(), end(), r.begin(), bind2nd(std::multiplies<value_type>(), dMul));
		return r;
	}
	//division
	const retMatrix operator /(value_type dMul) const
	{
		retMatrix r(rows_, cols_);
		std::transform(begin(), end(), r.begin(), bind2nd(std::divides<value_type>(), dMul));
		return r;
	}

	//standard multiplication of two matrices
	TEMPLATE_PARAM
	const retMatrix operator *(const PARAM_MATRIX& m) const
	{
		value_type dSum;
		retMatrix r;
		if(cols_ == m.row_num()) {
			r.NewMatrix(rows_, m.col_num());
			//new version
			cr_iterator r_it;
			retc_iterator res_it(r);
			typename PARAM_MATRIX::ccol_iterator l_it(m, true);
			for(size_type i=0; i<m.col_num(); ++i) {
				l_it = m.begin() + i;
				r_it = begin();
				while(r_it != end()) {
					dSum = 0;
					for(size_type j=0; j<cols_; ++j) {
						dSum += *r_it * (*l_it);
						++r_it; ++l_it;
					}
					*res_it = dSum;
					++res_it;
				}
			}
		}
		else r = *this;
		return r;
	}

	TEMPLATE_PARAM
	const retMatrix operator /(const PARAM_MATRIX& m) const
	{
		retMatrix r(rows_, cols_);
		if(rows_==m.rows_ && cols_==m.col_num()) {
			std::transform(begin(), end(), m.begin(), r.begin(), std::divides<value_type>());
		}
		else r = *this;
		return r;
	}

	TEMPLATE_PARAM
	const retMatrix operator +(const PARAM_MATRIX& m) const
	{
		retMatrix r(rows_, cols_);
		if(rows_==m.row_num() && cols_==m.col_num()) {
			std::transform(begin(), end(), m.begin(), r.begin(), std::plus<value_type>());
		}
		else r = *this;
		return r;
	}

	TEMPLATE_PARAM
	const retMatrix operator -(const PARAM_MATRIX& m) const
	{
		retMatrix r(rows_, cols_);
		if(rows_==m.row_num() && cols_==m.col_num()) {
			std::transform(begin(), end(), m.begin(), r.begin(), std::minus<value_type>());
		}
		else r = *this;
		return r;
	}

	//per-element multiplication of matrices with equal size
	TEMPLATE_PARAM
	const retMatrix Mul(const PARAM_MATRIX& m) const
	{
		retMatrix r(rows_, cols_);
		if(rows_==m.row_num() && cols_==m.col_num()) {
			std::transform(begin(), end(), m.begin(), r.begin(), std::multiplies<value_type>());
		}
		else r = *this;
		return r;
	}

	//negate operator
	const retMatrix operator -() const
	{
		retMatrix r(rows_, cols_);
		std::transform(begin(), end(), r.begin(), std::negate<value_type>());
		return r;
	}

	//unary matrix transpose operator
	const this_t operator !() const
	{
		this_t r(cols_, rows_);
		if(rows_ == 1 || cols_ == 1) {
			copy(buf_begin(), buf_end(), r.buf_begin());
		}
		else {
			//new version
			col_buf_iterator dst1(r);
			std::copy(buf_begin(), buf_end(), dst1);
		}
		return r;
	}

	//for quadratic matrix returns triangle matrix using Gauss alg
	Matrix operator ~(void) const
	{
		Matrix r;
		r = *this;
		if(rows_ == cols_) {
			size_type nOfs = 0;
			double dMul;
			for(size_type i=0; i<rows_-1; ++i) {			
				for(size_type j=i+1; j<rows_; ++j) {
					dMul = - r[j*cols_ + nOfs]/r[i*cols_ + nOfs];
					for(size_type k=nOfs; k<cols_; ++k)
						r[j*cols_ + k] += r[i*cols_ + k]*dMul;
				}
				++nOfs;
			}
		}
		return r;
	}

	//determinant of quadratic matrix
	double Det(void) const
	{
		double dDet = 0;
		if(rows_ == cols_) {
			dDet = 1;
			Matrix mTr = ~(*this);
			for(r_iterator pos(begin()); pos < end(); pos += cols_ + 1)
				dDet *= *pos;
		}
		return dDet;
	}

	//horizontal matrix concat
	const this_t operator &(const this_t& m) const
	{
		this_t r;
		if(size_ == 0) {
			r ^= m;
		}
		else if(m.size_ == 0 || cols_ != m.cols_) {
			r ^= *this;
		}
		else {
			r.NewMatrix(rows_ + m.rows_, cols_);
			buf_iterator pos = r.buf_begin();
			pos = copy(buf_begin(), buf_end(), pos);
			copy(m.buf_begin(), m.buf_end(), pos);
		}

		return r;
	}

	this_t& operator &=(const this_t& m)
	{
		if(size_ == 0) {
			*this ^= m;
		}
		else if(cols_ == m.cols_) {
			_pData->reserve(_pData->size() + m.size());
			_pData->insert(buf_end(), m.buf_begin(), m.buf_end());
			//copy(m.begin(), m.end(), back_inserter(*_pData));
			rows_ += m.rows_;
			size_ = _pData->size();
		}

		return *this;
	}

	//for matrices-vectors this is an analog to vector's push_back operation - works with buffer
	void push_back(cbuf_reference val, bool grow_column = true)
	{
		if(size_ == 0 || rows_ == 1 || cols_ == 1) {
			_pData->push_back(val);
			size_ = _pData->size();
			if(size_ == 1) {
				rows_ = cols_ = 1;
			}
			else {
				if(cols_ > rows_) grow_column = false;
				if(grow_column) ++rows_;
				else ++cols_;
			}
		}
	}

	//for matrices-vectors this is an analog to vector's insert operation - works with buffer
	void insert(cbuf_reference val, size_type where, bool grow_column = true)
	{
		if((size_ == 0 || rows_ == 1 || cols_ == 1) && where <= size_) {
			_pData->insert(buf_begin() + where, val);
			size_ = _pData->size();
			if(size_ == 1) {
				rows_ = cols_ = 1;
			}
			else {
				if(cols_ > rows_) grow_column = false;
				if(grow_column) ++rows_;
				else ++cols_;
			}
		}
	}

	//vertical concat
	const this_t operator |(const this_t& m) const
	{
		this_t r;
		if(size_ == 0) {
			r ^= m;
		}
		else if(m.size_ == 0 || rows_ != m.rows_) {
			r ^= *this;
		}
		else {
			r.NewMatrix(rows_, cols_ + m.cols_);
			cbuf_iterator src1(buf_begin()), src2(m.buf_begin());
			buf_iterator dst(r.buf_begin());
			//size_type src1_ofs = 0, src2_ofs = 0;
			for(size_type i=0; i<rows_; ++i) {
				dst = copy(src1, src1 + cols_, dst);
				src1 += cols_;
				dst = copy(src2, src2 + m.cols_, dst);
				src2 += m.cols_;
			}
		}

		return r;
	}

	this_t& operator |=(const this_t& m)
	{
		if(size_ == 0) {
			*this ^= m;
		}
		else if(rows_ == m.rows_) {
			_pData->reserve(_pData->size() + m.size());
			cbuf_iterator src(m.buf_begin());
			buf_iterator dst(buf_begin() + cols_);
			for(size_type i=0; i<rows_; ++i) {
				_pData->insert(dst, src, src + m.cols_);
				//copy(src, src + m.cols_, inserter(*_pData, dst));
				src += m.cols_;
				dst += (cols_ + m.cols_);
			}
			cols_ += m.cols_;
			size_ = _pData->size();
		}

		return *this;
	}

	//sum of all matrix elements
	value_type Sum() const
	{
		value_type sum = 0;
		for(cr_iterator pos(begin()); pos != end(); ++pos)
			sum += *pos;
		return sum;
	}

	retMatrix vSum(bool by_rows = false) const
	{
		retMatrix r;
		if(size_ == 0) return r;
		double dSum;

		if(by_rows) {
			r.NewMatrix(rows_, 1);
			cr_iterator p_src = begin();
			for(retr_iterator p_dst = r.begin(); p_dst != r.end(); ++p_dst) {
				dSum = 0;
				for(size_type j=0; j<cols_; ++j) {
					dSum += *p_src;
					++p_src;
				}
				*p_dst = dSum;
			}
		}
		else {
			r.NewMatrix(1, cols_);
			ccol_iterator p_src(*this);
			for(retr_iterator p_dst = r.begin(); p_dst != r.end(); ++p_dst) {
				dSum = 0;
				for(size_type i = 0; i < rows_; ++i) {
					dSum += *p_src;
					++p_src;
				}
				*p_dst = dSum;
			}
		}
		return r;
	}

	//cumulative sum by columns
	retMatrix CumSum(void) const
	{
		retMatrix r(rows_, cols_);
		value_type dSum = 0;
		//new version
		retc_iterator dst(r);
		size_type i = 0;
		for(col_iterator src(*this); src != end(); ++src) {
			dSum += *src;
			*dst = dSum;
			++dst; ++i;
			if(i == rows_) {
				i = 0; dSum = 0;
			}
		}
		return r;
	}

	this_t GetRows(size_type row, size_type num = 1) const
	{
		if(size_ == 0) return this_t();

		row = min(row, rows_ - 1);
		num = min(num, rows_ - row);
		//thisMPtr r(new thisMatrix(num, cols_));
		this_t r(num, cols_);
		copy(buf_begin() + row*cols_, buf_begin() + (row + num)*cols_, r.buf_begin());
		return r;
	}

	this_t GetColumns(size_type col, size_type num = 1) const
	{
		if(size_ == 0) return this_t();

		col = std::min<size_type>(col, cols_ - 1);
		num = std::min<size_type>(num, cols_ - col);
		this_t r(rows_, num);
		//new version
		ccol_buf_iterator src(*this);
		src = buf_begin() + col;
		for(col_buf_iterator dst(r); dst != r.buf_end(); ++dst) {
			*dst = *src;
			++src;
		}
		return r;
	}

	template<class r_buf_traits>
	void SetRows(const TMatrix<T, r_buf_traits>& m, size_type row, size_type num = 1)
	{
		if(m.col_num() != cols_ || m.row_num() < 1 || (row + num) > rows_) return;
		copy(m.begin(), m.begin() + num*cols_, begin() + row*cols_);
	}

	template<class r_buf_traits>
	void SetColumns(const TMatrix<T, r_buf_traits>& m, size_type nColumn, size_type nNum = 1)
	{
		if(m.row_num() != rows_ || m.col_num() < nNum || (nColumn + nNum) > cols_) return;
		typename TMatrix<T, r_buf_traits>::cr_iterator src = m.begin();
		r_iterator dst = begin() + nColumn;
		while(src < m.end()) {
			copy(src, src + nNum, dst);
			src += m.col_num();
			dst += cols_;
		}
	}

	void DelRows(size_type row, size_type num = 1)
	{
		if(row >= rows_) return;
		num = min(num, rows_ - row);
		_pData->erase(buf_begin() + row*cols_, buf_begin() + (row + num)*cols_);

		rows_ -= num;
		if(rows_ == 0) cols_ = 0;	
		size_ = _pData->size();
	}

	void DelColumns(size_type nColumn, size_type nNum = 1)
	{
		if(nColumn >= cols_) return;
		nNum = std::min<size_type>(nNum, cols_ - nColumn);
		buf_iterator pos = buf_begin() + nColumn;
		for(size_type i=0; i<rows_; ++i) {
			_pData->erase(pos, pos + nNum);
			pos = pos + cols_ - nNum;
		}
		cols_ -= nNum;
		if(cols_ == 0) rows_ = 0;
		size_ = _pData->size();
	}

	this_t& Resize(size_type newRows = 0, size_type newCols = 0, buf_value_type fill_val = 0)
	{
		//correct rows - just resize
		if(newRows > 0 && newRows != rows_) {
			rows_ = newRows;
			if(cols_ == 0) cols_ = 1;
			size_ = rows_*cols_;
			_pData->resize(size_);
		}
		//resize in columns
		if(newCols > 0 && newCols != cols_) {
			if(newCols > cols_) {
				//reserve memory
				if(rows_ == 0) rows_ = 1;
				_pData->reserve(rows_*newCols);
				//for all columns except last insert zero elements
				buf_iterator pos(buf_begin() + cols_);
				for(size_type i=0; i < rows_; ++i) {
					_pData->insert(pos, newCols - cols_, fill_val);
					pos += newCols;
				}
				//for last column add elements by resizing
				//_pData->resize(_pData->size() + newCols - cols_);
				size_ = _pData->size();
				cols_ = newCols;
			}
			else DelColumns(newCols, cols_ - newCols);
		}
		return *this;
	}

	template< class _predicate_t >
	retMatrix Sort(_predicate_t pr, bool ByRows = false) const
	{
		retMatrix r;
		r = *this;
		//thisMatrix t;

		if(rows_ == 1 || cols_ == 1)
			sort(r.begin(), r.end(), pr);
		else if(ByRows) {
			//sorting by rows
			for(retr_iterator pos(r.begin()); pos != r.end(); pos += cols_)
				sort(pos, pos + cols_, pr);
		}
		else {
			//sorting by columns
			//new version
			retc_iterator p_beg(r), p_end(r);
			retr_iterator pos = r.begin();
			for(size_type i = 0; i < cols_; ++i) {
				p_beg = pos; p_end = ++pos;
				if(i == cols_ - 1) p_end = r.end();
				std::sort(p_beg, p_end, pr);
			}
		}
		return r;
	}

	retMatrix Sort(bool ByRows = false) const {
		return Sort(std::less< value_type >(), ByRows);
	}

	template< class _predicate_t >
	indMatrix RawSort(_predicate_t pr)
	{
		if(size_ == 0) return indMatrix();
		indMatrix mInd(1, size_);
		value_type val_swap;
		size_type ind_swap;
		//new version
		for(size_type i=0; i< size_; ++i)
			mInd[i] = i;

		r_iterator pos1(begin()), pos2;
		for(size_type i = 0; i < size_ - 1; ++i) {
			pos2 = pos1 + 1;
			for(size_type j=i+1; j<size_; ++j) {
				if(pr(*pos2, *pos1)) {
					val_swap = *pos1;
					*pos1 = *pos2;
					*pos2 = val_swap;

					ind_swap = mInd[i];
					mInd[i] = mInd[j];
					mInd[j] = ind_swap;
				}
				++pos2;
			}
			++pos1;
		}

		return mInd;
	}

	indMatrix RawSort() {
		return RawSort(std::less< value_type >());
	}

	double Mean(void) const
	{
		double dSum = 0;
		for(cr_iterator pos(begin()); pos != end(); ++pos)
			dSum += *pos;

		if(size_) return dSum/size_;
		else return 0;
	}

	double Var() const
	{
		if(size_ == 0) return 0;
		double m = Mean();
		double dSum = 0;
		double dT;
		//new version
		for(cr_iterator pos(begin()); pos != end(); ++pos) {
			dT = *pos - m;
			dSum += dT*dT;
		}
		return (dSum/size_);
	}

	double Std(int type = 0) const
	{
		if(size_ == 0) return 0;
		double m = Mean();
		double dSum = 0;
		double dT;
		//new version
		for(cr_iterator pos(begin()); pos != end(); ++pos) {
			dT = *pos - m;
			dSum += dT*dT;
		}

		if(type == 1) return sqrt(dSum/(size_ - 1));
		else return sqrt(dSum/size_);
	}

	Matrix vMean(bool by_rows = false) const
	{
		Matrix r;
		if(size_ == 0) return r;
		double dSum;
		//size_type ofs = 0;
		if(by_rows) {
			r.NewMatrix(rows_, 1);
			cr_iterator p_src = begin();
			for(dr_iterator p_dst = r.begin(); p_dst != r.end(); ++p_dst) {
				dSum = 0;
				for(size_type j=0; j<cols_; ++j) {
					dSum += *p_src;
					++p_src;
				}
				*p_dst = dSum/cols_;
			}
		}
		else {
			r.NewMatrix(1, cols_);
			ccol_iterator p_src(*this);
			for(dr_iterator p_dst = r.begin(); p_dst != r.end(); ++p_dst) {
				dSum = 0;
				for(size_type i = 0; i < rows_; ++i) {
					dSum += *p_src;
					++p_src;
				}
				*p_dst = dSum/rows_;
			}
		}
		return r;
	}

	Matrix vStd(bool by_rows = false, int type = 1) const
	{
		Matrix r;
		if(size_ == 0) return r;
		Matrix m = vMean(by_rows);
		double dT, dSum;
		//new version
		dr_iterator pos_(m.begin()), dst;
		if(by_rows) {
			r.NewMatrix(rows_, 1);
			cr_iterator src(begin());
			for(dst = r.begin(); dst != r.end(); ++dst) {
				dSum = 0;
				for(size_type i=0; i<cols_; ++i) {
					dT = *src - *pos_;
					dSum += dT*dT;
					++src;
				}
				if(type == 1) *dst = sqrt(dSum/(cols_ - 1));
				else *dst = sqrt(dSum/cols_);
				++pos_;
			}
		}
		else {
			r.NewMatrix(1, cols_);
			ccol_iterator src(*this);
			for(dst = r.begin(); dst != r.end(); ++dst) {
				dSum = 0;
				for(size_type i=0; i<rows_; ++i) {
					dT = *src - *pos_;
					dSum += dT*dT;
					++src;
				}
				if(type == 1) *dst = sqrt(dSum/(cols_ - 1));
				else *dst = sqrt(dSum/cols_);
				++pos_;
			}
		}

		return r;
	}

	retMatrix SubMean(const Matrix& mean, bool by_rows = false) const
	{
		Matrix m = mean;
		if((by_rows && mean.size() != rows_) || (!by_rows && mean.size() != cols_)) 
			m = vMean(by_rows);
		retMatrix r(rows_, cols_);
		//new version
		//thisMatrix save;
		dr_iterator pos_(m.begin());
		cr_iterator src(begin());
		retr_iterator dst(r.begin());
		for(size_type i=0; i<rows_; ++i) {
			for(size_type j=0; j<cols_; ++j) {
				*dst = *src - *pos_;
				++src; ++dst;
				if(!by_rows) ++pos_;
			}
			if(!by_rows) pos_ = m.begin();
			else ++pos_;
		}
		return r;
	}

	retMatrix AddMean(const Matrix& mean, bool by_rows = false) const
	{
		retMatrix r(rows_, cols_);
		if((by_rows && mean.size() != rows_) || (!by_rows && mean.size() != cols_)) {
			r = *this;
			return r;
		}
		//new version
		//thisMatrix save;
		dr_iterator pos_(mean.begin());
		cr_iterator src(begin());
		retr_iterator dst(r.begin());
		for(size_type i=0; i<rows_; ++i) {
			for(size_type j=0; j<cols_; ++j) {
				*dst = *src + static_cast<value_type>(*pos_);
				++src; ++dst;
				if(!by_rows) ++pos_;
			}
			if(!by_rows) pos_ = mean.begin();
			else ++pos_;
		}
		return r;
	}

	retMatrix Abs(void) const
	{
		retMatrix r(rows_, cols_);
		//transform(begin(), end(), r->begin(), ptr_fun<value_type, value_type>(abs));
		transform(begin(), end(), r.begin(), my_abs<value_type>());
		return r;
	}

	value_type Min(void) const {
		return *min_element(begin(), end());
	}

	size_type min_ind() const {
		return min_element(begin(), end()) - begin();
	}

	value_type Max(void) const {
		return *max_element(begin(), end());
	}

	size_type max_ind() const {
		return max_element(begin(), end()) - begin();
	}

	retMatrix vMin(bool by_rows = false) const
	{
		retMatrix r;
		if(size_ == 0) return r;
		//new version;
		retr_iterator dst;
		if(by_rows) {
			r.NewMatrix(rows_, 1);
			dst = r.begin();
			for(cr_iterator src(begin()); src != end(); src += cols_) {
				*dst = *min_element(src, src + cols_);
				++dst;
			}
		}
		else {
			r.NewMatrix(1, cols_);
			dst = r.begin();
			ccol_iterator src_beg(*this), src_end(*this);
			for(size_type i=0; i<cols_; ++i) {
				src_beg = begin() + i;
				if(i + 1 < cols_)
					src_end = begin() + i + 1;
				else 
					src_end = end();
				*dst = *std::min_element< ccol_iterator >(src_beg, src_end);
				++dst;
			}
		}
		return r;
	}

	retMatrix vMax(bool by_rows = false) const
	{
		retMatrix r;
		if(size_ == 0) return r;
		//new version;
		retr_iterator dst;
		if(by_rows) {
			r.NewMatrix(rows_, 1);
			dst = r.begin();
			for(cr_iterator src(begin()); src != end(); src += cols_) {
				*dst = *max_element(src, src + cols_);
				++dst;
			}
		}
		else {
			r.NewMatrix(1, cols_);
			dst = r.begin();
			ccol_iterator src_beg(*this), src_end(*this);
			for(size_type i=0; i<cols_; ++i) {
				src_beg = begin() + i;
				if(i + 1 < cols_)
					src_end = begin() + i + 1;
				else 
					src_end = end();
				*dst = *std::max_element< ccol_iterator >(src_beg, src_end);
				++dst;
			}
		}
		return r;
	}

	retMatrix minmax(bool by_rows = false) const
	{
		retMatrix r;
		if(by_rows) {
			r = vMin(by_rows);
			r |= vMax(by_rows);
		}
		else {
			r = vMin(by_rows);
			r &= vMax(by_rows);
		}
		return r;
	}

	static retMatrix Read(std::istream& is, size_type max_rows = 0)
	{
		retMatrix r, row;
		value_type val;
		std::string s;
		std::istringstream sinp;
		size_type col_ind;

		bool first_row = true;
		while(getline(is, s)) {
			sinp.str(s);
			sinp.clear();
			col_ind = 0;
			while(sinp >> val) {
				if(first_row)
					row.push_back(val, false);
				else {
					row[col_ind++] = val;
					if(col_ind == row.size()) break;
				}
			}
			first_row = false;
			if(row.size() > 0)
				r &= row;
			else break;
			if(max_rows > 0 && r.row_num() == max_rows) break;
		}
		return r;
	}

	std::ostream& Print(std::ostream& outs, bool delimRows = true, int num_width = 0) const
	{
		return _print<r_iterator>(outs, delimRows, num_width);
	}

	std::ostream& PrintBuf(std::ostream& outs, bool delimRows = true, int num_width = 0) const
	{
		return _print<buf_iterator>(outs, delimRows, num_width);
	}

	std::ostream& operator ()(std::ostream& outs, bool delimRows = true) const {
		return Print(outs);
	}

	template<class r_buf_traits>
	sp_ind_vec FindRow(const TMatrix<T, r_buf_traits>& row) const
	{
		sp_ind_vec ind(new ind_vec);
		size_type i = 0;
		for(cr_iterator pos(begin()); pos != end(); pos += cols_) {
			if(equal(pos, pos + cols_, row.begin())) ind->push_back(i);
			++i;
		}
		return ind;
	}

	template<class r_buf_traits>
	sp_ind_vec FindCol(const TMatrix<T, r_buf_traits>& col) const
	{
		sp_ind_vec ind(new ind_vec);
		ccol_iterator p_beg(*this), p_end(*this);
		cr_iterator pos(begin());
		for(size_type i = 0; i < cols_; ++i) {
			p_beg = pos;
			if(cols_ - i > 1) p_end = pos + 1;
			else p_end = end();

			if(equal(p_beg, p_end, col.begin())) ind->push_back(i);
			++pos;
		}
		return ind;
	}

	template<class r_buf_traits>
	size_type RowInd(const TMatrix<T, r_buf_traits>& row) const
	{
		size_type i = 0;
		cr_iterator p_row(row.begin());
		for(cr_iterator pos(begin()); pos != end(); pos += cols_) {
			if(equal(pos, pos + cols_, p_row)) break;
			++i;
		}
		return i;
	}

	template<class r_buf_traits>
	size_type RRowInd(const TMatrix<T, r_buf_traits>& row) const
	{	
		cr_iterator p_row(row.begin()), pos(end() - cols_);
		size_type i = rows_ - 1;
		for(; i < rows_; --i) {
			if(equal(pos, pos + cols_, p_row)) break;
			pos -= cols_;
		}
		return i;
	}

	template<class r_buf_traits>
	size_type ColInd(const TMatrix<T, r_buf_traits>& col) const
	{
		ccol_iterator p_beg(*this), p_end(*this);
		cr_iterator pos(begin());
		size_type i = 0;
		for(; i < cols_; ++i) {
			p_beg = pos;
			if(cols_ - i > 1) p_end = pos + 1;
			else p_end = end();

			if(equal(p_beg, p_end, col.begin())) break;
			++pos;
		}
		return i;
	}

	size_type ElementInd(const_reference elem) const {
		return find(begin(), end(), elem) - begin();
	}

	size_type BufElementInd(cbuf_reference elem) const {
		return find(buf_begin(), buf_end(), elem) - buf_begin();
	}

	const retMatrix sign(void) const
	{
		retMatrix r;
		r = *this;
		replace_if(r.begin(), r.end(), bind2nd(std::less<value_type>(), 0), -1);
		replace_if(r.begin(), r.end(), bind2nd(std::greater<value_type>(), 0), 1);
		return r;
	}

	this_t& Reverse()
	{
		reverse(begin(), end());
		return *this;
	}

	void permutate_rows(size_type row1, size_type row2)
	{
		r_iterator p_row1(begin() + row1*cols_), p_row2(begin() + row2*cols_);
		copy(p_row1, p_row1 + cols_, p_row2);
	}

	void permutate_cols(size_type col1, size_type col2)
	{
		col_iterator p_col1(*this, begin() + col1), p_col2(*this, begin() + col2);
		copy(p_col1, p_col1 + rows_, p_col2);
	}

	double norm2() const
	{
		double res = 0;
		for(cr_iterator pos = begin(); pos != end(); ++pos)
			res += (*pos)*(*pos);
		return sqrt(res);
	}
};

//-----------------------------------TMatrix partial specializations ------------------------------------
#if defined(_WIN32) && (_MSC_VER >= 1200)
//MS compiler workaround - we need definitions right here
#define _MSC_INLINE
#include "matrix.cpp"
#undef _MSC_INLINE
#else
template<>
TMatrix<bool>::TMatrix(size_type rows, size_type cols, cbuf_pointer ptr);

template<>
void TMatrix<bool>::NewMatrix(size_type rows, size_type cols, cbuf_pointer ptr);

template<>
TMatrix<bool>::buf_pointer TMatrix<bool>::GetBuffer() const;

template<>
void TMatrix<bool>::SetBuffer(cbuf_pointer pBuf);
#endif

//----------------------------------comparision operators--------------------------------------------------
//#define TEMPLATE_MATVAL template< class Tr, template< class > class r_buf_traits, class V >
//comparision operators
TEMPLATE_PARAM
bool operator ==(const PARAM_MATRIX& m, Tr val)
{
	return (std::find_if(m.begin(), m.end(), std::bind2nd(std::not_equal_to< Tr >(), val)) == m.end());
}
TEMPLATE_PARAM
bool operator ==(Tr val, const PARAM_MATRIX& m)
{
	return (m == val);
}

TEMPLATE_PARAM
bool operator !=(const PARAM_MATRIX& m, Tr val) {
	return !(m == val);
}
TEMPLATE_PARAM
bool operator !=(Tr val, const PARAM_MATRIX& m) {
	return !(m == val);
}

TEMPLATE_PARAM
bool operator > (const PARAM_MATRIX& m, Tr val) {
	return (std::find_if(m.begin(), m.end(), std::bind2nd(std::less_equal< Tr >(), val)) == m.end());
}
TEMPLATE_PARAM
bool operator > (Tr val, const PARAM_MATRIX& m) {
	return (std::find_if(m.begin(), m.end(), std::bind1st(std::less_equal< Tr >(), val)) == m.end());
}

TEMPLATE_PARAM
bool operator < (const PARAM_MATRIX& m, Tr val) {
	return (std::find_if(m.begin(), m.end(), std::bind2nd(std::greater_equal< Tr >(), val)) == m.end());
}
TEMPLATE_PARAM
bool operator < (Tr val, const PARAM_MATRIX& m) {
	return (std::find_if(m.begin(), m.end(), std::bind1st(std::greater_equal< Tr >(), val)) == m.end());
}

TEMPLATE_PARAM
bool operator >= (const PARAM_MATRIX& m, Tr val) {
	return !(m < val)
	//return (std::find_if(m.begin(), m.end(), std::bind2nd(std::less< Tr >(), val)) == m.end());
}
TEMPLATE_PARAM
bool operator >= (Tr val, const PARAM_MATRIX& m) {
	return !(val < m);
}

TEMPLATE_PARAM
bool operator <= (const PARAM_MATRIX& m, Tr val) {
	return !(m > val);
	//return (std::find_if(m.begin(), m.end(), std::bind2nd(std::greater< Tr >(), val)) == m.end());
}
TEMPLATE_PARAM
bool operator <= (Tr val, const PARAM_MATRIX& m) {
	return !(val > m);
}

//----------------------------------global functions-------------------------------------------------------
//return matrix with buffer_ptr_traits with elements-pointers to this matrix elements
template<class T>
TMatrix<T, val_ptr_buffer> create_ptr_mat(const TMatrix<T>& m)
{
	TMatrix<T, val_ptr_buffer> r(m.row_num(), m.col_num());
	typename TMatrix<T, val_ptr_buffer>::buf_iterator p_ptr = r.buf_begin();
	for(typename TMatrix<T>::cr_iterator p_val = m.begin(); p_val != m.end(); ++p_val) {
		*p_ptr = &(*p_val);
		++p_ptr;
	}
	return r;
}

template<class T>
TMatrix<T, val_ptr_buffer> create_ptr_mat(const TMatrix<T, val_sp_buffer>& m)
{
	TMatrix<T, val_ptr_buffer> r(m.row_num(), m.col_num());
	typename TMatrix<T, val_ptr_buffer>::buf_iterator p_ptr = r.buf_begin();
	for(typename TMatrix<T, val_sp_buffer>::cbuf_iterator p_sp = m.buf_begin(); p_sp != m.buf_end(); ++p_sp) {
		*p_ptr = p_sp->get();
		++p_ptr;
	}
	return r;
}

//prints matrix to file or to cout
template<class T, template <class> class buf_traits_type>
void DumpMatrix(const TMatrix<T, buf_traits_type>& m, const char* pFname = NULL, int num_width = 0)
{
	if(pFname) {
		std::ofstream fd(pFname, ios::out | ios::trunc);
		if(num_width > 0) fd.precision(num_width);
		m.Print(fd, true, num_width);
	}
	else m.Print(std::cout, true, num_width);
}

//-----------------------------------some typedefs for TMatrix users
typedef TMatrix<double> Matrix;
typedef TMatrix<ulong> ulMatrix;
typedef TMatrix<int> iMatrix;
typedef TMatrix<bool> bitMatrix;

typedef TMatrix<double, val_ptr_buffer> MatrixPtr;
typedef TMatrix<ulong, val_ptr_buffer> ulMatrixPtr;
typedef TMatrix<bool, val_ptr_buffer> bitMatrixPtr;
//-------------------------------------------------------------------

/*
#undef INNER
#undef EXTERN
#undef TMATRIX
#undef TEMPLATE_PARAM
#undef PARAM_MATRIX
*/

#ifdef _WIN32
#pragma warning(pop)
#endif

#endif	//_MATRIX_H
