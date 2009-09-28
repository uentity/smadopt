#ifndef _MBUF_TRAITS
#define _MBUF_TRAITS

#include "common.h"

//-----------------------------------Matrix buffers traits begin-----------------------------------------
template< class T >
struct val_buffer
{
	typedef T value_type;
	typedef std::vector<T> buf_type;
	typedef typename buf_type::iterator r_iterator;
	typedef typename buf_type::const_iterator cr_iterator;
	typedef typename buf_type::reference reference;
	typedef typename buf_type::const_reference const_reference;
	typedef typename buf_type::pointer pointer;
	typedef typename buf_type::const_pointer const_pointer;
	//typedef T* pointer;
	//typedef const T* const_pointer;

	typedef typename buf_type::value_type buf_value_type;
	typedef typename buf_type::iterator buf_iterator;
	typedef typename buf_type::const_iterator cbuf_iterator;
	typedef typename buf_type::reference buf_reference;
	typedef typename buf_type::const_reference cbuf_reference;
	//typedef typename buf_type::pointer buf_pointer;
	//typedef typename buf_type::const_pointer cbuf_pointer;
	typedef buf_value_type* buf_pointer;
	typedef const buf_value_type* cbuf_pointer;

	typedef ulong size_type;
	typedef typename buf_type::difference_type diff_type;

	typedef smart_ptr<buf_type> sp_buf;
	typedef const sp_buf const_sp_buf;

	static reference val(sp_buf& buf, size_type ind) {
		return buf->operator[](static_cast< typename buf_type::size_type >(ind));
	}

	static const_reference val(const sp_buf& buf, size_type ind) {
		return buf->operator[](static_cast< typename buf_type::size_type >(ind));
	}

	static reference val(buf_reference buf_val) {
		return buf_val;
	}

	static const_reference val(cbuf_reference buf_val) {
		return buf_val;
	}
};

template<class T, class _buf_type = std::vector<T*> >
struct val_ptr_buffer_base
{
	typedef T value_type;
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef _buf_type buf_type;
	//typedef std::vector<pointer> buf_type;

	typedef typename buf_type::value_type buf_value_type;
	typedef typename buf_type::reference buf_reference;
	typedef typename buf_type::const_reference cbuf_reference;
	typedef typename buf_type::pointer buf_pointer;
	typedef typename buf_type::const_pointer cbuf_pointer;

	typedef typename buf_type::iterator buf_iterator;
	typedef typename buf_type::const_iterator cbuf_iterator;

	typedef ulong size_type;
	typedef typename buf_type::difference_type diff_type;

	typedef smart_ptr<buf_type> sp_buf;
	typedef const sp_buf const_sp_buf;

	//we need to define our own base iterator
	//--------------------------r_iterator for matrix of pointers begin
	template< class buf_iter_t >
	class r_iterator_base
		: public std::iterator< std::random_access_iterator_tag, value_type, diff_type, pointer, reference >
	{
	private:
		buf_iter_t pos_;

	public:

		//constructors
		r_iterator_base() {};

		r_iterator_base(const r_iterator_base& pos)
			: pos_(pos.pos_)
		{};

  		r_iterator_base(const buf_iter_t& pos)
			: pos_(pos)
		{};

		~r_iterator_base() {};


		//conversion
		operator buf_iter_t() {
			return pos_;
		}

		//common operators
		reference operator *() const {
			return (**pos_);
		}

		pointer operator ->() const {
			return (*pos_);
		}

		reference operator [](size_type ind) const {
			return *(pos_[ind]);
		}

		r_iterator_base& operator =(const r_iterator_base& pos) {
			pos_ = pos.pos_;
			return *this;
		}

		r_iterator_base& operator =(const buf_iter_t& pos) {
			pos_ = pos;
			return *this;
		}

		r_iterator_base& operator ++() {
			++pos_;
			return *this;
		}

		r_iterator_base operator ++(int) {
			r_iterator_base tmp(*this);
			++pos_;
			return tmp;
		}

		r_iterator_base& operator +=(size_type offset) {
			pos_ += offset;
			return *this;
		}

		r_iterator_base operator +(size_type offset) const {
			r_iterator_base tmp(*this);
			tmp.pos_ += offset;
			return tmp;
		}

		r_iterator_base& operator --() {
			--pos_;
			return *this;
		}

		r_iterator_base operator --(int) {
			r_iterator_base tmp(*this);
			--pos_;
			return tmp;
		}

		r_iterator_base& operator -=(size_type offset) {
			pos_ -= offset;
			return *this;
		}

		r_iterator_base operator -(size_type offset) const {
			r_iterator_base tmp(*this);
			tmp.pos_ -= offset;
			return tmp;
		}

		diff_type operator -(const r_iterator_base& pos) const {
			return (pos_ - pos.pos_);
		}
		diff_type operator -(const buf_iter_t& pos) const {
			return (pos_ - pos);
		}

		bool operator !=(const r_iterator_base& pos) const {
			return (pos_ != pos.pos_);
		}

		bool operator ==(const r_iterator_base& pos) const {
			return (pos_ == pos.pos_);
		}

		bool operator <(const r_iterator_base& pos) const {
			return (pos_ < pos.pos_);
		}

		bool operator >(const r_iterator_base& pos) const {
			return (pos_ > pos.pos_);
		}

		bool operator <=(const r_iterator_base& pos) const {
			return (pos_ <= pos.pos_);
		}

		bool operator >=(const r_iterator_base& pos) const {
			return (pos_ >= pos.pos_);
		}
	};
	//--------------------------r_iterator for matrix of pointers end
	typedef r_iterator_base< buf_iterator > r_iterator;
	typedef r_iterator_base< cbuf_iterator > cr_iterator;

	static reference val(sp_buf& buf, size_type ind) {
		return *(*buf)[static_cast< typename buf_type::size_type >(ind)];
	}

	static const_reference val(const sp_buf& buf, size_type ind) {
		return *(*buf)[static_cast< typename buf_type::size_type >(ind)];
	}

	static reference val(buf_reference buf_val) {
		return *buf_val;
	}

	static const_reference val(cbuf_reference buf_val) {
		return *buf_val;
	}

};

template< class T >
struct val_ptr_buffer : public val_ptr_buffer_base<T> {};

template< class T >
struct val_sp_buffer : public val_ptr_buffer_base<T, std::vector<smart_ptr<T> > > {};

template< class orig_buf_traits >
struct val_idx_buffer
{
	typedef typename orig_buf_traits::value_type value_type;
	typedef typename orig_buf_traits::reference reference;
	typedef typename orig_buf_traits::const_reference const_reference;
	typedef typename orig_buf_traits::pointer pointer;
	typedef typename orig_buf_traits::const_pointer const_pointer;

	typedef std::vector< unsigned long > buf_type;
	typedef typename buf_type::value_type buf_value_type;
	typedef typename buf_type::reference buf_reference;
	typedef typename buf_type::const_reference cbuf_reference;
	typedef typename buf_type::pointer buf_pointer;
	typedef typename buf_type::const_pointer cbuf_pointer;

	typedef typename buf_type::iterator buf_iterator;
	typedef typename buf_type::const_iterator cbuf_iterator;

//	typedef buf_value_type* buf_pointer;
//	typedef const buf_value_type* cbuf_pointer;

	typedef unsigned long size_type;
	typedef typename buf_type::difference_type diff_type;

	typedef smart_ptr< buf_type > sp_buf;
	typedef const sp_buf const_sp_buf;

protected:
	typename orig_buf_traits::sp_buf indexed_buf_;

public:

	void attach(const typename orig_buf_traits::sp_buf& underlying_buf) {
		indexed_buf_ = underlying_buf;
	}

	reference val(const sp_buf& buf, unsigned long ind) {
		assert(indexed_buf_);
		return (*indexed_buf_)[(*buf)[ind]];
	}

	const_reference val(const sp_buf& buf, unsigned long ind) const {
		assert(indexed_buf_);
		return (*indexed_buf_)[(*buf)[ind]];
	}

	reference val(buf_reference buf_val) {
		return (*indexed_buf_)[buf_val];
	}

	const_reference val(cbuf_reference buf_val) const {
		return (*indexed_buf_)[buf_val];
	}
};

#endif //_MBUF_TRAITS

