#ifndef __SMART_PTR_H__
#define __SMART_PTR_H__

//smart pointer template
template < class T >
class smart_ptr {
	typedef T* pointer_t;

	template< class ptr >
	struct usual_deleter {
		void operator()(ptr p) const {
			typedef char type_must_be_complete[ sizeof(ptr)? 1: -1 ];
			(void) sizeof(type_must_be_complete);
			if(p) delete p;
		}
	};

	struct ptr_autodeleter {
		explicit ptr_autodeleter(pointer_t p)
			: p_(p)
		{}

		~ptr_autodeleter() {
			usual_deleter< pointer_t >()(p_);
			//delete this;
		}

		pointer_t get() const {
			return p_;
		}

	private:
		pointer_t p_;
	};

	class ptr_holder {
	public:
		explicit ptr_holder(pointer_t p)
			: pd_(new ptr_autodeleter(p))
		{}

		template< class D >
		explicit ptr_holder(D* p)
			: pd_(new ptr_autodeleter(p))
		{}

		~ptr_holder() {
			delete pd_;
		}

		void swap(ptr_holder& lhs) {
			std::swap(pd_, lhs.pd_);
		}

		ptr_holder& operator=(const ptr_holder& lhs) {
			if(pd_->get() != lhs->get())
				ptr_holder(lhs).swap(*this);
			return *this;
		}

		ptr_holder& operator=(T* lhs) {
			if(pd_->get() != lhs)
				ptr_holder(lhs).swap(*this);
			return *this;
		}

		ptr_autodeleter* operator->() const {
			return pd_;
		}

		friend inline bool operator!=(const ptr_holder& lhs, const ptr_holder& rhs) {
			return lhs.pd_ != rhs.pd_;
		}

	private:
		ptr_autodeleter* pd_;
	};

public:
	T* p;
	long* count;

	//typedef T _PtrClass;
	smart_ptr(T* lp = NULL)
		: p(lp), count(new long(1))
	{}

	// copy ctor
	smart_ptr(const smart_ptr< T >& lp) throw()
		: p(lp.p), count(lp.count) 
	{
		++*count;
	}

	// ctor from smart_ptr of castable type
	template< class R >
	smart_ptr(const smart_ptr< R >& lp) throw()
		: p(lp.p), count(lp.count)
	{
		++*count;
	}

	~smart_ptr() throw() {
		dispose();
	}

    smart_ptr< T >& operator=(const smart_ptr< T >& lp) throw() {
		if(lp.p != p) {
			dispose();
			p = lp.p;
			count = lp.count;
			++*count;
		}
		return *this;
    }

    smart_ptr< T >& operator=(T* lp) throw() {
		if(lp != p) {
			dispose();
			p = lp;
			count = new long(1);
		}
		return *this;
    }

	T* get() const throw() { return p; }
    operator T*() const throw () {return p; }
	//operator const T*() const throw () {return p; }
    T& operator*() const throw() {return *p; }
    //T** operator&() const throw () {return (T**)&p; }
    T* operator->() const throw() {return p; }

	operator bool() const {
		return p != NULL;
	}

private:
	void dispose() {
		if(--(*count) == 0) {
			delete count;
			//usual_deleter< T >()(p);
			if(p) delete p;
		}
	}
};

//realization for void* - no dereferencing operator
//template< >
//class smart_ptr< void > {
//public:
//	void* p;
//	long* count;
//
//	//typedef T _PtrClass;
//	smart_ptr(void* lp = NULL)
//		: p(lp), count(new long(1))
//	{
//	}
//
//	smart_ptr(const smart_ptr<void>& lp) throw()
//		: p(lp.p), count(lp.count) {
//		++*count;
//	}
//
//	~smart_ptr() throw() {
//		dispose();
//	}
//
//    smart_ptr<void>& operator=(const smart_ptr<void>& lp) throw() {
//		if(lp.p != p) {
//			dispose();
//			p = lp.p;
//			count = lp.count;
//			++*count;
//		}
//		return *this;
//    }
//
//    smart_ptr<void>& operator=(void* lp) throw() {
//		if(lp != p) {
//			dispose();
//			p = lp;
//			count = new long(1);
//		}
//		return *this;
//    }
//
//	operator bool() const {
//		return p != NULL;
//	}
//
//	void* get() const throw() { return p; }
//    operator void*() const throw () {return p; }
//	operator const void*() const throw () {return p; }
//    void* operator->() const throw() {return p; }
//
//private:
//	void dispose() {
//		if(--*count == 0) {
//			delete count;
//			//usual_deleter< T >()(p);
//			if(p) delete p;
//		}
//	}
//};

// comparison operators
template< class T, class R >
bool operator ==(const smart_ptr< T >& lhs, const smart_ptr< R >& rhs) {
	return (void*)lhs.get() == (void*)rhs.get();
}

template< class T, class R >
bool operator !=(const smart_ptr< T >& lhs, const smart_ptr< R >& rhs) {
	return (void*)lhs.get() != (void*)rhs.get();
}

template< class T, class R >
bool operator >(const smart_ptr< T >& lhs, const smart_ptr< R >& rhs) {
	return (void*)lhs.get() > (void*)rhs.get();
}

template< class T, class R >
bool operator <(const smart_ptr< T >& lhs, const smart_ptr< R >& rhs) {
	return (void*)lhs.get() < (void*)rhs.get();
}

template< class T, class R >
bool operator>=(const smart_ptr< T >& lhs, const smart_ptr< R >& rhs) {
	return (void*)lhs.get() >= (void*)rhs.get();
}

template< class T, class R >
bool operator<=(const smart_ptr< T >& lhs, const smart_ptr< R >& rhs) {
	return (void*)lhs.get() <= (void*)rhs.get();
}

#endif

