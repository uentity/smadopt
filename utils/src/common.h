#ifndef _COMMON_H
#define _COMMON_H

#ifdef _WIN32
	#include <windows.h>
	#ifdef __EXPORTING
		#define _LIBAPI extern "C" __declspec(dllexport)
		#define _CLASS_DECLSPEC __declspec(dllexport)
	#else
		#define _LIBAPI extern "C" __declspec(dllimport)
		#define _CLASS_DECLSPEC __declspec(dllimport)
	#endif

	#ifdef _MSC_VER
		//class x needs dll-interface blah-blah-blah
		#pragma warning(disable: 4251)
		//this used in base member initializer list
		#pragma warning(disable: 4355)
		//c++ exception specification ignored
		#pragma warning(disable: 4290)

		#if (_MSC_VER >= 1400)
			//disable checked iterators for VS2005
			//#undef _SECURE_SCL
			#define _SECURE_SCL 0
			#define _HAS_ITERATOR_DEBUGGING 0

			//#define _CRT_SECURE_NO_DEPRECATE
			#define _SCL_SECURE_NO_WARNINGS
			#define _CRT_SECURE_NO_WARNINGS
		#endif
	#endif
#else
	#define _LIBAPI
	#define _CLASS_DECLSPEC
#endif

//common includes - used almost everywhere
#include <new>
#include <vector>
#include <string>
#include <iosfwd>

//this manipulator skips line
template <class charT, class traits>
inline
std::basic_istream <charT, traits>&
ignoreLine(std::basic_istream <charT, traits>& strm)
{
	strm.ignore(0x7fff, strm.widen('\n'));
	return strm;
}

//helper compile-time things
template< int v >
struct int2type {
	enum {value = v};
};

//smart pointer template
template <class T>
class smart_ptr
{
public:
	T* p;
	long* count;

	//typedef T _PtrClass;
	explicit smart_ptr(T* lp = NULL)
		: p(lp), count(new long(1))
	{
	}

	smart_ptr(const smart_ptr<T>& lp) throw()
		: p(lp.p), count(lp.count) {
		++*count;
	}

	~smart_ptr() throw() {
		dispose();
		//phT->dispose();
	}

    smart_ptr<T>& operator=(const smart_ptr<T>& lp) throw() {
		if(lp.p != p) {
			dispose();
			p = lp.p;
			count = lp.count;
			++*count;
		}
		return *this;
    }

    smart_ptr<T>& operator=(T* lp) throw() {
		if(lp != p) {
			dispose();
			p = lp;
			count = new long(1);
		}
		return *this;
    }

	T* get() const throw() { return p; }
    operator T*() const throw () {return p; }
	operator const T*() const throw () {return p; }
    T& operator*() const throw() {return *p; }
    //T** operator&() const throw () {return (T**)&p; }
    T* operator->() const throw() {return p; }

private:
	void dispose() {
		if(--(*count) == 0) {
			delete count;
			if(p) delete p;
		}
	}
};

//realization for void* - no dereferencing operator
template<>
class smart_ptr<void>
{
public:
	void* p;
	long* count;

	//typedef T _PtrClass;
	explicit smart_ptr(void* lp = NULL)
		: p(lp), count(new long(1))
	{
	}

	smart_ptr(const smart_ptr<void>& lp) throw()
		: p(lp.p), count(lp.count) {
		++*count;
	}

	~smart_ptr() throw() {
		dispose();
	}

    smart_ptr<void>& operator=(const smart_ptr<void>& lp) throw() {
		if(lp.p != p) {
			dispose();
			p = lp.p;
			count = lp.count;
			++*count;
		}
		return *this;
    }

    smart_ptr<void>& operator=(void* lp) throw() {
		if(lp != p) {
			dispose();
			p = lp;
			count = new long(1);
		}
		return *this;
    }

	void* get() const throw() { return p; }
    operator void*() const throw () {return p; }
	operator const void*() const throw () {return p; }
    void* operator->() const throw() {return p; }

private:
	void dispose() {
		if(--*count == 0) {
			delete count;
			if(p) delete p;
		}
	}
};

//common tyepdefs
typedef unsigned long ulong;
typedef unsigned int uint;
typedef std::vector<ulong> ul_vec;
typedef smart_ptr<ul_vec> sp_ul_vec;

#ifdef _WIN32
//heap manage class for Windows
template<class T>
class CHeapManage
{
	class heap_handle {
	public:

		void* s_hHeap;
		ulong s_uNumAllocsInHeap;

		heap_handle() {
			s_hHeap = NULL;
			s_uNumAllocsInHeap = 0;
		}
		~heap_handle() {
			delHeap();
		}

		void* MyNew(size_t size)
		{
			if(s_hHeap==NULL) {
				s_hHeap = (void*)HeapCreate(HEAP_NO_SERIALIZE, 0, 0);
				if(s_hHeap==NULL)
					return (NULL);
			}
			void* p = HeapAlloc((HANDLE)s_hHeap, 0, size);
			if(p!=NULL)	s_uNumAllocsInHeap++;
			return (p);
		}

		void MyDelete(void *p) {
			if(HeapFree((HANDLE)s_hHeap, 0, p))
				s_uNumAllocsInHeap--;
		}

		void delHeap() {
			if(HeapDestroy((HANDLE)s_hHeap))
				s_hHeap = NULL;
		}
	};

protected:

	static heap_handle& get_hh() {
		static heap_handle hh;

		return hh;
	}

	static void DeleteHeap() {
		get_hh().delHeap();
	}

public:
	CHeapManage() {};

	void* operator new[](size_t size) {
		return get_hh().MyNew(size);
	}
	void* operator new[](size_t size, void* _Where) {
		return _Where;
	}
	void* operator new(size_t size) {
		return get_hh().MyNew(size);
	}
	void* operator new(size_t size, void* _Where) {
		return _Where;
	}

	void operator delete[](void* p) {
		get_hh().MyDelete(p);
	}
	void operator delete(void* p) {
		get_hh().MyDelete(p);
	}
	void operator delete[](void* p, void* _Where) { };
	void operator delete(void* p, void* _Where) { };
};

#endif
namespace hybrid_adapt {
//misc functions
_LIBAPI void my_sprintf(std::string& buf, const char* fmt, ...);
_LIBAPI void DumpV(const ul_vec& v, const char* pFname = NULL);
_LIBAPI long ha_round(double dIn);
}

#endif //COMMON_H
