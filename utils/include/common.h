#ifndef _UTILS_COMMON_H
#define _UTILS_COMMON_H

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
#include <time.h>

#include "smart_ptr.h"

#define NW 13

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

//common typedefs
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
