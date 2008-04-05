#include "common.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdarg.h>
#include <math.h>

using namespace std;

//void* CHeapManage::s_hHeap = NULL;
//ulong CHeapManage::s_uNumAllocsInHeap = 0;

/*
int WordPos(const std::string& sSrc, const std::string& sWord)
{
	size_t nPos, nCur;
	nCur = 0;
	int nCnt = 0;
	if((nPos = sSrc.find(sWord)) != sSrc.npos) {
		while((nCur = sSrc.find(' ', nCur)) != sSrc.npos) {
			++nCnt;
			++nCur;
			if(nCur == nPos) break;
		}
	}
	return nCnt;
}

std::string _extractFname(std::string& s)
{
	string sret;
	string::size_type pos = s.find(';');
	if(pos != s.npos) {
		sret = s.substr(0, pos);
		s.erase(s.begin(), s.begin() + pos + 1);
	}
	else {
		sret = s;
		s.clear();
	}
	return sret;
}
*/

long round(double dIn)
{
	double dFloor, dMod;
	dMod = modf(dIn, &dFloor);
	if(fabs(dMod)>=0.5) {
		dMod >= 0 ? ++dFloor : --dFloor;
	}
	return (long)dFloor;
}

void my_sprintf(std::string& buf, const char* fmt, ...)
{
	//prepare string stream
	ostringstream ss;
	//start main loop
	va_list arg_list;
	va_start(arg_list, fmt);
	const char* term = fmt + strlen(fmt);
	char* beg = const_cast<char*>(fmt);
	char* pos;
	while((pos = strchr(beg, '%')) != NULL && ss) {
		ss.write(beg, static_cast<std::streamsize>(pos - beg));
		if(++pos >= term) break;
		switch(*pos) {
			case 'c':	//char
				if(*(pos + 1) == 'c')		//const char
					ss << va_arg(arg_list, const char*);
				else
					ss << va_arg(arg_list, char);
				break;
			case 's':	//c++ string
				ss << va_arg(arg_list, const char*);
				break;
			case 'i':	//integer
				ss << va_arg(arg_list, int);
				break;
			case 'l':	//long
				ss << va_arg(arg_list, long);
				break;
			case 'u':	//unsigned
				if(++pos >= term) break;
				switch(*pos) {
					case 'i':
						ss << va_arg(arg_list, uint);
						break;
					case 'l':
						ss << va_arg(arg_list, ulong);
						break;
					default:		//no type symbol found - step back
						--pos;
						break;
				}
				break;
			case 'f':	//double
				ss << va_arg(arg_list, double);
				break;
			case '%':	//%% - just type %
				ss << '%';
				break;
			default:		//no type symbol found - step back
				--pos;
				break;
		}
		if((beg = ++pos) >= term) break;
	}
	va_end(arg_list);
	//write latter end of format string
	ss << beg;
	ss.flush();
	buf = ss.str();
}

void DumpV(const ul_vec& v, const char* pFname)
{
	if(pFname) {
		ofstream fd(pFname, ios::out | ios::trunc);
		for(ul_vec::const_iterator pos(v.begin()); pos != v.end(); ++pos)
			fd << *pos << " ";
		fd << endl;		
	}
	else {
		for(ul_vec::const_iterator pos(v.begin()); pos != v.end(); ++pos)
			cout << *pos << " ";
		cout << endl;
	}
}
