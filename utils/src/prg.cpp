#include "common.h"
#include "prg.h"

#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <vector>

#define MODMUL(a, b, c, m, s) q = s/a; s = b*(s - q*q) - c*q; if(s < 0) s += m;
#define POLYNOMS_NUM 2

using namespace std;
using namespace prg;
using namespace hybrid_adapt;

typedef unsigned long ulong;

namespace prg {
const ulong os[] = {0x80000000 + 0x40 + 0x20 + 2,  //(32, 7, 6, 2, 0)
				//0x80000000 + 0x20000000 + 0x2000000 + 0x1000000, //(32, 30, 26, 25)
				0x80000000 + 0x40 + 0x10 + 4 + 2 + 1}; //(32, 7, 5, 3, 2, 1, 0)
				//0x80000000 + 0x40000000 + 0x20000000 + 0x10000000 + 0x4000000 + 0x1000000};	//(32, 31, 30, 29, 27, 25, 0)
}

//combinedLCG g_mainrg;

//prg g_prg;

//base class for pseudo-random generator
class prg::randgen
{
	friend void init();

protected:
	double _prev_randn;
	bool _saved;

	static bool _initialized;

public:
	randgen() : _saved(false) {};
	randgen(const randgen& rg) : _prev_randn(rg._prev_randn), _saved(rg._saved) {};
	virtual ~randgen() {};

	virtual void init()
	{
		unsigned int i, cnt;
		if(!_initialized) {
			//srand((unsigned)time(NULL));
			cnt = (unsigned)time(NULL);
			cnt ^= clock();
			srand(cnt);
			for(i = 0; i < 100; ++i) rand();
			_initialized = true;
		}
		i = rand(); i = (i << 16) | rand();
		srand(i);
		//loop random generator a little
		cnt = rand() & 0xFF;
		for(i = 0; i < cnt; ++i) rand();
	}

	virtual double rand01() {
		return (double)rand()/RAND_MAX;
	}

	virtual double randn(double mu = 0, double q2 = 1)
	{
		if(_saved) {
			_saved = false;
			return _prev_randn;
		}
		double x1, x2, r;
		do {
			x1 = rand01()*2 - 1;
			x2 = rand01()*2 - 1;
			r = x1*x1 + x2*x2;
		} while(r == 0 || r > 1);
		double s = sqrt(-2*log(r)/r);
		_prev_randn = mu + q2*x2*s;
		_saved = true;
		return mu + q2*x1*s;
	}

	virtual long randInt(long a, long b) {
		return ha_round(rand01()*(b - a) + a);
	}

	virtual ulong randIntUB(ulong upper_bound) {
		return ha_round(rand01()*(upper_bound - 1));
	}
};

bool randgen::_initialized = false;

//combined linear congruent generator - 10^18 - main random channel
class prg::combinedLCG : public randgen
{
private:
#if defined(_WIN32) && defined(_MSC_VER)
	__int32 _s1, _s2;
#else
	int32_t _s1, _s2;
#endif
	bool _initialized;
public:
	combinedLCG() : _initialized(false) { _s1 = 1; _s2 = 1; }
	combinedLCG(const combinedLCG& rg)
		: _s1(rg._s1), _s2(rg._s2),
		_initialized(rg._initialized)
		{
			_prev_randn = rg._prev_randn;
			_saved = rg._saved;
		};

	void init()
	{
		if(!_initialized) {
			if(!randgen::_initialized) randgen::init();
			//_s1 = 1 + round(((double)rand()/RAND_MAX) * (0x7FFFFFAA - 1));
			//_s2 = 1 + round(((double)rand()/RAND_MAX) * (0x7FFFFF06 - 1));

			_s1 = rand();
			_s1 = ((_s1 << 16) | (rand() & 0xFFFF)) & 0x7FFFFFAA;
			_s2 = rand();
			_s2 = ((_s2 << 16) | (rand() & 0xFFFF)) & 0x7FFFFF06;

			//_s1 = ((rand() << 16) | rand()) & 0x7FFFFFAA;
			//_s2 = ((rand() << 16) | rand()) & 0x7FFFFF06;
			_initialized = true;
		}
	}

	double rand01()
	{
#if defined(_WIN32) && defined(_MSC_VER)
		__int32 q, z;
#else
		int32_t q, z;
#endif
		MODMUL(53668, 40014, 12211, 0x7FFFFFABL, _s1)
		MODMUL(52774, 40692, 3791, 0x7FFFFF07L, _s2)
		z = _s1 - _s2;
		if(z < 1) z += 0x7FFFFFAA;
		return z*4.656613e-10;
	}
};

//shift register generator
class prg::shiftreg_rg : public randgen
{
	typedef unsigned long ulong;
private:
	ulong _shiftr, _os;
	bool _initialized;
public:
	shiftreg_rg() : _initialized(false) { _shiftr = 1; _os = os[0]; }
	shiftreg_rg(const shiftreg_rg& rg)
		: _shiftr(rg._shiftr), _os(rg._os),
		_initialized(rg._initialized)
		{
			_prev_randn = rg._prev_randn;
			_saved = rg._saved;
		};

	void init()
	{
		if(!_initialized) {
			if(!randgen::_initialized) randgen::init();
			_shiftr = (rand() << 16) | rand();
			//_os = os[1];
			_os = os[rand() & 1];
			_initialized = true;
		}
	}

	ulong rand_ul()
	{
		ulong os_bits, os_bit, ret = 0;
		for(ulong i=0; i<32; ++i) {
			os_bits = _shiftr & _os;
			os_bit = 0;
			for(int j=0; j<32; ++j)
				os_bit ^= (os_bits >> j);
			_shiftr = (_shiftr >> 1) | ((os_bit & 1) << 31);
			ret |= (_shiftr & 1) << i;
		}
		return ret;
	}

	double rand01()
	{
		return (double)rand_ul()/0xFFFFFFFF;
	}
};

class prg::xaoc_gen : public randgen
{
	double _x, _mult;
	bool _initialized;
public:
	xaoc_gen() : _initialized(false) { _x = 571; _mult = 0.697; }
	xaoc_gen(const xaoc_gen& rg) :
		_x(rg._x), _mult(rg._mult),
		_initialized(rg._initialized)
		{
			_prev_randn = rg._prev_randn;
			_saved = rg._saved;
		};

	void init()
	{
		if(!_initialized) {
			if(!randgen::_initialized) randgen::init();
			_mult = randgen::rand01()*1000;
			_x = randgen::rand01();
			_initialized = true;
		}
	}

	double rand01()
	{
		_x = _x*_mult;
		_x = _x - floor(_x);
		return _x;
	}
};

//--------------------------------------------------------------------------------------
randgen* create_rg(int rg_type)
{
	randgen* pRG;
	switch(rg_type) {
		default:
		case combined_lcg:
			pRG = new combinedLCG;
			break;
		case standart:
			pRG = new randgen;
			break;
		case shift_reg:
			pRG = new shiftreg_rg;
			break;
		case xaoc:
			pRG = new xaoc_gen;
			break;
	}
	pRG->init();
	return pRG;
}

class prg::prg_store
{
	vector<randgen*> _rg;
	randgen* _pRG;

public:
	prg_store(int rg_type = def_base_rg) {
		init(rg_type);
	}
	~prg_store() {
		for(ulong i=0; i<_rg.size(); ++i)
			delete _rg[i];
	}

	void init(int rg_type = def_base_rg) {
		_rg.clear();
		create_stream(rg_type);
		_pRG = _rg[0];
	}

	unsigned int create_stream(int rg_type)
	{
		_rg.push_back(create_rg(rg_type));
		return static_cast<uint>(_rg.size() - 1);
	}

	bool delete_stream(unsigned int nStream)
	{
		if(nStream >= _rg.size() || _rg.size() == 1) return false;
		if(_pRG == _rg[nStream]) _pRG = _rg[0];
		if(_rg[nStream]) delete _rg[nStream];
		_rg.erase(_rg.begin() + nStream);
		return true;
	}

	bool change_stream(unsigned int nStream, int new_rg_type)
	{
		if(nStream >= _rg.size()) return false;
		if(_pRG == _rg[nStream]) _pRG = NULL;
		if(_rg[nStream]) delete _rg[nStream];
		_rg[nStream] = create_rg(new_rg_type);
		if(_pRG == NULL) _pRG = _rg[nStream];
		return true;
	}

	bool switch_stream(unsigned int nStream)
	{
		if(nStream >= _rg.size()) return false;
		_pRG = _rg[nStream];
		return true;
	}

	bool init_stream(unsigned int nStream)
	{
		if(nStream >= _rg.size()) return false;
		_rg[nStream]->init();
		return true;
	}

	unsigned int streams_count() {
		return static_cast<uint>(_rg.size());
	}

	randgen* operator ->() {
		return _pRG;
	}
} prgs;

//global functions

void prg::init()
{
	//reset system generator
	unsigned int i;
	i = rand(); i = (i << 16) | rand();
	srand(i);
	//loop random generator a little
	unsigned int cnt = rand() & 0xFF;
	for(unsigned int i = 0; i < cnt; ++i) rand();

	//randgen::_initialized = false;
	prgs.init();
	//prgs->init();
}

unsigned int prg::create_stream(int rg_type)
{
	return prgs.create_stream(rg_type);
}

void prg::init_stream(unsigned int nStream)
{
	prgs.init_stream(nStream);
}

bool prg::switch_stream(unsigned int nStream)
{
	return prgs.switch_stream(nStream);
}

unsigned int prg::streams_count()
{
	return prgs.streams_count();
}

void prg::prepare_streams(unsigned int nStreams, int rg_type)
{
	int nAdd = nStreams - prgs.streams_count() + 1;
	for(int i=0; i<nAdd; ++i)
		create_stream(rg_type);
}

bool prg::change_stream(unsigned int nStream, int new_rg_type)
{
	return prgs.change_stream(nStream, new_rg_type);
}

double prg::rand01()
{
	return prgs->rand01();
}

double prg::randn(double mu, double q2)
{
	return prgs->randn(mu, q2);
}

long prg::randInt(long a, long b)
{
	return prgs->randInt(a, b);
}

ulong prg::randIntUB(ulong upper_bound)
{
	return prgs->randIntUB(upper_bound);
}
