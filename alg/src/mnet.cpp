#include "prg.h"
#include "mnet.h"

#include <time.h>
#include <cmath>
#include <algorithm>
#include <functional>
#include <sstream>
#include <iostream>
#ifdef __unix__
#include <stdarg.h>
#endif

const int MNET_CONST = 100;
#define SIGN 0x434E4E

using namespace std;
using namespace NN;

/*
void DumpM(const Matrix& m, const char* pFname = NULL)
{
	if(pFname) {
		ofstream fd(pFname, ios::out | ios::trunc);
		m.Print(fd);
	}
	else m.Print(cout);
}
*/

void MNet::_print_err(const char* pErr)
{
	if(!errFile_.is_open()) {
		errFile_.open(opt_.errFname_.c_str(), ios::trunc | ios::out);
		cerr.rdbuf(errFile_.rdbuf());
	}
	time_t cur_time = time(NULL);
	string sTime = ctime(&cur_time);
	string::size_type pos;
	if((pos = sTime.rfind('\n')) != string::npos)
		sTime.erase(pos);
	cerr << sTime << ": " << pErr << endl;
	cout << pErr << endl;
	_state.lastError = pErr;
}

/*
void MNet::_setDefOpt()
{
	opt_.nu = 0.5; opt_.mu = 0.05; opt_.goal = 0.01;
	opt_.limit = 0.0000001;
	opt_.batch = false; opt_.adaptive = false;
	opt_.saturate = false;
	opt_.showPeriod = 10;
	opt_.maxCycles = 5000;
	opt_.normInp = false; opt_.noise = 0;
	opt_.thresh01 = 0.7; opt_.thresh11 = 0.1;
	opt_.initFun = random;
	opt_.perfFun = sse;
	opt_.wiRange = 0.01;
	opt_.logsig_a = 1;
	opt_.tansig_a = 1.7159; opt_.tansig_b = 2./3.;
	opt_.tansig_e = 0.7159; opt_.logsig_e = 0.1;
	opt_.rp_delt_dec = 0.5; opt_.rp_delt_inc = 1.2;
	opt_.rp_delta0 = 0.07; opt_.rp_deltamax = 50;
	opt_.learnFun = BP;
	opt_.useSimpleRP = true;

	opt_.flags_ = MLP;
	iniFname_ = "nn.ini";
	opt_.errFname_ = "nn_err.txt";

	_state.status = not_learned;
}
*/

MNet::MNet(void)// : mInp()
{
	_nLayers = 0; _nInpSize = 0;
	_state.status = not_learned;
	//_setDefOpt();

	//_LTypes = NULL;
	//_W = NULL; _B = NULL;
	//_Out = NULL; _D = NULL; _BD = NULL;
}

MNet::MNet(const MNet& net)
{
	operator =(const_cast<MNet&>(net));
}

MNet& MNet::operator =(MNet& net)
{
	_nLayers = net._nLayers;
	_nInpSize = net._nInpSize;
	opt_.set_wrapper_opt(net.opt_);
	_state = net._state;
	opt_.flags_ = net.opt_.flags_;

	_LTypes = net._LTypes;
	_W = net._W;
	_B = net._B;
	_L = net._L;

	for(ulong i=0; i<_nLayers; ++i) {
		_D.push_back(*(new Matrix(net._D[i].row_num(), net._D[i].col_num())));
		_BD.push_back(*(new Matrix(net._BD[i].row_num(), net._BD[i].col_num())));
		_LD.push_back(*(new Matrix(net._LD[i].row_num(), net._LD[i].col_num())));
		_Out.push_back(*(new Matrix(net._Out[i].row_num(), net._Out[i].col_num())));
	}
	_Out.push_back(*(new Matrix(net._Out[_nLayers].row_num(), net._Out[_nLayers].col_num())));

	return *this;
}

MNet::MNet(ulong nLayNum, ulong nInpSize, ...)
{
	//MNet();
	SetLayersNum(nLayNum);
	//_nLayers = nLayNum;
	//_W = new Matrix[nLayNum];
	//_LTypes = new int[nLayNum];
	//_B = new Matrix[nLayNum];
	//_Out = new Matrix[nLayNum+1];
	//_D = new Matrix[nLayNum];
	//_BD = new Matrix[nLayNum];

	//mInp.NewMatrix(nInpSize, 1);
	SetInputSize(nInpSize);
	va_list varlist;
	va_start(varlist, nInpSize);
	for(ulong i=0; i<nLayNum; ++i)
		SetLayer(i, va_arg(varlist, ulong), tansig);
	va_end(varlist);

	opt_.set_def_opt();
	//_state.status = not_learned;
}

MNet::~MNet(void)
{
	if(errFile_.is_open()) errFile_.close();
	//_dealloc();
	//Matrix::delHeap();
}

bool MNet::SetLayersNum(ulong nNum)
{
	//_dealloc();
	_nLayers = nNum;

	//_W = new Matrix[nNum];
	//_LTypes = new int[nNum];
	//_B = new Matrix[nNum];
	//_Out = new Matrix[nNum+1];
	//_D = new Matrix[nNum];
	//_BD = new Matrix[nNum];

	_W.resize(nNum);
	//_B.resize(nNum);
	_D.resize(nNum);
	//if(opt_.flags_ & useBiases) _BD.resize(nNum);
	_Out.resize(nNum + 1);
	_LTypes.resize(nNum);
	//if(opt_.flags_ & useLateral) _L.resize(nNum);
	return true;
}

ulong MNet::AddLayer(ulong nNeurons, int nType)
{
	if(_nLayers == 0) {
		//check if no input size specified
		if(_nInpSize == 0) {
			//_state.lastError = nn_except::explain_error(NoInputSize);
			_print_err(nn_except::explain_error(NoInputSize));
			throw nn_except(NoInputSize);
		}
		//reserve input
		_Out.push_back(Matrix(_nInpSize, 1));
		//add weights
		_W.push_back(Matrix(nNeurons, _nInpSize));
		_D.push_back(Matrix(nNeurons, _nInpSize));
	}
	else {
		_W.push_back(Matrix(nNeurons, _W[_nLayers - 1].row_num()));
		_D.push_back(Matrix(nNeurons, _W[_nLayers - 1].row_num()));
	}
	//layer's output
	_Out.push_back(Matrix(nNeurons, 1));
	//if(opt_.flags_ & useBiases) {
	//	//add biases
	//	_B.push_back(Matrix(nNeurons, 1));
	//	_BD.push_back(Matrix(nNeurons, 1));
	//}
	//layer type
	_LTypes.push_back(nType);
	//_LTypes[_nLayers] = nType;
	//lateral connection
	//if(opt_.flags_ & useLateral) _L.push_back(Matrix(nNeurons, 1));
	++_nLayers;

	return _nLayers;
}

void MNet::_constructBiases()
{
	if(_B.size() != _nLayers) {
		_B.resize(_nLayers); _BD.resize(_nLayers);
	}
	vm_iterator pm_b(_B.begin()), pm_bd(_BD.begin());
	for(vm_iterator pm_out(_Out.begin() + 1); pm_out != _Out.end(); ++pm_out) {
		if((*pm_b).row_num() != (*pm_out).row_num()) {
			(*pm_b).NewMatrix((*pm_out).row_num(), 1);
			(*pm_bd).NewMatrix((*pm_out).row_num(), 1);
		}
		++pm_b; ++pm_bd;
	}
}

void MNet::_constructLateral()
{
	if(_L.size() != _nLayers) {
		_L.resize(_nLayers); _LD.resize(_nLayers);
	}
	vm_iterator pm_l(_L.begin()), pm_ld(_LD.begin());
	for(vm_iterator pm_out(_Out.begin() + 1); pm_out != _Out.end(); ++pm_out) {
		if((*pm_l).row_num() != (*pm_out).row_num()) {
			(*pm_l).NewMatrix((*pm_out).row_num(), 1);
			(*pm_ld).NewMatrix((*pm_out).row_num(), 1);
		}
		++pm_l; ++pm_ld;
	}
}

void MNet::_constructGrad()
{
	if(_G.size() != _W.size()) _G.resize(_W.size());
	if(_BG.size() != _B.size()) _BG.resize(_B.size());
	if(_LG.size() != _L.size()) _LG.resize(_L.size());
	vm_iterator pm_g(_G.begin()), pm_bg(_BG.begin()), pm_lg(_LG.begin());
	vm_iterator pm_b(_B.begin()), pm_l(_L.begin());
	bool recreate;
	for(vm_iterator pm_w(_W.begin()); pm_w != _W.end(); ++pm_w) {
		recreate = false;
		if((*pm_g).row_num() != (*pm_w).row_num() || (*pm_g).col_num() != (*pm_w).col_num()) {
			(*pm_g).NewMatrix((*pm_w).row_num(), (*pm_w).col_num());
			recreate = true;
		}
		++pm_g;
		if(pm_bg != _BG.end()) {
			if(recreate) (*pm_bg).NewMatrix((*pm_b).row_num(), (*pm_b).col_num());
			++pm_bg; ++pm_b;
		}
		if(pm_lg != _LG.end()) {
			if(recreate) (*pm_lg).NewMatrix((*pm_l).row_num(), (*pm_l).col_num());
			++pm_lg; ++pm_l;
		}
	}
}

void MNet::SetInputSize(ulong nSize)
{
	//mInp.row_num() = _Out[0].row_num() = nSize;
	//mInp.col_num() = _Out[0].col_num() = 1;
	_nInpSize = nSize;
	try {
		if(GetNeuronsNum(1) > 0)
			SetLayer(0, GetNeuronsNum(1), _LTypes[0]);
	}
	catch (alg_except& ex) {
	}
}

bool MNet::SetLayer(ulong nLayNum, ulong nNeurons, int nType)
{
	bool res = false;
	try {
		if(nLayNum >= _nLayers) throw nn_except(InvalidLayer);
		if(nNeurons==0) throw nn_except(InvalidParameter, "Neurons number must be > 0");
		if(nLayNum>0) {
			if(!_W[nLayNum-1].row_num()) throw nn_except(InvalidLayer,
				"Preceding layer must exists when creating a new one");
			_W[nLayNum].NewMatrix(nNeurons, _W[nLayNum-1].row_num());
			_D[nLayNum].NewMatrix(nNeurons, _D[nLayNum-1].row_num());
		}
		else if(_nInpSize==0) throw nn_except(NoInputSize);
		else {
			_W[nLayNum].NewMatrix(nNeurons, _nInpSize);
			_D[nLayNum].NewMatrix(nNeurons, _nInpSize);
		}
		if(nLayNum + 1 < _nLayers) {
			_W[nLayNum+1].NewMatrix(_W[nLayNum+1].row_num(), nNeurons);
			_D[nLayNum+1].NewMatrix(_W[nLayNum+1].row_num(), nNeurons);
		}
		_LTypes[nLayNum] = nType;
		//_B[nLayNum].NewMatrix(nNeurons, 1);
		//_BD[nLayNum].NewMatrix(nNeurons, 1);
		_Out[nLayNum+1].NewMatrix(nNeurons, 1);
		_D[nLayNum] = 0;
		//_BD[nLayNum] = 0;
		res = true;
	}
	catch(alg_except& ex) {
		_print_err(ex.what());
		throw;
	}
	return res;
}

void MNet::InitWeightsRandom()
{
	//srand((unsigned)time(NULL));
	for(ulong i=0; i<_nLayers; ++i) {
		generate(_W[i].begin(), _W[i].end(), prg::rand01);
		_W[i] -= 0.5; _W[i] *= opt_.wiRange;
		if(opt_.flags_ & useBiases) {
			generate(_B[i].begin(), _B[i].end(), prg::rand01);
			_B[i] -= 0.5; _B[i] *= opt_.wiRange;
		}
	}
}

void MNet::InitNW()
{
	//return if biases don't used
	if(_B.size() == 0) {
		InitWeightsRandom();
		return;
	}

	if(_inp_range.row_num() != _nInpSize || _inp_range.col_num() != 2) {
		Matrix ir_col(_nInpSize, 1);
		ir_col = -1;
		_inp_range |= ir_col;
		ir_col = 1;
		_inp_range |= ir_col;
	}

	vm_iterator pm_w(_W.begin()), pm_b(_B.begin());
	r_iterator p_w, p_b;
	Matrix ir, afr, wDir, wDir2;
	double dMag, x, y, nmax, nmin;
	ir = _inp_range;

	//debug
	ofstream wf("init_w.txt", ios::out | ios::trunc);

	////Simple version
	//for(ulong i=0; i<_nLayers; ++i) {
	//	dMag = 0.7 * pow((*pm_w).row_num(), 1./(*pm_w).col_num());
	//	generate((*pm_w).begin(), (*pm_w).end(), rand);
	//	//scale to +- 0.5
	//	*pm_w /= RAND_MAX; *pm_w -= 0.5;
	//	//scale to magnitude
	//	transform((*pm_w).begin(), (*pm_w).end(), (*pm_w).begin(), bind2nd(multiplies<double>(), dMag));
	//	//generate biases
	//	generate((*pm_b).begin(), (*pm_b).end(), rand);
	//	//scale to +- 1
	//	*pm_b /= (double)RAND_MAX/2; *pm_b -= 1;
	//	//scale to magnitude
	//	transform((*pm_b).begin(), (*pm_b).end(), (*pm_b).begin(), bind2nd(multiplies<double>(), dMag));
	//	//scale weights to input range
	//	wDir = (ir.GetColumns(1) - ir.GetColumns(0)).Abs();
	//	p_w = (*pm_w).begin();
	//	for(ulong j = 0; j<(*pm_w).row_num(); ++j)
	//		p_w = transform(p_w, p_w + (*pm_w).col_num(), wDir.begin(), p_w, divides<double>());

	//	//debug
	//	wf << i << " layer weights" << endl;
	//	(*pm_w).Print(wf);
	//	wf << i << " layer biases" << endl;
	//	(*pm_b).Print(wf);
	//	//

	//	++pm_w; ++pm_b;
	//}

	for(ulong i=0; i<_nLayers; ++i) {
		if(_LTypes[i] == logsig || _LTypes[i] == tansig) {
			//calc magnitude
			dMag = 0.7 * pow((*pm_w).row_num(), 1./(*pm_w).col_num());
			//calc directions
			generate((*pm_w).begin(), (*pm_w).end(), prg::rand01);
			*pm_w *= 2; *pm_w -= 1;
			//normalize in rows & calc weights
			wDir2 = (*pm_w).Mul(*pm_w);
			p_w = (*pm_w).begin();
			for(ulong j=0; j<(*pm_w).row_num(); ++j) {
				x = dMag/sqrt(wDir2.GetRows(j).Sum());
				p_w = transform(p_w, p_w + (*pm_w).col_num(), p_w, bind2nd(multiplies<double>(), x));
			}
			//calc biases
			if((*pm_w).row_num() == 1)
				*pm_b = 0;
			else {
				double b = -1;
				double delta = 2./((*pm_b).size() - 1);
				r_iterator p_w((*pm_w).begin());
				for(r_iterator p_b((*pm_b).begin()); p_b != (*pm_b).end(); ++p_b) {
					if(*p_w == 0) *p_b = 0;
					else {
						*p_b = b*dMag;
						if(*p_w < 0) *p_b = - *p_b;
					}
					b += delta;
					p_w += (*pm_w).col_num();
				}
			}
			//Conversion of net inputs of [-1 1] to [Nmin Nmax]
			//determine activation function region
			afr = _active_af_region(_LTypes[i]);
			x = 0.5*(afr[1] - afr[0]);
			y = 0.5*(afr[1] + afr[0]);
			*pm_w *= x;
			*pm_b = *pm_b * x + y;

			//Conversion of inputs of PR to [-1 1]
			wDir = ir.GetColumns(1) - ir.GetColumns(0);
			transform(wDir.begin(), wDir.end(), wDir.begin(), bind1st(divides<double>(), 2));
			wDir2 = - ir.GetColumns(1).Mul(wDir) + 1;
			//final biases
			*pm_b = (*pm_w)*wDir2 + *pm_b;
			//final weights
			p_w = (*pm_w).begin();
			for(ulong j=0; j<(*pm_w).row_num(); ++j)
				p_w = transform(p_w, p_w + (*pm_w).col_num(), wDir.begin(), p_w, multiplies<double>());
		}
		else {
			//simple random for unconstrained act. fun.
			generate((*pm_w).begin(), (*pm_w).end(), prg::rand01);
			*pm_w -= 0.5; *pm_w *= opt_.wiRange;
			generate((*pm_b).begin(), (*pm_b).end(), prg::rand01);
			*pm_b -= 0.5; *pm_b *= opt_.wiRange;
		}

		//debug
		wf << i << " layer weights" << endl;
		(*pm_w).Print(wf);
		wf << i << " layer biases" << endl;
		(*pm_b).Print(wf);
		//

		if(i < _nLayers - 1) {
			ir.NewMatrix((*pm_w).row_num(), 2);
			wDir.NewMatrix((*pm_w).row_num(), 1);
			switch(_LTypes[i]) {
				case logsig:
					wDir = 0; if(opt_.saturate) wDir += opt_.logsig_e;
					ir.SetColumns(wDir, 0);
					wDir = 1; if(opt_.saturate) wDir -= opt_.logsig_e;
					ir.SetColumns(wDir, 1);
					break;
				case tansig:
					wDir = -opt_.tansig_a; if(opt_.saturate) wDir += opt_.tansig_e;
					ir.SetColumns(wDir, 0);
					wDir = opt_.tansig_a; if(opt_.saturate) wDir -= opt_.tansig_e;
					ir.SetColumns(wDir, 1);
					break;
				default:
					wDir = -1;
					ir.SetColumns(wDir, 0);
					wDir = 1;
					ir.SetColumns(wDir, 1);
			}
		}

		++pm_w; ++pm_b;
	}
}

//void MNet::Randomize(double dRange)
//{
//	srand((unsigned)time(NULL));
//	for(ulong i=0;i<_nLayers;i++) {
//		for(ulong j=0;j<_W[i].size();j++) {
//			_W[i][j] = dRange*((double)rand()/RAND_MAX-0.5);
//			if(j<_B[i].row_num())
//				_B[i][j] = dRange*((double)rand()/RAND_MAX-0.5);
//		}
//	}
//}

void MNet::InitWeights()
{
	switch(opt_.initFun) {
		default:
		case if_random:
			InitWeightsRandom();
			break;
		case nw:
			InitNW();
			break;
	}
}

bool MNet::SetInput(const Matrix& mInp)
{
	if(mInp.col_num() != 1 || mInp.row_num() != _nInpSize) {
		//_state.lastError = nn_except::explain_error(SizesMismatch);
		_print_err(nn_except::explain_error(SizesMismatch));
		throw nn_except(SizesMismatch);
	}
	//_Out[0] = mInp;
	_Out[0].NewExtern(mInp);

	//if(nSize)
	//	//_Out[0].NewMatrix(nSize, 1, pInp);
	//else if(_nInpSize)
	//	//_Out[0].NewMatrix(_nInpSize, 1, pInp);
	//else {
	//	_state.status = ErrorNoSize;
	//	return false;
	//}
	return true;
}

void MNet::Propagate(void)
{
	vm_iterator pmout(++_Out.begin()), pmb;
	if(opt_.flags_ & useBiases) pmb = _B.begin();
	vector<int>::iterator ptype(_LTypes.begin());
	for(vm_iterator pmw(_W.begin()); pmw != _W.end(); ++pmw) {
		*pmout = *pmw * (*(pmout - 1));
		if(opt_.flags_ & useBiases) {
			*pmout += *pmb;
			++pmb;
		}
		ActFunc(*pmout, *ptype);
		++pmout; ++ptype;
	}

	//for(ulong i=0; i<_nLayers; ++i) {
	//	_Out[i + 1] = _W[i]*_Out[i];
	//	if(opt_.flags_ & useBiases) _Out[i + 1] += _B[i];
	//	ActFunc(_Out[i + 1], _LTypes[i]);
	//}
}

void MNet::BackPropagate(void)
{
	for(ulong i=_nLayers - 1; ; --i) {
		if(opt_.flags_ & useBiases) _Out[i + 1] += _B[i];
		ActFunc(_Out[i + 1], _LTypes[i]);
		_Out[i] = !_W[i]*_Out[i + 1];
		if(i == 0) break;
	}
}

bool MNet::SetLayerType(ulong nLayer, int nType)
{
	if(nLayer >= _nLayers) {
		//_state.lastError = nn_except::explain_error(InvalidLayer);
		_print_err(nn_except::explain_error(InvalidLayer));
		throw nn_except(InvalidLayer);
	}
	_LTypes[nLayer] = nType;
	return true;
}

Matrix MNet::BPCalcError(std::vector<Matrix>& input, std::vector<Matrix>& targets, ulong nPattern)
{
	Matrix T;
	Matrix Err(targets[0].row_num(), 1);
	if(opt_.batch) {
		for(ulong i=0; i<input.size(); ++i) {
			if(!SetInput(input[i])) return Matrix();
			T = targets[i] - _Out[_nLayers];
			Err += T.Mul(T);
		}
		Err = Err*0.5;
	}
	else {
		if(!SetInput(input[nPattern])) return Matrix();
		Err = targets[nPattern] - _Out[_nLayers];
		Err = Err.Mul(Err)*0.5;
	}
	return Err;
}

void MNet::BPCalcSpeed(ulong cur_layer, Matrix& Er)
{
	//if(!opt_.adaptive || (opt_.batch && cur_layer != _nLayers)) return;
	if(opt_.adaptive && (!opt_.batch || (opt_.batch && cur_layer==_nLayers))) {
		//Matrix mOut(_Out[cur_layer]);
		Matrix mOut2 = _Out[cur_layer].Mul(_Out[cur_layer]);

		double dTNu = 0;
		double dT, dT1, dSum = 0, dSum2 = 0;
		for(ulong m = 0; m < Er.row_num(); ++m) {
			dT1 = Er[m]*Er[m];
			dSum += dT1;
			if((dT = 1 - mOut2[m])!=0)
				dTNu += dT1/dT;
		}
		for(ulong m = 0; m < _Out[cur_layer - 1].row_num(); ++m) {
			dT = _Out[cur_layer - 1][m];
			dSum2 += dT*dT;
		}
		if(dSum!=0)
			dTNu /= (1 + dSum2)*dSum;
		else dTNu = 0;
		if(opt_.batch) _state.nu = dTNu;
		else opt_.nu = dTNu;
	}
}

double MNet::BPLearnStep(const Matrix& mInput, const Matrix& mTarget, const Matrix& mask)
{
	//Matrix Er, T;
	double dPerf;
	SetInput(mInput);
	if(opt_.noise > 0) AddNoise(true);

	Propagate();
	Matrix Er = mTarget - _Out[_nLayers];
	//if mask specified
	cr_iterator _pos;
	r_iterator pEr;
	if(mask.size() == mTarget.size()) {
		_pos = mask.begin();
		for(pEr = Er.begin(); pEr != Er.end(); ++pEr) {
			if(*_pos == 0) *pEr = 0;
			++_pos;
		}
		_pos = mask.begin();
	}
	else _pos = mask.end();

	//calc perfomance and function being mimimized
	dPerf = Er.Mul(Er).Sum();
	if(opt_.perfFun == mse) dPerf /= Er.size();
	//return if perfomance is enough
	if(dPerf < opt_.goal) return dPerf;

	double dT;
	//matrices iterators
	vm_iterator pm_outp(_Out.begin() + _nLayers - 1), pm_w(_W.begin() + _nLayers - 1),
		pm_d(_D.begin() + _nLayers - 1), pm_bd;
	if(opt_.flags_ & useBiases) pm_bd = _BD.begin() + _nLayers - 1;
	vector<int>::iterator p_lt(_LTypes.begin() + _nLayers - 1);
	//values iterators
	Matrix::r_iterator d_pos, bd_pos, outp_pos;
	//main cycle starts
	for(vm_iterator pm_out = pm_outp + 1; pm_out != _Out.begin(); --pm_out) {
		//calc nu
		//BPCalcSpeed(i + 1, Er);

		//calc local gradient
		D_Sigmoid(*pm_out, *p_lt);
		Er <<= Er.Mul(*pm_out);
		//D_Sigmoid(_Out[i + 1], _LTypes[i]);
		//Er = Er.Mul(_Out[i + 1]);

		//set deltas iterators positions
		d_pos = (*pm_d).begin();
		if(opt_.flags_ & useBiases)
			bd_pos = (*pm_bd).begin();
		//calc weights deltas
		for(pEr = Er.begin(); pEr != Er.end(); ++pEr) {
			//if(_pos != mask.end() && *(_pos++) == 0) {
			//	//no deltas for this row
			//	fill_n(d_pos, (*pm_d).col_num(), 0);
			//	d_pos += (*pm_d).col_num();
			//	if(opt_.flags_ & useBiases) {
			//		*bd_pos = 0;
			//		++bd_pos;
			//	}
			//	continue;
			//}

			for(outp_pos = (*pm_outp).begin(); outp_pos != (*pm_outp).end(); ++outp_pos) {
				*d_pos = opt_.nu*(*pEr)*(*outp_pos) + opt_.mu*(*d_pos);
				//dT = opt_.nu*(*pEr)*(*outp_pos) + opt_.mu*(*d_pos);
				//if(opt_.batch) *d_pos += dT;
				//else *d_pos = dT;
				++d_pos;
			}

			if(opt_.flags_ & useBiases) {
				*bd_pos = opt_.nu*(*pEr) + opt_.mu*(*bd_pos);
				//dT = opt_.nu*(*pEr) + opt_.mu*(*bd_pos);
				//if(opt_.batch) *bd_pos += dT;
				//else *bd_pos = dT;
				++bd_pos;
			}
		}
		if(pm_w != _W.begin()) {
			Er <<= !(*pm_w)*Er;
			--pm_w;
			//Er = !_W[i]*Er;

			--pm_d;
			if(opt_.flags_ & useBiases) --pm_bd;
			--pm_outp;
			--p_lt;
		}
	}
	//if(!opt_.batch) {
		Update();
		//if(IsConverged())
		//	_state.status = stop_palsy;
		//else Update();
	//}
	return dPerf;
}

double MNet::CalcGrad(const Matrix& mInput, const Matrix& mTarget, const Matrix& mask)
{
	double dPerf;
	SetInput(mInput);
	if(opt_.noise > 0) AddNoise(true);

	Propagate();
	Matrix Er = mTarget - _Out[_nLayers];
	//if mask specified
	cr_iterator _pos;
	ulong n_excluded = 0;
	if(mask.size() == mTarget.size()) {
		_pos = mask.begin();
		for(r_iterator pEr(Er.begin()); pEr != Er.end(); ++pEr) {
			if(*_pos == 0) {
				*pEr = 0;
				++n_excluded;
			}
			++_pos;
		}
		_pos = mask.begin();
	}
	else _pos = mask.end();

	//matrices iterators
	vm_iterator pm_outp(_Out.begin() + _nLayers - 1), pm_w(_W.begin() + _nLayers - 1),
		pm_g(_G.begin() + _nLayers - 1), pm_bg(_BG.begin() + _nLayers - 1);
	//layers type iterator
	vector<int>::iterator p_lt(_LTypes.begin() + _nLayers - 1);
	//values iterators
	r_iterator p_g, p_bg, outp_pos;

	//save grad here
	//Matrix save_g(n_excluded, (*pm_g).col_num()), save_bg(n_excluded, (*pm_bg).col_num());
	//r_iterator p_sg(save_g.begin()), p_sbg(save_bg.begin());

	//calc perfomance and function being mimimized
	dPerf = Er.Mul(Er).Sum();
	//if(opt_.perfFun == mse) dPerf /= Er.size();
	//return if perfomance is enough
	//if(dPerf < opt_.goal) return dPerf;

	//main cycle starts
	//Matrix Grad, BGrad;
	//double dT;
	bool palsy = true;
	for(vm_iterator pm_out = pm_outp + 1; pm_out != _Out.begin(); --pm_out) {
		//save grad for last layer if mask specified
		//if(n_excluded > 0) {
		//	p_g = (*pm_g).begin();
		//	p_bg = (*pm_bg).begin();
		//	for(; _pos != mask.end() && p_sbg != save_bg.end(); ++_pos) {
		//		if(*_pos == 0) {
		//			//save current grad & deltas
		//			p_sg = copy(p_g, p_g + (*pm_g).col_num(), p_sg);
		//			//zero actual grad & deltas
		//			//fill_n(p_g, (*pm_g).col_num(), 0);
		//			//same for biases
		//			*p_sbg = *p_bg;
		//			++p_sbg;
		//		}
		//		p_g += (*pm_g).col_num();
		//		++p_bg;
		//	}
		//}

		//calc error
		D_Sigmoid(*pm_out, *p_lt);
		Er <<= Er.Mul(*pm_out);

		//calc weights gradient
		if(opt_.batch) {
			*pm_g += Er * !(*pm_outp);
			*pm_bg += Er;
		}
		else {
			*pm_g = Er * !(*pm_outp);
			*pm_bg = Er;
		}

		//palsy detection
		if(palsy) {
			for(p_g = pm_g->begin(); p_g != pm_g->end(); ++p_g)
				if(abs(*p_g) > opt_.limit) {
					palsy = false;
					goto next_layer_grad;
				}
			for(p_bg = pm_bg->begin(); p_bg != pm_bg->end(); ++p_bg)
				if(abs(*p_bg) > opt_.limit) {
					palsy = false;
					goto next_layer_grad;
				}
		}

next_layer_grad:
		if(pm_w != _W.begin()) {
			Er <<= !(*pm_w)*Er;
			--pm_w;
			//Er = !_W[i]*Er;

			--pm_g;	--pm_bg;
			--pm_outp;
			--p_lt;
		}
	}
	//restore deltas & gradients
	//if(n_excluded > 0) {
	//	//ulong col_num = _Out[_nLayers - 1].size();
	//	p_g = _G[_nLayers - 1].begin();
	//	p_bg = _BG[_nLayers - 1].begin();
	//	p_sg = save_g.begin();
	//	p_sbg = save_bg.begin();

	//	for(_pos = mask.begin(); _pos != mask.end() && p_sbg != save_bg.end(); ++_pos) {
	//		if(*_pos == 0) {
	//			copy(p_sg, p_sg + save_g.col_num(), p_g);
	//			p_sg += save_g.col_num();

	//			*p_bg = *p_sbg;
	//			++p_sbg;
	//		}
	//		p_g += save_g.col_num();
	//		++p_bg;
	//	}
	//}
	if(palsy)
		_state.status = stop_palsy;
	return dPerf;
}

void MNet::_rp_classic(Matrix& grad, Matrix& old_grad, Matrix& deltas, Matrix& weights)
{
/*
	Matrix gc = grad.Mul(old_grad);
	//extract only changes in grad sign
	Matrix gc_lz(gc.row_num(), gc.col_num(), gc.GetBuffer());
	replace_if(gc_lz.begin(), gc_lz.end(), bind2nd(greater<double>(), 0), 0);
	replace_if(gc_lz.begin(), gc_lz.end(), bind2nd(less<double>(), 0), -1);
	//revert these steps back
	weights += deltas.Mul(gc_lz.Mul(old_grad.sign()));
	//zero grad where changes in sign happen
	grad += grad.Mul(gc_lz);

	//correct deltas
	replace_if(gc.begin(), gc.end(), bind2nd(greater<double>(), 0), opt_.rp_delt_inc);
	replace_if(gc.begin(), gc.end(), bind2nd(less<double>(), 0), opt_.rp_delt_dec);
	replace_if(gc.begin(), gc.end(), bind2nd(equal_to<double>(), 0), 1);
	//deltas += deltas.Mul(gc);
	deltas *= gc;
	//update weights
	weights += deltas.Mul(grad.sign());
*/

	//iterator form

	r_iterator p_w = weights.begin(), p_g = grad.begin(), pold_g = old_grad.begin();
	double dT;
	for(r_iterator p_d = deltas.begin(); p_d != deltas.end(); ++p_d) {
		dT = (*p_g)*(*pold_g);
		if(dT >= 0) {
			if(dT > 0)
				*p_d = min(*p_d * opt_.rp_delt_inc, opt_.rp_deltamax);
			if(*p_g > 0) dT = *p_d;
			else if(*p_g < 0) dT = - *p_d;
			else dT = 0;
		}
		else {
			if(*pold_g > 0) dT = - *p_d;
			else dT = *p_d;
			*p_d *= opt_.rp_delt_dec;
			*p_g = 0;
		}

		*p_w += dT;
		++p_g; ++pold_g;
		++p_w;
	}
}

void MNet::_rp_simple(const Matrix& grad, const Matrix& old_grad, Matrix& deltas, Matrix& weights)
{
/*
	Matrix gc = grad.Mul(old_grad);
	replace_if(gc.begin(), gc.end(), bind2nd(greater<double>(), 0), opt_.rp_delt_inc);
	replace_if(gc.begin(), gc.end(), bind2nd(less<double>(), 0), opt_.rp_delt_dec);
	replace_if(gc.begin(), gc.end(), bind2nd(equal_to<double>(), 0), 1);
	//deltas += deltas.Mul(gc);
	deltas *= gc;
	replace_if(deltas.begin(), deltas.end(), bind2nd(greater<double>(), opt_.rp_deltamax), opt_.rp_deltamax);
	//update weights
	weights += deltas.Mul(grad.sign());
*/

	//iterator form

	cr_iterator p_g = grad.begin(), pold_g = old_grad.begin();
	r_iterator p_w = weights.begin();
	double dT;
	for(r_iterator p_d = deltas.begin(); p_d != deltas.end(); ++p_d) {
		dT = (*p_g)*(*pold_g);
		if(dT > 0)
			*p_d = min(*p_d * opt_.rp_delt_inc, opt_.rp_deltamax);
		else if(dT < 0)
			*p_d *= opt_.rp_delt_dec;
		if(*p_g > 0) dT = *p_d;
		else if(*p_g < 0) dT = - *p_d;
		else dT = 0;

		*p_w += dT;
		++p_g; ++pold_g;
		++p_w;
	}

}

void MNet::R_BPUpdate(vm_type& old_G, vm_type& old_BG)
{
	//if(IsConverged()) {
	//	_state.status = stop_palsy;
	//	return;
	//}
	if(_state.status == stop_palsy) return;

	//matrices iterators
	vm_iterator pm_d(_D.begin()), pm_bd(_BD.begin()), pm_g(_G.begin()), pm_bg(_BG.begin()),
		pm_og(old_G.begin()), pm_obg(old_BG.begin()),
		pm_w(_W.begin()), pm_b(_B.begin());
	//values iterators
	//r_iterator p_d, p_bd, p_g, p_bg, pold_g, pold_bg, p_w, p_b;
	//save deltas here
	//Matrix save_d, save_bd;

	//if mask specified - save last layer's deltas
	//r_iterator _pos;
	//ulong n_excluded = 0;
	//if(mask.size() > 0) {
	//	pm_d = _D.begin() + _nLayers - 1;
	//	pm_bd = _BD.begin() + _nLayers - 1;
	//	for(_pos = mask.begin(); _pos != mask.end(); ++_pos) {
	//		if(*_pos == 0) {
	//			save_d &= (*pm_d).GetRows(_pos - mask.begin());
	//			//zero this row
	//			fill_n((*pm_d).begin() + (_pos - mask.begin())*(*pm_d).col_num(), (*pm_d).col_num(), 0);
	//			save_bd &= (*pm_bd).GetRows(_pos - mask.begin());
	//			//zero this bias
	//			*((*pm_bd).begin() + (_pos - mask.begin())) = 0;
	//			++n_excluded;
	//		}
	//	}
	//}

	//main cycle starts
	for(; pm_d != _D.end(); ++pm_d) {
		if(_state.cycle == 0) {
			//old grad = current grad
			*pm_og = *pm_g;
			*pm_obg = *pm_bg;
		}

		//calc weights deltas
		if(opt_.useSimpleRP) {
			//debug simple scheme
			/*
			Matrix d_copy, w_copy, b_copy, bd_copy;
			d_copy = *pm_d; w_copy = *pm_w;
			b_copy = *pm_b, bd_copy = *pm_bd;
			DumpM(*pm_g, "grad.txt"); DumpM(*pm_og, "old_grad.txt");
			DumpM(*pm_d, "deltas.txt");
			_rp_simple(*pm_g, *pm_og, d_copy, w_copy);
			_rp_simplem(*pm_g, *pm_og, *pm_d, *pm_w);
			if(d_copy != *pm_d) {
				cout << "Deltas not equal! " << _state.cycle << endl;
				DumpM(d_copy, "delt_it.txt");
				DumpM(*pm_d, "delt_m.txt");
				//(d_copy - *pm_d).Print(cout);
				DumpM(d_copy - *pm_d, "diff_deltas.txt");
			}
			if(w_copy != *pm_w)
				cout << "Weights not equal!" << _state.cycle << endl;
			_rp_simple(*pm_bg, *pm_obg, bd_copy, b_copy);
			_rp_simplem(*pm_bg, *pm_obg, *pm_bd, *pm_b);
			if(bd_copy != *pm_bd)
				cout << "Bias deltas not equal!" << _state.cycle << endl;
			if(b_copy != *pm_b)
				cout << "Biases not equal!" << _state.cycle << endl;
			*/

			//Matlab update rule - simplier
			_rp_simple(*pm_g, *pm_og, *pm_d, *pm_w);
			//biases
			_rp_simple(*pm_bg, *pm_obg, *pm_bd, *pm_b);
		}
		else {
			//classic RP rule
			_rp_classic(*pm_g, *pm_og, *pm_d, *pm_w);
			//biases
			_rp_classic(*pm_bg, *pm_obg, *pm_bd, *pm_b);
		}

		//copy current grad to old
		*pm_og = *pm_g; *pm_obg = *pm_bg;
		if(opt_.batch) {
			*pm_g = 0; *pm_bg = 0;
		}

		++pm_bd;
		++pm_g; ++pm_bg;
		++pm_og; ++pm_obg;
		++pm_w; ++pm_b;
	}

	//restore deltas
	//if(n_excluded > 0) {
	//	//ulong col_num = _Out[_nLayers - 1].size();
	//	p_d = _D[_nLayers - 1].begin();
	//	p_bd = _BD[_nLayers - 1].begin();
	//	r_iterator p_sd = save_d.begin();
	//	r_iterator p_sbd = save_bd.begin();

	//	for(_pos = mask.begin(); _pos != mask.end(); ++_pos) {
	//		if(*_pos == 0) {
	//			copy(p_sd, p_sd + save_d.col_num(), p_d);
	//			p_sd += save_d.col_num();

	//			*p_bd = *p_sbd;
	//			++p_sbd;
	//		}
	//		p_d += save_d.col_num();
	//		++p_bd;
	//	}
	//}
}

void MNet::BPUpdate()
{
	//if(IsConverged()) {
	//	_state.status = stop_palsy;
	//	return;
	//}
	if(_state.status == stop_palsy) return;

	//matrices iterators
	vm_iterator pm_d(_D.begin()), pm_bd(_BD.begin()), pm_g(_G.begin()), pm_bg(_BG.begin()),
		pm_w(_W.begin()), pm_b(_B.begin());
	//values iterators
	//r_iterator p_d, p_bd, p_g, p_bg;

	//main cycle starts
	//double dT;
	for(; pm_d != _D.end(); ++pm_d) {
		//calc nu
		//BPCalcSpeed(i + 1, Er);

		*pm_d = *pm_g * opt_.nu + *pm_d * opt_.mu;
		*pm_bd = *pm_bg * opt_.nu + *pm_bd * opt_.mu;

		*pm_w += *pm_d;
		*pm_b += *pm_bd;
		////set values iterators
		//p_g = (*pm_g).begin();
		//p_bg = (*pm_bg).begin();

		////calc weights deltas
		//for(p_d = (*pm_d).begin(); p_d != (*pm_d).end(); ++p_d) {
		//	dT = -opt_.nu*(*p_g) + opt_.mu*(*p_d);
		//	if(opt_.batch) *p_d += dT;
		//	else *p_d = dT;
		//	++p_g;
		//}
		////biases
		//for(p_bd = (*pm_bd).begin(); p_bd != (*pm_bd).end(); ++p_bd) {
		//	dT = -opt_.nu*(*p_bg) + opt_.mu*(*p_bd);
		//	if(opt_.batch) *p_bd += dT;
		//	else *p_bd = dT;
		//	++p_bg;
		//}

		if(opt_.batch) {
			*pm_g = 0; *pm_bg = 0;
		}

		++pm_bd;
		++pm_g; ++pm_bg;
		++pm_w; ++pm_b;
	}
}

double MNet::GHALearnStep(const Matrix& input)
{
	double dPerf, dFeedBack;
	ulong ind;

	SetInput(input);

	Propagate();

	//dPerf = 0;
	//main cycle starts
	for(long i = _nLayers - 1; i >= 0; --i) {
		//calc weights deltas
		for(ulong k=0; k<_D[i].col_num(); ++k) {
			dFeedBack = 0;
			ind = k;
			for(ulong j=0; j<_D[i].row_num(); ++j) {
				//ind = j*_D[i].col_num() + k;
				dFeedBack += _W[i][ind]*_Out[i + 1][j];
				_D[i][ind] = opt_.nu*_Out[i + 1][j]*(_Out[i][k] - dFeedBack);
				//dPerf += _D[i][ind]*_D[i][ind];
				ind += _D[i].col_num();
			}
		}
		//calc norm of deltas vector
		//dPerf += sqrt(_D[i].Mul(_D[i]).Sum());
	}

	Update();
	//dPerf = sqrt(dPerf);
	//if(_nLayers) dPerf /= _nLayers;

	return 0;
}

double MNet::APEXLearnStep(const Matrix& input)
{
	double dPerf, dFeedBack;
	ulong ind = 0;

	SetInput(input);

	Propagate();

	//main cycle starts
	dPerf = 0;
	for(long i = _nLayers - 1; i >= 0; --i) {
		//calc weights deltas
		for(ulong k=0; k<_D[i].col_num(); ++k) {
			dFeedBack = 0;
			for(ulong j=0; j<_D[i].row_num(); ++j) {
				//ind = j*_D[i].col_num() + k;
				dFeedBack += _W[i][ind]*_Out[i][j];
				_D[i][ind] = opt_.nu*_Out[i][j]*(_Out[i - 1][k] - dFeedBack);
				++ind;
			}
		}
		//calc norm of deltas vector
		dPerf += sqrt(_D[i].Mul(_D[i]).Sum());
	}

	Update();
	if(_nLayers) dPerf /= _nLayers;

	return dPerf;
}

int MNet::BPLearn(const Matrix& input, const Matrix& targets, bool init_weights, pLearnInformer pProc, const Matrix& mask) throw(alg_except)
{
	try {
		//DumpMatrix(input, "input.txt");
		//DumpMatrix(targets, "targets.txt");
		if(_state.status == learning)
			throw nn_except(NN_Busy, "This network is already in learning state");
		//_state.perf = opt_.goal + 1;
		_state.status = learning;
		_state.cycle = 0;
		_state.lastPerf = 0;
		double dPerf;

		vm_type prev_G, prev_BG;
		//add biases
		if(opt_.flags_ & useBiases)
			_constructBiases();
		//add lateral if needed
		if(opt_.flags_ & useLateral)
			_constructLateral();
		//add gradients & set initial deltas
		_constructGrad();
		if(opt_.learnFun == R_BP) {
			prev_G.resize(_G.size());
			prev_BG.resize(_BG.size());
			for(ulong i=0; i<_nLayers; ++i) {
				_D[i] = opt_.rp_delta0; _BD[i] = opt_.rp_delta0;
				prev_G[i].NewMatrix(_G[i].row_num(), _G[i].col_num());
				prev_BG[i].NewMatrix(_BG[i].row_num(), _BG[i].col_num());
			}
		}
		//else if(opt_.batch)
		//	for(ulong i=0;i<_nLayers;i++) {
		//		_D[i] = 0; _BD[i] = 0;
		//	}

		if(init_weights) InitWeights();

		//create patterns present order
		vector<ulong> order(input.col_num());
		vector<ulong>::iterator p_order;
		for(ulong i = 0; i < order.size(); ++i)
			order[i] = i;

		//main learn cycle
		do {
			double dBatchNu = 0;
			//shuffle order in serial mode
			if(!opt_.batch) random_shuffle(order.begin(), order.end());
			_state.perf = 0;
			for(p_order = order.begin(); p_order != order.end() && _state.status != stop_palsy; ++p_order) {
				//if(opt_.normInp && _state.cycle==0) {
				//	SetInput(input.GetColumns(*p_order));
				//	Normalize(0);
				//}

				dPerf = CalcGrad(input.GetColumns(*p_order), targets.GetColumns(*p_order), mask.GetColumns(*p_order));

				if(!opt_.batch) {
					if(opt_.learnFun == BP)
						BPUpdate();
						//BPLearnStep(input.GetColumns(*p_order), targets.GetColumns(*p_order), mask.GetColumns(*p_order));
					else if(opt_.learnFun == R_BP) {
						R_BPUpdate(prev_G, prev_BG);
						//prev_G = _G;
						//prev_BG = _BG;
					}
				}

				_state.perf += dPerf;
				//if(dPerf > _state.perf) {
				//	_state.perf = dPerf;
				//	dBatchNu = _state.nu;
				//}
			}
			//calc final performance
			if(opt_.perfFun == mse)
				_state.perf /= input.col_num();
			if(opt_.batch && _state.perf >= opt_.goal) {
				if(opt_.learnFun == BP) {
					for(ulong i=0; i<_nLayers; ++i) {
						_G[i] /= input.col_num();
						_BG[i] /= input.col_num();
					}
					BPUpdate();
					//BPLearnStep(input.GetColumns(*p_order), targets.GetColumns(*p_order), mask.GetColumns(*p_order));
					//Update(input.col_num());
				}
				else if(opt_.learnFun == R_BP) {
					R_BPUpdate(prev_G, prev_BG);
					//prev_G = _G;
					//prev_BG = _BG;
				}

				//for(ulong i=0; i<_nLayers; ++i) {
				//	_G[i] = 0; _BG[i] = 0;
				//}

				//if(IsConverged() && _state.perf > opt_.goal) {
				//	_state.status = stop_palsy;
				//	break;
				//}
				//Update(input.col_num());
				if(opt_.adaptive) {
					opt_.nu = dBatchNu;
				}
			}
			_state.lastPerf = _state.perf;
			_state.cycle++;
			if((_state.cycle%opt_.showPeriod==0) && pProc && !pProc(_state.cycle, _state.perf, (void*)this)) {
				_state.status = stop_breaked;
				break;
			}
		} while(_state.perf > opt_.goal && _state.cycle < opt_.maxCycles && _state.status != stop_palsy);

		if(_state.status != stop_palsy) {
			if(_state.perf <= opt_.goal) _state.status = learned;
			else _state.status = stop_maxcycle;
		}
	}
	//errors handling
	catch(alg_except& ex) {
		_state.status = error;
		//_state.lastError = ex.what();
		_print_err(ex.what());
		throw;
	}
	catch(exception ex) {
		_state.status = error;
		//_state.lastError = ex.what();
		_print_err(ex.what());
		throw nn_except(ex.what());
	}
	catch(...) {
		_print_err("Unknown run-time exception thrown");
		throw nn_except(-1, _state.lastError.c_str());
	}

	return _state.status;
}

int MNet::PCALearn(const Matrix& input, bool init_weights, pLearnInformer pProc) throw(alg_except)
{
	if(_state.status == learning)
		throw nn_except(NN_Busy, "This network is already in learning state");
	//_state.perf = opt_.goal + 1;
	_state.status = learning;
	_state.cycle = 0;
	_state.lastPerf = 0;
	opt_.nu = 0.0001;
	double dPerf;
	Matrix one;

	if(init_weights) InitWeights();

	//create patterns present order
	vector<ulong> order(input.col_num());
	for(ulong i=0; i<order.size(); ++i)
		order[i] = i;

	try {
		//main learn cycle
		do {
			//shuffle order
			random_shuffle(order.begin(), order.end());
			dPerf = 0;
			for(unsigned i=0; i<order.size(); ++i) {
				GHALearnStep(input.GetColumns(order[i]));
			}

			ulong ind;
			for(ulong i=0; i<_nLayers; ++i) {
				one <<= _W[i]*!_W[i];
				ind = 0;
				for(ulong i=0; i<one.row_num(); ++i) {
					one[ind] -= 1;
					ind = ind + one.col_num() + 1;
				}
				dPerf += sqrt(one.Mul(one).Sum());
			}

			if(opt_.adaptive) {
				if(dPerf < _state.lastPerf)
					opt_.nu *= 1.2;
				else if(dPerf > _state.lastPerf) opt_.nu *= 0.5;
				if(opt_.nu < 0.0001) opt_.nu = 0.0001;
			}

			_state.perf = dPerf;
			_state.lastPerf = dPerf;
			_state.cycle++;
			if((_state.cycle%opt_.showPeriod==0) && pProc && !pProc(_state.cycle, _state.perf, (void*)this)) {
				_state.status = stop_breaked;
				break;
			}
		} while(_state.perf > opt_.goal && _state.cycle < opt_.maxCycles);

		if(_state.perf <= opt_.goal) _state.status = learned;
		else _state.status = stop_maxcycle;
	}
	//errors handling
	catch(alg_except& ex) {
		_state.status = error;
		//_state.lastError = ex.what();
		_print_err(ex.what());
		throw;
	}
	catch(exception ex) {
		_state.status = error;
		//_state.lastError = ex.what();
		_print_err(ex.what());
		throw nn_except(ex.what());
	}
	catch(...) {
		//_state.lastError = "Unknown run-time exception thrown";
		_print_err("Unknown run-time exception thrown");
		throw nn_except(-1, _state.lastError.c_str());
	}

	return _state.status;
}

void MNet::Update(ulong nNum)
{
	vm_iterator pmd(_D.begin()), pmbd, pmld, pmb, pml;
	if(opt_.flags_ & useBiases) {
		pmbd = _BD.begin();
		pmb = _B.begin();
	}
	if(opt_.flags_ & useLateral) {
		pmld = _LD.begin();
		pml = _L.begin();
	}
	for(vm_iterator pmw(_W.begin()); pmw != _W.end(); ++pmw) {
		if(opt_.batch) {
			*pmd /= nNum;
			if(opt_.flags_ & useBiases)
				*pmbd /= nNum;
			if(opt_.flags_ & useLateral)
				*pmld /= nNum;
		}

		*pmw += *pmd;
		//if(opt_.batch) *pmd = 0;
		++pmd;

		if(opt_.flags_ & useBiases) {
			*pmb += *pmbd;
			//if(opt_.batch) *pmbd = 0;
			++pmb; ++pmbd;
		}
		if(opt_.flags_ & useLateral) {
			*pml += *pmld;
			//if(opt_.batch) *pmld = 0;
			++pml; ++pmld;
		}
	}

	//for(ulong i=0; i<_nLayers; ++i) {
	//	if(opt_.batch) {
	//		_D[i] /= (double)nNum;
	//		if(opt_.flags_ & useBiases)
	//			_BD[i] /= (double)nNum;
	//		if(opt_.flags_ & useLateral)
	//			_L[i] /= (double)nNum;
	//	}
	//	_W[i] += _D[i];
	//	if(opt_.flags_ & useBiases)
	//		_B[i] += _BD[i];
	//	if(opt_.flags_ & useLateral)
	//		_L[i] += _LD[i];
	//	if(opt_.batch) {
	//		_D[i] = 0;
	//		if(opt_.flags_ & useBiases)
	//			_BD[i] = 0;
	//		if(opt_.flags_ & useLateral)
	//			_LD[i] = _LD[i];
	//	}
	//}
}

bool MNet::IsConverged(void)
{
	//new version - check gradient
	for(vm_iterator pm_g(_G.begin()); pm_g != _G.end(); ++pm_g) {
		for(r_iterator p_g((*pm_g).begin()); p_g != (*pm_g).end(); ++p_g)
			if(*p_g > opt_.limit) return false;
	}
	if(opt_.flags_ & useBiases) {
		for(vm_iterator pm_bg(_BG.begin()); pm_bg != _BG.end(); ++pm_bg) {
			for(r_iterator p_bg((*pm_bg).begin()); p_bg != (*pm_bg).end(); ++p_bg)
				if(*p_bg > opt_.limit) return false;
		}
	}

	//vm_iterator pmd(_D.begin()), pmbd, pmld;
	//if(opt_.flags_ & useBiases) pmbd = _BD.begin();
	//if(opt_.flags_ & useLateral) pmld = _LD.begin();
	//r_iterator pd, pbd, pld;
	//for(; pmd != _D.end(); ++pmd) {
	//	for(pd = (*pmd).begin(); pd != (*pmd).end(); ++pd)
	//		if(abs(*pd) > opt_.limit) return false;
	//	if(opt_.flags_ & useBiases) {
	//		for(pbd = (*pmbd).begin(); pbd != (*pmbd).end(); ++pbd)
	//			if(abs(*pbd) > opt_.limit) return false;
	//		++pmbd;
	//	}
	//	if(opt_.flags_ & useLateral) {
	//		for(pld = (*pmld).begin(); pld != (*pmld).end(); ++pld)
	//			if(abs(*pld) > opt_.limit) return false;
	//		++pmld;
	//	}
	//}

	//ulong ind = 0;
	//for(ulong i=0; i<_nLayers; ++i) {
	//	while(ind < _D[i].size()) {
	//		if(abs(_D[i][ind]) > opt_.limit)
	//			return false;
	//		if(ind < _BD[i].size() && _BD[i][ind] > opt_.limit)
	//			return false;
	//		++ind;
	//	}
	//}
	return true;
}

void MNet::Normalize(ulong nLayer)
{
	if(nLayer>_nLayers) return;
	double dSum, dMin, dMax;
	dSum = (_Out[nLayer].Mul(_Out[nLayer])).Sum();
	dSum = sqrt(dSum);
	_Out[nLayer]/=dSum;
	/*
	dMax = dMin = _Out[nLayer][0];
	for(ulong i=1;i<_Out[nLayer].row_num();i++) {
		if(_Out[nLayer][i]>dMax)
			dMax = _Out[nLayer][i];
		if(fabs(_Out[nLayer][i])<dMin)
			dMin = fabs(_Out[nLayer][i]);
	}
	dSum = (dMax - dMin)/2;
	for(ulong i=0;i<_Out[nLayer].row_num();i++) {
		_Out[nLayer][i] = _Out[nLayer][i]/dSum - 0,5;
	}
	_Out[nLayer]=_Out[nLayer]-0,5;
	*/
}

void MNet::AddNoise(bool bSaveInput)
{
	//if(_Out[0]._nAlloc==EXTERN && bSaveInput)
	if(bSaveInput)
		_Out[0].NewMatrix(_Out[0].row_num(), _Out[0].col_num(), &_Out[0][0]);
	srand((unsigned)time(NULL));
	for(ulong j=0;j<_Out[0].row_num();j++)
		_Out[0][j] += opt_.noise*2*((double)rand()/RAND_MAX-0.5);
}

void MNet::_dealloc(void)
{
	_nLayers = 0;
}

bool MNet::VerifyNet(void)
{
	ulong utek,uprev;
	bool bRes = false;
	try {
		if(_nLayers == 0 || (uprev = _Out[0].row_num())==0) throw 0;
		for(ulong i=0;i<_nLayers;i++) {
			if((utek = _Out[i+1].row_num())==0) throw 0;
			if(_B[i].row_num()!=utek || _BD[i].row_num()!=utek) throw 0;
			if(_W[i].row_num()!=utek || _W[i].col_num()!=uprev) throw 0;
			if(_D[i].row_num()!=utek || _D[i].col_num()!=uprev) throw 0;
			uprev = utek;
		}
		bRes = true;
	}
	catch(int) {
	}
	return bRes;
}

ulong MNet::GetLayersNum(void)
{
	return _nLayers;
}

ulong MNet::GetNeuronsNum(ulong nLayer)
{
	if(nLayer>_nLayers) return 0;
	else if(nLayer==0) return _nInpSize;
	else return _Out[nLayer].row_num();
}

Matrix MNet::Sim(const Matrix& mInp)
{
	Matrix mOutp;
	if(mInp.row_num() == _nInpSize) {
		mOutp.NewMatrix(_Out[_nLayers].size(), mInp.col_num());
		for(ulong i=0; i<mInp.col_num(); ++i) {
			SetInput(mInp.GetColumns(i));
			Propagate();
			mOutp.SetColumns(_Out[_nLayers], i);
		}
	}
	return mOutp;
}

Matrix MNet::ReverseSim(Matrix& mInp)
{
	Matrix mOutp;
	if(mInp.row_num() == _Out[_nLayers].size()) {
		mOutp.NewMatrix(_nInpSize, mInp.col_num());
		for(ulong i=0; i<mInp.col_num(); ++i) {
			_Out[_nLayers] = mInp.GetColumns(i);
			BackPropagate();
			mOutp.SetColumns(_Out[0], i);
		}
	}
	return mOutp;
}

bool MNet::SimBinary(char* pCode)
{
	Propagate();
	r_iterator pRes = _Out[_nLayers].begin();
	double dMax = *pRes;
	double dMin = dMax;
	for(ulong i=0; i<_Out[_nLayers].row_num(); ++i) {
		if(*pRes > dMax) dMax = *pRes;
		if(abs(dMax - *pRes) < opt_.thresh11) {
			if(*pRes < dMin) dMin = *pRes;
		}
		if(dMin < dMax - opt_.thresh11) dMin = dMax;
		++pRes;
	}
	char b = 0x80;
	*pCode = 0;
	pRes = _Out[_nLayers].begin();
	for(ulong i=0; i<_Out[_nLayers].row_num(); ++i) {
		if(*pRes >= dMin) *pCode |= b;
		else if(*pRes > dMin - opt_.thresh01) return false;
		b >>= 1;
		if(i%8 == 7) {
			b = 0x80;
			++pCode; *pCode = 0;
		}
	}
	return true;
}

void MNet::ActFunc(Matrix& m, int ActFunType) {
	//ulong nSize = m.row_num()*m.col_num();
	r_iterator pos(m.begin());
	switch(ActFunType)
	{
	case tansig:
		for(; pos != m.end(); ++pos)
			*pos = opt_.tansig_a*tanh(opt_.tansig_b*(*pos));
		if(opt_.saturate) {
			replace_if(m.begin(), m.end(), bind2nd(greater<Matrix::value_type>(), opt_.tansig_a - opt_.tansig_e), opt_.tansig_a - opt_.tansig_e);
			replace_if(m.begin(), m.end(), bind2nd(less<Matrix::value_type>(), opt_.tansig_e - opt_.tansig_a), opt_.tansig_e - opt_.tansig_a);
		}
		break;
	case logsig:
		for(; pos != m.end(); ++pos)
			*pos = 1/(1 + exp(-opt_.logsig_a*(*pos)));
		if(opt_.saturate) {
			replace_if(m.begin(), m.end(), bind2nd(greater<Matrix::value_type>(), 1 - opt_.logsig_e), 1 - opt_.logsig_e);
			replace_if(m.begin(), m.end(), bind2nd(less<Matrix::value_type>(), opt_.logsig_e - 1), opt_.logsig_e - 1);
		}
		break;
	case poslin:
		for(; pos != m.end(); ++pos)
			if(*pos < 0) *pos = 0;
		break;
	case radbas:
		for(; pos != m.end(); ++pos)
			*pos = exp(-(*pos)*(*pos));
		break;
	//case HARDLIMIT:
	//	for(ulong i=0;i<nSize;i++)
	//		if(m[i]>0) m[i] = 1;
	//		else if(m[i]<0) m[i] = -1;
	//	break;
	//case THRESHOLD:
	//	for(ulong i=0;i<nSize;i++)
	//		if(m[i]>0.5) m[i] = 1;
	//		else if(m[i]<-0.5) m[i] = -1;
	//	break;
	}
}

void MNet::D_Sigmoid(Matrix& m, int ActFunType) {
	//ulong nSize = m.row_num()*m.col_num();
	r_iterator pos(m.begin());
	switch(ActFunType)
	{
	case tansig:
		for(; pos != m.end(); ++pos)
			*pos = opt_.tansig_b*(opt_.tansig_a - *pos)*(opt_.tansig_a + *pos)/opt_.tansig_a;
		break;
	case logsig:
		for(; pos != m.end(); ++pos)
			*pos = opt_.logsig_a*(*pos)*(1 - *pos);
			//m[i] = m[i]*(1-m[i]);
		break;
	case poslin:
		for(; pos != m.end(); ++pos) {
			if(*pos < 0) *pos = 0;
			else *pos = 1;
		}
		break;
	case purelin:
		m = 1;
		break;
	}
}

Matrix MNet::_active_af_region(int ActFunType)
{
	Matrix res(1, 2);
	double a = 0, b = 0;
	switch(ActFunType) {
		case logsig:
			a = -opt_.logsig_a*log(1./opt_.logsig_e - 1);
			b = -opt_.logsig_a*log(1./(1 - opt_.logsig_e) - 1);
			break;
		case tansig:
			a = 1./(2*opt_.tansig_b)*log(opt_.tansig_e/(2*opt_.tansig_a - opt_.tansig_e));
			b = 1./(2*opt_.tansig_b)*log((2*opt_.tansig_a - opt_.tansig_e)/opt_.tansig_e);
			break;
	}

	if(b > a) {
		res[0] = a; res[1] = b;
	}
	else {
		res[0] = b; res[1] = a;
	}
	return res;
}

double MNet::CalcPerfomance(void)
{
	double dSSE = _Out[_nLayers].Mul(_Out[_nLayers]).Sum();
	if(opt_.perfFun == mse) return dSSE/_Out[_nLayers].size();
	else return dSSE;
}

void MNet::LinPCA(const Matrix& input, std::auto_ptr<MNet>& net, pLearnInformer pProc, ulong comp_num, int learnType)
{
	try {
		bool init_weights = false;
		if(net.get() == NULL) {
			net.reset(new MNet);
			net->opt_.initFun = if_random;
			net->opt_.wiRange = 0.01;
			net->opt_.learnFun = learnType;
			if(net->opt_.flags_ & useBiases) net->opt_.flags_ -= useBiases;
			if(learnType == APEX) net->opt_.flags_ |= useLateral;
			else if(net->opt_.flags_ & useLateral) net->opt_.flags_ -= useLateral;
			net->opt_.nu = 0.0001;
			net->opt_.goal = 0.01;
			net->opt_.maxCycles = 50000;
			net->opt_.showPeriod = 100;

			net->SetInputSize(input.row_num());
			if(comp_num) net->AddLayer(comp_num, purelin);
			else net->AddLayer(input.row_num(), purelin);

			init_weights = true;
		}

		net->PCALearn(input, init_weights, pProc);
	}
	catch(alg_except ex) {
		_state.status = error;
		//_state.lastError = ex.what();
		_print_err(ex.what());
		throw;
	}
}

/*
void MNet::ReadOptions(const char* pFName)
{
	if(pFName && *pFName != 0) iniFname_ = pFName;
	//string s;
	//if(pFName && *pFName != 0) {
	//	s = pFName;
	//	iniFname_ = _extractFname(s);
	//}

	ifstream inif(iniFname_.c_str());
	string sOpts = " LearningRate MomentumConst Goal MinGrad Batch Adaptive Saturate NormInput ShowPeriod";
	sOpts += " MaxCycles Threshold01 Threshold11 WeightsInitRange WeightsInitFun PerfomanceFun LearningFun";
	sOpts += " TansigA TansigB TansigE LogsigA LogsigE RPDeltInc RPDeltDec RPDelta0 RPDeltaMax";
	string sInitFun = " Random NW";
	string sPerfFun = " SSE MSE";
	string sLearnFun = " BackProp ResilientBP GHA";

	string sWord, sTmp;
	std::istringstream is;
	int nPos;
	while(inif >> sWord) {
		if(sWord.size() == 0 || sWord.find(';') == 0) {
			inif >> ignoreLine;
			continue;
		}
		inif >> sTmp;
		if(sTmp[0] != '=') {
			inif >> ignoreLine;
			continue;
		}
		if((nPos = WordPos(sOpts, sWord)) <= 0) {
			inif >> ignoreLine;
			continue;
		}
		switch(nPos) {
			case 1:		//Nu
				inif >> opt_.nu;
				break;
			case 2:		//Mu
				inif >> opt_.mu;
				break;
			case 3:		//Goal
				inif >> opt_.goal;
				break;
			case 4:		//limit
				inif >> opt_.limit;
				break;
			case 5:		//Batch
				inif >> opt_.batch;
				break;
			case 6:		//Adaptive
				inif >> opt_.adaptive;
				break;
			case 7:
				inif >> opt_.saturate;
				break;
			case 8:
				inif >> opt_.normInp;
				break;
			case 9:
				inif >> opt_.showPeriod;
				break;
			case 10:
				inif >> opt_.maxCycles;
				break;
			case 11:
				inif >> opt_.thresh01;
				break;
			case 12:
				inif >> opt_.thresh11;
				break;
			case 13:
				inif >> opt_.wiRange;
				break;
			case 14:		//initFun
				inif >> sWord;
				if((nPos = WordPos(sInitFun, sWord)) > 0) opt_.initFun = nPos;
				break;
			case 15:		//perfFun
				inif >> sWord;
				if((nPos = WordPos(sPerfFun, sWord)) > 0) opt_.perfFun = nPos;
				break;
			case 16:		//learnFun
				inif >> sWord;
				if((nPos = WordPos(sLearnFun, sWord)) > 0) opt_.learnFun = nPos;
				break;
			case 17:
				inif >> opt_.tansig_a;
				break;
			case 18:
				inif >> opt_.tansig_b;
				break;
			case 19:
				inif >> opt_.tansig_e;
				break;
			case 20:
				inif >> opt_.logsig_a;
				break;
			case 21:
				inif >> opt_.logsig_e;
				break;
			case 22:
				inif >> opt_.rp_delt_inc;
				break;
			case 23:
				inif >> opt_.rp_delt_dec;
				break;
			case 24:
				inif >> opt_.rp_delta0;
				break;
			case 25:
				inif >> opt_.rp_deltamax;
				break;
		}	//main options

		inif >> ignoreLine;
	}
	//return s;
}
*/
