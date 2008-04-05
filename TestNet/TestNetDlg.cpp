// TestNetDlg.cpp : implementation file
//

#include "stdafx.h"
#include "TestNet.h"
#include ".\testnetdlg.h"
#include "matrix.h"
#include "objnet.h"
#include "kmeans.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>

// CAboutDlg dialog used for App About
using namespace NN;
using namespace KM;
using namespace std;

CTestNetDlg* pTDlg;
bool do_learn;

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()


// CTestNetDlg dialog



CTestNetDlg::CTestNetDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CTestNetDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	pTDlg = this;
}

void CTestNetDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CTestNetDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_CREATE, OnBnClickedCreate)
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_KAR, OnBnClickedKar)
	ON_BN_CLICKED(IDC_STOP, OnBnClickedStop)
END_MESSAGE_MAP()


// CTestNetDlg message handlers

BOOL CTestNetDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	p_net = NULL;
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CTestNetDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CTestNetDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CTestNetDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CTestNetDlg::OnOK()
{
	//CDialog::OnOK();
}

/*
void Test1()
{
	//net = new MNet(3, 5, 3, 3, 2);
	double dTest[] = {1, 2, 3, 4, 5};
	//MPtr pmTest(new Matrix(5, 1));
	//Matrix tmp = *pmTest;
	ofstream f("test.txt", ios::out | ios::trunc);
	Matrix tmp1(1, 5, &dTest[0]);
	Matrix tmp(1, 5);
	//fill(tmp.begin(), tmp.end(), 1);
	generate(tmp.begin(), tmp.end(), rand);
	generate(tmp1.begin(), tmp1.end(), rand);
	f << "tmp = ";
	tmp.Print(f) << endl;
	f << "tmp1 = ";
	tmp1.Print(f) << endl;	
	Matrix res = tmp + tmp1;
	f << "tmp + tmp1" << endl;
	res.Print(f);
	res = tmp - tmp1;
	f << "tmp - tmp1" << endl;
	res.Print(f);
	res = tmp.Mul(tmp1);
	f << "tmp.Mul(tmp1)" << endl;
	res.Print(f);
	res = tmp/tmp1;
	f << "tmp / tmp1" << endl;
	res.Print(f);
	res = tmp + 2.3;
	f << "tmp + 2.3" << endl;
	res.Print(f);
	res = tmp - 2.3;
	f << "tmp - 2.3" << endl;
	res.Print(f);
	tmp += tmp1;
	f << "tmp += tmp1" << endl;
	tmp.Print(f);
	tmp -= tmp1;
	f << "tmp -= tmp1" << endl;
	tmp.Print(f);
	//tmp += 2.3;
	//f << "tmp += 2.3" << endl;
	//tmp.Print(f);
	//tmp -= 2.3;
	//f << "tmp -= 2.3" << endl;
	//tmp.Print(f);
	res = tmp*2.3;
	f << "tmp * 2.3" << endl;
	res.Print(f) << endl;
	res = tmp/2.3;
	f << "tmp / 2.3" << endl;
	res.Print(f) << endl;

	res = !tmp*tmp1;
	f << "!tmp * tmp1" << endl;
	res.Print(f) << endl;
	res = !res;
	res.Print(f);

	fstream f1("test1.txt", ios::out | ios::in | ios::trunc);
	res.Print(f1);
	f1.seekg(0, ios::beg);
	istream finp(f1.rdbuf());
	Matrix tmp2 = Matrix::Read(finp);
	f << "tmp2" << endl;
	tmp2.Print(f);

	//net->Randomize(0.01);
	//Matrix mOutp = net->Sim(pmTest.get());
	//delete pmTest;
}

void Test2()
{
	double dTest[] = {1, 2, 3, 4, 5};
	bool bTest[] = {true, true, false, true, false};
	ofstream f("test.txt", ios::out | ios::trunc);
	Matrix tmp(1, 5), tmp1(1, 5, &dTest[0]);
	//test vector<bool>
	bitMatrix b_tmp(1, 5, bTest);
	b_tmp *= true;
	b_tmp += false;
	f << "b_tmp = ";
	b_tmp.Print(f) << endl;
	bitMatrix bit_res = b_tmp.Mul(b_tmp);
	f << "b_tmp = ";
	b_tmp.Print(f) << endl;
	//make matrix of pointers
	MatrixPtr p_tmp(1, 5);
	bitMatrixPtr bp_tmp(1, 5);
	TMatrix<double, val_sp_buffer> sp_tmp(1, 5);
	uint i = 0;
	for(uint i=0; i<tmp1.size(); ++i) {
		p_tmp.at_buf(i) = dTest + i;
		bp_tmp.at_buf(i) = bTest + i;
		sp_tmp.at_buf(i) = dTest + i;
	}
	MatrixPtr p_tmp1;
	bitMatrixPtr bp_tmp1;
	bp_tmp1.NewMatrix(1, 5, bp_tmp.GetBuffer());
	p_tmp1.NewMatrix(1, 5, (double*)1);
	p_tmp1.NewMatrix(1, 5, &p_tmp.at_buf(i));

	generate(tmp.begin(), tmp.end(), rand);
	generate(tmp1.begin(), tmp1.end(), rand);
	p_tmp = tmp1;
	tmp1 = p_tmp;
	p_tmp1 = p_tmp;
	generate(p_tmp.begin(), p_tmp.end(), rand);
	f << "tmp = ";
	tmp.Print(f) << endl;
	f << "tmp1 = ";
	tmp1.Print(f) << endl;	
	f << "p_tmp = ";
	p_tmp.Print(f) << endl;

	Matrix res = tmp1 + p_tmp;
	f << "tmp1 + p_tmp" << endl;
	res.Print(f);

	res = p_tmp - tmp;
	f << "tmp - p_tmp" << endl;
	res.Print(f);

	res = p_tmp.Mul(p_tmp);
	f << "p_tmp.Mul(p_tmp)" << endl;
	res.Print(f);

	res = (!p_tmp)*tmp1;
	f << "p_tmp / tmp1" << endl;
	res.Print(f);

	res = p_tmp + 2.3;
	f << "tmp + 2.3" << endl;
	res.Print(f);

	res = tmp - 2.3;
	f << "tmp - 2.3" << endl;
	res.Print(f);

	tmp += tmp1;
	f << "tmp += tmp1" << endl;
	tmp.Print(f);

	tmp -= tmp1;
	f << "tmp -= tmp1" << endl;
	tmp.Print(f);
	//tmp += 2.3;
	//f << "tmp += 2.3" << endl;
	//tmp.Print(f);
	//tmp -= 2.3;
	//f << "tmp -= 2.3" << endl;
	//tmp.Print(f);
	res = tmp*2.3;
	f << "tmp * 2.3" << endl;
	res.Print(f) << endl;
	res = tmp/2.3;
	f << "tmp / 2.3" << endl;
	res.Print(f) << endl;

	res = !tmp*tmp1;
	f << "!tmp * tmp1" << endl;
	res.Print(f) << endl;
	res = !res;
	res.Print(f);

	fstream f1("test1.txt", ios::out | ios::in | ios::trunc);
	res.Print(f1);
	f1.seekg(0, ios::beg);
	istream finp(f1.rdbuf());
	Matrix tmp2 = Matrix::Read(finp);
	f << "tmp2" << endl;
	tmp2.Print(f);

	//net->Randomize(0.01);
	//Matrix mOutp = net->Sim(pmTest.get());
	//delete pmTest;
}
*/

void TestKmeans()
{
	kmeans km; 
	ifstream f1("t.txt");
	Matrix t = Matrix::Read(f1);
	f1.close(); f1.clear();
	f1.open("f.txt");
	Matrix f = Matrix::Read(f1);
	km.opt_.ReadOptions();
	km.opt_.seed_t = KM::sample;
	//km.find_clusters(t, 10, 200);

	km.find_clusters_f(t, f, f.size()*0.3, 200);
	const Matrix c = km.get_centers();

	//Matrix c;
	//double quant_mult;
	//vector< Matrix > c_trials;
	//vector< ulong > c_rows;
	//for(double quant_mult = 0.01; quant_mult <= 1; quant_mult+=0.05) {
	//	c <<= km.drops_hetero_simple(t, f, 0.8, 200, quant_mult);
	//	//c <<= km.drops_homo(t, f, 0.8, 200, quant_mult);
	//	c_trials.push_back(c);
	//	c_rows.push_back(c.row_num());
	//}
	//sort(c_rows.begin(), c_rows.end());
	//ulong median = c_rows[c_rows.size()/2];
	//for(ulong i = 0; i < c_trials.size(); ++i) {
	//	if(c_trials[i].row_num() == median) {
	//		c <<= c_trials[i];
	//		break;
	//	}
	//}

	ofstream f2("centers.txt", ios::out | ios::trunc);
	//km.get_centers().Print(f2);
	c.Print(f2);
	f2.close(); f2.clear();
	f2.open("ind.txt", ios::out | ios::trunc);
	km.get_ind().Print(f2);
}

void CTestNetDlg::OnBnClickedCreate()
{
	//Test2();
	TestKmeans();
	return;

	//string buf;
	//my_sprintf(buf, "test %cc", "Hello World!");
	//return;
}

void CTestNetDlg::OnClose()
{
	// TODO: Add your message handler code here and/or call default
	CDialog::OnClose();
}

/*
bool NNLearnInformer(ulong uCycle, double dSSE, void* pNet)
{
	CListBox* pList = (CListBox*)pTDlg->GetDlgItem(IDC_LB_STATE);
	CString str;
	str.Format("Cycle %i, error %.4f", uCycle, dSSE);
	pList->AddString(str);
	pList->SelectString(pList->GetCount() - 1, str);
	return do_learn;
}

UINT KarotazhThread(LPVOID p)
{
	CTestNetDlg* pTDlg = (CTestNetDlg*)p;
	//read karotazh data
	ifstream fsrc("karotazh.txt");
	Matrix kar = Matrix::Read(fsrc);
	Matrix mask(kar.row_num(), kar.col_num());
	mask = 1;
	Matrix::r_iterator pmask(mask.begin());
	for(Matrix::r_iterator pkar(kar.begin()); pkar != kar.end(); ++pkar) {
		if(*pkar == -999.25) {
			*pkar = 0;
			*pmask = 0;
		}
		++pmask;
	}
	//normilize data
	Matrix mean, kar_col;
	//kar.SubMean(mean);
	Matrix kar2 = kar.Mul(kar);
	double dT;
	Matrix rad(1, kar.col_num());
	for(ulong i = kar.col_num() - 1; ; --i) {
		kar_col = kar.GetColumns(i);
		rad[i] = kar_col.Abs().Max();
		//rad[i] = sqrt(kar2.GetColumns(i).Sum());
		if(rad[i] > 0) {
			kar_col = kar.GetColumns(i)/rad[i];
			kar.SetColumns(kar_col, i);
		}
		else
			kar.DelColumns(i);
		if(i==0) break;
	}

	//exclude again -999.25
	//pmask = mask.begin();
	//for(Matrix::r_iterator pkar(kar.begin()); pkar != kar.end(); ++pkar) {
	//	if(*pmask == 0)
	//		*pkar = 0;
	//	++pmask;
	//}

	ofstream pp_src("pp_kar.txt", ios::out | ios::trunc);
	kar.Print(pp_src);

	Matrix tar;
	tar = kar;

	((CListBox*)pTDlg->GetDlgItem(IDC_LB_STATE))->ResetContent();

	auto_ptr<MNet> pNet(new MNet);
	pTDlg->p_net = pNet.get();
	try {
		pNet->SetInputSize(kar.row_num());
		//pNet->AddLayer(tar.row_num() / 2, logsig);
		pNet->AddLayer(tar.row_num(), tansig);
		pNet->AddLayer(tar.row_num() / 2, tansig);
		pNet->AddLayer(tar.row_num(), tansig);
		//pNet->AddLayer(tar.row_num() / 2, tansig);
		//pNet->AddLayer(tar.row_num() / 2, logsig);
		//pNet->AddLayer(50, tansig);
		//pNet->AddLayer(10, logsig);
		//pNet->AddLayer(10, logsig);
		//pNet->AddLayer(50, tansig);
		pNet->AddLayer(tar.row_num(), tansig);
		pNet->_opt.tansig_a = 1.2;
		pNet->_opt.tansig_b = 1;
		pNet->_opt.tansig_e = 0.2;
		pNet->_opt.nu = 0.7;
		pNet->_opt.mu = 0.3;
		pNet->_opt.adaptive = false;
		pNet->_opt.goal = 0.1;
		pNet->_opt.showPeriod = 1;
		pNet->_opt.wiRange = 0.01;
		pNet->_opt.maxCycles = 5000;
		pNet->_opt.batch = true;
		pNet->_opt.saturate = true;
		pNet->_opt.learnFun = R_BP;

		pNet->_inp_range = kar.minmax(true);
		pNet->_opt.initFun = NN::if_random;

		pNet->BPLearn(kar, tar, true, NNLearnInformer, mask);

		nnState state = pNet->state();
		CString str;
		str.Format("Cycle %i, error %.4f", state.cycle, state.perf);
		((CListBox*)pTDlg->GetDlgItem(IDC_LB_STATE))->AddString(str);

		//pNet->BPLearn(kar, tar, NNLearnInformer, mask);

		Matrix res = pNet->Sim(kar);
		//restore data
		for(ulong i=0; i<kar.col_num(); ++i) {
			//if(rad[i] != 0)
			kar_col = res.GetColumns(i)*rad[i];
			res.SetColumns(kar_col, i);
		}
		//res.AddMean(mean);
		ofstream netf("net_kar.txt", ios::out | ios::trunc);
		res.Print(netf);
	}
	catch(nn_except ex) {
		((CListBox*)pTDlg->GetDlgItem(IDC_LB_STATE))->AddString(ex.what());
		return 1;
	}
	return 0;
}
*/

void CTestNetDlg::OnBnClickedKar()
{
	do_learn = true;
	//AfxBeginThread(KarotazhThread, this);
}

void CTestNetDlg::OnBnClickedStop()
{
	do_learn = false;
}
