// TestNetDlg.h : header file
//

#pragma once

#include "mnet.h"

// CTestNetDlg dialog
class CTestNetDlg : public CDialog
{
	
	//Matrix* pmTest;
	//DumbClass* pDC;

// Construction
public:
	NN::MNet* p_net;

	CTestNetDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
	enum { IDD = IDD_TESTNET_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
protected:
	virtual void OnOK();
public:
	afx_msg void OnBnClickedCreate();
	afx_msg void OnClose();
	afx_msg void OnBnClickedKar();
	afx_msg void OnBnClickedStop();
};
