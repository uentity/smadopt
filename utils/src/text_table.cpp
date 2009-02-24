#include "text_table.h"
#include "matrix.h"

#include <string.h>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <list>
//#ifdef __GNUC__
//#include <bits/stl_multimap.h>
//#else
#include <map>
//#endif
#include <iomanip>
#include <numeric>

using namespace std;
//using namespace hybrid_adapt;

// text_table class implementation
// pimpl

// namespace hybrid_adapt {
namespace {

//inline
//string trim(const string& s) {
//	istringstream is;
//	ostringstream os;
//	string tmp;
//	// trim initial spaces
//	is.str(s);
//	is >> skipws >> tmp; os << tmp;
//	is >> noskipws >> tmp; os << tmp;
//	// rotate string
//	tmp = os.str();
//	reverse(tmp.begin(), tmp.end());
//	// read rotated string
//	is.str(tmp);
//	is.clear();
//	os.str("");
//	os.clear();
//	is >> skipws >> tmp; os << tmp;
//	is >> noskipws >> tmp; os << tmp;
//	// rotate again to get normal result
//	tmp = os.str();
//	reverse(tmp.begin(), tmp.end());
//	return tmp;
//}

const char* spaces = " \n\r\t";

inline
string trim(const string& ss) {
	string s = ss;
	while(s.size() > 0 && strchr(spaces, s[0]) != NULL)
		s.erase(s.begin());
	while(s.size() > 0 && strchr(spaces, s[s.size() - 1]) != NULL)
		s.erase(s.size() - 1);
	return s;
}

inline
string trim_fill(const string& ss, char fill_c = '_') {
	string s = ss;
	string::size_type pos = 0;
	for(string::iterator p = s.begin(); p != s.end() && strchr(spaces, *p) != NULL; ++p)
		*p = fill_c;
	for(string::reverse_iterator p = s.rbegin(); p != s.rend() && strchr(spaces, *p) != NULL; ++p)
		*p = fill_c;
	return s;
}

}	// end of private namespace

// fmt_flags default ctor
text_table::fmt_flags::fmt_flags()
	: sep_cols(false), sep_rows(false), w(NW), wrap(true), align(0)
{}

class text_table::tt_impl {
public:
	typedef vector< string > svector;

	// format flags
	text_table::fmt_flags fmt_;

	// header and body spec
	//string head_;
	typedef std::list< string > body_t;
	body_t body_;

	// this flag is set once for each table in adjust_wb
	bool wb_ready_;
	// cols lengths
	typedef vector< int > ivector;
	ivector w_;
	// indexes of hlines
	typedef multimap< int, int > hlines_t;
	hlines_t hl_ids_;

	// string stream for formatted output
	ostringstream os_;
	// last row goes here - needed for online mode
	string lastr_;

	// cells extracted from table spec comes here
	typedef TMatrix< string > str_mat;
	str_mat cells_;
	// cols delimiters
	str_mat delims_;

	// def ctor assumes def fmt flags
	tt_impl() : wb_ready_(false) {}
	// ctor with custom formatting
	tt_impl(const text_table::fmt_flags& fmt)
		: fmt_(fmt), wb_ready_(false)
	{}

	// functions
//	void set_header(const string& h) {
//		head_ = h;
//		// process header to count columns
//		ulong cols = 0;
//		string::size_type pos = 0;
//		while((pos = head_.find('&', pos)) != string::npos) {
//			if(head_[pos - 1] != '\\')
//				++cols;
//			++pos;
//		}
//		//		++cols;
//		w_.resize(cols);
//		fill(w_.begin(), w_.end(), fmt_.w);
//	}

	void set_header(const string& h) {
		istringstream is;
		is.str(h);

		// clear all fmt info
		w_.clear();
		delims_.clear();
		cells_.clear();
		body_.clear();
		hl_ids_.clear();

		// process header to count columns & borders
		string d;
		int w;
		while(is) {
			// read delimiter
			is >> d;
			if(d == "_")
				delims_.push_back(" ", false);
			else if(d == "-")
				delims_.push_back("", false);
			else
				delims_.push_back(d, false);

			// read width
			if(is && (is >> w))
				w_.push_back(w);
		}

		// ensure that size of delimiters = size(w_) + 1
		if(delims_.size() <= w_.size())
			delims_.Resize(1, w_.size() + 1, "");

		// if we are online-make corrections now
		wb_ready_ = false;
		if(fmt_.online)
			adjust_wb();

		// set global stream fmt
		if(fmt_.align == 0 || fmt_.align == 2)
			os_ << left;
		else
			os_ << right;
	}

	void set_width(const ivector& w) {
		if(w.size() < w_.size()) return;
		copy(w.begin(), w.begin() + w_.size(), w_.begin());
	}

	void add_row(const string& l) {
		body_.push_back(l);
	}

	void add_hline() {
		body_.push_back("\\hline");
	}

	bool rem_row(ulong line_n) {
		if(line_n >= body_.size()) return false;
		body_t::iterator r(body_.begin());
		advance(r, line_n);
		body_.erase(r);
		return true;
	}

//	// format functions
//	void draw_hline(bool side_borders = false) {
//		// calc full length of all row
//		ulong full_w = accumulate(w_.begin(), w_.end(), 0);
//		for(ulong i = 0; i <= w_.size(); ++i)
//			full_w += delims_[i].size();
//
//		// draw hline
//		if(side_borders && full_w > 1) {
//			os_ << "|";
//			full_w -= 2;
//		}
//		os_ << setw(full_w) << setfill('_') << "";
//		if(side_borders)
//			os_ << "|";
//
//		// restore default fill
//		os_ << setfill(' ') << std::endl;
//	}

	void draw_hline(char fill_c = '_', bool side_borders = false) {
		if(w_.size() == 0) return;

		// calc full length of all row
		long full_w = accumulate(w_.begin(), w_.end(), 0);
		for(ulong i = 0; i <= w_.size(); ++i)
			full_w += delims_[i].size();

		ostringstream os;
		// copy format from global stream
		os.copyfmt(os_);
		// draw left border
		string b1, b2;
		if(side_borders) {
			b1 = trim(delims_[0]);
			b2 = trim(delims_[w_.size()]);
			os << b1;
			full_w = max< long >(full_w - b1.size() - b2.size(), 0);
		}
		// draw hline
		os << setw(full_w) << setfill(fill_c) << "";
		//draw right border
		if(side_borders) {
			os << b2;
		}
		os << std::endl;
		// restore default fill
		//os_ << setfill(' ') << std::endl;

		lastr_ = os.str();
		os_ << lastr_;
	}

	void draw_inner_hline(char fill_c = '_') {
		// new algo - trims whitespaces in delimiters
		ostringstream os;
		// copy format from global stream
		os.copyfmt(os_);
		os << setfill(fill_c);

		string d;
		for(ulong i = 0; i < w_.size(); ++i) {
			// replace whitespaces in delimiter
			os << trim_fill(delims_[i], fill_c);
			// draw hline in field
			os << setw(w_[i]) << "";
		}
		os << trim_fill(delims_[w_.size()], fill_c) << endl;

		lastr_ = os.str();
		os_ << lastr_;

//		// copy
//		// prepare filled row
//		str_mat hline(1, w_.size());
//		for(ulong i = 0; i < w_.size(); ++i) {
//			hline[i].resize(w_[i], fill_c);
//		}
//		print_row(hline);
	}

	// formatted field output
	void print_row(str_mat& sv) {
		bool wrap2next = false;
		ostringstream os;
		// copy format from global stream
		os.copyfmt(os_);
		string f;
		for(ulong i = 0; i < w_.size(); ++i)
		{
			// print field
			//if(fmt_.sep_cols) os_ << setw(1) << '|';
			// print column delimiter
			os << delims_[i];
			// print field
			os << setw(w_[i]);
			if(i < sv.size()) {
				// process no more than field width
				f = sv[i].substr(0, w_[i]);
				// field can span multiple rows so we need to handle it
				// correctly
				string::size_type eol_pos = f.find('\n', 0);
				if(eol_pos != string::npos) {
					f = f.substr(0, eol_pos);
					// remove eol symbol from source
					sv[i].erase(eol_pos, 1);
					wrap2next = true;
				}
				// now check for a wrapping condition
				if(fmt_.wrap || wrap2next) {
					sv[i].erase(0, f.size());
					if(sv[i].size() > 0) wrap2next = true;
				}
				// add whitespaces for centering if needed
				if(fmt_.align == 2) {
					string::size_type ws_sz = (w_[i] - f.size()) >> 1;
					f.insert(0, string(ws_sz, ' '));
				}
				// print field
				os << f;
			}
			else
				os << "";
		}
		// finish line
		os << delims_[w_.size()];
		// if(fmt_.sep_cols) os_ << setw(1) << '|';
		os << std::endl;

		//save line
		lastr_ += os.str();
		os_ << os.str();
		// recursively print wrapped text on the next line
		if(wrap2next) print_row(sv);
	}

	void format_row(const string& l) {
		if(l.size() == 0) return;
		string::size_type a = 0, b;

		// extracted fields
		string f;
		// row of fields
		str_mat cr;
		// if hline detected
		//bool is_hline = false;
		do {
			b = l.find('&', a);
			// extract field
			f = trim(l.substr(a, b - a));
			a = b + 1;
			// check if this is an hline command
			if(f == "\\hline") {
				hl_ids_.insert(hlines_t::value_type(cells_.row_num(), 0));
				//is_hline = true;
				return;
			}
			else if(f == "\\eline") {
				hl_ids_.insert(hlines_t::value_type(cells_.row_num(), 1));
				//is_hline = true;
				return;
			}
			cr.push_back(f, false);
		} while(b != string::npos);

		// check row size
		if(cr.col_num() != w_.size())
			cr.Resize(1, w_.size(), string(""));
		// append new row
		cells_ &= cr;
		// if we are working online - print record
	}

	// adjust widths and borders
	// returns full table width in chars
	ulong adjust_wb() {
		// check if all is done for current table
		if(wb_ready_) return 0;
		wb_ready_ = true;

		// calc full tbl width
		str_mat tmp;
		ulong full_w = 0;
		// if width == -2 then use default width
		replace(w_.begin(), w_.end(), -2, fmt_.w);
		// if we are working online - then no auto-width is allowed
		if(fmt_.online)
			replace_if(w_.begin(), w_.end(), bind2nd(less< int >(), 1), fmt_.w);
		for(ulong i = 0; i < w_.size(); ++i) {
			if(w_[i] <= 0) {
				ulong max_colw = 0;
				for(ulong j = 0; j < cells_.row_num(); ++j) {
					string& cell = cells_(j, i);
					// find max line length in multiple-line row
					ulong max_linew = 0;
					string::size_type line_a = 0, line_b;
					while(line_a < cell.size() && (line_b = cell.find('\n', line_a)) != string::npos) {
						max_linew = max(max_linew, line_b - line_a);
						line_a = line_b + 1;
					}
					if(max_linew == 0) max_linew = cell.size();
					// find max length in column
					if(j == 0 || max_linew > max_colw)
						max_colw = max_linew;
				}

				if(w_[i] == 0) {
					w_[i] = max_colw;
					// add extra space that will be removed
					if(fmt_.sep_cols) w_[i] += 2;
				}
				// if w_[i] < 0 then auto-width with global limitation is used
				else if(fmt_.sep_cols) w_[i] = min< ulong >(max_colw + 2, fmt_.w);
				else w_[i] = min< ulong >(max_colw, fmt_.w);
			}
			// width was specified explicitly
			full_w += w_[i];

			// adjust borders
			if(fmt_.sep_cols) {
				// add space to left border
				if(delims_.size() > i) delims_[i] += " ";
				// add space to right border
				if(delims_.size() > i + 1) delims_[i + 1].insert(0, " ");
				// decrease field width
				w_[i] -= 2;
			}
		}
		// ensure that all fields at least 1 char long
		replace_if(w_.begin(), w_.end(), bind2nd(less< int >(), 1), 1);
		return full_w;
	}

	std::string format(ulong start_row, ulong how_many) {
		// convert table spec to cells matrix
		cells_.clear();
		hl_ids_.clear();
		for(body_t::const_iterator p = body_.begin(); p != body_.end(); ++p)
			format_row(*p);

		// clear output stream
		os_.str("");
		os_.clear();
		if(!cells_.row_num()) return "";

		// calc full tbl width
		adjust_wb();

		// print body
		start_row = min(start_row, cells_.row_num() - 1);
		if(!how_many) how_many = cells_.row_num();
		how_many = min(start_row + how_many, cells_.row_num());

		str_mat row;
		char hfill[] = {'_', ' '};
		pair< hlines_t::const_iterator, hlines_t::const_iterator > hl_rng;
		for(ulong i = start_row; i <= how_many; ++i) {
			// draw hlines
			if(fmt_.sep_rows)
				draw_hline();
			hl_rng = hl_ids_.equal_range(i);
			for(;hl_rng.first != hl_rng.second; ++hl_rng.first) {
				if(i == start_row)
					draw_hline();
				else if(i == how_many)
					draw_hline(hfill[0], true);
				else
					draw_inner_hline(hfill[hl_rng.first->second]);
			}

			// print row
			if(i == how_many) break;
			row <<= cells_.GetRows(i);
			lastr_ = "";
			print_row(row);
//			if(fmt_.sep_rows || hl_ids_.find(i + 1) != hl_ids_.end()) {
//				if(i == cells_.row_num() - 1)
//					draw_hline('_', true);
//				else
//					draw_inner_hline();
//			}
		}

		return os_.str();
	}

	str_mat content(ulong start_row, ulong how_many) {
		// convert table spec to cells matrix
		cells_.clear();
		for(body_t::const_iterator p = body_.begin(); p != body_.end(); ++p)
			format_row(*p);

		if(!cells_.row_num()) return str_mat();
		// return body
		start_row = min(start_row, cells_.row_num() - 1);
		if(!how_many) how_many = cells_.row_num();
		how_many = min(how_many, cells_.row_num() - start_row);

		return cells_.GetRows(start_row, how_many);
	}
};

// ----------------------------------------- text_table implementation -------------------------------------------------
text_table::text_table(bool online)
	: pimpl_(new tt_impl), inp_state_(1)
{
	pimpl_->fmt_.online = online;
}

text_table::text_table(const fmt_flags& fmt)
	: pimpl_(new tt_impl(fmt)), inp_state_(1)
{}

text_table::text_table(const string& header)
	: pimpl_(new tt_impl), inp_state_(1)
{
	set_header(header);
}

text_table::text_table(const text_table& lhs)
	: pimpl_(lhs.pimpl_), inp_state_(lhs.inp_state_)
{
	line_.str(lhs.line_.str());
	line_.clear();
}

text_table::fmt_flags& text_table::fmt() {
	return pimpl_->fmt_;
}

// set table's format
//void text_table::set_fmt(const fmt_flags& f) {
//	pimpl_->fmt_ = f;
//}

// set header for table
void text_table::set_header(const std::string& h) {
	pimpl_->set_header(h);
}

// add next line
void text_table::add_row(const std::string& l) {
	pimpl_->add_row(l);
}

// remove n-th line
bool text_table::rem_row(ulong line_num) {
	return pimpl_->rem_row(line_num);
}

// get ready-to-use table in a string form
std::string text_table::format(ulong start_row, ulong how_many) {
	return pimpl_->format(start_row, how_many);
}

std::string text_table::last_row() const {
	return pimpl_->lastr_;
}

TMatrix< std::string > text_table::content(ulong start_row, ulong how_many) {
	return pimpl_->content(start_row, how_many);
}

ostream& text_table::row_stream() {
	return line_;
}

std::ostream& operator <<(std::ostream& os, text_table& tt) {
	os << tt.format();
	return os;
}

//}	// end of hybrid_adapt namespace
