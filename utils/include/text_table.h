#ifndef __TEXT_TABLE_H__
#define __TEXT_TABLE_H__
 
#include "common.h"
#include "matrix.h"
#include <sstream>

// simple class to format text tables using TEX-like notation
class _CLASS_DECLSPEC text_table {
public:
	struct fmt_flags {
		bool sep_cols;		// whether to visually separate columns
		bool sep_rows;		// whether to visually separate rows
		int w;				// default width for all fields
		bool wrap;			// whether to wrap long text on several lines
		int align;			// 0 = right, 1 = left
		bool online;		// if true, rows are printed immidiately
		bool no_borders;	// if true, borders are replaced by whitespaces

		char hline_char;
		//std::ios::fmtflags f;	// custom formatting flags (applied to all fields)

		// standard settings here
		fmt_flags();
	};

	// manipulators to ease header & lines formatting
	struct begh {};
	struct begr {};
	struct endr {};
	struct endrh {};
	struct hline {};
	struct eline {};
	struct borders {
		borders(int draw_borders) : b_(draw_borders) {}
		int b_;
	};

	// text table ctors
	text_table(bool online = false);
	text_table(const fmt_flags& fmt);
	// construct with header spec
	text_table(const std::string& header);
	// copy ctor
	text_table(const text_table&);

	// table's format accessor
	fmt_flags& fmt();

	// set header for table
	void set_header(const std::string& h);
	// add next row with content
	void add_row(const std::string& l);
	// remove n-th line
	bool rem_row(ulong line_num);
	// return rows number
	ulong row_num() const;
	// retrieve row
	std::string get_row(ulong num) const;
	// clear table
	void clear();

	// get last formatted row - usable for online mode
	std::string last_row() const;
	// get stream, used to inupt data via << operator
	std::ostream& row_stream();
	// how many rows were added?
	ulong size() const;
	// get ready-to-use table in a string form
	std::string format(ulong start_row = 0, ulong how_many = 0);
	// get formatted table content in matrix form
	TMatrix< std::string > content(ulong start_row = 0, ulong how_many = 0);

	// overload operators << for any type
	template< class T >
	text_table& operator <<(const T& v) {
		row_manip< T >::put(*this, v);
		return *this;
	}

private:
	template< class T, class = void >
	struct row_manip {
		// overload operators << for any type
		static void put(text_table& tt, const T& v) {
			tt.line_ << v;
		}
	};

	// copy data from one table to another
	template< class unused >
	struct row_manip< text_table, unused > {
		static void put(text_table& dst, const text_table& src) {
			ulong n = src.row_num();
			for(ulong i = 0; i < n; ++i)
				dst.add_row(src.get_row(i));
		}
	};

	// explicit spec for manipulators
	template< class unused >
	struct row_manip< text_table::begh, unused > {
		static void put(text_table& tt, const text_table::begh&) {
			tt.line_.str("");
			tt.line_.clear();
			tt.inp_state_ = 1;
		}
	};

	template< class unused >
	struct row_manip< text_table::begr, unused > {
		static void put(text_table& tt, const text_table::begr&) {
			tt.line_.str("");
			tt.line_.clear();
			tt.inp_state_ = 0;
		}
	};

	template< class unused >
	struct row_manip< text_table::endr, unused > {
		static void put(text_table& tt, const text_table::endr&) {
			if(tt.inp_state_ == 1) {
				tt.set_header(tt.line_.str());
				// clear existing table contents
				tt.clear();
				// auto-switch to rows input
				tt.inp_state_ = 0;
			}
			else
				tt.add_row(tt.line_.str());
			// clear line
			tt.line_.str("");
			tt.line_.clear();
		}
	};

	template< class unused >
	struct row_manip< text_table::endrh, unused > {
		static void put(text_table& tt, const text_table::endrh&) {
			// end row as usual
			tt << endr();
			// add hline
			tt << "\\hline" << endr();
		}
	};

	template< class unused >
	struct row_manip< text_table::hline, unused > {
		static void put(text_table& tt, const text_table::hline&) {
			// add hline
			tt << "\\hline" << endr();
		}
	};

	template< class unused >
	struct row_manip< text_table::eline, unused > {
		static void put(text_table& tt, const text_table::eline&) {
			// add hline
			tt << "\\eline" << endr();
		}
	};

	template< class unused >
	struct row_manip< text_table::borders, unused > {
		static void put(text_table& tt, const text_table::borders& b) {
			if(b.b_ > 0)
				tt.fmt().no_borders = false;
			else
				tt.fmt().no_borders = true;
		}
	};

	// pimpl
	class tt_impl;
	smart_ptr< tt_impl > pimpl_;

	// formatters
	int inp_state_;
	std::ostringstream line_;
};

// manipulators typedefs for easier typing
typedef text_table::begh tt_begh;
typedef text_table::begr tt_begr;
typedef text_table::endr tt_endr;
typedef text_table::endrh tt_endrh;
typedef text_table::hline tt_hline;
typedef text_table::eline tt_eline;

// overload operator << for text_table
_CLASS_DECLSPEC std::ostream& operator <<(std::ostream& os, text_table& tt);

#endif

