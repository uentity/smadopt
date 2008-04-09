#include "ga_common.h"
#include <algorithm>

using namespace GA;
using namespace std;

ga_stat::ga_stat(ulong iterations)
{
	if(iterations) reserve(iterations);
}

void ga_stat::add_record(ulong chrom_cnt, double best_ff, double mean_ff, ulong stall_g)
{
	chrom_cnt_.push_back(chrom_cnt);
	best_ff_.push_back(best_ff);
	mean_ff_.push_back(mean_ff);
	stall_cnt_.push_back(stall_g);
}

void ga_stat::reserve(ulong iterations)
{
	chrom_cnt_.reserve(iterations);
	best_ff_.reserve(iterations);
	mean_ff_.reserve(iterations);
	stall_cnt_.reserve(iterations);
}

void ga_stat::clear()
{
	chrom_cnt_.clear();
	best_ff_.clear();
	mean_ff_.clear();
	stall_cnt_.clear();
}

ulong ga_stat::size() const
{
	return chrom_cnt_.size();
}

/*
ga_stat& ga_stat::operator =(const ga_stat& s)
{
	chrom_cnt_ = s.chrom_cnt_;
	best_ff_.assign(s.best_ff_.begin(), s.best_ff_.end());
	mean_ff_.assign(s.mean_ff_.begin(), s.mean_ff_.end());
	stall_count_.assign(s.stall_count_.begin(), s.stall_count_.end());
	return *this;
}
*/

const ga_stat& ga_stat::operator +=(const ga_stat& s)
{
	if(size() == 0) return (*this = s);
	else if(size() != s.size()) return (*this);
	chrom_cnt_ += s.chrom_cnt_;
	best_ff_ += s.best_ff_;
	mean_ff_ += s.mean_ff_;
	stall_cnt_ += s.stall_cnt_;
	/*
	transform(chrom_count_.begin(), chrom_count_.end(), s.chrom_count_.begin(), chrom_count_.begin(), 
		plus< ulong >());
	transform(best_ff_.begin(), best_ff_.end(), s.best_ff_.begin(), best_ff_.begin(),
		plus< double >());
	transform(mean_ff_.begin(), mean_ff_.end(), s.mean_ff_.begin(), mean_ff_.begin(),
		plus< double >());
	transform(stall_count_.begin(), stall_count_.end(), s.stall_count_.begin(), stall_count_.begin(),
		plus< ulong >());
	*/
	return (*this);
}

const ga_stat& ga_stat::operator /=(ulong cnt)
{
	chrom_cnt_ /= cnt;
	best_ff_ /= cnt;
	mean_ff_ /= cnt;
	stall_cnt_ /= cnt;
	/*
	transform(chrom_count_.begin(), chrom_count_.end(), chrom_count_.begin(), bind2nd(divides< ulong >(), cnt));
	transform(best_ff_.begin(), best_ff_.end(), best_ff_.begin(), bind2nd(divides< double >(), cnt));
	transform(mean_ff_.begin(), mean_ff_.end(), mean_ff_.begin(), bind2nd(divides< double >(), cnt));
	transform(stall_count_.begin(), stall_count_.end(), stall_count_.begin(), bind2nd(divides< ulong >(), cnt));
	*/
	return *this;
}
