#include "ga_common.h"
#include <algorithm>

using namespace GA;
using namespace std;

ga_stat::ga_stat(ulong iterations)
	: startt_(0), timer_flushed_(true)
{
	if(iterations) reserve(iterations);
}

void ga_stat::add_record(ulong chrom_cnt, double best_ff, double mean_ff, ulong stall_g)
{
	chrom_cnt_.push_back(chrom_cnt);
	best_ff_.push_back(best_ff);
	mean_ff_.push_back(mean_ff);
	stall_cnt_.push_back(stall_g);
	//timer operations
	//save elapsed time in seconds
	timer_.push_back(double(clock()) / CLOCKS_PER_SEC);
	if(timer_flushed_) {
		startt_ = timer_[timer_.size() - 1];
		timer_flushed_ = false;
	}
}

void ga_stat::reserve(ulong iterations)
{
	chrom_cnt_.reserve(iterations);
	best_ff_.reserve(iterations);
	mean_ff_.reserve(iterations);
	stall_cnt_.reserve(iterations);
	timer_.reserve(iterations);
}

void ga_stat::clear()
{
	chrom_cnt_.clear();
	best_ff_.clear();
	mean_ff_.clear();
	stall_cnt_.clear();
	timer_.clear();
	reset_timer();
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
	timer_ += s.timer_;
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
	timer_ /= cnt;
	/*
	transform(chrom_count_.begin(), chrom_count_.end(), chrom_count_.begin(), bind2nd(divides< ulong >(), cnt));
	transform(best_ff_.begin(), best_ff_.end(), best_ff_.begin(), bind2nd(divides< double >(), cnt));
	transform(mean_ff_.begin(), mean_ff_.end(), mean_ff_.begin(), bind2nd(divides< double >(), cnt));
	transform(stall_count_.begin(), stall_count_.end(), stall_count_.begin(), bind2nd(divides< ulong >(), cnt));
	*/
	return *this;
}

void ga_stat::reset_timer() {
	startt_ = 0;
	timer_flushed_ = true;
}

double ga_stat::sec_elapsed(ulong epoch) const {
	epoch = min(epoch, timer_.size() - 1);
	return timer_[epoch] - startt_;
}

ostream& ga_stat::print_elapsed(ostream& outs, ulong epoch) const {
	double sec = sec_elapsed(epoch);
	double min = 0, hour = 0;
	if(sec >= 60) {
		min = floor(sec / 60);
		sec -= 60 * min;
	}
	if(min >= 60) {
		hour = floor(min / 60);
		min -= 60 * hour;
	}
	//print time
	std::ostringstream buf;
	buf.str("");
	if(hour > 0) buf << hour << " h ";
	if(min > 0) buf << min << " m ";
	buf << sec << " s";
	//std::string s = buf.str();
	outs << std::right << buf.str();
}

ostream& ga_stat::print(ostream& outs, bool print_header, bool decorate_time) {
	if(print_header) {
		outs << setw(NW) << "Generation" << ' ' << setw(NW) << "f-count" << ' ' << setw(NW) << "Best f(x)" << ' ';
		outs << setw(NW) << "Mean" << ' ' << setw(NW) << "StallGen" << ' ' << setw(TIME_NW) << "Time elapsed" << endl;
	}
	//print data
	for(ulong i = 0; i < chrom_cnt_.size(); ++i) {
		outs << setw(NW) << i << ' ' << setw(NW) << chrom_cnt_[i] << ' ' << setw(NW) << best_ff_[i] << ' ';
		outs << setw(NW) << mean_ff_[i] << ' ' << setw(NW) << stall_cnt_[i] << ' ';
		if(decorate_time) {
			outs << setw(TIME_NW);
			print_elapsed(outs, i);
		}
		else
			outs << setw(NW) << timer_[i];
		outs << endl;
	}
	return outs;
}
