#ifndef DETERM_ANNEALING_H_
#define DETERM_ANNEALING_H_

#include "common.h"
#include "matrix.h"

namespace DA {

class _CLASS_DECLSPEC determ_annealing
{
public:
	typedef std::vector<ul_vec> vvul;

	determ_annealing();
	~determ_annealing();

	//clusterization using deterministic annealing
	void find_clusters(const Matrix& data, const Matrix& f, ulong clust_num, ulong maxiter);

	const Matrix& get_centers() const;

	const ulMatrix& get_ind() const;

	const Matrix& get_norms() const;

	const vvul& get_aff() const;

private:
	class da_impl;
	smart_ptr< da_impl > pimpl_;
};

}	//end of namespace DA

#endif // DETERM_ANNEALING_H_
