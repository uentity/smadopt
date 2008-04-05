#ifndef _KMEANS_H
#define _KMEANS_H

#include "kmeans_common.h"

namespace KM {

	class _CLASS_DECLSPEC kmeans
	{
	public:
		typedef std::vector<ul_vec> vvul;

	private:
		class kmeans_impl;
		smart_ptr< kmeans_impl > pimpl_;

	public:
		//options
		km_opt opt_;

		kmeans();
		~kmeans() {};

		void set_def_opt();

		void find_clusters(const Matrix& data, ulong clust_num, ulong maxiter = 1000, 
			bool skip_online = false, const Matrix* pCent = NULL, bool use_prev_cent = false);
		void find_clusters_f(const Matrix& data, const Matrix& f, ulong clust_num, 
			ulong maxiter = 1000, const Matrix* pCent = NULL, bool use_prev_cent = false);
		void restart(const Matrix& data, ulong clust_num = 0, ulong maxiter = 100, 
			bool skip_online = false, const Matrix* pCent = NULL, bool use_prev_cent = false);

		Matrix drops_homo(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100, double quant_mult = 0.3);
		Matrix drops_hetero(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100, double quant_mult = 0.3);
		Matrix drops_hetero_map(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100, double quant_mult = 0.3);
		Matrix drops_hetero_simple(const Matrix& data, const Matrix& f, double drops_mult = 0.8, ulong maxiter = 100, double quant_mult = 0.3);

		const Matrix& get_centers() const;

		const ulMatrix& get_ind() const;

		const Matrix& get_norms() const;

		const vvul& get_aff() const;
	};
}

#endif	// _KMEANS_H
