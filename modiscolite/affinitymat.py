# affinitymat.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import sklearn
import sklearn.manifold

import numpy as np

import scipy
from scipy.sparse import coo_matrix

from numba import njit
from numba import prange

from . import util
from . import gapped_kmer


@njit('float64(float64[:], int64[:], int64[:], float64[:], int64[:], int64[:], int64, int64)')
def _sparse_vv_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, i, j):
	xi = X_indptr[i]
	yj = Y_indptr[j]
	dot = 0.0

	while xi < X_indptr[i+1] and yj < Y_indptr[j+1]:
		x_col = X_indices[xi]
		x_data = X_data[xi]

		y_col = Y_indices[yj]
		y_data = Y_data[yj]

		if x_col == y_col:
			dot += x_data * y_data
			xi += 1
			yj += 1

		elif x_col < y_col:
			xi += 1

		else:
			yj += 1

	return dot

@njit(parallel=True)
def _sparse_mm_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, k):
	n_rows = len(Y_indptr) - 1

	neighbors = np.empty((n_rows, k), dtype='int32')
	sims = np.empty((n_rows, k), dtype='float64')

	for i in prange(n_rows):
		dot = np.zeros(n_rows, dtype='float64')

		for j in range(n_rows):
			xdot = _sparse_vv_dot(X_data, X_indices, X_indptr, X_data, X_indices, X_indptr, i, j)
			ydot = _sparse_vv_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, i, j)
			dot[j] = max(xdot, ydot)

		dot_argsort = np.argsort(-dot, kind='mergesort')[:k]
		neighbors[i] = dot_argsort
		sims[i] = dot[dot_argsort]

	return sims, neighbors

@njit(parallel=True)
def _indexed_topk(X_data, X_indices, X_indptr, XT_data, XT_rows, XT_indptr,
	YT_data, YT_rows, YT_indptr, n_rows, k, n_blocks):
	"""_sparse_mm_dot's result, from an inverted index instead of all pairs.

	For row i, every row j sharing a gapped k-mer f with it gets
	X[i, f] * X[j, f] (and X[i, f] * Y[j, f]) added, f in increasing order:
	the products _sparse_vv_dot's merge adds, in the order it adds them, so
	each dot product is bit-identical; rows sharing no k-mer stay at exactly
	0.0 without being visited. The top k are then ordered as
	np.argsort(-dot, kind='mergesort') orders the full row: positive dots by
	value (ties by index), then the zero dots by index, then the negative ones.
	"""

	neighbors = np.empty((n_rows, k), dtype='int32')
	sims = np.empty((n_rows, k), dtype='float64')
	block = (n_rows + n_blocks - 1) // n_blocks

	for b in prange(n_blocks):
		start = b * block
		end = min(n_rows, start + block)
		xacc = np.zeros(n_rows, dtype='float64')
		yacc = np.zeros(n_rows, dtype='float64')
		touched = np.zeros(n_rows, dtype=np.bool_)
		hits = np.empty(n_rows, dtype='int64')

		for i in range(start, end):
			n_hits = 0
			for xi in range(X_indptr[i], X_indptr[i+1]):
				f = X_indices[xi]
				a = X_data[xi]
				for t in range(XT_indptr[f], XT_indptr[f+1]):
					j = XT_rows[t]
					xacc[j] += a * XT_data[t]
					if not touched[j]:
						touched[j] = True
						hits[n_hits] = j
						n_hits += 1
				for t in range(YT_indptr[f], YT_indptr[f+1]):
					j = YT_rows[t]
					yacc[j] += a * YT_data[t]
					if not touched[j]:
						touched[j] = True
						hits[n_hits] = j
						n_hits += 1

			js = np.sort(hits[:n_hits])
			d = np.empty(n_hits, dtype='float64')
			for h in range(n_hits):
				j = js[h]
				d[h] = max(xacc[j], yacc[j])

			# positive dots, by value then index (js is in index order and
			# the sort is stable)
			pos = np.where(d > 0)[0]
			order = pos[np.argsort(-d[pos], kind='mergesort')]
			n_out = 0
			for h in order[:k]:
				neighbors[i, n_out] = js[h]
				sims[i, n_out] = d[h]
				n_out += 1

			# then the zero dots, visited or not, by index
			j = 0
			while n_out < k and j < n_rows:
				if touched[j]:
					v = max(xacc[j], yacc[j])
					if v == 0:
						neighbors[i, n_out] = j
						sims[i, n_out] = v
						n_out += 1
				else:
					neighbors[i, n_out] = j
					sims[i, n_out] = 0.0
					n_out += 1
				j += 1

			# then the negative dots, closest to zero first
			if n_out < k:
				neg = np.where(d < 0)[0]
				order = neg[np.argsort(-d[neg], kind='mergesort')]
				for h in order[:k - n_out]:
					neighbors[i, n_out] = js[h]
					sims[i, n_out] = d[h]
					n_out += 1

			for h in range(n_hits):
				j = hits[h]
				xacc[j] = 0.0
				yacc[j] = 0.0
				touched[j] = False

	return sims, neighbors


def _sparse_mm_dot_indexed(X, Y, k):
	"""_sparse_mm_dot(X.data, X.indices, X.indptr, Y.data, Y.indices,
	Y.indptr, k), computed through an inverted index (see _indexed_topk).

	The gapped k-mer matrices are 5**max_len columns wide, so the columns in
	use are renumbered first; the renumbering keeps their order, and with it
	the order the products are added in. _sparse_vv_dot's merge assumes each
	row's columns are sorted; if they are not, the original is used.
	"""

	if not (X.has_sorted_indices and Y.has_sorted_indices):
		return _sparse_mm_dot(X.data, X.indices.astype('int64'), X.indptr.astype('int64'),
			Y.data, Y.indices.astype('int64'), Y.indptr.astype('int64'), k)

	n = X.shape[0]
	cols, inverse = np.unique(np.concatenate([X.indices, Y.indices]), return_inverse=True)
	Xr = scipy.sparse.csr_matrix((X.data, inverse[:len(X.indices)], X.indptr), shape=(n, len(cols)))
	Yr = scipy.sparse.csr_matrix((Y.data, inverse[len(X.indices):], Y.indptr), shape=(n, len(cols)))
	XT = Xr.tocsc()
	YT = Yr.tocsc()
	XT.sort_indices()
	YT.sort_indices()

	n_blocks = min(n, 1024)
	return _indexed_topk(Xr.data.astype('float64'), Xr.indices.astype('int64'), Xr.indptr.astype('int64'),
		XT.data.astype('float64'), XT.indices.astype('int64'), XT.indptr.astype('int64'),
		YT.data.astype('float64'), YT.indices.astype('int64'), YT.indptr.astype('int64'),
		n, k, n_blocks)


def cosine_similarity_from_seqlets(seqlets, n_neighbors, sign, topn=20, 
	min_k=4, max_k=6, max_gap=15, max_len=15, max_entries=500, 
	alphabet_size=4):

	X_fwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, 
		min_k, max_k, max_gap, max_len, max_entries, True, sign)

	X_bwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, min_k, max_k, max_gap, 
			max_len, max_entries, False, sign)

	X = sklearn.preprocessing.normalize(X_fwd, norm='l2', axis=1)
	Y = sklearn.preprocessing.normalize(X_bwd, norm='l2', axis=1)

	n, d = X.shape
	k = min(n_neighbors+1, n)
	return _sparse_mm_dot_indexed(X, Y, k)


def jaccard_from_seqlets(seqlets, min_overlap, filter_seqlets=None, 
	seqlet_neighbors=None):

	all_fwd_data, all_rev_data = util.get_2d_data_from_patterns(seqlets)

	if filter_seqlets is None:
		filter_seqlets = seqlets
		filters_all_fwd_data = all_fwd_data
		filters_all_rev_data = all_rev_data
	else:
		filters_all_fwd_data, filters_all_rev_data = util.get_2d_data_from_patterns(filter_seqlets)

	if seqlet_neighbors is None:
		seqlet_neighbors = [list(range(len(filter_seqlets)))
							for x in seqlets] 

	#apply the cross metric
	affmat_fwd = jaccard(seqlet_neighbors=seqlet_neighbors, 
		X=filters_all_fwd_data,
		Y=all_fwd_data, min_overlap=min_overlap, func=int, 
		return_sparse=True)

	affmat_rev = jaccard(seqlet_neighbors=seqlet_neighbors,
		X=filters_all_rev_data, Y=all_fwd_data,
		min_overlap=min_overlap, func=int,
		return_sparse=True) 

	affmat = np.maximum(affmat_fwd, affmat_rev)
	return affmat


def jaccard(X, Y, min_overlap=None, seqlet_neighbors=None, func=np.ceil, 
	return_sparse=False):

	if seqlet_neighbors is None:
		seqlet_neighbors = np.tile(np.arange(X.shape[0]), (Y.shape[0], 1))

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0,0), (n_pad, n_pad), (0,0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")
	else:
		n_pad = 0 

	len_output = 1 + Y.shape[1] - X.shape[1] 

	X = X.astype('float32')
	Y = Y.astype('float32')

	seqlet_neighbors = seqlet_neighbors.astype('int32')
	scores = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], len_output), dtype='float32')
	_jaccard(X, Y, seqlet_neighbors, scores)

	if return_sparse == True:
		return scores.max(axis=-1)

	argmaxs = np.argmax(scores, axis=-1)
	idxs = np.arange(seqlet_neighbors.shape[1])
	results = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], 2))
	for i in range(Y.shape[0]):
		results[i, :, 0] = scores[i][idxs, argmaxs[i]]
		results[i, :, 1] = argmaxs[i] - n_pad

	return results

@njit(parallel=True)
def pairwise_jaccard(X, k):
	n, m = X.shape

	jaccards = np.empty((n, k), dtype='float64')
	neighbors = np.empty((n, k), dtype='int32')

	for i in prange(n):
		jaccard_ = np.empty(n, dtype='float64')

		for j in range(n):
			min_sum = 0.0
			max_sum = 0.0

			for l in range(m):
				sign = np.sign(X[i, l]) * np.sign(X[j, l])
				xi = abs(X[i, l])
				xj = abs(X[j, l])

				if xi > xj:
					min_sum += xj * sign
					max_sum += xi
				else:
					min_sum += xi * sign
					max_sum += xj 

			jaccard_[j] = min_sum / max_sum

		idxs = np.argsort(-jaccard_, kind='mergesort')[:k]

		jaccards[i] = jaccard_[idxs]
		neighbors[i] = idxs

	return jaccards, neighbors


@njit('void(float32[:, :, :], float32[:, :, :], int32[:, :], float32[:, :, :])', parallel=True)
def _jaccard(X, Y, neighbors, scores):
	nx, d, m = X.shape
	ny = Y.shape[0]
	len_output = scores.shape[-1]

	for l in prange(ny):
		for idx in range(len_output):
			for i in range(neighbors.shape[1]):
				min_sum = 0.0
				max_sum = 0.0
				neighbor_li = neighbors[l, i]

				for j in range(idx, idx+d):
					j_idx = j - idx

					for k in range(m):
						sign = np.sign(X[neighbor_li, j_idx, k]) * np.sign(Y[l, j, k])

						x = abs(X[neighbor_li, j_idx, k])
						y = abs(Y[l, j, k])

						if y > x:
							min_sum += x * sign
							max_sum += y
						else:
							min_sum += y * sign
							max_sum += x

				scores[l, i, idx] = min_sum / max_sum



def pearson_correlation(X, Y, min_overlap=None, func=np.ceil):
	if X.ndim == 2:
		X = X[None, :, :]
	if Y.ndim == 2:
		Y = Y[None, :, :]

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0, 0), (n_pad, n_pad), (0, 0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")

	n, d, _ = X.shape
	len_output = 1 + Y.shape[1] - d 
	scores = np.zeros((n, len_output))

	for idx in range(len_output):
		Y_ = Y[:, idx:idx+d]

		scores_ = np.dot((X / np.linalg.norm(X)).ravel(),
				  (Y_ / np.linalg.norm(Y_)).ravel()) 
		scores_ = np.nan_to_num(scores_)
		scores[:,idx] = scores_

	argmaxs = np.argmax(scores, axis=1)
	idxs = np.arange(len(scores))
	return np.array([[scores[idxs, argmaxs], argmaxs - n_pad]]).transpose(0, 2, 1)


class NNTsneConditionalProbs():
	def __init__(self, perplexity):
		self.perplexity = perplexity 

	def __call__(self, affinity_mat, nearest_neighbors):
		distmat_nn = np.log((1.0/(0.5*np.maximum(affinity_mat, 0.0000001)))-1)
		distmat_nn = np.maximum(distmat_nn, 0.0) #eliminate tiny neg floats

		# Compute the number of nearest neighbors to find.
		# LvdM uses 3 * perplexity as the number of neighbors.
		# In the event that we have very small # of points
		# set the neighbors to n - 1.
		n_samples = distmat_nn.shape[0]
		k = min(n_samples - 1, int(3. * self.perplexity + 1))

		P = self.tsne_probs_calc(distances_nn=distmat_nn[:,1:(k+1)],
								 neighbors_nn=[row[1:(k+1)] for row in 
											   nearest_neighbors])
		return P

	def tsne_probs_calc(self, distances_nn, neighbors_nn):
		# Compute conditional probabilities such that they approximately match
		# the desired perplexity
		n_samples, k = len(neighbors_nn),len(neighbors_nn[0])
		distances = distances_nn.astype(np.float32, copy=False)
		neighbors = neighbors_nn
		
		conditional_P = sklearn.manifold._utils._binary_search_perplexity(
			distances, self.perplexity, verbose=False)

		eps = 1e-8
		marginal_sum = conditional_P.sum(axis=-1)
		marginal_sum[marginal_sum < eps] = eps

		#normalize the conditional_P to sum to 1 across the rows
		conditional_P = conditional_P / marginal_sum[:,None]

		data = []
		rows = []
		cols = []
		for row_idx,(ps,neigh_row) in enumerate(zip(conditional_P, neighbors)):
			data.extend([p for p,neighbor in zip(ps, neigh_row)])
			rows.extend([row_idx for neighbor in neigh_row])
			cols.extend([neighbor for neighbor in neigh_row])

		P = coo_matrix((data, (rows, cols)),
					   shape=(len(neighbors), len(neighbors)))
		return P

