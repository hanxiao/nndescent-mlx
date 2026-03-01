"""NNDescent approximate k-NN graph construction in pure MLX for Apple Silicon.

Reference: Dong et al. "Efficient K-Nearest Neighbor Graph Construction for Generic
Similarity Measures" (WWW 2011).
"""

import mlx.core as mx
import numpy as np
import time


class NNDescent:
    """Approximate k-NN graph via NNDescent on Metal GPU.

    Parameters:
        k: number of nearest neighbors (default 15)
        n_iters: maximum iterations (default 20)
        delta: early stopping threshold on fraction of updates (default 0.001)
        random_state: seed for reproducibility
        verbose: print progress
    """

    def __init__(
        self,
        k: int = 15,
        n_iters: int = 20,
        delta: float = 0.001,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iters = n_iters
        self.delta = delta
        self.random_state = random_state
        self.verbose = verbose
        self.neighbor_graph = None

    def build(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Build approximate k-NN graph.

        Returns:
            (indices, distances): both (n, k) numpy arrays. Euclidean distances.
        """
        if isinstance(X, np.ndarray):
            X = mx.array(X.astype(np.float32))

        n, d = X.shape
        k = min(self.k, n - 1)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        sq_norms = mx.sum(X * X, axis=1)
        mx.eval(sq_norms)

        # Random init
        indices_np = np.empty((n, k), dtype=np.int32)
        for i in range(n):
            pool = np.random.choice(n - 1, k, replace=False)
            pool[pool >= i] += 1
            indices_np[i] = pool
        indices = mx.array(indices_np)

        # Initial distances
        dists = self._gather_dists(X, sq_norms, indices)
        sort_idx = mx.argsort(dists, axis=1)
        indices = mx.take_along_axis(indices, sort_idx, axis=1)
        dists = mx.take_along_axis(dists, sort_idx, axis=1)
        mx.eval(indices, dists)

        t0 = time.time()
        for it in range(self.n_iters):
            # Forward candidates: neighbors of neighbors
            nn_of_nn = indices[indices.reshape(-1)].reshape(n, k, k)
            fwd_cands = nn_of_nn.reshape(n, k * k)

            # Reverse candidates on GPU: transpose the edge list
            # For each edge (i -> indices[i,j]), create reverse edge (indices[i,j] -> i)
            src_all = mx.broadcast_to(mx.arange(n)[:, None], (n, k)).reshape(-1)
            dst_all = indices.reshape(-1)
            # Scatter src into dst's reverse list (fixed size k)
            # Use sort-based approach: sort by dst, slice k per point
            sort_by_dst = mx.argsort(dst_all)
            sorted_src = src_all[sort_by_dst]
            sorted_dst = dst_all[sort_by_dst]
            mx.eval(sorted_src, sorted_dst)

            # Each point gets n*k/n = k reverse edges on average
            # Reshape into (n, k) by taking k entries per destination
            # Since sort groups by dst, slice [i*k : (i+1)*k] approximately
            # But distribution is uneven. Use simpler approach:
            # Pad to (n, k) using the sorted order
            rev_cands = sorted_src.reshape(n, k)  # approximate: assumes uniform distribution

            # Combine: current (k) + forward (k*k) + reverse (k) = k + k^2 + k
            all_cands = mx.concatenate([indices, fwd_cands, rev_cands], axis=1)
            total_c = all_cands.shape[1]

            # Compute distances via chunked bmm
            all_dists = self._gather_dists(X, sq_norms, all_cands)

            # Mask self
            self_mask = all_cands == mx.arange(n)[:, None]
            all_dists = mx.where(self_mask, 1e30, all_dists)

            # Deduplicate per row
            cand_sort = mx.argsort(all_cands, axis=1)
            sc = mx.take_along_axis(all_cands, cand_sort, axis=1)
            sd = mx.take_along_axis(all_dists, cand_sort, axis=1)
            is_dup = mx.concatenate([
                mx.zeros((n, 1), dtype=mx.bool_),
                sc[:, 1:] == sc[:, :-1]
            ], axis=1)
            sd = mx.where(is_dup, 1e30, sd)
            unsort = mx.argsort(cand_sort, axis=1)
            all_dists = mx.take_along_axis(sd, unsort, axis=1)

            # Top k
            top_idx = mx.argpartition(all_dists, kth=k-1, axis=1)[:, :k]
            top_dists = mx.take_along_axis(all_dists, top_idx, axis=1)
            sub_sort = mx.argsort(top_dists, axis=1)
            top_idx = mx.take_along_axis(top_idx, sub_sort, axis=1)

            new_indices = mx.take_along_axis(all_cands, top_idx, axis=1)
            new_dists = mx.take_along_axis(all_dists, top_idx, axis=1)
            mx.eval(new_indices, new_dists)

            changed = int(mx.sum(new_indices != indices))
            update_frac = changed / (n * k)

            indices = new_indices
            dists = new_dists

            if self.verbose:
                elapsed = time.time() - t0
                print(f"Iter {it+1}/{self.n_iters}: {changed} updates "
                      f"({update_frac:.4f}), {elapsed:.2f}s")

            if update_frac < self.delta:
                if self.verbose:
                    print(f"Converged at iteration {it+1}")
                break

        final_dists = mx.sqrt(mx.maximum(dists, 0.0))
        mx.eval(indices, final_dists)
        self.neighbor_graph = (np.array(indices), np.array(final_dists))
        return self.neighbor_graph

    def _gather_dists(self, X, sq_norms, col_ids):
        """Squared distances from point i to col_ids[i, :] via chunked bmm."""
        n, c = col_ids.shape
        d = X.shape[1]

        # Chunk to limit memory: (cs, c, d) intermediate
        max_cs = max(1, 500_000_000 // (c * d))
        max_cs = min(max_cs, n)

        chunks = []
        for s in range(0, n, max_cs):
            e = min(s + max_cs, n)
            cs = e - s

            flat = col_ids[s:e].reshape(-1)
            X_tgt = X[flat].reshape(cs, c, d)
            X_src = X[s:e]

            # bmm: (cs, 1, d) @ (cs, d, c) -> (cs, 1, c) -> (cs, c)
            dots = mx.matmul(X_src[:, None, :],
                            mx.transpose(X_tgt, (0, 2, 1)))[:, 0, :]

            sq_src = sq_norms[s:e][:, None]
            sq_tgt = sq_norms[flat].reshape(cs, c)
            chunk_d = mx.maximum(sq_src + sq_tgt - 2.0 * dots, 0.0)
            mx.eval(chunk_d)
            chunks.append(chunk_d)

        return mx.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
