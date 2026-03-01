"""NNDescent approximate k-NN graph construction in pure MLX for Apple Silicon.

Algorithm: iteratively refine a random k-NN graph by exploring neighbors-of-neighbors.
The key insight: "a neighbor of my neighbor is likely my neighbor too."

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

        Args:
            X: (n, d) data matrix (numpy or mlx array)

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

        # Random initialization (fully vectorized)
        # Sample k random integers in [0, n-1) for each point, then shift to avoid self
        indices_np = np.random.randint(0, n - 1, (n, k), dtype=np.int32)
        # For point i, candidates >= i should be incremented (to skip i itself)
        i_vals = np.arange(n, dtype=np.int32)[:, None]  # (n, 1)
        indices_np = np.where(indices_np >= i_vals, indices_np + 1, indices_np)
        indices = mx.array(indices_np)

        # Initial distances
        dists = self._gather_dists(X, sq_norms, indices)
        sort_idx = mx.argsort(dists, axis=1)
        indices = mx.take_along_axis(indices, sort_idx, axis=1)
        dists = mx.take_along_axis(dists, sort_idx, axis=1)
        mx.eval(indices, dists)

        t0 = time.time()
        for it in range(self.n_iters):
            # Neighbors-of-neighbors: for each point, gather its neighbors' neighbors
            # indices: (n, k) -> nn_of_nn[i,j,:] = indices[indices[i,j], :]
            nn_of_nn = indices[indices.reshape(-1)].reshape(n, k, k)  # (n, k, k)
            # Flatten to (n, k*k) candidates
            candidates = nn_of_nn.reshape(n, k * k)

            # Merge: current neighbors + candidates
            all_cands = mx.concatenate([indices, candidates], axis=1)  # (n, k + k*k)
            total_c = all_cands.shape[1]

            # Compute distances (chunked to avoid OOM)
            all_dists = self._gather_dists(X, sq_norms, all_cands)

            # Mask self-references
            self_mask = all_cands == mx.arange(n)[:, None]
            all_dists = mx.where(self_mask, 1e30, all_dists)

            # Deduplicate: sort by candidate index, mark duplicates
            cand_sort = mx.argsort(all_cands, axis=1)
            sc = mx.take_along_axis(all_cands, cand_sort, axis=1)
            sd = mx.take_along_axis(all_dists, cand_sort, axis=1)
            is_dup = mx.concatenate([
                mx.zeros((n, 1), dtype=mx.bool_),
                sc[:, 1:] == sc[:, :-1]
            ], axis=1)
            sd = mx.where(is_dup, 1e30, sd)
            # Unsort back
            unsort = mx.argsort(cand_sort, axis=1)
            all_dists = mx.take_along_axis(sd, unsort, axis=1)
            mx.eval(all_dists)

            # Select top k
            top_idx = mx.argpartition(all_dists, kth=k-1, axis=1)[:, :k]
            # Sort the top k
            top_dists = mx.take_along_axis(all_dists, top_idx, axis=1)
            sub_sort = mx.argsort(top_dists, axis=1)
            top_idx = mx.take_along_axis(top_idx, sub_sort, axis=1)

            new_indices = mx.take_along_axis(all_cands, top_idx, axis=1)
            new_dists = mx.take_along_axis(all_dists, top_idx, axis=1)
            mx.eval(new_indices, new_dists)

            # Count updates
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
        """Compute squared distances from each point i to col_ids[i, :].

        Chunked to limit memory usage.

        Args:
            X: (n, d)
            sq_norms: (n,)
            col_ids: (n, c) indices

        Returns:
            (n, c) squared distances
        """
        n, c = col_ids.shape
        d = X.shape[1]

        # Chunk size: limit intermediate (cs, c, d) to ~500MB
        max_cs = max(1, 125_000_000 // (c * d))  # 500MB / 4 bytes
        max_cs = min(max_cs, n)

        chunks = []
        for s in range(0, n, max_cs):
            e = min(s + max_cs, n)
            cs = e - s

            flat = col_ids[s:e].reshape(-1)
            X_tgt = X[flat].reshape(cs, c, d)
            X_src = X[s:e]

            # Batched matmul: (cs, 1, d) @ (cs, d, c) -> (cs, 1, c) -> (cs, c)
            dots = mx.matmul(X_src[:, None, :],
                            mx.transpose(X_tgt, (0, 2, 1)))[:, 0, :]

            sq_src = sq_norms[s:e][:, None]
            sq_tgt = sq_norms[flat].reshape(cs, c)
            chunk_d = mx.maximum(sq_src + sq_tgt - 2.0 * dots, 0.0)
            mx.eval(chunk_d)
            chunks.append(chunk_d)

        return mx.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
