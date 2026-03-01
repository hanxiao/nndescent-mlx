"""NNDescent approximate k-NN graph construction in pure MLX for Apple Silicon.

Reference: Dong et al. "Efficient K-Nearest Neighbor Graph Construction for Generic
Similarity Measures" (WWW 2011).

Entire pipeline on Metal GPU via MLX. No numpy in the hot path.
"""

import mlx.core as mx
import numpy as np
import time


class NNDescent:
    """Approximate k-NN graph via NNDescent on Metal GPU."""

    def __init__(
        self,
        k: int = 15,
        n_iters: int = 20,
        max_candidates: int | None = None,
        delta: float = 0.001,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_iters = n_iters
        self.max_candidates = max_candidates
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
        mc = self.max_candidates or k

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Precompute squared norms
        sq_norms = mx.sum(X * X, axis=1)  # (n,)

        # Random init (numpy for random, then convert)
        idx_np = np.random.randint(0, n - 1, (n, k), dtype=np.int32)
        i_vals = np.arange(n, dtype=np.int32)[:, None]
        idx_np = np.where(idx_np >= i_vals, idx_np + 1, idx_np).astype(np.int32)
        indices = mx.array(idx_np)  # (n, k)

        # Initial distances
        dists = _gather_dists(X, sq_norms, indices)  # (n, k)

        # flags: 1 = new, 0 = old
        flags = mx.ones((n, k), dtype=mx.uint8)

        # Sort
        si = mx.argsort(dists, axis=1)
        indices = mx.take_along_axis(indices, si, axis=1)
        dists = mx.take_along_axis(dists, si, axis=1)
        mx.eval(indices, dists, flags, sq_norms)

        t0 = time.time()
        for it in range(self.n_iters):
            # ---- Build candidates via forward + reverse edges ----
            # Forward: each (i, indices[i,j]) is an edge with flag[i,j]
            # Reverse: each (indices[i,j], i) is also a candidate

            # For each point, gather neighbors-of-neighbors as candidates
            # This is equivalent to local join but GPU-friendly:
            # candidates[i] = union(indices[indices[i,j], :]) for all j
            nn_of_nn = indices[indices.reshape(-1)].reshape(n, k, k)  # (n, k, k)

            # Reverse candidates: for each edge (i -> j), add i as candidate for j
            # Build (n, k) reverse candidate array via scatter
            src_all = mx.broadcast_to(mx.arange(n)[:, None], (n, k)).reshape(-1)  # (n*k,)
            dst_all = indices.reshape(-1)  # (n*k,)

            # Sort by destination to group reverse edges
            rev_order = mx.argsort(dst_all)
            rev_src = src_all[rev_order]
            rev_dst = dst_all[rev_order]
            mx.eval(rev_src, rev_dst)

            # Count edges per destination point
            # Use searchsorted on sorted dst to find boundaries
            # Build (n, k) reverse candidate array without Python loop
            # rev_src is sorted by rev_dst. For each destination point,
            # take the first k sources.
            # Assign a within-group index to each edge
            mx.eval(rev_dst)
            rev_dst_np = np.array(rev_dst)
            # Compute within-group position
            group_starts = np.searchsorted(rev_dst_np, np.arange(n))
            positions = np.arange(len(rev_dst_np)) - group_starts[rev_dst_np]
            # Keep only first k per group
            keep_mask = positions < k
            kept_dst = rev_dst_np[keep_mask]
            kept_src = np.array(rev_src)[keep_mask]
            kept_pos = positions[keep_mask]
            # Scatter into (n, k) array
            rev_cands_np = np.zeros((n, k), dtype=np.int32)
            rev_cands_np[kept_dst, kept_pos] = kept_src
            rev_cands = mx.array(rev_cands_np)

            # Combine: current neighbors (k) + nn-of-nn (k*k) + reverse (k)
            all_cands = mx.concatenate([
                indices,                    # (n, k) current
                nn_of_nn.reshape(n, k * k),  # (n, k^2) forward
                rev_cands,                   # (n, k) reverse
            ], axis=1)  # (n, k^2 + 2k)
            total_c = all_cands.shape[1]

            # Compute distances to all candidates
            all_dists = _gather_dists(X, sq_norms, all_cands)  # (n, total_c)

            # Mask self-references
            self_mask = all_cands == mx.arange(n)[:, None]
            all_dists = mx.where(self_mask, 1e30, all_dists)

            # Deduplicate per row: for duplicate candidate indices, keep the
            # one with smallest distance. Sort by (candidate, distance), then
            # mark later occurrences of same candidate as inf.
            # Two-key sort: primary = candidate index, secondary = distance
            # Use lexsort-like approach: sort by distance first, then stable sort by candidate
            dist_order = mx.argsort(all_dists, axis=1)
            all_cands_ds = mx.take_along_axis(all_cands, dist_order, axis=1)
            all_dists_ds = mx.take_along_axis(all_dists, dist_order, axis=1)

            cand_sort = mx.argsort(all_cands_ds, axis=1)
            sorted_c = mx.take_along_axis(all_cands_ds, cand_sort, axis=1)
            sorted_d = mx.take_along_axis(all_dists_ds, cand_sort, axis=1)

            # Now for each run of same candidate, the first has smallest distance
            is_dup = mx.concatenate([
                mx.zeros((n, 1), dtype=mx.bool_),
                sorted_c[:, 1:] == sorted_c[:, :-1]
            ], axis=1)
            sorted_d = mx.where(is_dup, 1e30, sorted_d)

            # Unsort back to original order
            unsort2 = mx.argsort(cand_sort, axis=1)
            all_dists_ds = mx.take_along_axis(sorted_d, unsort2, axis=1)
            unsort1 = mx.argsort(dist_order, axis=1)
            all_dists = mx.take_along_axis(all_dists_ds, unsort1, axis=1)

            # Select top k
            top_idx = mx.argpartition(all_dists, kth=k - 1, axis=1)[:, :k]
            top_dists = mx.take_along_axis(all_dists, top_idx, axis=1)
            sub_sort = mx.argsort(top_dists, axis=1)
            top_idx = mx.take_along_axis(top_idx, sub_sort, axis=1)

            new_indices = mx.take_along_axis(all_cands, top_idx, axis=1)
            new_dists = mx.take_along_axis(all_dists, top_idx, axis=1)

            # Compute flags: mark entries that changed as "new"
            # An entry is new if it wasn't in the previous neighbor set
            new_flags = mx.ones((n, k), dtype=mx.uint8)
            # Check each new neighbor against all old neighbors
            # old_expanded: (n, 1, k), new_expanded: (n, k, 1)
            # match if any old neighbor equals this new neighbor
            matches = new_indices[:, :, None] == indices[:, None, :]  # (n, k, k)
            was_old = mx.any(matches, axis=2)  # (n, k) True if was already a neighbor
            new_flags = mx.where(was_old, mx.zeros_like(new_flags), new_flags)

            mx.eval(new_indices, new_dists, new_flags)

            # Count updates
            changed = int(mx.sum(new_indices != indices))
            update_frac = changed / (n * k)

            indices = new_indices
            dists = new_dists
            flags = new_flags

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


def _gather_dists(X, sq_norms, col_ids):
    """Squared distances from each point i to col_ids[i, :].

    Chunked to limit memory. All MLX, single eval at end.

    Args:
        X: (n, d) data
        sq_norms: (n,) precomputed ||x||^2
        col_ids: (n, c) neighbor indices

    Returns:
        (n, c) squared distances
    """
    n, c = col_ids.shape
    d = X.shape[1]

    # Chunk to keep intermediate (cs, c, d) under ~500MB
    max_cs = max(1, 125_000_000 // (c * d))
    max_cs = min(max_cs, n)

    if max_cs >= n:
        # Single chunk - no eval boundary
        flat = col_ids.reshape(-1)
        X_tgt = X[flat].reshape(n, c, d)
        X_src = X
        dots = mx.matmul(X_src[:, None, :], mx.transpose(X_tgt, (0, 2, 1)))[:, 0, :]
        return mx.maximum(sq_norms[:, None] + sq_norms[flat].reshape(n, c) - 2.0 * dots, 0.0)

    chunks = []
    for s in range(0, n, max_cs):
        e = min(s + max_cs, n)
        cs = e - s
        flat = col_ids[s:e].reshape(-1)
        X_tgt = X[flat].reshape(cs, c, d)
        X_src = X[s:e]
        dots = mx.matmul(X_src[:, None, :], mx.transpose(X_tgt, (0, 2, 1)))[:, 0, :]
        chunk_d = mx.maximum(sq_norms[s:e][:, None] + sq_norms[flat].reshape(cs, c) - 2.0 * dots, 0.0)
        mx.eval(chunk_d)
        chunks.append(chunk_d)
    return mx.concatenate(chunks, axis=0)
