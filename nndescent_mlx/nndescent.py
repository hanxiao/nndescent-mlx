"""NNDescent approximate k-NN graph construction in pure MLX for Apple Silicon.

Reference: Dong et al. "Efficient K-Nearest Neighbor Graph Construction for Generic
Similarity Measures" (WWW 2011).

This implementation uses a hybrid approach:
- Forward + reverse candidate gathering (the core NNDescent idea)
- Local join: compare (new, new) and (new, old) pairs
- Vectorized merge-sort for graph updates (instead of per-element heap ops)
- GPU distance computation via MLX batched matmul on Metal
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
            X_mx = mx.array(X.astype(np.float32))
        else:
            X_mx = X

        n, d = X_mx.shape
        k = min(self.k, n - 1)
        mc = self.max_candidates or k

        if self.random_state is not None:
            np.random.seed(self.random_state)

        sq_norms = mx.sum(X_mx * X_mx, axis=1)
        mx.eval(sq_norms)

        # Random init
        indices = np.random.randint(0, n - 1, (n, k), dtype=np.int32)
        i_vals = np.arange(n, dtype=np.int32)[:, None]
        indices = np.where(indices >= i_vals, indices + 1, indices).astype(np.int32)

        # Initial distances on GPU
        dists = self._gpu_dists_rows(X_mx, sq_norms, mx.array(indices), n, k, d)
        flags = np.ones((n, k), dtype=np.uint8)

        si = np.argsort(dists, axis=1)
        indices = np.take_along_axis(indices, si, axis=1)
        dists = np.take_along_axis(dists, si, axis=1)

        t0 = time.time()
        for it in range(self.n_iters):
            # ---- Build candidate lists ----
            # All forward + reverse edges
            src = np.repeat(np.arange(n, dtype=np.int32), k)
            dst = indices.ravel().astype(np.int32)
            flg = flags.ravel()

            # Both directions
            all_src = np.concatenate([src, dst])
            all_dst = np.concatenate([dst, src])
            all_flg = np.concatenate([flg, flg])

            # Remove self
            mask = all_src != all_dst
            all_src, all_dst, all_flg = all_src[mask], all_dst[mask], all_flg[mask]

            # Group by source point
            order = np.argsort(all_src, kind='stable')
            all_src, all_dst, all_flg = all_src[order], all_dst[order], all_flg[order]
            bounds = np.searchsorted(all_src, np.arange(n + 1))

            # Extract new and old candidates per point (capped at mc each)
            new_cands = np.full((n, mc), -1, dtype=np.int32)
            old_cands = np.full((n, mc), -1, dtype=np.int32)
            _build_cands(bounds, all_dst, all_flg, new_cands, old_cands, n, mc)

            # Reset flags
            flags[:] = 0

            # Count actual new per point
            new_valid = new_cands >= 0  # (n, mc)
            nc_per_pt = new_valid.sum(axis=1)
            max_nc = int(nc_per_pt.max()) if nc_per_pt.max() > 0 else 0
            if max_nc == 0:
                if self.verbose:
                    print(f"Iter {it+1}: no new candidates")
                break

            # ---- Generate local join pairs (vectorized) ----
            nc_eff = min(max_nc, mc)
            oc_eff = min(mc, int((old_cands >= 0).sum(axis=1).max()))

            # new-new pairs: upper triangle
            ii, jj = np.triu_indices(nc_eff, k=1)
            p_nn = new_cands[:, ii].ravel()
            q_nn = new_cands[:, jj].ravel()
            valid = (p_nn >= 0) & (q_nn >= 0) & (p_nn != q_nn)
            p_nn, q_nn = p_nn[valid], q_nn[valid]

            # new-old pairs: cartesian product
            if oc_eff > 0:
                p_no = np.repeat(new_cands[:, :nc_eff], oc_eff, axis=1).ravel()
                q_no = np.tile(old_cands[:, :oc_eff], (1, nc_eff)).ravel()
                valid = (p_no >= 0) & (q_no >= 0) & (p_no != q_no)
                p_no, q_no = p_no[valid], q_no[valid]
                p_all = np.concatenate([p_nn, p_no])
                q_all = np.concatenate([q_nn, q_no])
            else:
                p_all, q_all = p_nn, q_nn

            # Deduplicate pairs
            lo = np.minimum(p_all, q_all)
            hi = np.maximum(p_all, q_all)
            pair_hash = lo.astype(np.int64) * n + hi.astype(np.int64)
            _, uniq_idx = np.unique(pair_hash, return_index=True)
            p_all, q_all = p_all[uniq_idx], q_all[uniq_idx]

            # ---- Compute distances on GPU ----
            pair_dists = self._gpu_dists_pairs(X_mx, sq_norms, p_all, q_all)

            # ---- Scatter updates into per-point proposal arrays ----
            # Each pair (p, q, d) generates proposals: (p <- q, d) and (q <- p, d)
            tgt = np.concatenate([p_all, q_all])
            cand = np.concatenate([q_all, p_all])
            d_arr = np.concatenate([pair_dists, pair_dists])

            # For each target point, merge current neighbors + proposals, dedup, keep top k
            n_updates = self._merge_updates(indices, dists, flags, tgt, cand, d_arr, n, k)

            update_frac = n_updates / (n * k)

            if self.verbose:
                elapsed = time.time() - t0
                print(f"Iter {it+1}/{self.n_iters}: {n_updates} updates "
                      f"({update_frac:.4f}), {len(p_all)} pairs, {elapsed:.2f}s")

            if update_frac < self.delta:
                if self.verbose:
                    print(f"Converged at iteration {it+1}")
                break

        final_dists = np.sqrt(np.maximum(dists, 0.0)).astype(np.float32)
        self.neighbor_graph = (indices.copy(), final_dists)
        return self.neighbor_graph

    @staticmethod
    def _merge_updates(indices, dists, flags, targets, candidates, d_vals, n, k):
        """Merge proposals into the graph using vectorized sort-merge.

        For each target point, combines current neighbors with new proposals,
        removes duplicates, and keeps the closest k.
        """
        # Pre-filter: only keep proposals better than target's worst
        worst = dists[targets, -1]
        keep = d_vals < worst
        targets, candidates, d_vals = targets[keep], candidates[keep], d_vals[keep]

        if len(targets) == 0:
            return 0

        # Group by target
        order = np.argsort(targets, kind='stable')
        targets, candidates, d_vals = targets[order], candidates[order], d_vals[order]

        unique_targets, tgt_starts = np.unique(targets, return_index=True)
        tgt_ends = np.append(tgt_starts[1:], len(targets))

        n_updates = 0
        for i, t in enumerate(unique_targets):
            s, e = tgt_starts[i], tgt_ends[i]
            prop_cands = candidates[s:e]
            prop_dists = d_vals[s:e]

            # Merge: current (k) + proposals (e-s)
            all_c = np.concatenate([indices[t], prop_cands])
            all_d = np.concatenate([dists[t], prop_dists])

            # Deduplicate by candidate index: keep first occurrence
            # Set self-reference distance to inf
            all_d = np.where(all_c == t, np.float32(np.inf), all_d)

            uniq_c, first_idx = np.unique(all_c, return_index=True)
            uniq_d = all_d[first_idx]

            # Keep top k
            if len(uniq_c) > k:
                top_k = np.argpartition(uniq_d, k)[:k]
                top_k = top_k[np.argsort(uniq_d[top_k])]
            else:
                top_k = np.argsort(uniq_d)[:k]

            new_idx = uniq_c[top_k].astype(np.int32)
            new_dst = uniq_d[top_k].astype(np.float32)

            # Pad if needed
            if len(new_idx) < k:
                pad = k - len(new_idx)
                new_idx = np.pad(new_idx, (0, pad), constant_values=-1)
                new_dst = np.pad(new_dst, (0, pad), constant_values=np.inf)

            # Count changes
            changed = np.sum(new_idx != indices[t])
            if changed > 0:
                # Mark new entries
                old_set = set(indices[t])
                new_flags = np.array([1 if int(c) not in old_set else 0
                                     for c in new_idx], dtype=np.uint8)
                indices[t] = new_idx
                dists[t] = new_dst
                flags[t] = new_flags
                n_updates += int(changed)

        return n_updates

    def _gpu_dists_rows(self, X, sq_norms, col_ids_mx, n, c, d):
        """Squared distances per row. Returns (n, c) numpy."""
        max_cs = max(1, 125_000_000 // (c * d))
        chunks = []
        for s in range(0, n, max_cs):
            e = min(s + max_cs, n)
            cs = e - s
            flat = col_ids_mx[s:e].reshape(-1)
            X_tgt = X[flat].reshape(cs, c, d)
            X_src = X[s:e]
            dots = mx.matmul(X_src[:, None, :], mx.transpose(X_tgt, (0, 2, 1)))[:, 0, :]
            chunk_d = mx.maximum(sq_norms[s:e][:, None] + sq_norms[flat].reshape(cs, c) - 2.0 * dots, 0.0)
            mx.eval(chunk_d)
            chunks.append(np.array(chunk_d))
        return np.concatenate(chunks, axis=0)

    def _gpu_dists_pairs(self, X, sq_norms_mx, p_arr, q_arr):
        """Squared distances for pairs. Returns (m,) numpy."""
        m = len(p_arr)
        chunk = 1000000
        chunks = []
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            p_mx, q_mx = mx.array(p_arr[s:e]), mx.array(q_arr[s:e])
            d = mx.sum((X[p_mx] - X[q_mx]) ** 2, axis=1)
            mx.eval(d)
            chunks.append(np.array(d))
        return np.concatenate(chunks)


def _build_cands(bounds, all_dst, all_flg, new_cands, old_cands, n, mc):
    """Fill new/old candidate arrays from sorted edge list."""
    for v in range(n):
        s, e = bounds[v], bounds[v + 1]
        nc, oc = 0, 0
        for idx in range(s, e):
            if all_flg[idx]:
                if nc < mc:
                    new_cands[v, nc] = all_dst[idx]
                    nc += 1
            else:
                if oc < mc:
                    old_cands[v, oc] = all_dst[idx]
                    oc += 1
            if nc >= mc and oc >= mc:
                break
