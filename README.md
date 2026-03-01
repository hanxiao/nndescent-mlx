# nndescent-mlx

NNDescent approximate k-NN graph construction in pure MLX for Apple Silicon.

## What is NNDescent?

NNDescent (Nearest Neighbor Descent) is an algorithm for building approximate k-nearest neighbor graphs efficiently. Instead of computing all O(n²) pairwise distances, it starts with a random graph and iteratively refines it using the principle: **"a neighbor of my neighbor is likely my neighbor too."**

Each iteration:
1. For each point, gather its neighbors' neighbors (candidates)
2. Compute distances to all candidates
3. Keep the closest k

Converges in ~10-20 iterations to 80-95% recall.

Reference: [Dong et al. WWW 2011](https://www.cs.princeton.edu/cass/papers/www11.pdf)

## Performance (M3 Ultra, Fashion-MNIST PCA-50)

```
N      k   nndescent-mlx   pynndescent   recall
10K    15   0.11s           4.7s          93%
70K    15   0.65s           N/A           91%
```

**42x faster** than pynndescent on 10K points. Entire hot path runs on Metal GPU via MLX.

## Install

```bash
git clone https://github.com/hanxiao/nndescent-mlx.git && cd nndescent-mlx
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Usage

```python
from nndescent_mlx import NNDescent
import numpy as np

X = np.random.randn(10000, 50).astype(np.float32)
nn = NNDescent(k=15, n_iters=20, verbose=True)
indices, distances = nn.build(X)  # indices: (10000, 15), distances: (10000, 15)
```

Parameters:
- `k`: number of nearest neighbors (default 15)
- `n_iters`: maximum iterations (default 20)
- `delta`: early stopping threshold (default 0.001)
- `random_state`: seed
- `verbose`: print progress

## How it works

1. Random initialization: each point gets k random neighbors
2. For each iteration (all on Metal GPU via MLX):
   - Forward candidates: gather neighbors-of-neighbors (k² per point)
   - Reverse candidates: if j is i's neighbor, i becomes j's candidate
   - Compute squared distances via chunked batched matmul on GPU
   - Deduplicate per row: sort by candidate index, keep smallest distance
   - Merge current + candidates, keep top k by distance
   - Track new/old flags: only new edges propagate information
3. Converge when < 0.1% of edges change

Entire pipeline runs on Metal GPU via MLX. Reverse candidate scatter uses `mx.cummax` + `at[].add()`.

Dependencies: `mlx` and `numpy` only.

## License

MIT
