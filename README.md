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
N      k   iters   nndescent-mlx   pynndescent   recall
10K    15   10     0.08s           4.7s          90%
70K    15   13     0.35s           N/A           81%
```

nndescent-mlx is **60x faster** than pynndescent (CPU) on 10K points, at the cost of ~10% recall.

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
2. For each iteration:
   - Gather neighbors-of-neighbors (k² candidates per point)
   - Merge with current neighbors (k + k² total)
   - Deduplicate: sort by index, mark duplicates, set distance to inf
   - Compute distances via chunked batched matmul on GPU
   - Keep top k by distance
3. Converge when < 0.1% of edges change

All distance computation runs on Metal GPU via MLX. Graph structure (indices) stays on CPU to minimize memory.

Dependencies: `mlx` and `numpy` only.

## License

MIT
