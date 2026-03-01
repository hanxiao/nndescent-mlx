"""Benchmark nndescent-mlx vs brute force and pynndescent."""
import numpy as np
import time
from nndescent_mlx import NNDescent
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import pynndescent

# Load Fashion-MNIST, PCA to 50D (once)
print("Loading Fashion-MNIST...")
mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X_full = PCA(n_components=50).fit_transform(mnist.data.astype(np.float32))
print(f"Loaded {X_full.shape}\n")

for n in [10000, 70000]:
    X = X_full[:n]
    print(f"{'='*60}")
    print(f"N = {n}, k = 15")
    print(f"{'='*60}")

    # nndescent-mlx
    t0 = time.time()
    nn = NNDescent(k=15, verbose=False, random_state=42)
    idx_mlx, dist_mlx = nn.build(X)
    t_mlx = time.time() - t0

    # Brute force on sample (for recall)
    sample_size = min(1000, n)
    sample = np.random.RandomState(42).choice(n, sample_size, replace=False)
    
    nbrs = NearestNeighbors(n_neighbors=15, algorithm='brute').fit(X)
    _, idx_true = nbrs.kneighbors(X[sample])

    # Compute recall on sample
    recall = np.mean([len(set(idx_mlx[sample[i]]) & set(idx_true[i])) / 15 
                     for i in range(sample_size)])

    print(f"nndescent-mlx:  {t_mlx:.2f}s  recall@15: {recall:.4f} (sample {sample_size})")

    # pynndescent (only for 10K, too slow for 70K)
    if n == 10000:
        t0 = time.time()
        pynn = pynndescent.NNDescent(X, n_neighbors=15, random_state=42, verbose=False)
        idx_pynn = pynn.neighbor_graph[0]
        t_pynn = time.time() - t0
        
        idx_true_full = nbrs.kneighbors(X, return_distance=False)
        recall_pynn = np.mean([len(set(idx_pynn[i]) & set(idx_true_full[i])) / 15 
                               for i in range(n)])
        print(f"pynndescent:    {t_pynn:.2f}s  recall@15: {recall_pynn:.4f}")
        print(f"Speedup:        {t_pynn / t_mlx:.1f}x faster than pynndescent")
    print()
