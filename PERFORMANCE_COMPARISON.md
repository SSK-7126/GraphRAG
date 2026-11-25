# Performance Comparison

Only measured runs are shown; absent backends have no simulated result.
The included fixture uses `MockLLM` and is a plumbing/scheduling check, not a production LLM throughput claim.

| Backend | Run ID | Dataset | Wall-clock seconds | Speedup vs serial | Documents/s | Chunks/s |
|---|---|---|---:|---:|---:|---:|
| asyncio | asyncio-mock | all | 0.018714 | 0.857 | 0.0000 | 0.0000 |
| asyncio | asyncio-mock | msft | 0.014792 | 0.936 | 67.6041 | 67.6041 |
| dask | dask-gpu-machine | all | 0.016198 | 0.990 | 0.0000 | 0.0000 |
| dask | dask-gpu-machine | msft | 0.150848 | 0.092 | 6.6292 | 6.6292 |
| optimized_serial | optimized-mock | all | 0.020270 | 0.791 | 0.0000 | 0.0000 |
| optimized_serial | optimized-mock | msft | 0.011071 | 1.250 | 90.3220 | 90.3220 |
| ray | ray-gpu-machine | all | 0.016917 | 0.948 | 0.0000 | 0.0000 |
| ray | ray-gpu-machine | msft | 3.635627 | 0.004 | 0.2751 | 0.2751 |
| serial | baseline-mock | all | 0.016037 | 1.000 | 0.0000 | 0.0000 |
| serial | baseline-mock | msft | 0.013840 | 1.000 | 72.2569 | 72.2569 |
