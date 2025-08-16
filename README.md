# Ray Serve Custom Router API Benchmarks

A benchmarking suite for comparing different request router implementations in Ray's LLM serving system. Compares Power-of-2 routers vs Prefix-Aware routers across various replica counts.

## Project Structure

```
custom-router-api-benchmarks/
├── scripts/
│   ├── engine_metrics.py      # Metrics collection and parsing from Ray /metrics endpoint
│   ├── sweep_replicas.py      # Main replica scaling benchmark script
│   └── visualize_replica_sweep.py  # Visualization and analysis of results
└── results/
    ├── replica_sweep/        # Current benchmark results (JSON + raw metrics)
    └── visualizations/       # Generated plots and charts
```

## Usage

### Run Benchmark
```bash
cd scripts/
python sweep_replicas.py
```

### Generate Visualizations
```bash
cd scripts/
python visualize_replica_sweep.py
```

## Requirements
- Ray cluster, see k8s install steps [here](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html)
- Docker image: rayproject/ray-llm:nightly-py311-cu128
- Ray [nightly](https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies) wheel

# Set the following environment variables in an Anyscale Service `runtime_env` for optimal performance:
- ANYSCALE_RAY_SERVE_THROUGHPUT_OPT=1
- RAYLLM_ROUTER_TO_MODEL_REPLICA_RATIO=8