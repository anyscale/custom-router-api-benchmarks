"""
Sweep over different router configurations.
"""

import os
import subprocess
import time

from ray import serve
from ray.serve._private.constants import SERVE_DEFAULT_APP_NAME
from ray.serve.schema import ApplicationStatus
from ray.serve.llm import LLMConfig, LLMServingArgs, build_openai_app, ModelLoadingConfig

from engine_metrics import EngineMetrics


BASE_LLM_CONFIG = LLMConfig(
    model_loading_config=ModelLoadingConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    runtime_env=dict(
        env_vars={
            "VLLM_USE_V1": "1",
        },
    ),
    engine_kwargs=dict(
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_model_len=32000
    ),
    log_engine_metrics=True,
)

cmd = [
    "vllm",
    "bench",
    "serve",
    "--backend",
    "openai",
    "--model",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "--dataset-name",
    "prefix_repetition",
    "--prefix-repetition-prefix-len",
    "512",
    "--prefix-repetition-suffix-len",
    "128",
    "--prefix-repetition-output-len",
    "128",
    "--save-result",
    "--append-result",
]

ROUTER_MAP = {
    # "pow_2": "ray.serve._private.request_router.pow_2_router.PowerOfTwoChoicesRequestRouter",
    "prefix_aware": "ray.serve.llm.request_router.PrefixCacheAffinityRouter",
}

CONCURRENCY_PER_REPLICA = 32
NUM_PROMPTS_PER_REPLICA = 256
NUM_PREFIXES_PER_REPLICA = 16

def main():
    os.makedirs("../results/replica_sweep", exist_ok=True)
    
    metrics = EngineMetrics()

    for router_name in ROUTER_MAP:
        llm_config = BASE_LLM_CONFIG.model_copy(deep=True)

        for replicas in [1]:
            llm_config.deployment_config = dict(
                autoscaling_config=dict(
                    min_replicas=replicas,
                    max_replicas=replicas,
                ),
                max_ongoing_requests=8192,
                request_router_config=dict(
                    request_router_class=ROUTER_MAP[router_name],
                    request_router_kwargs=dict(
                        imbalanced_threshold=100000,
                    ),
                ),
            )
            app = build_openai_app(LLMServingArgs(llm_configs=[llm_config]))
            serve.run(app, blocking=False)
            status = ApplicationStatus.NOT_STARTED
            while status != ApplicationStatus.RUNNING:
                time.sleep(1)
                status = serve.status().applications[SERVE_DEFAULT_APP_NAME].status

            print(f"Application status: {status}")

            full_cmd = cmd + [
                "--num-prompts",
                str(replicas * NUM_PROMPTS_PER_REPLICA),
                "--max-concurrency",
                str(replicas * CONCURRENCY_PER_REPLICA),
                "--prefix-repetition-num-prefixes",
                str(replicas * NUM_PREFIXES_PER_REPLICA),
                "--result-filename",
                f"../results/replica_sweep/{replicas}_replicas_{router_name}.json",
            ]

            subprocess.run(full_cmd, check=True)

            metrics.collect_gpu_prefix_cache_metrics(replicas, router_name)

            time.sleep(10)
            serve.shutdown()
            time.sleep(10)


if __name__ == "__main__":
    main()