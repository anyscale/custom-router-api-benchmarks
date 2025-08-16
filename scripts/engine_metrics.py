"""
Engine metrics collection and parsing for vLLM Prometheus endpoints.
"""

import os
import requests
import re
import json
import time
import ray
from typing import Optional, Tuple, Dict, Any


class WorkerNodeDiscovery:
    """Automatically discover worker nodes in the Ray cluster."""

    @staticmethod
    def discover_worker_nodes() -> list:
        """
        Automatically discover worker node IPs from the Ray cluster.

        Returns:
            List of worker node IP addresses
        """
        try:
            # Connect to Ray if not already connected
            if not ray.is_initialized():
                ray.init(address='auto', ignore_reinit_error=True)

            nodes = ray.nodes()
            worker_ips = []

            print("Discovering Ray cluster nodes:")
            for node in nodes:
                node_ip = node.get('NodeManagerAddress', 'unknown')
                resources = node.get('Resources', {})
                alive = node.get('alive', True)

                # Head node usually has 'node:__internal_head__' resource
                is_head = 'node:__internal_head__' in resources

                print(f"  Node: {node_ip} - Head: {is_head} - Alive: {alive}")

                if not is_head and alive and node_ip != 'unknown':
                    worker_ips.append(node_ip)

            print(f"Worker nodes discovered: {worker_ips}")
            return worker_ips

        except Exception as e:
            print(f"Error discovering worker nodes: {e}")
            return []


class VLLMMetricsCollector:
    """Collector for vLLM Prometheus metrics."""

    def __init__(self, base_url: str = "http://localhost:8085"):
        self.base_url = base_url
        self.metrics_url = f"{base_url}/metrics"

        # Always auto-discover worker nodes
        print("Auto-discovering worker nodes...")
        self.worker_nodes = WorkerNodeDiscovery.discover_worker_nodes()

        # Track last used worker to avoid stale metrics
        self.last_worker_id = None
        self.last_worker_ip = None

    def fetch_raw_metrics_from_all_workers(self, timeout: int = 10) -> Dict[str, str]:
        """
        Fetch raw Prometheus metrics from all worker nodes.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Dictionary mapping worker_ip -> raw metrics text
        """
        all_metrics = {}

        # Fetch from all worker nodes via SSH
        for worker_ip in self.worker_nodes:
            print(f"Fetching metrics via SSH from worker {worker_ip}...")
            response_text = self.fetch_raw_metrics_ssh(worker_ip, timeout)
            if response_text:
                all_metrics[worker_ip] = response_text
            else:
                print(f"Failed to fetch from worker {worker_ip}")

        return all_metrics

    def find_different_worker_metrics(self, timeout: int = 10) -> Optional[str]:
        """
        Find metrics from a worker with a different/newer worker ID than last time.
        This ensures we get fresh metrics from the current benchmark run.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Raw metrics text from a different worker, or None if failed
        """
        print(f"🔍 Looking for worker with different worker ID than last time...")
        if self.last_worker_id:
            print(f"  Last used: worker_id {self.last_worker_id[:16]}... on {self.last_worker_ip}")
        else:
            print(f"  First run - no previous worker ID to compare")

        # Get metrics from all workers
        all_worker_metrics = self.fetch_raw_metrics_from_all_workers(timeout)

        if not all_worker_metrics:
            return None

        # Find workers with different worker IDs (prefer newer ones)
        candidate_workers = []

        for worker_ip, metrics_text in all_worker_metrics.items():
            # Find workers with prefix cache metrics
            prefix_pattern = r'ray_vllm:gpu_prefix_cache_queries[^\n]*WorkerId="([^"]+)"[^\n]*'
            workers_with_metrics = re.findall(prefix_pattern, metrics_text)

            if not workers_with_metrics:
                print(f"  ❌ Worker {worker_ip}: no prefix cache metrics")
                continue

            current_worker_id = workers_with_metrics[-1]  # Most recent worker ID on this node

            # Check if this worker ID is different from last time
            if self.last_worker_id is None or current_worker_id != self.last_worker_id:
                print(f"  ✅ Worker {worker_ip}: different worker_id {current_worker_id[:16]}...")
                candidate_workers.append((worker_ip, current_worker_id, metrics_text))
            else:
                print(f"  ⚠️  Worker {worker_ip}: same worker_id {current_worker_id[:16]}... (potentially stale)")

        if not candidate_workers:
            print("  🔄 No workers with different worker IDs found, using any available worker")
            # Fallback: use any worker with metrics
            for worker_ip, metrics_text in all_worker_metrics.items():
                prefix_pattern = r'ray_vllm:gpu_prefix_cache_queries[^\n]*WorkerId="([^"]+)"[^\n]*'
                workers_with_metrics = re.findall(prefix_pattern, metrics_text)
                if workers_with_metrics:
                    current_worker_id = workers_with_metrics[-1]
                    candidate_workers.append((worker_ip, current_worker_id, metrics_text))
                    break

        if candidate_workers:
            # Prefer the worker with the lexicographically "newest" worker ID
            best_worker = max(candidate_workers, key=lambda x: x[1])  # Sort by worker_id
            selected_ip, selected_worker_id, selected_metrics = best_worker

            print(f"🏆 Selected worker: {selected_ip} with worker_id {selected_worker_id[:16]}...")

            # Remember this worker for next time
            self.last_worker_id = selected_worker_id
            self.last_worker_ip = selected_ip

            return selected_metrics
        else:
            print("❌ No workers with valid metrics found")
            return None

    def fetch_raw_metrics(self, timeout: int = 10) -> Optional[str]:
        """
        Fetch raw Prometheus metrics, preferring workers with different worker IDs
        than the last collection to avoid stale metrics.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Raw metrics response text from a fresh worker, or None if failed
        """
        # Check if we have worker nodes to analyze
        if self.worker_nodes:
            # Find worker with different worker ID than last time
            fresh_metrics = self.find_different_worker_metrics(timeout)
            if fresh_metrics:
                return fresh_metrics

        # Fallback to local head node
        try:
            print(f"Fetching metrics from local head node {self.metrics_url}...")
            response = requests.get(self.metrics_url, timeout=timeout)

            if response.status_code == 200:
                print(f"Successfully fetched metrics from local ({len(response.text)} characters)")
                return response.text
        except requests.RequestException as e:
            print(f"Local fetch failed: {e}")

        print("Failed to fetch metrics from any source")
        return None

    def fetch_raw_metrics_ssh(self, worker_ip: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch metrics from worker node via SSH.

        Args:
            worker_ip: Worker node IP address
            timeout: Request timeout in seconds

        Returns:
            Raw metrics response text or None if failed
        """
        import subprocess
        try:
            # Add -o StrictHostKeyChecking=no to automatically accept new hosts
            cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-p", "2222",
                worker_ip,
                "curl -s localhost:8085/metrics"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0 and result.stdout:
                print(f"Successfully fetched metrics via SSH from {worker_ip} ({len(result.stdout)} characters)")
                return result.stdout
            else:
                print(f"SSH fetch failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"SSH fetch error: {e}")
            return None

    def save_raw_metrics(self, filename: str, timeout: int = 10) -> bool:
        """
        Fetch and save raw Prometheus metrics to a file.

        Args:
            filename: Output filename for the raw response
            timeout: Request timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        response_text = self.fetch_raw_metrics(timeout)

        if response_text is None:
            return False

        try:
            with open(filename, "w") as f:
                f.write(response_text)

            print(f"Raw metrics saved to {filename}")
            return True

        except Exception as e:
            print(f"Error saving metrics to {filename}: {e}")
            return False


class GPUPrefixCacheMetricsParser:
    """Parser for GPU prefix cache metrics from vLLM Prometheus output."""

    @staticmethod
    def find_newest_worker_id(response_text: str) -> Optional[str]:
        """Find the last (most recent) WorkerId that has prefix cache metrics."""
        # Find workers that have prefix cache metrics
        prefix_pattern = r'ray_vllm:gpu_prefix_cache_queries[^\n]*WorkerId="([^"]+)"[^\n]*'
        workers_with_metrics = re.findall(prefix_pattern, response_text)

        if workers_with_metrics:
            # Use the LAST worker that has prefix cache metrics (most recent)
            target_worker = workers_with_metrics[-1]
            unique_workers = set(workers_with_metrics)
            print(f"Found {len(unique_workers)} worker(s) with prefix cache metrics, using last (most recent): {target_worker[:16]}...")
            return target_worker

        # Fallback: find any WorkerId if no prefix cache metrics exist
        worker_pattern = r'WorkerId="([^"]+)"'
        worker_ids = re.findall(worker_pattern, response_text)

        if not worker_ids:
            print("No WorkerId found in metrics")
            return None

        # Use the last worker ID found (most recent)
        newest_worker = worker_ids[-1]
        unique_workers = set(worker_ids)
        print(f"Found {len(unique_workers)} worker IDs, using last (most recent): {newest_worker[:16]}...")
        return newest_worker

    @staticmethod
    def parse_metrics(response_text: str) -> Optional[Tuple[float, float, float]]:
        """
        Parse GPU prefix cache metrics from Prometheus response for the newest WorkerId.
        Handles missing hit metrics gracefully.

        Args:
            response_text: Raw Prometheus metrics response

        Returns:
            Tuple of (hits, queries, hit_rate) or None if metrics not found
        """
        # Find the newest WorkerId
        newest_worker_id = GPUPrefixCacheMetricsParser.find_newest_worker_id(response_text)
        if not newest_worker_id:
            return None

        hits = None
        queries = None

        # Parse queries for specific WorkerId
        queries_pattern = f'ray_vllm:gpu_prefix_cache_queries_total{{[^}}]*WorkerId="{re.escape(newest_worker_id)}"[^}}]*}}\\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)'
        queries_match = re.search(queries_pattern, response_text)
        if queries_match:
            queries = float(queries_match.group(1))
            print(f"Found queries: {queries}")
        else:
            print("No GPU prefix cache queries found")
            return None

        # Parse hits for specific WorkerId - try both new and deprecated metric names
        hit_patterns = [
            f'ray_vllm:gpu_prefix_cache_hits_total{{[^}}]*WorkerId="{re.escape(newest_worker_id)}"[^}}]*}}\\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)',
            f'ray_vllm:gpu_prefix_cache_hits{{[^}}]*WorkerId="{re.escape(newest_worker_id)}"[^}}]*}}\\s+([0-9.]+(?:[eE][+-]?[0-9]+)?)'
        ]

        for pattern in hit_patterns:
            hits_match = re.search(pattern, response_text)
            if hits_match:
                hits = float(hits_match.group(1))
                print(f"Found hits: {hits}")
                break

        if hits is None:
            print("No GPU prefix cache hits found - cache may not be configured or no hits yet")
            hits = 0.0  # Default to 0 hits if metric missing

        if queries > 0:
            hit_rate = hits / queries
            print(f"WorkerId {newest_worker_id[:16]}... - Hits: {hits:,.0f}, Queries: {queries:,.0f}, Hit Rate: {hit_rate:.4f}")
            return hits, queries, hit_rate
        else:
            print(f"No queries found for WorkerId {newest_worker_id[:16]}...")
            return None

    @staticmethod
    def create_hit_rate_data(replicas: int, router_name: str, hits: float, queries: float, hit_rate: float, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create structured hit rate data dictionary.

        Args:
            replicas: Number of replicas
            router_name: Router name
            hits: Number of cache hits
            queries: Number of cache queries
            hit_rate: Calculated hit rate (0.0 to 1.0)
            worker_id: WorkerId for this deployment

        Returns:
            Dictionary with structured hit rate data
        """
        return {
            "replicas": replicas,
            "router_name": router_name,
            "worker_id": worker_id,
            "gpu_prefix_cache_hits": hits,
            "gpu_prefix_cache_queries": queries,
            "gpu_prefix_cache_hit_rate": hit_rate,
            "gpu_prefix_cache_hit_rate_percent": hit_rate * 100,
            "timestamp": time.time()
        }


class EngineMetrics:
    """Main class for collecting and analyzing vLLM engine metrics."""

    def __init__(self, base_url: str = "http://localhost:8085"):
        self.collector = VLLMMetricsCollector(base_url)
        self.parser = GPUPrefixCacheMetricsParser()
        self.worker_nodes = self.collector.worker_nodes  # Use the discovered nodes

    def collect_gpu_prefix_cache_metrics(
        self,
        replicas: int,
        router_name: str,
        raw_metrics_filename: Optional[str] = None,
        hit_rate_filename: Optional[str] = None,
        save_raw: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Collect and parse GPU prefix cache hit rate metrics.
        Uses SSH for worker nodes, local for head node.

        Args:
            replicas: Number of replicas
            router_name: Router name
            raw_metrics_filename: Optional filename for raw metrics (auto-generated if None)
            hit_rate_filename: Optional filename for hit rate JSON (auto-generated if None)
            save_raw: Whether to save raw metrics to file

        Returns:
            Hit rate data dictionary or None if failed
        """
        # Create replica_sweep directory if it doesn't exist
        os.makedirs("../results/replica_sweep", exist_ok=True)
        
        # Generate filenames if not provided
        if raw_metrics_filename is None:
            raw_metrics_filename = f"../results/replica_sweep/{replicas}_replicas_{router_name}_raw_metrics.txt"
        if hit_rate_filename is None:
            hit_rate_filename = f"../results/replica_sweep/{replicas}_replicas_{router_name}_hit_rate.json"

        # Fetch raw metrics (SSH to workers first, then local)
        response_text = self.collector.fetch_raw_metrics()

        if response_text is None:
            print("Failed to retrieve metrics from any source")
            return None

        # Save raw metrics if requested
        if save_raw:
            try:
                with open(raw_metrics_filename, "w") as f:
                    f.write(response_text)
                print(f"Raw metrics saved to {raw_metrics_filename}")
            except Exception as e:
                print(f"Error saving raw metrics: {e}")

        # Parse metrics
        try:
            metrics = self.parser.parse_metrics(response_text)

            if metrics:
                hits, queries, hit_rate = metrics
                worker_id = self.parser.find_newest_worker_id(response_text)
                hit_rate_data = self.parser.create_hit_rate_data(replicas, router_name, hits, queries, hit_rate, worker_id)

                # Save hit rate data to JSON
                with open(hit_rate_filename, 'w') as f:
                    json.dump(hit_rate_data, f, indent=2)

                print(f"GPU Prefix Cache Hit Rate: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
                print(f"  Hits: {hits:,.0f}")
                print(f"  Queries: {queries:,.0f}")
                print(f"Hit rate data saved to {hit_rate_filename}")

                return hit_rate_data
            else:
                print("Could not find GPU prefix cache metrics in the response")
                return None

        except Exception as e:
            print(f"Error parsing metrics: {e}")
            return None

    def collect_from_file(
        self,
        raw_metrics_file: str,
        replicas: int,
        router_name: str,
        hit_rate_filename: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse metrics from an existing raw metrics file using newest WorkerId.

        Args:
            raw_metrics_file: Path to raw Prometheus metrics file
            replicas: Number of replicas
            router_name: Router name
            hit_rate_filename: Optional filename for hit rate JSON (auto-generated if None)

        Returns:
            Hit rate data dictionary or None if failed
        """
        # Create replica_sweep directory if it doesn't exist
        os.makedirs("../results/replica_sweep", exist_ok=True)
        
        if hit_rate_filename is None:
            hit_rate_filename = f"../results/replica_sweep/{replicas}_replicas_{router_name}_hit_rate.json"

        try:
            with open(raw_metrics_file, 'r') as f:
                response_text = f.read()

            metrics = self.parser.parse_metrics(response_text)

            if metrics:
                hits, queries, hit_rate = metrics
                worker_id = self.parser.find_newest_worker_id(response_text)
                hit_rate_data = self.parser.create_hit_rate_data(replicas, router_name, hits, queries, hit_rate, worker_id)

                with open(hit_rate_filename, 'w') as f:
                    json.dump(hit_rate_data, f, indent=2)

                print(f"GPU Prefix Cache Hit Rate: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
                print(f"  Hits: {hits:,.0f}")
                print(f"  Queries: {queries:,.0f}")
                print(f"Hit rate data saved to {hit_rate_filename}")

                return hit_rate_data
            else:
                print("Could not find GPU prefix cache metrics in the response")
                return None

        except Exception as e:
            print(f"Error parsing metrics from file: {e}")
            return None
