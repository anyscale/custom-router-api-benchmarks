#!/usr/bin/env python3
"""
Visualize replica sweep results, comparing different numbers of replicas.
Uses the same four plots as visualize_concurrency_sweep.py but with replica scaling.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import re

def load_replica_data_from_directories(directories):
    """Load replica sweep data from multiple directories and combine them."""
    combined_data = {'pow_2': [], 'prefix_aware': []}
    combined_hit_rate_data = {'pow_2': [], 'prefix_aware': []}
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist, skipping")
            continue
            
        print(f"Loading data from {directory}")
        data, hit_rate_data = load_replica_data_single_directory(directory)
        
        # Combine data from this directory
        for router_type in ['pow_2', 'prefix_aware']:
            combined_data[router_type].extend(data[router_type])
            combined_hit_rate_data[router_type].extend(hit_rate_data[router_type])
    
    # Sort by replica count for consistent plotting
    for router_type in ['pow_2', 'prefix_aware']:
        combined_data[router_type].sort(key=lambda x: x['replica_count'])
        combined_hit_rate_data[router_type].sort(key=lambda x: x['replica_count'])
    
    return combined_data, combined_hit_rate_data

def load_replica_data_single_directory(directory):
    """Load replica sweep data from a single directory and organize by router type."""
    data = {'pow_2': [], 'prefix_aware': []}
    hit_rate_data = {'pow_2': [], 'prefix_aware': []}

    # Automatically infer replica counts from directory contents
    replica_counts = set()
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            # Match pattern: {replica_count}_replicas_{router_type}.json
            match = re.match(r'(\d+)_replicas_(?:pow_2|prefix_aware)(?:_hit_rate)?\.json$', filename)
            if match:
                replica_count = int(match.group(1))
                replica_counts.add(replica_count)

    replica_counts = sorted(list(replica_counts))

    if not replica_counts:
        print(f"Warning: No replica sweep files found in {directory}")
        return data, hit_rate_data

    print(f"  Found replica counts: {replica_counts}")

    # Try to load consolidated cache hit rates file
    cache_hit_rates_file = os.path.join(directory, 'cache_hit_rates.json')
    consolidated_hit_rates = None
    if os.path.exists(cache_hit_rates_file):
        try:
            with open(cache_hit_rates_file, 'r') as f:
                consolidated_hit_rates = json.load(f)
                print(f"  Loaded consolidated cache hit rates from {cache_hit_rates_file}")
        except Exception as e:
            print(f"  Warning: Could not load {cache_hit_rates_file}: {e}")

    for replica_count in replica_counts:
        for router_type in ['pow_2', 'prefix_aware']:
            # Load main benchmark data
            filename = f"{replica_count}_replicas_{router_type}.json"
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    # Handle multiple JSON objects in one file (take the last one)
                    if '\n{' in content:
                        json_objects = []
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and line.startswith('{'):
                                try:
                                    json_objects.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                        if json_objects:
                            entry = json_objects[-1]  # Use the last (most recent) entry
                        else:
                            continue
                    else:
                        entry = json.loads(content)
                    
                    entry['replica_count'] = replica_count
                    entry['source_directory'] = os.path.basename(directory)  # Track source
                    data[router_type].append(entry)
            except FileNotFoundError:
                print(f"  Warning: {filepath} not found")
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse JSON in {filepath}: {e}")

            # Load hit rate data - try consolidated file first, then individual files
            hit_rate_entry = None
            
            if consolidated_hit_rates:
                # Use consolidated cache hit rates file
                router_key = f"{router_type}_cache_hit_rates"
                if router_key in consolidated_hit_rates:
                    hit_rate_value = consolidated_hit_rates[router_key].get(str(replica_count))
                    if hit_rate_value is not None:
                        hit_rate_entry = {
                            'replica_count': replica_count,
                            'gpu_prefix_cache_hit_rate_percent': hit_rate_value * 100,  # Convert to percentage
                            'source_directory': os.path.basename(directory)
                        }
            
            if hit_rate_entry is None:
                # Fall back to individual hit rate files
                hit_rate_filename = f"{replica_count}_replicas_{router_type}_hit_rate.json"
                hit_rate_filepath = os.path.join(directory, hit_rate_filename)

                try:
                    with open(hit_rate_filepath, 'r') as f:
                        hit_rate_entry = json.load(f)
                        hit_rate_entry['replica_count'] = replica_count
                        hit_rate_entry['source_directory'] = os.path.basename(directory)
                except FileNotFoundError:
                    # For direct ingress full run, create dummy hit rate data with 0% hit rate
                    hit_rate_entry = {
                        'replica_count': replica_count,
                        'gpu_prefix_cache_hit_rate_percent': 0.0,
                        'source_directory': os.path.basename(directory)
                    }
                    print(f"  Info: {hit_rate_filepath} not found, using 0% hit rate for direct ingress")
            
            if hit_rate_entry:
                hit_rate_data[router_type].append(hit_rate_entry)

    return data, hit_rate_data

def load_replica_data(directory):
    """Legacy function for backward compatibility - loads from single directory."""
    return load_replica_data_single_directory(directory)

def calculate_metrics(data):
    """Calculate theoretical throughput metrics based on TPOT/TTFT latencies."""
    replica_counts = [d['replica_count'] for d in data]
    median_tpot_ms = [d['median_tpot_ms'] for d in data]
    median_ttft_ms = [d['median_ttft_ms'] for d in data]
    num_prompts = [d['num_prompts'] for d in data]
    max_concurrency = [d['max_concurrency'] for d in data]

    # Calculate effective concurrency (can't exceed number of requests)
    effective_concurrency = [min(conc, num_req) for conc, num_req in zip(max_concurrency, num_prompts)]

    # Calculate tokens per request for each data point
    input_tokens_per_request = [d['total_input_tokens'] / d['num_prompts'] for d in data]
    output_tokens_per_request = [d['total_output_tokens'] / d['num_prompts'] for d in data]

    # Calculate theoretical throughput metrics
    # 1. Output token throughput based on TPOT: concurrency * 1000 / TPOT_ms
    output_throughput_tpot_based = [(eff_conc * 1000) / tpot for eff_conc, tpot in zip(effective_concurrency, median_tpot_ms)]

    # 2. Input token throughput based on TTFT: (input_tokens_per_request × 1000 / TTFT_ms) × concurrency
    input_throughput_ttft_based = [(in_tokens * 1000 / ttft) * eff_conc for in_tokens, ttft, eff_conc in zip(input_tokens_per_request, median_ttft_ms, effective_concurrency)]

    # Normalize throughput by number of replicas
    output_throughput_per_replica = [throughput / replica_count for throughput, replica_count in zip(output_throughput_tpot_based, replica_counts)]
    input_throughput_per_replica = [throughput / replica_count for throughput, replica_count in zip(input_throughput_ttft_based, replica_counts)]

    # Calculate total token throughput per replica from raw data
    total_input_tokens = [d['total_input_tokens'] for d in data]
    total_output_tokens = [d['total_output_tokens'] for d in data]
    duration = [d['duration'] for d in data]
    total_throughput_per_replica = [
        (input_tokens + output_tokens) / (dur * replica_count)
        for input_tokens, output_tokens, dur, replica_count
        in zip(total_input_tokens, total_output_tokens, duration, replica_counts)
    ]

    # 3. TPOT as tokens per second: 1000 / TPOT_ms
    tpot_tokens_per_sec = [1000 / tpot for tpot in median_tpot_ms]

    # 4. TTFT as tokens per second: input_tokens_per_request × 1000 / TTFT_ms
    ttft_tokens_per_sec = [in_tokens * 1000 / ttft for in_tokens, ttft in zip(input_tokens_per_request, median_ttft_ms)]

    return {
        'replica_counts': replica_counts,
        'median_tpot_ms': median_tpot_ms,
        'median_ttft_ms': median_ttft_ms,
        'tpot_tokens_per_sec': tpot_tokens_per_sec,
        'ttft_tokens_per_sec': ttft_tokens_per_sec,
        'output_throughput_tpot_based': output_throughput_tpot_based,
        'input_throughput_ttft_based': input_throughput_ttft_based,
        'output_throughput_per_replica': output_throughput_per_replica,
        'input_throughput_per_replica': input_throughput_per_replica,
        'total_throughput_per_replica': total_throughput_per_replica,
        'input_tokens_per_request': input_tokens_per_request,
        'output_tokens_per_request': output_tokens_per_request,
        'max_concurrency': max_concurrency,
        'num_prompts': num_prompts
    }

def create_comparison_plots(pow2_data, prefix_aware_data, pow2_hit_rate_data, prefix_aware_hit_rate_data, replica_counts):
    """Create five plots comparing both router types across replica counts."""

    # Calculate metrics for both datasets
    pow2_metrics = calculate_metrics(pow2_data)
    prefix_metrics = calculate_metrics(prefix_aware_data)

    # Determine if we should use log scale (only if all replica counts are powers of 2)
    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0

    use_log_scale = len(replica_counts) > 1 and all(is_power_of_2(x) for x in replica_counts)

    # Prepare tick settings
    if use_log_scale:
        tick_positions = replica_counts
        tick_labels = [str(x) for x in replica_counts]
    else:
        tick_positions = replica_counts
        tick_labels = [str(x) for x in replica_counts]

    # Create subplots - 2 rows, 3 columns for landscape layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    replica_range = f"{min(replica_counts)}-{max(replica_counts)}" if len(replica_counts) > 1 else str(replica_counts[0])

    # Extract model_id from the data (use first available entry)
    model_id = "Unknown Model"
    if pow2_data and len(pow2_data) > 0:
        model_id = pow2_data[0].get('model_id', 'Unknown Model')
    elif prefix_aware_data and len(prefix_aware_data) > 0:
        model_id = prefix_aware_data[0].get('model_id', 'Unknown Model')

    fig.suptitle(f'Replica Scaling: Power-of-2 vs Prefix-Aware ({replica_range} Replicas)\nvLLM Backend ({model_id}), Combined 16-Replica Dataset', fontsize=16, fontweight='bold')

    # Helper function to set x-axis consistently
    def setup_x_axis(ax):
        if use_log_scale:
            ax.set_xscale('log', base=2)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    # Plot 1: TPOT in milliseconds (Top Left)
    ax1 = axes[0, 0]
    ax1.plot(pow2_metrics['replica_counts'], pow2_metrics['median_tpot_ms'],
             'o-', linewidth=3, markersize=8, color='blue', label='Power-of-2 Router')
    ax1.plot(prefix_metrics['replica_counts'], prefix_metrics['median_tpot_ms'],
             's-', linewidth=3, markersize=8, color='red', label='Prefix-Aware Router')
    ax1.set_xlabel('Number of Replicas')
    ax1.set_ylabel('TPOT (milliseconds)')
    ax1.set_title('Time Per Output Token (p50)')
    ax1.grid(True, alpha=0.3)
    setup_x_axis(ax1)
    ax1.legend()

    # Add value labels for both lines
    for x, y in zip(pow2_metrics['replica_counts'], pow2_metrics['median_tpot_ms']):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=8, color='blue')
    for x, y in zip(prefix_metrics['replica_counts'], prefix_metrics['median_tpot_ms']):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    # Plot 2: TTFT in milliseconds (Top Middle)
    ax2 = axes[0, 1]
    ax2.plot(pow2_metrics['replica_counts'], pow2_metrics['median_ttft_ms'],
             'o-', linewidth=3, markersize=8, color='blue', label='Power-of-2 Router')
    ax2.plot(prefix_metrics['replica_counts'], prefix_metrics['median_ttft_ms'],
             's-', linewidth=3, markersize=8, color='red', label='Prefix-Aware Router')
    ax2.set_xlabel('Number of Replicas')
    ax2.set_ylabel('TTFT (milliseconds)')
    ax2.set_title('Time To First Token (p50)')
    ax2.grid(True, alpha=0.3)
    setup_x_axis(ax2)
    ax2.legend()

    # Add value labels for both lines
    for x, y in zip(pow2_metrics['replica_counts'], pow2_metrics['median_ttft_ms']):
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=8, color='blue')
    for x, y in zip(prefix_metrics['replica_counts'], prefix_metrics['median_ttft_ms']):
        ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    # Plot 3: TPOT-Based Output Token Throughput Per Replica (Bottom Left)
    ax3 = axes[1, 0]
    ax3.plot(pow2_metrics['replica_counts'], pow2_metrics['output_throughput_per_replica'],
             'o-', linewidth=3, markersize=8, color='blue', label='Power-of-2 Router')
    ax3.plot(prefix_metrics['replica_counts'], prefix_metrics['output_throughput_per_replica'],
             's-', linewidth=3, markersize=8, color='red', label='Prefix-Aware Router')
    ax3.set_xlabel('Number of Replicas')
    ax3.set_ylabel('Output Tokens/s Per Replica')
    ax3.set_title('TPOT-Based Output Throughput Per Replica\n(Concurrency × 1000) / (TPOT_ms × Replicas)')
    ax3.grid(True, alpha=0.3)
    setup_x_axis(ax3)
    # Format y-axis to show clean integer labels
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax3.legend()

    # Add value labels for both lines
    for x, y in zip(pow2_metrics['replica_counts'], pow2_metrics['output_throughput_per_replica']):
        ax3.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=8, color='blue')
    for x, y in zip(prefix_metrics['replica_counts'], prefix_metrics['output_throughput_per_replica']):
        ax3.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    # Plot 4: TTFT-Based Input Token Throughput Per Replica (Bottom Middle)
    ax4 = axes[1, 1]
    ax4.plot(pow2_metrics['replica_counts'], pow2_metrics['input_throughput_per_replica'],
             'o-', linewidth=3, markersize=8, color='blue', label='Power-of-2 Router')
    ax4.plot(prefix_metrics['replica_counts'], prefix_metrics['input_throughput_per_replica'],
             's-', linewidth=3, markersize=8, color='red', label='Prefix-Aware Router')
    ax4.set_xlabel('Number of Replicas')
    ax4.set_ylabel('Input Tokens/s Per Replica')
    ax4.set_title('TTFT-Based Input Throughput Per Replica\n(Input_tokens × 1000 / TTFT_ms) × Concurrency / Replicas')
    ax4.grid(True, alpha=0.3)
    setup_x_axis(ax4)
    # Format y-axis to show clean integer labels
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax4.legend()

    # Add value labels for both lines
    for x, y in zip(pow2_metrics['replica_counts'], pow2_metrics['input_throughput_per_replica']):
        ax4.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=8, color='blue')
    for x, y in zip(prefix_metrics['replica_counts'], prefix_metrics['input_throughput_per_replica']):
        ax4.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    # Plot 5: GPU Prefix Cache Hit Rate (Top Right)
    ax5 = axes[0, 2]

    # Extract hit rate data
    pow2_hit_rates = []
    pow2_replica_counts = []
    prefix_hit_rates = []
    prefix_replica_counts = []

    for hit_data in pow2_hit_rate_data:
        pow2_hit_rates.append(hit_data['gpu_prefix_cache_hit_rate_percent'])
        pow2_replica_counts.append(hit_data['replica_count'])

    for hit_data in prefix_aware_hit_rate_data:
        prefix_hit_rates.append(hit_data['gpu_prefix_cache_hit_rate_percent'])
        prefix_replica_counts.append(hit_data['replica_count'])

    if pow2_hit_rates:
        ax5.plot(pow2_replica_counts, pow2_hit_rates,
                 'o-', linewidth=3, markersize=8, color='blue', label='Power-of-2 Router')
    if prefix_hit_rates:
        ax5.plot(prefix_replica_counts, prefix_hit_rates,
                 's-', linewidth=3, markersize=8, color='red', label='Prefix-Aware Router')

    ax5.set_xlabel('Number of Replicas')
    ax5.set_ylabel('GPU Prefix Cache Hit Rate (%)')
    ax5.set_title('GPU Prefix Cache Hit Rate by Replica Count')
    ax5.grid(True, alpha=0.3)
    setup_x_axis(ax5)
    if pow2_hit_rates or prefix_hit_rates:
        ax5.legend()

    # Add value labels
    for x, y in zip(pow2_replica_counts, pow2_hit_rates):
        ax5.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=8, color='blue')
    for x, y in zip(prefix_replica_counts, prefix_hit_rates):
        ax5.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    # Plot 6: Total Token Throughput Per Replica (Bottom Right)
    ax6 = axes[1, 2]
    ax6.plot(pow2_metrics['replica_counts'], pow2_metrics['total_throughput_per_replica'],
             'o-', linewidth=3, markersize=8, color='blue', label='Power-of-2 Router')
    ax6.plot(prefix_metrics['replica_counts'], prefix_metrics['total_throughput_per_replica'],
             's-', linewidth=3, markersize=8, color='red', label='Prefix-Aware Router')
    ax6.set_xlabel('Number of Replicas')
    ax6.set_ylabel('Total Tokens/s Per Replica')
    ax6.set_title('Total Token Throughput Per Replica\n(Input + Output Tokens) / (Duration × Replicas)')
    ax6.grid(True, alpha=0.3)
    setup_x_axis(ax6)
    # Format y-axis to show clean integer labels
    ax6.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax6.legend()

    # Add value labels for both lines
    for x, y in zip(pow2_metrics['replica_counts'], pow2_metrics['total_throughput_per_replica']):
        ax6.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,15), ha='center', fontsize=8, color='blue')
    for x, y in zip(prefix_metrics['replica_counts'], prefix_metrics['total_throughput_per_replica']):
        ax6.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('../results/visualizations', exist_ok=True)
    
    plt.savefig('../results/visualizations/replica_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print('Replica scaling plots saved as ../results/visualizations/replica_scaling_analysis.png')

def print_comparison_summary(pow2_data, prefix_aware_data, replica_counts):
    """Print a summary comparing both router types across replica counts."""
    replica_range = f"{min(replica_counts)}-{max(replica_counts)}" if len(replica_counts) > 1 else str(replica_counts[0])
    print(f"\n=== REPLICA SCALING SUMMARY ({replica_range} Replicas) ===")

    # Calculate metrics for both datasets
    pow2_metrics = calculate_metrics(pow2_data)
    prefix_metrics = calculate_metrics(prefix_aware_data)

    print(f"\nReplica scaling configuration:")
    for i, replica_count in enumerate(pow2_metrics['replica_counts']):
        print(f"  {replica_count} replicas: {pow2_metrics['max_concurrency'][i]} concurrency, {pow2_metrics['num_prompts'][i]} prompts")

    print(f"\nTheoretical Throughput Per Replica:")
    print(f"{'Replicas':<8} {'POW2 Output':<12} {'PREFIX Output':<12} {'POW2 Input':<12} {'PREFIX Input':<12}")
    print(f"{'Count':<8} {'Per Replica':<12} {'Per Replica':<12} {'Per Replica':<12} {'Per Replica':<12}")
    print("-" * 76)

    # Create mapping by replica count for both datasets
    pow2_by_replicas = {count: i for i, count in enumerate(pow2_metrics['replica_counts'])}
    prefix_by_replicas = {count: i for i, count in enumerate(prefix_metrics['replica_counts'])}

    for replica_count in replica_counts:
        pow2_output = pow2_metrics['output_throughput_per_replica'][pow2_by_replicas[replica_count]] if replica_count in pow2_by_replicas else 0
        prefix_output = prefix_metrics['output_throughput_per_replica'][prefix_by_replicas[replica_count]] if replica_count in prefix_by_replicas else 0
        pow2_input = pow2_metrics['input_throughput_per_replica'][pow2_by_replicas[replica_count]] if replica_count in pow2_by_replicas else 0
        prefix_input = prefix_metrics['input_throughput_per_replica'][prefix_by_replicas[replica_count]] if replica_count in prefix_by_replicas else 0

        print(f"{replica_count:<8} {pow2_output:<12.0f} {prefix_output:<12.0f} {pow2_input:<12.0f} {prefix_input:<12.0f}")

    print(f"\nTheoretical Total System Throughput:")
    print(f"{'Replicas':<8} {'POW2 Output':<12} {'PREFIX Output':<12} {'POW2 Input':<12} {'PREFIX Input':<12}")
    print(f"{'Count':<8} {'Total':<12} {'Total':<12} {'Total':<12} {'Total':<12}")
    print("-" * 76)

    for replica_count in replica_counts:
        pow2_output = pow2_metrics['output_throughput_tpot_based'][pow2_by_replicas[replica_count]] if replica_count in pow2_by_replicas else 0
        prefix_output = prefix_metrics['output_throughput_tpot_based'][prefix_by_replicas[replica_count]] if replica_count in prefix_by_replicas else 0
        pow2_input = pow2_metrics['input_throughput_ttft_based'][pow2_by_replicas[replica_count]] if replica_count in pow2_by_replicas else 0
        prefix_input = prefix_metrics['input_throughput_ttft_based'][prefix_by_replicas[replica_count]] if replica_count in prefix_by_replicas else 0

        print(f"{replica_count:<8} {pow2_output:<12.0f} {prefix_output:<12.0f} {pow2_input:<12.0f} {prefix_input:<12.0f}")

    # Calculate scaling efficiency based on per-replica throughput (skip if missing baseline data)
    if replica_counts and replica_counts[0] in pow2_by_replicas and replica_counts[0] in prefix_by_replicas:
        print(f"\nPer-Replica Efficiency (relative to single replica):")
        print(f"{'Replicas':<8} {'POW2 Output':<12} {'PREFIX Output':<12} {'POW2 Input':<12} {'PREFIX Input':<12}")
        print(f"{'Count':<8} {'Per-Replica':<12} {'Per-Replica':<12} {'Per-Replica':<12} {'Per-Replica':<12}")
        print("-" * 76)

        # Find the baseline (first available replica count for both)
        baseline_replica = replica_counts[0]
        pow2_base_output = pow2_metrics['output_throughput_per_replica'][pow2_by_replicas[baseline_replica]]
        prefix_base_output = prefix_metrics['output_throughput_per_replica'][prefix_by_replicas[baseline_replica]]
        pow2_base_input = pow2_metrics['input_throughput_per_replica'][pow2_by_replicas[baseline_replica]]
        prefix_base_input = prefix_metrics['input_throughput_per_replica'][prefix_by_replicas[baseline_replica]]

        for replica_count in replica_counts:
            if replica_count in pow2_by_replicas and replica_count in prefix_by_replicas:
                pow2_output_scale = pow2_metrics['output_throughput_per_replica'][pow2_by_replicas[replica_count]] / pow2_base_output
                prefix_output_scale = prefix_metrics['output_throughput_per_replica'][prefix_by_replicas[replica_count]] / prefix_base_output
                pow2_input_scale = pow2_metrics['input_throughput_per_replica'][pow2_by_replicas[replica_count]] / pow2_base_input
                prefix_input_scale = prefix_metrics['input_throughput_per_replica'][prefix_by_replicas[replica_count]] / prefix_base_input

                print(f"{replica_count:<8} {pow2_output_scale:<12.2f} {prefix_output_scale:<12.2f} {pow2_input_scale:<12.2f} {prefix_input_scale:<12.2f}")
            else:
                print(f"{replica_count:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    # Find best performing configurations (based on per-replica throughput)
    pow2_max_output_idx = max(range(len(pow2_metrics['output_throughput_per_replica'])),
                             key=lambda i: pow2_metrics['output_throughput_per_replica'][i])
    prefix_max_output_idx = max(range(len(prefix_metrics['output_throughput_per_replica'])),
                               key=lambda i: prefix_metrics['output_throughput_per_replica'][i])

    print(f"\nBest Per-Replica Configurations:")
    print(f"Power-of-2 Router: {pow2_metrics['output_throughput_per_replica'][pow2_max_output_idx]:.0f} tokens/s per replica at {pow2_metrics['replica_counts'][pow2_max_output_idx]} replicas")
    print(f"Prefix-Aware Router: {prefix_metrics['output_throughput_per_replica'][prefix_max_output_idx]:.0f} tokens/s per replica at {prefix_metrics['replica_counts'][prefix_max_output_idx]} replicas")

    # Calculate overall performance comparison (per-replica averages)
    pow2_avg_output = sum(pow2_metrics['output_throughput_per_replica']) / len(pow2_metrics['output_throughput_per_replica'])
    prefix_avg_output = sum(prefix_metrics['output_throughput_per_replica']) / len(prefix_metrics['output_throughput_per_replica'])

    print(f"\nAverage Per-Replica Performance Across All Replica Counts:")
    print(f"Power-of-2 Router: {pow2_avg_output:.0f} tokens/s per replica")
    print(f"Prefix-Aware Router: {prefix_avg_output:.0f} tokens/s per replica")

    improvement = ((prefix_avg_output - pow2_avg_output) / pow2_avg_output) * 100
    print(f"Prefix-Aware vs Power-of-2: {improvement:+.1f}% average per-replica improvement")

def main():
    """Main function to run the replica scaling analysis."""
    import sys
    
    # Allow directory to be specified as command line argument
    if len(sys.argv) > 1:
        replica_dir = sys.argv[1]
    else:
        replica_dir = '../results/replica_sweep'
    
    print(f"Loading data from: {replica_dir}")

    try:
        data, hit_rate_data = load_replica_data(replica_dir)

        if not data['pow_2'] and not data['prefix_aware']:
            print(f"Error: No data found in {replica_dir}/")
            return

        # Get replica counts from available data
        all_replica_counts = set()
        if data['pow_2']:
            all_replica_counts.update([d['replica_count'] for d in data['pow_2']])
        if data['prefix_aware']:
            all_replica_counts.update([d['replica_count'] for d in data['prefix_aware']])
        
        replica_counts = sorted(list(all_replica_counts))
        
        if not replica_counts:
            print("Error: No replica count data found")
            return
        
        print(f"Found data for replica counts: {replica_counts}")

        create_comparison_plots(data['pow_2'], data['prefix_aware'],
                              hit_rate_data['pow_2'], hit_rate_data['prefix_aware'], replica_counts)
        print_comparison_summary(data['pow_2'], data['prefix_aware'], replica_counts)

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()