"""
Visualization utilities for path planning analysis and paper figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd


def plot_algorithm_performance(metrics_dict: Dict[str, dict], 
                               save_path: str = "algorithm_performance.png"):
    """
    Compare A* and RRT* performance metrics
    
    Args:
        metrics_dict: Dictionary with algorithm names as keys and metrics as values
    """
    
    algorithms = list(metrics_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Path Planning Algorithm Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Planning Time
    ax = axes[0, 0]
    planning_times = [metrics_dict[algo]['planning_time_ms'] for algo in algorithms]
    colors = ['#3498db', '#e74c3c']
    ax.bar(algorithms, planning_times, color=colors, alpha=0.7)
    ax.set_ylabel('Planning Time (ms)', fontweight='bold')
    ax.set_title('Computation Speed')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(planning_times):
        ax.text(i, v, f'{v:.1f}ms', ha='center', va='bottom')
    
    # Plot 2: Nodes Explored
    ax = axes[0, 1]
    nodes = [metrics_dict[algo]['nodes_explored'] for algo in algorithms]
    ax.bar(algorithms, nodes, color=colors, alpha=0.7)
    ax.set_ylabel('Nodes Explored', fontweight='bold')
    ax.set_title('Search Efficiency')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(nodes):
        ax.text(i, v, f'{v}', ha='center', va='bottom')
    
    # Plot 3: Path Length
    ax = axes[1, 0]
    path_lengths = [metrics_dict[algo]['path_length'] for algo in algorithms]
    ax.bar(algorithms, path_lengths, color=colors, alpha=0.7)
    ax.set_ylabel('Path Length (nodes)', fontweight='bold')
    ax.set_title('Path Quality')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(path_lengths):
        ax.text(i, v, f'{v}', ha='center', va='bottom')
    
    # Plot 4: Total Cost
    ax = axes[1, 1]
    costs = [metrics_dict[algo]['total_cost'] for algo in algorithms]
    ax.bar(algorithms, costs, color=colors, alpha=0.7)
    ax.set_ylabel('Total Path Cost', fontweight='bold')
    ax.set_title('Terrain-Weighted Cost')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(costs):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved algorithm performance to {save_path}")


def plot_path_visualization(hazard_map: np.ndarray,
                            paths: Dict[str, List[Tuple[int, int]]],
                            start: Tuple[int, int],
                            goal: Tuple[int, int],
                            save_path: str = "path_visualization.png"):
    """
    Create publication-quality path visualization
    
    Args:
        hazard_map: 2D numpy array with terrain classes
        paths: Dictionary of algorithm_name -> path (list of positions)
        start: Start position (row, col)
        goal: Goal position (row, col)
    """
    
    # Color mapping for terrain
    colors = {
        0: [0, 0, 0],         # Background - Black
        1: [255, 0, 0],       # Safe - Red
        2: [0, 255, 0],       # Rocks - Green
        3: [255, 255, 0],     # Crater - Yellow
    }
    
    h, w = hazard_map.shape
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colors.items():
        rgb_map[hazard_map == cls] = color
    
    n_plots = len(paths)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    path_colors = ['#2980b9', '#e67e22', '#27ae60']
    
    for idx, (algo_name, path) in enumerate(paths.items()):
        ax = axes[idx]
        
        # Display terrain
        ax.imshow(rgb_map)
        
        # Draw path
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 
                   color=path_colors[idx % len(path_colors)], 
                   linewidth=3, label='Planned Path', alpha=0.9)
        
        # Mark start and goal
        ax.plot(start[1], start[0], 'go', markersize=15, 
               label='Start', markeredgecolor='white', markeredgewidth=2, zorder=10)
        ax.plot(goal[1], goal[0], 'r*', markersize=20, 
               label='Goal', markeredgecolor='white', markeredgewidth=2, zorder=10)
        
        ax.set_title(f'{algo_name} Path Planning', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.axis('off')
    
    # Add legend for terrain
    legend_elements = [
        mpatches.Patch(color='red', label='Safe Zone'),
        mpatches.Patch(color='green', label='Rocks'),
        mpatches.Patch(color='yellow', label='Craters'),
        mpatches.Patch(color='black', label='Background')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=4, fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved path visualization to {save_path}")


def plot_risk_analysis(risk_data_dict: Dict[str, dict],
                      save_path: str = "risk_analysis.png"):
    """
    Visualize path risk metrics for different algorithms
    
    Args:
        risk_data_dict: Dictionary with algorithm names and their risk data
    """
    
    algorithms = list(risk_data_dict.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Terrain Distribution
    ax = axes[0]
    
    terrain_types = ['Safe', 'Rocks', 'Craters', 'Background']
    x = np.arange(len(terrain_types))
    width = 0.35
    
    colors_terrain = ['#2ecc71', '#f39c12', '#e74c3c', '#34495e']
    
    for idx, (algo, data) in enumerate(risk_data_dict.items()):
        values = [
            data['safe_pixels'],
            data['rock_pixels'],
            data['crater_pixels'],
            data['background_pixels']
        ]
        offset = width * (idx - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, values, width, label=algo, alpha=0.8)
    
    ax.set_ylabel('Number of Pixels', fontweight='bold')
    ax.set_xlabel('Terrain Type', fontweight='bold')
    ax.set_title('Path Terrain Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(terrain_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Risk Score Comparison
    ax = axes[1]
    
    risk_scores = [risk_data_dict[algo]['risk_score'] for algo in algorithms]
    safe_percentages = [risk_data_dict[algo]['safe_percentage'] for algo in algorithms]
    
    x_pos = np.arange(len(algorithms))
    
    # Dual axis plot
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x_pos - 0.2, risk_scores, 0.4, label='Risk Score', 
                   color='#e74c3c', alpha=0.7)
    bars2 = ax2.bar(x_pos + 0.2, safe_percentages, 0.4, label='Safe %', 
                    color='#2ecc71', alpha=0.7)
    
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('Risk Score', fontweight='bold', color='#e74c3c')
    ax2.set_ylabel('Safe Terrain %', fontweight='bold', color='#2ecc71')
    ax.set_title('Path Safety Analysis', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms)
    
    ax.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved risk analysis to {save_path}")


def generate_performance_table(metrics_dict: Dict[str, dict]) -> pd.DataFrame:
    """
    Generate performance comparison table for paper
    
    Returns:
        DataFrame ready for LaTeX export
    """
    
    data = {
        'Algorithm': [],
        'Planning Time (ms)': [],
        'Nodes Explored': [],
        'Path Length': [],
        'Total Cost': [],
        'Success': []
    }
    
    for algo, metrics in metrics_dict.items():
        data['Algorithm'].append(algo)
        data['Planning Time (ms)'].append(f"{metrics['planning_time_ms']:.2f}")
        data['Nodes Explored'].append(metrics['nodes_explored'])
        data['Path Length'].append(metrics['path_length'])
        data['Total Cost'].append(f"{metrics['total_cost']:.2f}")
        data['Success'].append('✓' if metrics['success'] else '✗')
    
    df = pd.DataFrame(data)
    
    print("\n📊 Performance Comparison Table:")
    print(df.to_string(index=False))
    print("\n📄 LaTeX Format:")
    print(df.to_latex(index=False))
    
    return df


def plot_latency_breakdown(inference_time: float = 75.0,
                          astar_time: float = 35.0,
                          rrt_time: float = 250.0,
                          save_path: str = "latency_breakdown.png"):
    """
    Visualize system latency breakdown
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # A* Pipeline
    stages_astar = ['Hazard\nDetection', 'A* Path\nPlanning', 'Path\nSmoothing']
    times_astar = [inference_time, astar_time, 10]
    colors_astar = ['#3498db', '#2ecc71', '#f39c12']
    
    ax1.bar(stages_astar, times_astar, color=colors_astar, alpha=0.8)
    ax1.set_ylabel('Time (ms)', fontweight='bold')
    ax1.set_title('A* Pipeline Latency Breakdown', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(times_astar) * 1.2)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(times_astar):
        ax1.text(i, v, f'{v:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    total_astar = sum(times_astar)
    ax1.text(1, max(times_astar) * 1.1, f'Total: {total_astar:.1f}ms', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # RRT* Pipeline
    stages_rrt = ['Hazard\nDetection', 'RRT* Path\nPlanning', 'Path\nSmoothing']
    times_rrt = [inference_time, rrt_time, 15]
    colors_rrt = ['#3498db', '#e74c3c', '#f39c12']
    
    ax2.bar(stages_rrt, times_rrt, color=colors_rrt, alpha=0.8)
    ax2.set_ylabel('Time (ms)', fontweight='bold')
    ax2.set_title('RRT* Pipeline Latency Breakdown', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(times_rrt) * 1.2)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(times_rrt):
        ax2.text(i, v, f'{v:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    total_rrt = sum(times_rrt)
    ax2.text(1, max(times_rrt) * 1.1, f'Total: {total_rrt:.1f}ms', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved latency breakdown to {save_path}")


if __name__ == "__main__":
    print("📊 Generating publication-ready figures...\n")
    
    # Example metrics for algorithm comparison
    example_metrics = {
        "A*": {
            "planning_time_ms": 35.2,
            "nodes_explored": 1547,
            "path_length": 89,
            "total_cost": 127.5,
            "success": True
        },
        "RRT*": {
            "planning_time_ms": 245.8,
            "nodes_explored": 1823,
            "path_length": 94,
            "total_cost": 115.2,
            "success": True
        }
    }
    
    plot_algorithm_performance(example_metrics)
    
    # Example risk data
    example_risk = {
        "A*": {
            "safe_pixels": 81,
            "rock_pixels": 6,
            "crater_pixels": 2,
            "background_pixels": 0,
            "risk_score": 12.3,
            "safe_percentage": 91.0
        },
        "RRT*": {
            "safe_pixels": 85,
            "rock_pixels": 7,
            "crater_pixels": 2,
            "background_pixels": 0,
            "risk_score": 10.1,
            "safe_percentage": 90.4
        }
    }
    
    plot_risk_analysis(example_risk)
    
    # Generate performance table
    generate_performance_table(example_metrics)
    
    # Generate latency breakdown
    plot_latency_breakdown()
    
    print("\n✅ All figures generated successfully!")
    print("📁 Files created:")
    print("   - algorithm_performance.png")
    print("   - risk_analysis.png")
    print("   - latency_breakdown.png")
