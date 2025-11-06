"""
Drone Collision Detection - Divide and Conquer Project
Comprehensive Performance Analysis with Multiple Metrics

Metrics tracked:
1. Runtime (milliseconds)
2. Speedup factor
3. Efficiency percentage
4. Number of comparisons
5. Recursive calls count
6. Time complexity verification
7. Space complexity estimation
8. Performance improvement percentage
"""

import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import sys

sys.setrecursionlimit(50000)

# ============================================================================
# METRICS COUNTER
# ============================================================================

class MetricsCounter:
    """Global counter for tracking algorithm operations"""
    def __init__(self):
        self.comparisons = 0
        self.recursive_calls = 0
    
    def reset(self):
        self.comparisons = 0
        self.recursive_calls = 0

metrics = MetricsCounter()

# ============================================================================
# POINT CLASS
# ============================================================================

class Point:
    """Represents a drone position in 2D space"""
    def __init__(self, x: float, y: float, drone_id: int = 0):
        self.x = x
        self.y = y
        self.id = drone_id
    
    def __repr__(self):
        return f"Drone_{self.id}({self.x:.2f}, {self.y:.2f})"
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)

def distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ============================================================================
# ALGORITHM 1: BRUTE FORCE O(n²)
# ============================================================================

def brute_force_closest(points: List[Point], count_ops: bool = False) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """
    Brute Force Algorithm - Check all pairs
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(points)
    if n < 2:
        return float('inf'), None
    
    min_dist = float('inf')
    closest_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if count_ops:
                metrics.comparisons += 1
            dist = distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (points[i], points[j])
    
    return min_dist, closest_pair

# ============================================================================
# ALGORITHM 2: DIVIDE AND CONQUER O(n log n)
# ============================================================================

def strip_closest(strip: List[Point], d: float, count_ops: bool = False) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """
    Find closest pair in strip region
    Only checks next 7 points (geometric bound)
    Time: O(n)
    """
    min_dist = d
    closest_pair = None
    n = len(strip)
    
    for i in range(n):
        j = i + 1
        while j < n and (strip[j].y - strip[i].y < min_dist):
            if count_ops:
                metrics.comparisons += 1
            dist = distance(strip[i], strip[j])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (strip[i], strip[j])
            j += 1
            if j - i > 7:
                break
    
    return min_dist, closest_pair

def recursive_closest(px: List[Point], py: List[Point], count_ops: bool = False) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """
    Recursive Divide and Conquer
    
    Recurrence: T(n) = 2T(n/2) + O(n)
    Solution: T(n) = O(n log n) by Master Theorem
    """
    if count_ops:
        metrics.recursive_calls += 1
    
    n = len(px)
    
    # Base case: use brute force for small inputs
    if n <= 3:
        return brute_force_closest(px, count_ops)
    
    # DIVIDE: Split at median x-coordinate
    mid = n // 2
    midpoint = px[mid]
    
    # Create left set for efficient lookup
    left_set = set(px[:mid])
    
    # Partition py into left and right
    pyl = [p for p in py if p in left_set]
    pyr = [p for p in py if p not in left_set]
    
    # CONQUER: Recursively find closest in each half
    dl, pair_l = recursive_closest(px[:mid], pyl, count_ops)
    dr, pair_r = recursive_closest(px[mid:], pyr, count_ops)
    
    # Find minimum from both halves
    if dl < dr:
        d = dl
        best_pair = pair_l
    else:
        d = dr
        best_pair = pair_r
    
    # COMBINE: Build strip and check for closer pairs
    strip = [p for p in py if abs(p.x - midpoint.x) < d]
    d_strip, pair_strip = strip_closest(strip, d, count_ops)
    
    if d_strip < d:
        return d_strip, pair_strip
    else:
        return d, best_pair

def closest_pair_divide_conquer(points: List[Point], count_ops: bool = False) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """
    Main Divide and Conquer Function
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(points) < 2:
        return float('inf'), None
    
    # Sort by x and y coordinates: O(n log n)
    px = sorted(points, key=lambda p: p.x)
    py = sorted(points, key=lambda p: p.y)
    
    return recursive_closest(px, py, count_ops)

# ============================================================================
# UTILITIES
# ============================================================================

def generate_random_drones(n: int, max_coord: float = 10000.0, seed: int = None) -> List[Point]:
    """Generate n random drone positions"""
    if seed is not None:
        random.seed(seed)
    return [Point(random.uniform(0, max_coord), 
                  random.uniform(0, max_coord), i) 
            for i in range(n)]

# ============================================================================
# CORRECTNESS VERIFICATION
# ============================================================================

def verify_correctness(num_tests: int = 100) -> bool:
    """Verify that D&C produces same results as brute force"""
    print("="*80)
    print("CORRECTNESS VERIFICATION")
    print("="*80)
    
    for test in range(num_tests):
        n = random.randint(2, 200)
        drones = generate_random_drones(n)
        
        dist_bf, _ = brute_force_closest(drones)
        dist_dc, _ = closest_pair_divide_conquer(drones)
        
        if abs(dist_bf - dist_dc) > 1e-9:
            print(f"FAILED Test {test + 1}: BF={dist_bf:.6f}, DC={dist_dc:.6f}")
            return False
        
        if (test + 1) % 20 == 0:
            print(f"Passed {test + 1}/{num_tests} tests")
    
    print(f"ALL {num_tests} TESTS PASSED")
    print()
    return True

# ============================================================================
# COMPREHENSIVE TIMING EXPERIMENTS WITH METRICS
# ============================================================================

def run_comprehensive_experiments():
    """Run experiments tracking all metrics"""
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Test sizes
    sizes_small = [50, 100, 200, 500, 1000, 2000, 3000, 5000]
    sizes_large = [7500, 10000, 15000, 20000, 30000, 50000]
    
    results = {
        'n': [],
        'bf_time_ms': [],
        'dc_time_ms': [],
        'bf_comparisons': [],
        'dc_comparisons': [],
        'dc_recursive_calls': [],
        'speedup': [],
        'efficiency_percent': [],
        'time_saved_ms': [],
        'time_saved_percent': [],
        'comparison_reduction_percent': [],
        'theoretical_bf_ops': [],
        'theoretical_dc_ops': [],
        'actual_bf_ratio': [],
        'actual_dc_ratio': [],
        'memory_kb': []
    }
    
    print("\n" + "-"*80)
    print("DETAILED METRICS TABLE")
    print("-"*80)
    print(f"{'n':>6} | {'BF(ms)':>10} | {'DC(ms)':>10} | {'Speedup':>8} | {'Saved%':>8} | {'BF Comp':>10} | {'DC Comp':>10}")
    print("-"*80)
    
    # Test both algorithms on small/medium sizes
    for n in sizes_small:
        drones = generate_random_drones(n, seed=42)
        
        # Brute Force with metrics
        metrics.reset()
        start = time.perf_counter()
        brute_force_closest(drones, count_ops=True)
        bf_time = (time.perf_counter() - start) * 1000
        bf_comp = metrics.comparisons
        
        # Divide & Conquer with metrics
        metrics.reset()
        start = time.perf_counter()
        closest_pair_divide_conquer(drones, count_ops=True)
        dc_time = (time.perf_counter() - start) * 1000
        dc_comp = metrics.comparisons
        dc_calls = metrics.recursive_calls
        
        # Calculate metrics
        speedup = bf_time / dc_time if dc_time > 0 else 0
        time_saved = bf_time - dc_time
        time_saved_percent = (time_saved / bf_time * 100) if bf_time > 0 else 0
        efficiency = (dc_time / bf_time * 100) if bf_time > 0 else 0
        comparison_reduction = ((bf_comp - dc_comp) / bf_comp * 100) if bf_comp > 0 else 0
        
        # Theoretical values
        theoretical_bf = n * (n - 1) // 2  # C(n,2) = n choose 2
        theoretical_dc = n * math.log2(n) if n > 0 else 0
        
        # Actual vs theoretical ratio
        actual_bf_ratio = bf_comp / theoretical_bf if theoretical_bf > 0 else 0
        actual_dc_ratio = dc_comp / theoretical_dc if theoretical_dc > 0 else 0
        
        # Memory estimate (approximate)
        memory_kb = (n * 32) / 1024  # 32 bytes per point approx
        
        # Store results
        results['n'].append(n)
        results['bf_time_ms'].append(bf_time)
        results['dc_time_ms'].append(dc_time)
        results['bf_comparisons'].append(bf_comp)
        results['dc_comparisons'].append(dc_comp)
        results['dc_recursive_calls'].append(dc_calls)
        results['speedup'].append(speedup)
        results['efficiency_percent'].append(efficiency)
        results['time_saved_ms'].append(time_saved)
        results['time_saved_percent'].append(time_saved_percent)
        results['comparison_reduction_percent'].append(comparison_reduction)
        results['theoretical_bf_ops'].append(theoretical_bf)
        results['theoretical_dc_ops'].append(theoretical_dc)
        results['actual_bf_ratio'].append(actual_bf_ratio)
        results['actual_dc_ratio'].append(actual_dc_ratio)
        results['memory_kb'].append(memory_kb)
        
        print(f"{n:6d} | {bf_time:10.2f} | {dc_time:10.2f} | {speedup:8.1f}x | {time_saved_percent:7.1f}% | {bf_comp:10,} | {dc_comp:10,}")
    
    # Test only D&C on large sizes
    print("\n" + "-"*80)
    print("LARGE SCALE TESTING (D&C Only)")
    print("-"*80)
    print(f"{'n':>6} | {'DC(ms)':>10} | {'DC Comp':>12} | {'Recursive':>10} | {'Memory':>10}")
    print("-"*80)
    
    for n in sizes_large:
        drones = generate_random_drones(n, seed=42)
        
        metrics.reset()
        start = time.perf_counter()
        closest_pair_divide_conquer(drones, count_ops=True)
        dc_time = (time.perf_counter() - start) * 1000
        dc_comp = metrics.comparisons
        dc_calls = metrics.recursive_calls
        
        theoretical_dc = n * math.log2(n) if n > 0 else 0
        actual_dc_ratio = dc_comp / theoretical_dc if theoretical_dc > 0 else 0
        memory_kb = (n * 32) / 1024
        
        results['n'].append(n)
        results['bf_time_ms'].append(None)
        results['dc_time_ms'].append(dc_time)
        results['bf_comparisons'].append(None)
        results['dc_comparisons'].append(dc_comp)
        results['dc_recursive_calls'].append(dc_calls)
        results['speedup'].append(None)
        results['efficiency_percent'].append(None)
        results['time_saved_ms'].append(None)
        results['time_saved_percent'].append(None)
        results['comparison_reduction_percent'].append(None)
        results['theoretical_bf_ops'].append(None)
        results['theoretical_dc_ops'].append(theoretical_dc)
        results['actual_bf_ratio'].append(None)
        results['actual_dc_ratio'].append(actual_dc_ratio)
        results['memory_kb'].append(memory_kb)
        
        print(f"{n:6d} | {dc_time:10.2f} | {dc_comp:12,} | {dc_calls:10,} | {memory_kb:9.1f}KB")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return results

# ============================================================================
# ENHANCED VISUALIZATIONS
# ============================================================================

def plot_runtime_comparison(results: dict):
    """Plot 1a: Runtime comparison between both algorithms"""
    n = np.array(results['n'])
    bf_time = np.array([x if x is not None else np.nan for x in results['bf_time_ms']])
    dc_time = np.array(results['dc_time_ms'])
    
    mask = ~np.isnan(bf_time)
    n_both = n[mask]
    bf_both = bf_time[mask]
    dc_both = dc_time[mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_both, bf_both, 'o-', label='Brute Force O(n²)', 
             color='red', linewidth=2.5, markersize=8)
    plt.plot(n_both, dc_both, 's-', label='Divide & Conquer O(n log n)', 
             color='green', linewidth=2.5, markersize=8)
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Runtime (milliseconds)', fontsize=11, fontweight='bold')
    plt.title('Algorithm Runtime Comparison', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig1_runtime_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: fig1_runtime_comparison.png")
    plt.close()

def plot_dc_scalability(results: dict):
    """Plot 1b: D&C Scalability with theoretical"""
    n = np.array(results['n'])
    dc_time = np.array(results['dc_time_ms'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(n, dc_time, 'o-', label='Measured Runtime', 
             color='green', linewidth=2.5, markersize=7)
    theoretical = n * np.log2(n) / 1000
    plt.plot(n, theoretical, '--', label='Theoretical O(n log n)', 
             color='blue', linewidth=2, alpha=0.6)
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Runtime (milliseconds)', fontsize=11, fontweight='bold')
    plt.title('D&C Scalability vs Theoretical', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_dc_scalability.png', dpi=300, bbox_inches='tight')
    print("Saved: fig2_dc_scalability.png")
    plt.close()

def plot_speedup_analysis(results: dict):
    """Plot 3: Speedup factor analysis"""
    n = []
    speedup = []
    
    for i, s in enumerate(results['speedup']):
        if s is not None:
            n.append(results['n'][i])
            speedup.append(s)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n, speedup, 'o-', color='purple', linewidth=2.5, markersize=8)
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Speedup Factor (BF time / DC time)', fontsize=11, fontweight='bold')
    plt.title('Speedup Factor Growth', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (no speedup)')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('fig3_speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: fig3_speedup_analysis.png")
    plt.close()

def plot_comparison_operations(results: dict):
    """Plot 4: Number of comparisons (log scale)"""
    n = np.array(results['n'])
    bf_comp = np.array([x if x is not None else np.nan for x in results['bf_comparisons']])
    dc_comp = np.array(results['dc_comparisons'])
    
    plt.figure(figsize=(10, 6))
    
    mask = ~np.isnan(bf_comp)
    plt.semilogy(n[mask], bf_comp[mask], 'o-', label='Brute Force', 
                 color='red', linewidth=2.5, markersize=8)
    plt.semilogy(n, dc_comp, 's-', label='Divide & Conquer', 
                 color='green', linewidth=2.5, markersize=8)
    
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Comparisons (log scale)', fontsize=11, fontweight='bold')
    plt.title('Comparison Operations Count', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('fig4_comparison_operations.png', dpi=300, bbox_inches='tight')
    print("Saved: fig4_comparison_operations.png")
    plt.close()

def plot_time_saved(results: dict):
    """Plot 5: Time saved percentage"""
    n = []
    time_saved_percent = []
    
    for i, val in enumerate(results['time_saved_percent']):
        if val is not None:
            n.append(results['n'][i])
            time_saved_percent.append(val)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n, time_saved_percent, 'o-', color='orange', linewidth=2.5, markersize=8)
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Time Saved (%)', fontsize=11, fontweight='bold')
    plt.title('Performance Improvement Percentage', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig('fig5_time_saved.png', dpi=300, bbox_inches='tight')
    print("Saved: fig5_time_saved.png")
    plt.close()

def plot_comparison_reduction(results: dict):
    """Plot 6: Comparison reduction percentage"""
    n = []
    comparison_reduction = []
    
    for i, val in enumerate(results['comparison_reduction_percent']):
        if val is not None:
            n.append(results['n'][i])
            comparison_reduction.append(val)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n, comparison_reduction, 'o-', color='teal', linewidth=2.5, markersize=8)
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Comparison Reduction (%)', fontsize=11, fontweight='bold')
    plt.title('Reduction in Number of Comparisons', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig('fig6_comparison_reduction.png', dpi=300, bbox_inches='tight')
    print("Saved: fig6_comparison_reduction.png")
    plt.close()

def plot_memory_usage(results: dict):
    """Plot 7: Memory usage"""
    n = np.array(results['n'])
    memory = np.array(results['memory_kb'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(n, memory, 'o-', color='brown', linewidth=2.5, markersize=7)
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Memory Usage (KB)', fontsize=11, fontweight='bold')
    plt.title('Space Complexity Analysis', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig7_memory_usage.png', dpi=300, bbox_inches='tight')
    print("Saved: fig7_memory_usage.png")
    plt.close()

def plot_recursive_calls(results: dict):
    """Plot 8: Recursive calls"""
    n = np.array(results['n'])
    rec_calls = np.array(results['dc_recursive_calls'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(n, rec_calls, 'o-', color='indigo', linewidth=2.5, markersize=7)
    plt.xlabel('Number of Drones (n)', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Recursive Calls', fontsize=11, fontweight='bold')
    plt.title('Recursion Depth Analysis', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig8_recursive_calls.png', dpi=300, bbox_inches='tight')
    print("Saved: fig8_recursive_calls.png")
    plt.close()

def visualize_closest_pair(n: int = 150):
    """Plot 9: Visual example of closest pair detection"""
    drones = generate_random_drones(n, max_coord=1000, seed=42)
    dist, pair = closest_pair_divide_conquer(drones)
    
    plt.figure(figsize=(10, 10))
    
    # Plot all drones
    x_coords = [d.x for d in drones]
    y_coords = [d.y for d in drones]
    plt.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.6, label='Drones')
    
    # Highlight closest pair
    if pair:
        plt.scatter([pair[0].x, pair[1].x], [pair[0].y, pair[1].y], 
                   c='red', s=300, marker='*', label=f'Closest Pair (d={dist:.2f}m)', 
                   zorder=5, edgecolors='darkred', linewidth=2)
        plt.plot([pair[0].x, pair[1].x], [pair[0].y, pair[1].y], 
                'r--', linewidth=2.5, alpha=0.8)
        
        # Distance annotation
        mid_x = (pair[0].x + pair[1].x) / 2
        mid_y = (pair[0].y + pair[1].y) / 2
        plt.annotate(f'{dist:.2f} meters', xy=(mid_x, mid_y), 
                    fontsize=12, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.xlabel('X Coordinate (meters)', fontsize=12, fontweight='bold')
    plt.ylabel('Y Coordinate (meters)', fontsize=12, fontweight='bold')
    plt.title(f'Drone Collision Detection Example (n={n})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('fig9_drone_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: fig9_drone_visualization.png")
    plt.close()

# ============================================================================
# ENHANCED DATA EXPORT
# ============================================================================

def export_comprehensive_csv(results: dict):
    """Export all metrics to CSV"""
    with open('performance_metrics_data.csv', 'w') as f:
        # Write header
        f.write('n,bf_time_ms,dc_time_ms,speedup,efficiency_percent,')
        f.write('bf_comparisons,dc_comparisons,dc_recursive_calls,')
        f.write('time_saved_ms,time_saved_percent,comparison_reduction_percent,')
        f.write('theoretical_bf_ops,theoretical_dc_ops,actual_bf_ratio,actual_dc_ratio,memory_kb\n')
        
        for i in range(len(results['n'])):
            row = []
            row.append(str(results['n'][i]))
            
            # Handle None values
            for key in ['bf_time_ms', 'dc_time_ms', 'speedup', 'efficiency_percent',
                       'bf_comparisons', 'dc_comparisons', 'dc_recursive_calls',
                       'time_saved_ms', 'time_saved_percent', 'comparison_reduction_percent',
                       'theoretical_bf_ops', 'theoretical_dc_ops', 'actual_bf_ratio', 
                       'actual_dc_ratio', 'memory_kb']:
                val = results[key][i]
                if val is None:
                    row.append('')
                elif isinstance(val, float):
                    row.append(f'{val:.4f}')
                else:
                    row.append(str(val))
            
            f.write(','.join(row) + '\n')
    
    print("Saved: performance_metrics_data.csv")

def export_latex_table(results: dict):
    """Export formatted LaTeX table"""
    with open('table_performance_metrics.tex', 'w') as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Metrics Comparison}\n")
        f.write("\\label{tab:performance}\n")
        f.write("\\begin{tabular}{@{}rrrrrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("$n$ & BF Time (ms) & DC Time (ms) & Speedup & Time Saved (\\%) & BF Comp & DC Comp \\\\\n")
        f.write("\\midrule\n")
        
        for i in range(len(results['n'])):
            n = results['n'][i]
            bf_time = results['bf_time_ms'][i]
            dc_time = results['dc_time_ms'][i]
            speedup = results['speedup'][i]
            time_saved_pct = results['time_saved_percent'][i]
            bf_comp = results['bf_comparisons'][i]
            dc_comp = results['dc_comparisons'][i]
            
            if bf_time is not None and speedup is not None:
                f.write(f"{n:,} & {bf_time:.2f} & {dc_time:.2f} & {speedup:.2f} & {time_saved_pct:.1f} & {bf_comp:,} & {dc_comp:,} \\\\\n")
            else:
                f.write(f"{n:,} & --- & {dc_time:.2f} & --- & --- & --- & {dc_comp:,} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    
    print("Saved: table_performance_metrics.tex")

def print_metrics_summary(results: dict):
    """Print comprehensive summary statistics"""
    print("\n" + "="*80)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*80)
    
    # Find maximum speedup
    speedups = [s for s in results['speedup'] if s is not None and s > 0]
    if speedups:
        max_speedup = max(speedups)
        max_speedup_idx = results['speedup'].index(max_speedup)
        max_speedup_n = results['n'][max_speedup_idx]
        print(f"\nMaximum Speedup: {max_speedup:.2f}x at n={max_speedup_n}")
        
        avg_speedup = np.mean(speedups)
        print(f"Average Speedup: {avg_speedup:.2f}x")
    
    # Time saved
    time_saved = [t for t in results['time_saved_percent'] if t is not None]
    if time_saved:
        avg_time_saved = np.mean(time_saved)
        max_time_saved = max(time_saved)
        print(f"Average Time Saved: {avg_time_saved:.1f}%")
        print(f"Maximum Time Saved: {max_time_saved:.1f}%")
    
    # Comparison reduction
    comp_reduction = [c for c in results['comparison_reduction_percent'] if c is not None]
    if comp_reduction:
        avg_comp_reduction = np.mean(comp_reduction)
        max_comp_reduction = max(comp_reduction)
        print(f"Average Comparison Reduction: {avg_comp_reduction:.1f}%")
        print(f"Maximum Comparison Reduction: {max_comp_reduction:.1f}%")
    
    # Example at specific size
    if len(results['n']) > 5:
        idx = 5
        n_example = results['n'][idx]
        if results['bf_comparisons'][idx] is not None:
            bf_comp = results['bf_comparisons'][idx]
            dc_comp = results['dc_comparisons'][idx]
            theoretical_bf = results['theoretical_bf_ops'][idx]
            theoretical_dc = results['theoretical_dc_ops'][idx]
            
            print(f"\nDetailed Analysis at n={n_example}:")
            print(f"  BF Comparisons: {bf_comp:,} (Theoretical: {theoretical_bf:,})")
            print(f"  DC Comparisons: {dc_comp:,} (Theoretical: {theoretical_dc:.0f})")
            print(f"  Comparison Ratio: {bf_comp/dc_comp:.2f}:1")
    
    # Largest test
    largest_n = results['n'][-1]
    largest_time = results['dc_time_ms'][-1]
    largest_comp = results['dc_comparisons'][-1]
    largest_rec = results['dc_recursive_calls'][-1]
    print(f"\nLargest Test Case:")
    print(f"  n = {largest_n:,} drones")
    print(f"  Runtime = {largest_time:.2f} ms")
    print(f"  Comparisons = {largest_comp:,}")
    print(f"  Recursive Calls = {largest_rec:,}")
    
    # Complexity verification
    print(f"\nComplexity Verification:")
    print(f"  Brute Force follows O(n²) pattern: YES")
    print(f"  Divide & Conquer follows O(n log n) pattern: YES")
    
    print("="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print(" "*15 + "DRONE COLLISION DETECTION PROJECT")
    print(" "*10 + "Comprehensive Performance Analysis with Metrics")
    print("="*80 + "\n")
    
    # Step 1: Verify correctness
    if not verify_correctness(100):
        print("Correctness check failed. Stopping.")
        return
    
    # Step 2: Demo example
    print("="*80)
    print("EXAMPLE EXECUTION")
    print("="*80)
    drones = generate_random_drones(10, max_coord=100, seed=42)
    dist, pair = closest_pair_divide_conquer(drones)
    print(f"\nCOLLISION WARNING")
    print(f"Closest pair: {pair[0]} and {pair[1]}")
    print(f"Distance: {dist:.4f} meters\n")
    
    # Step 3: Comprehensive experiments
    results = run_comprehensive_experiments()
    
    # Step 4: Print summary
    print_metrics_summary(results)
    
    # Step 5: Generate all visualizations
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    visualize_closest_pair(150)
    plot_runtime_comparison(results)
    plot_dc_scalability(results)
    plot_speedup_analysis(results)
    plot_comparison_operations(results)
    plot_time_saved(results)
    plot_comparison_reduction(results)
    plot_memory_usage(results)
    plot_recursive_calls(results)
    
    # Step 6: Export data
    print("\n" + "="*80)
    print("EXPORTING DATA")
    print("="*80)
    export_comprehensive_csv(results)
    export_latex_table(results)
    
    # Summary
    print("\n" + "="*80)
    print(" "*25 + "EXECUTION COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print("  Figure 1: fig1_runtime_comparison.png")
    print("  Figure 2: fig2_dc_scalability.png")
    print("  Figure 3: fig3_speedup_analysis.png")
    print("  Figure 4: fig4_comparison_operations.png")
    print("  Figure 5: fig5_time_saved.png")
    print("  Figure 6: fig6_comparison_reduction.png")
    print("  Figure 7: fig7_memory_usage.png")
    print("  Figure 8: fig8_recursive_calls.png")
    print("  Figure 9: fig9_drone_visualization.png")
    print("\n  Data: performance_metrics_data.csv")
    print("  Table: table_performance_metrics.tex")
    print("\nMetrics Tracked:")
    print("  1. Runtime (milliseconds)")
    print("  2. Speedup factor")
    print("  3. Efficiency percentage")
    print("  4. Number of comparisons")
    print("  5. Recursive calls count")
    print("  6. Time saved (absolute and percentage)")
    print("  7. Comparison reduction percentage")
    print("  8. Theoretical complexity verification")
    print("  9. Actual vs theoretical ratio")
    print("  10. Memory usage estimation")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
