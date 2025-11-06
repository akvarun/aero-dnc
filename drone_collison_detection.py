"""
Drone Collision Detection - Divide and Conquer Project
Closest Pair of Points Problem

This script implements and compares:
1. Brute Force O(n²) algorithm
2. Divide & Conquer O(n log n) algorithm

Outputs:
- Timing comparison plots
- Visualization of closest pair
- CSV data file
"""

import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

import sys
sys.setrecursionlimit(50000)


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


def brute_force_closest(points: List[Point]) -> Tuple[float, Optional[Tuple[Point, Point]]]:
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
            dist = distance(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (points[i], points[j])
    
    return min_dist, closest_pair


def strip_closest(strip: List[Point], d: float) -> Tuple[float, Optional[Tuple[Point, Point]]]:
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
            dist = distance(strip[i], strip[j])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (strip[i], strip[j])
            j += 1
            if j - i > 7:  # Geometric optimization
                break
    
    return min_dist, closest_pair

def recursive_closest(px: List[Point], py: List[Point]) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """
    Recursive Divide and Conquer
    
    Recurrence: T(n) = 2T(n/2) + O(n)
    Solution: T(n) = O(n log n) by Master Theorem
    """
    n = len(px)
    
    if n <= 3:
        return brute_force_closest(px)
    
    mid = n // 2
    midpoint = px[mid]
    
    # Create left set for efficient lookup
    left_set = set(px[:mid])
    
    # Partition py into left and right
    pyl = [p for p in py if p in left_set]
    pyr = [p for p in py if p not in left_set]
    
    # CONQUER: Recursively find closest in each half
    dl, pair_l = recursive_closest(px[:mid], pyl)
    dr, pair_r = recursive_closest(px[mid:], pyr)
    
    # Find minimum from both halves
    if dl < dr:
        d = dl
        best_pair = pair_l
    else:
        d = dr
        best_pair = pair_r
    
    # COMBINE: Build strip and check for closer pairs
    strip = [p for p in py if abs(p.x - midpoint.x) < d]
    d_strip, pair_strip = strip_closest(strip, d)
    
    if d_strip < d:
        return d_strip, pair_strip
    else:
        return d, best_pair

def closest_pair_divide_conquer(points: List[Point]) -> Tuple[float, Optional[Tuple[Point, Point]]]:
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
    
    return recursive_closest(px, py)


def generate_random_drones(n: int, max_coord: float = 10000.0, seed: int = None) -> List[Point]:
    """Generate n random drone positions"""
    if seed is not None:
        random.seed(seed)
    return [Point(random.uniform(0, max_coord), 
                  random.uniform(0, max_coord), i) 
            for i in range(n)]


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
            print(f"✓ Passed {test + 1}/{num_tests} tests")
    
    print(f"ALL {num_tests} TESTS PASSED!\n")
    return True


def run_timing_experiments():
    """Run comprehensive timing experiments"""
    print("="*80)
    print("TIMING EXPERIMENTS")
    print("="*80)
    
    # Test sizes
    sizes_small = [50, 100, 200, 500, 1000, 2000, 3000, 5000]
    sizes_large = [7500, 10000, 15000, 20000, 30000, 50000]
    
    results = {
        'n': [],
        'brute_force_ms': [],
        'dc_ms': [],
        'speedup': []
    }
    
    # Test both algorithms on small/medium sizes
    print("\nTesting with both algorithms:")
    print("-"*80)
    for n in sizes_small:
        drones = generate_random_drones(n, seed=42)
        
        # Time Brute Force
        start = time.perf_counter()
        brute_force_closest(drones)
        bf_time = (time.perf_counter() - start) * 1000
        
        # Time Divide & Conquer
        start = time.perf_counter()
        closest_pair_divide_conquer(drones)
        dc_time = (time.perf_counter() - start) * 1000
        
        speedup = bf_time / dc_time
        
        results['n'].append(n)
        results['brute_force_ms'].append(bf_time)
        results['dc_ms'].append(dc_time)
        results['speedup'].append(speedup)
        
        print(f"n={n:6d} | BF: {bf_time:8.2f}ms | DC: {dc_time:8.2f}ms | Speedup: {speedup:6.1f}x")
    
    # Test only D&C on large sizes
    print("\nTesting D&C only (large sizes):")
    print("-"*80)
    for n in sizes_large:
        drones = generate_random_drones(n, seed=42)
        
        start = time.perf_counter()
        closest_pair_divide_conquer(drones)
        dc_time = (time.perf_counter() - start) * 1000
        
        results['n'].append(n)
        results['brute_force_ms'].append(None)
        results['dc_ms'].append(dc_time)
        results['speedup'].append(None)
        
        print(f"n={n:6d} | DC: {dc_time:8.2f}ms")
    
    print(f"\n{'='*80}")
    print("TIMING COMPLETE\n")
    return results


def plot_timing_comparison(results: dict):
    """Create timing comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    n = np.array(results['n'])
    bf = np.array([x if x is not None else np.nan for x in results['brute_force_ms']])
    dc = np.array(results['dc_ms'])
    
    # Plot 1: Both algorithms (where BF data exists)
    mask = ~np.isnan(bf)
    n_both = n[mask]
    bf_both = bf[mask]
    dc_both = dc[mask]
    
    ax1.plot(n_both, bf_both, 'o-', label='Brute Force O(n²)', 
             color='red', linewidth=2.5, markersize=8)
    ax1.plot(n_both, dc_both, 's-', label='Divide & Conquer O(n log n)', 
             color='green', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Number of Drones (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (milliseconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: D&C Scalability
    ax2.plot(n, dc, 'o-', label='Divide & Conquer', 
             color='green', linewidth=2.5, markersize=7)
    
    # Theoretical O(n log n) line
    theoretical = n * np.log2(n) / 1000
    ax2.plot(n, theoretical, '--', label='Theoretical O(n log n)', 
             color='blue', linewidth=2, alpha=0.6)
    
    ax2.set_xlabel('Number of Drones (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Runtime (milliseconds)', fontsize=12, fontweight='bold')
    ax2.set_title('D&C Scalability (Large n)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timing_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: timing_comparison.png")
    plt.show()

def plot_speedup_chart(results: dict):
    """Create speedup comparison chart"""
    n = []
    speedup = []
    
    for i, s in enumerate(results['speedup']):
        if s is not None:
            n.append(results['n'][i])
            speedup.append(s)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n, speedup, 'o-', color='purple', linewidth=2.5, markersize=8)
    plt.xlabel('Number of Drones (n)', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup (BF time / DC time)', fontsize=12, fontweight='bold')
    plt.title('Divide & Conquer Speedup vs Brute Force', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('speedup_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: speedup_chart.png")
    plt.show()

def visualize_closest_pair(n: int = 150):
    """Visualize drone positions and closest pair"""
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
    plt.title(f'Drone Collision Detection (n={n})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('drone_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: drone_visualization.png")
    plt.show()


def export_csv(results: dict):
    """Export timing results to CSV"""
    with open('timing_results.csv', 'w') as f:
        f.write('n,brute_force_ms,dc_ms,speedup\n')
        for i in range(len(results['n'])):
            n = results['n'][i]
            bf = results['brute_force_ms'][i]
            dc = results['dc_ms'][i]
            speedup = results['speedup'][i]
            
            bf_str = f"{bf:.4f}" if bf is not None else ""
            speedup_str = f"{speedup:.2f}" if speedup is not None else ""
            
            f.write(f"{n},{bf_str},{dc:.4f},{speedup_str}\n")
    
    print("Saved: timing_results.csv")


def main():
    print("\n" + "="*80)
    print(" "*20 + "DRONE COLLISION DETECTION PROJECT")
    print(" "*15 + "Divide and Conquer: Closest Pair Problem")
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
    print(f"\n⚠️  COLLISION WARNING!")
    print(f"Closest pair: {pair[0]} and {pair[1]}")
    print(f"Distance: {dist:.4f} meters\n")
    
    # Step 3: Timing experiments
    results = run_timing_experiments()
    
    # Step 4: Generate visualizations
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    visualize_closest_pair(150)
    plot_timing_comparison(results)
    plot_speedup_chart(results)
    
    # Step 5: Export data
    print("\n" + "="*80)
    print("EXPORTING DATA")
    print("="*80)
    export_csv(results)
    
    # Summary
    print("\n" + "="*80)
    print(" "*25 + "ALL COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("   timing_comparison.png  - Algorithm comparison")
    print("   speedup_chart.png      - Speedup analysis")
    print("   drone_visualization.png - Visual example")
    print("   timing_results.csv     - Raw data")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
