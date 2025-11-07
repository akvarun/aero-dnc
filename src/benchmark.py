import csv
from .closest_pair import brute_force, divide_and_conquer
from .utils import generate_points, time_function

def run_benchmark(output_csv="results/runtime_data.csv"):
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    results = []

    for n in sizes:
        points = generate_points(n)

        # Brute force
        (_, _), t_brute = time_function(brute_force, points)

        # Divide and conquer
        (_, _), t_dc = time_function(divide_and_conquer, points)

        results.append((n, t_brute, t_dc))
        print(f"[n={n}] Brute: {t_brute:.4f}s | D&C: {t_dc:.4f}s")

    # Write results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "BruteForceTime", "DivideConquerTime"])
        writer.writerows(results)

    print(f"\nResults saved to {output_csv}")
    print("You can plot the results using the provided plotting script.")
    return results


