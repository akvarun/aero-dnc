import matplotlib.pyplot as plt
import csv

def plot_runtime(csv_path, output_path="results/runtime_plot.png"):
    n_vals, brute_times, dc_times = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_vals.append(int(row["n"]))
            brute_times.append(float(row["BruteForceTime"]))
            dc_times.append(float(row["DivideConquerTime"]))

    plt.figure(figsize=(8, 5))
    plt.plot(n_vals, brute_times, "r-o", label="Brute Force (O(n²))")
    plt.plot(n_vals, dc_times, "b-o", label="Divide & Conquer (O(n log n))")
    plt.title("Runtime Comparison: Drone Collision Detection Algorithms")
    plt.xlabel("Number of Drones (n)")
    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Plot saved to {output_path}")
