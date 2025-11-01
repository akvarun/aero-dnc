from src.benchmark import run_benchmark
from src.visualize import plot_runtime

if __name__ == "__main__":
    csv_path = "results/runtime_data.csv"
    plot_path = "results/runtime_plot.png"

    results = run_benchmark(output_csv=csv_path)
    plot_runtime(csv_path, output_path=plot_path)

