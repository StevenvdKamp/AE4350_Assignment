import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering without a display
import matplotlib.pyplot as plt
import os

# Load data directly from the provided CSV (as a string or a file)
data = """
param_type,param_value,run_name,max_ep_rew_mean,max_explained_variance
baseline,default,RPPO_baseline_1,15.93019962310791,0.979974091053009
batch_size,1024,RPPO_batch_size1024_1,18.3651065826416,0.9964086413383484
batch_size,2048,RPPO_batch_size2048_1,10.136608123779297,0.891704261302948
batch_size,512,RPPO_batch_size512_1,12.460915565490723,0.8950569033622742
batch_size,64,RPPO_batch_size64_1,18.006696701049805,0.9933639764785767
learning_rate,1e-3,RPPO_learning_rate1e-3_1,16.684526443481445,0.9814180135726929
learning_rate,1e-4,RPPO_learning_rate1e-4_1,17.740467071533203,0.9729681611061096
learning_rate,1e-5,RPPO_learning_rate1e-5_1,3.9458744525909424,0.8319805264472961
n_epochs,20,RPPO_n_epochs20_1,18.010990142822266,0.999168336391449
n_epochs,40,RPPO_n_epochs40_1,6.387020587921143,0.9053543210029602
n_epochs,5,RPPO_n_epochs5_1,17.567150115966797,0.8934041261672974
n_steps,1024,RPPO_n_steps1024_1,17.186376571655273,0.7545499801635742
n_steps,2048,RPPO_n_steps2048_1,11.039470672607422,0.3067514896392822
n_steps,512,RPPO_n_steps512_1,11.949692726135254,0.5608861446380615
n_steps,64,RPPO_n_steps64_1,17.73711585998535,0.9980178475379944
"""

# Convert the string data into a DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# --- Plotting ---
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

print("Starting plotting")
for metric in ["max_ep_rew_mean", "max_explained_variance"]:
    print(f"Plotting {metric}")
    
    # Process baseline separately, always making sure it comes first in the plot
    baseline = df[df['param_type'] == 'baseline']
    if not baseline.empty:
        print(f"Processing baseline: {baseline}")
    
    # Process non-baseline parameters
    non_baseline_df = df[df['param_type'] != 'baseline']
    for param in sorted(non_baseline_df['param_type'].unique()):
        # Update parameter names to more descriptive names
        param_name_map = {
            'batch_size': 'Batch Size',
            'learning_rate': 'Learning Rate',
            'n_epochs': 'Number of Epochs',
            'n_steps': 'Number of Steps'
        }
        
        param_desc = param_name_map.get(param, param)  # Default to the original name if not found
        
        print(f"Processing parameter: {param_desc}")
        subset = non_baseline_df[non_baseline_df['param_type'] == param].copy()
        if subset.empty:
            print(f"No data for parameter: {param_desc}")
            continue

        subset['param_value'] = subset['param_value'].astype(str)
        print(f"Plotting {metric} for parameter {param_desc} with values: {subset['param_value'].tolist()}")

        # Sort the non-baseline values numerically
        def safe_float(val):
            try:
                return float(val)
            except ValueError:
                return None

        subset['sort_val'] = subset['param_value'].apply(lambda x: safe_float(x) if x != "default" else None)
        subset = subset.sort_values('sort_val')

        # Combine baseline with the sorted non-baseline entries (ensuring baseline is first)
        sorted_subset = pd.concat([baseline, subset])

        # Drop rows with NaN values in 'param_value' or the metric of interest
        sorted_subset = sorted_subset.dropna(subset=['param_value', metric])
        if sorted_subset.empty:
            print(f"No valid data to plot for {param_desc}")
            continue

        print(f"param_value: {sorted_subset['param_value'].values}")
        print(f"{metric}: {sorted_subset[metric].values}")

        # Update the plot metric name based on the selected metric
        if metric == "max_ep_rew_mean":
            metric_label = "Episode Mean Reward"
        elif metric == "max_explained_variance":
            metric_label = "Max Explained Variance"

        default_dict = {'learning_rate': 3e-4, 'batch_size': 128, 'n_epochs': 10, 'n_steps': 128}
        bar_name = sorted_subset['param_value'].replace("default", f"{default_dict[f"{param}"]} (default)")

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.bar(bar_name, sorted_subset[metric], color='skyblue')
        plt.title(f"{metric_label} vs {param_desc}")
        plt.xlabel(param_desc)
        plt.ylabel(metric_label)
        plt.grid(True)
        plt.tight_layout()

        # Save as PDF
        plot_path = os.path.join(output_dir, f"{metric_label.replace(' ', '_')}_vs_{param_desc.replace(' ', '_')}.pdf")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")

# Optional: Export data summary to CSV
df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
print("Saved: summary_metrics.csv")
