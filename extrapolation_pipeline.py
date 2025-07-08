import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.isotonic import IsotonicRegression
from ipywidgets import interact, FloatSlider
import warnings

# --- HELPER FUNCTIONS (The "Engine") ---

def enforce_monotonicity(x, y, increasing=True):
    """Enforces monotonicity on a data series using Isotonic Regression (PAVA)."""
    iso_reg = IsotonicRegression(out_of_bounds="clip", increasing=increasing)
    return iso_reg.fit_transform(x, y)

def whittaker_smoother(y, lambd, d=2):
    """A robust smoother based on penalizing the d-th differences of the signal."""
    m = len(y)
    E = sparse.eye(m, format='csc')
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - d, m), format='csc')
    A = E + lambd * D.T @ D
    z = spsolve(A, y)
    return z

# --- PRIMARY ANALYSIS CLASS (The "Engine") ---

class MonotonicPipeline:
    """
    A class that encapsulates a full analysis pipeline:
    1. Enforce monotonicity with Isotonic Regression.
    2. Smooth the monotonic data with a Whittaker Smoother.
    3. Extrapolate the final smoothed data.
    4. Plot all stages of the analysis.
    """

    def __init__(self, x_data, y_data, increasing, lambda_smooth, max_x, 
                 x_label="X-Axis", y_label="Y-Axis", known_convergent_value=None):
        """Initializes and runs the entire analysis pipeline."""
        self.x_data = np.array(x_data)
        self.original_y_data = np.array(y_data)
        self.max_x = max_x
        self.x_label = x_label
        self.y_label = y_label
        self.is_increasing = increasing
        self.known_convergent_value = known_convergent_value

        self.y_monotonic = enforce_monotonicity(self.x_data, self.original_y_data, increasing=self.is_increasing)
        self.y_final_smooth = whittaker_smoother(self.y_monotonic, lambd=lambda_smooth)
        self.extrapolator = self._GeometricExtrapolator(self.x_data, self.y_final_smooth, self.max_x)
        
        self.convergent_value = self.extrapolator.convergent_value
        self.error = self.extrapolator.error
        self._calculate_plot_bounds()

    def _calculate_plot_bounds(self):
        """Calculate the overall data range for interactive sliders."""
        all_y = np.concatenate([self.original_y_data, self.y_monotonic, self.y_final_smooth, self.extrapolator.extrapolated_values])
        if np.isfinite(self.extrapolator.error):
            all_y = np.concatenate([all_y, self.extrapolator.extrapolated_values - self.extrapolator.extrapolated_errors, self.extrapolator.extrapolated_values + self.extrapolator.extrapolated_errors])
        if self.known_convergent_value is not None:
             all_y = np.append(all_y, self.known_convergent_value)

        self.y_min_bound, self.y_max_bound = np.nanmin(all_y), np.nanmax(all_y)
        y_range = self.y_max_bound - self.y_min_bound
        self.y_min_bound -= y_range * 0.1
        self.y_max_bound += y_range * 0.1

    def plot_analysis(self, x_min, x_max, y_min, y_max):
        """Plots all stages of the analysis with interactive zoom and pan."""
        plt.figure(figsize=(16, 9))
        trend_str = "increasing" if self.is_increasing else "decreasing"
        monotonic_label = f'2. Monotonic Data ({trend_str})'
        
        plt.plot(self.x_data, self.original_y_data, 'o', color='gray', markersize=6, alpha=0.5, label='1. Original Noisy Data')
        plt.plot(self.x_data, self.y_monotonic, '.-', color='orange', markersize=7, label=monotonic_label)
        plt.plot(self.x_data, self.y_final_smooth, 'o-', color='blue', label='3. Final Smoothed Data')

        ext = self.extrapolator
        plt.errorbar(ext.target_x_values, ext.extrapolated_values, yerr=ext.extrapolated_errors, fmt='o', mfc='red', ecolor="lightcoral", capsize=3, label="4. Extrapolated", zorder=3)

        if np.isfinite(ext.convergent_value):
            plt.axhline(ext.convergent_value, color='red', linestyle='--', label=f'Extrapolated CV ({ext.convergent_value:.6f})', zorder=2)
            plt.fill_between([self.x_data[0], self.max_x], ext.convergent_value - ext.error, ext.convergent_value + ext.error, color='red', alpha=0.15, label=f'Uncertainty (±{ext.error:.6f})', zorder=1)
        
        if self.known_convergent_value is not None:
            plt.axhline(self.known_convergent_value, color='black', linestyle=':', linewidth=2.5, label=f'Known CV ({self.known_convergent_value:.6f})', zorder=2)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Full Monotonic Smoothing Pipeline for '{self.y_label}'", fontsize=16)
        plt.xlabel(self.x_label, fontsize=12)
        plt.ylabel(self.y_label, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.show()

    class _GeometricExtrapolator:
        """A nested helper class to handle the geometric extrapolation logic."""
        def __init__(self, x_data, y_data, max_x):
            self.x_data, self.y_data, self.max_x = x_data, y_data, max_x
            self.step_size = x_data[1] - x_data[0] if len(x_data) > 1 else 1.0
            self._precompute_values()

        def _precompute_values(self):
            self.deltas = np.diff(self.y_data)
            valid_deltas = self.deltas[np.abs(self.deltas) > 1e-15]
            if len(valid_deltas) < 2: self.r, self.ratios = 1.0, []
            else:
                self.ratios = valid_deltas[1:] / valid_deltas[:-1]
                self.r = np.mean(self.ratios)
            self.last_delta = self.deltas[-1] if len(self.deltas) > 0 else 0
            self.current_value = self.y_data[-1]
            self.convergent_value = self.get_convergent_value()
            self.target_x_values = np.arange(self.x_data[-1] + self.step_size, self.max_x + 1, self.step_size)
            self.extrapolated_values = self.extrapolate_to_specific_sizes()
            r_std = np.std(self.ratios) if hasattr(self, 'ratios') and len(self.ratios) > 1 else 0.01
            if not np.isfinite(self.convergent_value):
                self.error, self.extrapolated_errors = np.inf, np.full_like(self.extrapolated_values, np.nan)
            else:
                denominator = (1 - self.r)**2
                self.error = abs(self.last_delta * r_std / denominator)
                steps = ((self.target_x_values - self.x_data[-1]) / self.step_size).astype(int)
                derivative = self.last_delta * (1 - steps * self.r**(steps - 1) + (steps - 1) * self.r**steps) / denominator
                self.extrapolated_errors = np.abs(derivative) * r_std

        def get_convergent_value(self):
            if self.r >= 1 or abs(1 - self.r) < 1e-9: return np.inf
            return self.current_value + self.last_delta / (1 - self.r)

        def extrapolate_to_specific_sizes(self):
            steps = ((self.target_x_values - self.x_data[-1]) / self.step_size).astype(int)
            if self.r >= 1 or abs(1 - self.r) < 1e-9:
                return self.current_value + np.mean(self.deltas) * steps
            return self.current_value + self.last_delta * (1 - self.r**steps) / (1 - self.r)


# --- THE SINGLE KEYWORD FUNCTION ---

def extrapolate(df):
    """
    This single function runs the entire interactive analysis pipeline.
    It has been renamed from 'run_analysis' to 'extrapolate'.
    """
    # Removed the problematic warnings line completely
    
    # 1. Get user input for the column to analyze
    available_columns = df.columns.drop('basis size').tolist()
    print("Available columns to analyze:")
    for col in available_columns:
        print(f"- {col}")
    print("-" * 30)

    while True:
        column_name = input("Please enter the name of the column to analyze: ")
        if column_name in available_columns: break
        else: print(f"Error: Invalid column name '{column_name}'.")

    # 2. Get user parameters for the analysis
    while True:
        direction_input = input("Should the trend be 'increasing' or 'decreasing'? ").lower().strip()
        if direction_input in ['increasing', 'decreasing']:
            is_increasing = True if direction_input == 'increasing' else False
            break
        else: print("Invalid input. Please enter 'increasing' or 'decreasing'.")

    lambda_val = float(input("Enter the lambda for the final Whittaker smooth (e.g., 1.0): "))
    max_x_val = int(input("Enter the extrapolation limit for the x-axis (e.g., 20000): "))

    known_cv = None
    cv_input = input("Enter the exact convergent value if known, otherwise type 'None': ").strip().lower()
    if cv_input != 'none':
        try:
            known_cv = float(cv_input)
        except ValueError:
            print("Invalid number for convergent value. It will be ignored.")

    # 3. Initialize and run the full pipeline
    pipeline = MonotonicPipeline(
        x_data=df['basis size'].values,
        y_data=df[column_name].values,
        lambda_smooth=lambda_val,
        max_x=max_x_val,
        increasing=is_increasing,
        x_label="Basis Size",
        y_label=column_name,
        known_convergent_value=known_cv
    )

    # 4. Display the results
    print("-" * 50)
    print(f"Analysis for '{pipeline.y_label}' (Specified Trend: {'increasing' if pipeline.is_increasing else 'decreasing'})")
    print(f"Extrapolated Convergent Value: {pipeline.convergent_value:.15f}")
    print(f"Associated Uncertainty:        +/- {pipeline.error:.15f}")
    if pipeline.known_convergent_value is not None:
        print(f"Known Convergent Value:        {pipeline.known_convergent_value:.15f}")
    print("-" * 50)

    # 5. Create the interactive plot
    interact(
        pipeline.plot_analysis,
        x_min=FloatSlider(min=pipeline.x_data[0], max=pipeline.max_x, step=(pipeline.max_x - pipeline.x_data[0])/200, value=pipeline.x_data[0], description='X Min'),
        x_max=FloatSlider(min=pipeline.x_data[0], max=pipeline.max_x, step=(pipeline.max_x - pipeline.x_data[0])/200, value=pipeline.max_x, description='X Max'),
        y_min=FloatSlider(min=pipeline.y_min_bound, max=pipeline.y_max_bound, step=(pipeline.y_max_bound - pipeline.y_min_bound)/200, value=pipeline.y_min_bound, description='Y Min'),
        y_max=FloatSlider(min=pipeline.y_min_bound, max=pipeline.y_max_bound, step=(pipeline.y_max_bound - pipeline.y_min_bound)/200, value=pipeline.y_max_bound, description='Y Max')
    )
