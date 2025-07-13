import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.isotonic import IsotonicRegression
from ipywidgets import interact, FloatSlider
import warnings

# --- CORE HELPER FUNCTIONS ---

def enforce_monotonicity(x, y, increasing=True):
    """Enforces monotonicity on a data series using Isotonic Regression (PAVA)."""
    iso_reg = IsotonicRegression(out_of_bounds="clip", increasing=increasing)
    return iso_reg.fit_transform(x, y)

def whittaker_smoother(y, lambd, d=2, weights=None):
    """A robust weighted Whittaker smoother."""
    m = len(y)
    if m < d + 1: return y
    if weights is None: weights = np.ones(m)
    W = sparse.diags(weights, format='csc')
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(m - d, m), format='csc')
    A = W + lambd * (D.T @ D)
    weighted_y = W @ y
    z = spsolve(A, weighted_y)
    return z

def find_optimal_lambda(y_monotonic, lambda_candidates):
    """
    Automatically finds the smallest lambda that produces stable extrapolation ratios.
    Anchoring the last point is now the default, robust behavior.
    """
    print("Searching for an optimal smoothing lambda...")
    weights = np.ones(len(y_monotonic))
    if len(weights) > 0:
        # Set a very high weight for the last point to anchor it
        weights[-1] = 1e2
    
    print("INFO: Last data point is anchored by default for robust smoothing.")
    
    for lambd in lambda_candidates:
        y_smooth = whittaker_smoother(y_monotonic, lambd, weights=weights)
        deltas = np.diff(y_smooth)
        valid_deltas = deltas[np.abs(deltas) > 1e-15]
        if len(valid_deltas) < 2: continue
        ratios = valid_deltas[1:] / valid_deltas[:-1]
        if len(ratios) > 0 and np.all((ratios > 0) & (ratios < 1)):
            print(f"✅ Suitable lambda found: {lambd:.4f}")
            return lambd
            
    best_lambda = lambda_candidates[-1]
    print(f" {best_lambda:.4f}")
    return best_lambda

# --- PRIMARY ANALYSIS CLASS ---

class MonotonicPipeline:
    """A class that encapsulates a full analysis pipeline with advanced plotting."""
    def __init__(self, x_data, y_data, increasing, lambda_smooth, max_x,
                 x_label="X-Axis", y_label="Y-Axis", known_convergent_value=None):
        self.x_data = np.array(x_data)
        self.original_y_data = np.array(y_data)
        self.lambda_smooth = lambda_smooth
        self.max_x = max_x
        self.x_label = x_label
        self.y_label = y_label
        self.is_increasing = increasing
        self.known_convergent_value = known_convergent_value
        self.anchor_last_point = True # Anchoring is now default
        
        weights = np.ones(len(self.x_data))
        if len(weights) > 0:
            weights[-1] = 1e2

        self.y_monotonic = enforce_monotonicity(self.x_data, self.original_y_data, increasing=self.is_increasing)
        self.y_final_smooth = whittaker_smoother(self.y_monotonic, lambd=self.lambda_smooth, weights=weights)
        self.extrapolator = self._GeometricExtrapolator(self.x_data, self.y_final_smooth, self.max_x)
        
        self.convergent_value = self.extrapolator.convergent_value
        self.error = self.extrapolator.error
        self.all_ratios_computed = self.extrapolator.ratios
        self._calculate_plot_bounds()

    def _calculate_plot_bounds(self):
        """Calculate the overall data range for the main plot."""
        all_y = np.concatenate([self.original_y_data, self.y_monotonic, self.y_final_smooth, self.extrapolator.extrapolated_values])
        if np.isfinite(self.extrapolator.error):
            all_y = np.concatenate([all_y, self.extrapolator.extrapolated_values - self.extrapolator.extrapolated_errors, self.extrapolator.extrapolated_values + self.extrapolator.extrapolated_errors])
        if self.known_convergent_value is not None:
            all_y = np.append(all_y, self.known_convergent_value)
        self.y_min_bound, self.y_max_bound = np.nanmin(all_y), np.nanmax(all_y)
        y_range = self.y_max_bound - self.y_min_bound
        y_padding = y_range * 0.1 if y_range > 0 else 1.0
        self.y_min_bound -= y_padding
        self.y_max_bound += y_padding

    def _draw_plots_on_axis(self, ax):
        """Helper method to draw the core data plots on a given matplotlib axis."""
        trend_str = "increasing" if self.is_increasing else "decreasing"
        monotonic_label = f'2. Monotonic Data ({trend_str})'
        smooth_label = '3. Final Smoothed Data (Anchored)'

        ax.plot(self.x_data, self.original_y_data, 'o', color='gray', markersize=5, alpha=0.6, label='1. Original Data')
        ax.plot(self.x_data, self.y_monotonic, '.-', color='orange', markersize=6, alpha=0.8, label=monotonic_label)
        ax.plot(self.x_data, self.y_final_smooth, 'o-', color='blue', markersize=5, label=smooth_label)

        ext = self.extrapolator
        ax.errorbar(ext.target_x_values, ext.extrapolated_values, yerr=ext.extrapolated_errors, fmt='o', ms=5, mfc='red', ecolor="lightcoral", capsize=3, label="4. Extrapolated", zorder=3)

        if np.isfinite(ext.convergent_value):
            ax.axhline(ext.convergent_value, color='red', linestyle='--', label=f'Extrapolated CV ({ext.convergent_value:.6f})', zorder=2)
            xlim = ax.get_xlim()
            ax.fill_between(xlim, ext.convergent_value - ext.error, ext.convergent_value + ext.error, color='red', alpha=0.15, label=f'Uncertainty (±{ext.error:.6f})', zorder=1)
            ax.set_xlim(xlim)

        if self.known_convergent_value is not None:
            ax.axhline(self.known_convergent_value, color='black', linestyle=':', linewidth=2.5, label=f'Known CV ({self.known_convergent_value:.6f})', zorder=2)
        
        ax.set_xlabel(self.x_label, fontsize=12)
        ax.set_ylabel(self.y_label, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')

    def plot_analysis(self, pan_x_percent=0, view_width_percent=25):
        """Plots a dual view with more intuitive and stable pan/zoom controls."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
        fig.suptitle(f"Monotonic Extrapolation Analysis for '{self.y_label}' (λ={self.lambda_smooth:.2f})", fontsize=18)

        self._draw_plots_on_axis(ax1)
        ax1.set_title("Overall View", fontsize=14)
        ax1.set_xlim(self.x_data[0], self.max_x)
        ax1.set_ylim(self.y_min_bound, self.y_max_bound)

        self._draw_plots_on_axis(ax2)
        ax2.set_title("Zoomed View (Interactive)", fontsize=14)

        x_start_extrap = self.x_data[-1] if len(self.x_data) > 0 else 0
        x_extrap_range = self.max_x - x_start_extrap
        if x_extrap_range <= 0: x_extrap_range = x_start_extrap * 0.1 if x_start_extrap > 0 else 1.0

        x_center = x_start_extrap + (x_extrap_range * pan_x_percent / 100.0)
        view_width = x_extrap_range * (view_width_percent / 100.0)
        zoom_xlim = (x_center - view_width / 2, x_center + view_width / 2)
        
        all_x_points = np.concatenate([self.x_data, self.extrapolator.target_x_values])
        all_y_points = np.concatenate([self.y_final_smooth, self.extrapolator.extrapolated_values])
        visible_mask = (all_x_points >= zoom_xlim[0]) & (all_x_points <= zoom_xlim[1])
        y_values_in_view = list(all_y_points[visible_mask])
        
        if np.isfinite(self.convergent_value):
            y_values_in_view.append(self.convergent_value)
            if np.isfinite(self.error):
                y_values_in_view.extend([self.convergent_value - self.error, self.convergent_value + self.error])

        if not y_values_in_view:
            y_center = self.y_final_smooth[-1] if len(self.y_final_smooth) > 0 else 0
            y_range = (self.y_max_bound - self.y_min_bound) * 0.1
        else:
            min_y_in_view, max_y_in_view = np.min(y_values_in_view), np.max(y_values_in_view)
            y_range = max_y_in_view - min_y_in_view
            y_center = min_y_in_view + y_range / 2

        total_y_range = self.y_max_bound - self.y_min_bound
        min_y_range = total_y_range * 0.05 if total_y_range > 0 else 1.0
        if y_range < min_y_range: y_range = min_y_range
            
        y_padding = y_range * 0.1
        zoom_ylim = (y_center - (y_range / 2) - y_padding, y_center + (y_range / 2) + y_padding)
        
        ax2.set_xlim(zoom_xlim)
        ax2.set_ylim(zoom_ylim)

        rect = patches.Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                                 linewidth=1.5, edgecolor='green', facecolor='green', alpha=0.2)
        ax1.add_patch(rect)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    class _GeometricExtrapolator:
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
                self.error = abs(self.last_delta * r_std / denominator) if denominator > 1e-9 else np.inf
                steps = ((self.target_x_values - self.x_data[-1]) / self.step_size).astype(int)
                derivative = self.last_delta * (1 - steps * self.r**(steps - 1) + (steps - 1) * self.r**steps) / denominator if denominator > 1e-9 else np.zeros_like(steps, dtype=float)
                self.extrapolated_errors = np.abs(derivative) * r_std
        def get_convergent_value(self):
            if self.r >= 1 or abs(1 - self.r) < 1e-9: return np.inf
            return self.current_value + self.last_delta / (1 - self.r)
        def extrapolate_to_specific_sizes(self):
            steps = ((self.target_x_values - self.x_data[-1]) / self.step_size).astype(int)
            if self.r >= 1 or abs(1 - self.r) < 1e-9:
                return self.current_value + np.mean(self.deltas) * steps
            return self.current_value + self.last_delta * (1 - self.r**steps) / (1 - self.r)

# --- THE MAIN USER-FACING FUNCTION ---

def extrapolate(df):
    """Runs the entire interactive analysis pipeline with the dual-plot view."""
    available_columns = df.columns.drop('basis size', errors='ignore').tolist()
    if not available_columns:
        print("ERROR: DataFrame must contain a 'basis size' column and at least one other data column.")
        return
        
    print("Available columns to analyze:", ", ".join(available_columns))
    column_name = ""
    while column_name not in available_columns:
        column_name = input("Please enter the name of the column to analyze: ")
        if column_name not in available_columns: print(f"Error: Invalid column name '{column_name}'.")
    print("-" * 30)

    x_data = df['basis size'].values
    y_data_raw = df[column_name].values
    is_increasing = np.mean(np.diff(y_data_raw)) > 0 if len(y_data_raw) > 1 else True
    print(f"Automatically determined trend for '{column_name}': {'increasing' if is_increasing else 'decreasing'}")

    max_x_val = int(input("Enter the extrapolation limit for the x-axis (e.g., 20000): "))
    
    known_cv = None
    cv_input = input("Enter the exact convergent value if known, otherwise press Enter: ").strip()
    if cv_input:
        try: known_cv = float(cv_input)
        except ValueError: print("Invalid number for convergent value. It will be ignored.")
    print("-" * 30)

    y_monotonic = enforce_monotonicity(x_data, y_data_raw, increasing=is_increasing)
    lambda_candidates = np.logspace(-2, 4, 100)
    best_lambda = find_optimal_lambda(y_monotonic, lambda_candidates=lambda_candidates)

    pipeline = MonotonicPipeline(x_data, y_data_raw, is_increasing, best_lambda, max_x_val, "Basis Size", column_name, known_cv)

    print("-" * 50)
    print(f"Analysis for '{pipeline.y_label}' (Using auto-selected Lambda: {best_lambda:.4f})")
    print(f"Extrapolated Convergent Value: {pipeline.convergent_value:.15f}")
    print(f"Associated Uncertainty:         +/- {pipeline.error:.15f}")
    if pipeline.known_convergent_value is not None:
        print(f"Known Convergent Value:         {pipeline.known_convergent_value:.15f}")
    print(f"\nFinal Ratio Constants: {np.round(pipeline.all_ratios_computed, 4).tolist()}")
    print("-" * 50)

    interact(
        pipeline.plot_analysis,
        pan_x_percent=FloatSlider(min=0, max=100, step=1, value=0, description='Pan X-Axis %', continuous_update=False),
        view_width_percent=FloatSlider(min=1, max=100, step=1, value=25, description='View Width %', continuous_update=False)
    )
    return pipeline
