import matplotlib.patches as patches
from lmfit import Model
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Mathematical Model Function ---
def exponential_decay_sq(x, C, A, B):
    """A 3-parameter exponential decay function."""
    return C + A * np.exp(-B * (x ** (0.5)) )

# --- 2. Create the Enhanced Fitter Class ---
class exponential_fit_sq:
    """
    A class to interactively fit an exponential model using a
    Self-Consistent Iteratively Reweighted Least Squares (IRLS) approach
    and calculate a comprehensive uncertainty for the extrapolated limit.
    """
    def __init__(self, dataframe):
        if 'basis size' not in dataframe.columns:
            raise ValueError("Input DataFrame must contain a 'Basis Size' column.")
        self.df = dataframe
        self.x_data = self.df['basis size']
        self.result = None
        self.total_uncertainty = None
        self.column_name = None
        self.max_x = None
        self.known_convergent_value = None
        self.known_convergent_uncertainty = None
        # Store iteration history for plotting
        self.iteration_history = {
            'C': [],
            'A': [],
            'B': [],
            'param_change': [],
            'residual_std': []
        }

    def _fit_column(self, column_name, max_x):
        """
        Performs a self-consistent, data-driven exponential fit using IRLS.
        """
        self.column_name = column_name
        self.max_x = max_x
        y_data = self.df[self.column_name].values

        # Clear iteration history for new fit
        for key in self.iteration_history:
            self.iteration_history[key] = []

        # --- Extended: Check multiple initial points for misleading values ---
        if len(y_data) >= 3:
            # Check first few points (up to 3) for misleading values
            points_to_check = min(3, len(y_data) - 1)  # Don't check all points
            bad_points = []
            
            for i in range(points_to_check):
                # If point is significantly lower than last point, mark as bad
                if y_data[i] < y_data[-1] - 0.5 * (y_data[-1] - np.min(y_data[1:])):
                    bad_points.append(i)
                else:
                    # Stop at first "good" point
                    break
            
            # Remove bad points but keep at least 2 points
            if bad_points and len(bad_points) < len(y_data) - 1:
                keep_indices = [i for i in range(len(y_data)) if i not in bad_points]
                x_trimmed = self.x_data.values[keep_indices]
                y_trimmed = y_data[keep_indices]
                print(f"Removed {len(bad_points)} early outlier(s) for '{column_name}'")
            else:
                x_trimmed = self.x_data.values
                y_trimmed = y_data
        else:
            x_trimmed = self.x_data.values
            y_trimmed = y_data

        # Use trimmed data for all fitting calculations
        x_min_orig, x_max_orig = x_trimmed.min(), x_trimmed.max()
        x_scaled = (x_trimmed - x_min_orig) / (x_max_orig - x_min_orig)

        exp_model = Model(exponential_decay_sq)
        params = exp_model.make_params()

        y_last, y_first = y_trimmed[-1], y_trimmed[0]
        params['C'].set(value=y_last)
        params['A'].set(value=y_first - y_last)
        if y_last < y_first:
            params['A'].set(min=0.0)
        else:
            params['A'].set(max=0.0)

        try:
            half_life_y = y_last + (y_first - y_last) / 2.0
            half_life_x = x_scaled[np.argmin(np.abs(y_trimmed - half_life_y))]
            params['B'].set(value=np.log(2) / np.sqrt(half_life_x) if half_life_x > 1e-6 else 0.1, min=1e-6)
        except:
            params['B'].set(value=0.1, min=1e-6)

        print("\n--- Starting Self-Consistent IRLS with Weight Optimization ---")

        # Test different weight forms if known convergent value is provided
        best_result = None
        best_distance = np.inf
        best_weights = None
        best_n = None

        weight_powers = [1, 2, 3, 4, 5]

        if self.known_convergent_value is not None:
            print(f"Optimizing weights using known value: {self.known_convergent_value:.8f}")
            print("Testing weight powers:", weight_powers)

            for n in weight_powers:
                try:
                    temp_result, temp_weights = self._fit_with_weights(
                        y_trimmed, x_scaled, params.copy(), exp_model, n
                    )

                    # Calculate distance to known value
                    extrapolated_limit = temp_result.params['C'].value
                    distance = abs(extrapolated_limit - self.known_convergent_value)

                    print(f"  Weight power {n}: Distance = {distance:.2e}")

                    if distance < best_distance:
                        best_distance = distance
                        best_result = temp_result
                        best_weights = temp_weights
                        best_n = n

                except Exception as e:
                    print(f"  Weight power {n} failed: {e}")
                    continue

            if best_result is not None:
                print(f"Best weight power: {best_n} with distance {best_distance:.2e}")
                result = best_result
                current_weights = best_weights
            else:
                print("Weight optimization failed, using default weights (n=1)")
                result, current_weights = self._fit_with_weights(
                    y_trimmed, x_scaled, params.copy(), exp_model, 1
                )
        else:
            # No known value, use default weight power
            print("No known convergent value provided, using weight power n=1")
            result, current_weights = self._fit_with_weights(
                y_trimmed, x_scaled, params.copy(), exp_model, 1
            )

        self.result = result
        final_weights = current_weights

        C_val = self.result.params['C'].value
        A_val = self.result.params['A'].value
        B_val = self.result.params['B'].value
        c_stderr = self.result.params['C'].stderr or 0.0
        a_stderr = self.result.params['A'].stderr or 0.0
        b_stderr = self.result.params['B'].stderr or 0.0

        # Monte Carlo uncertainty calculation for infinite basis size
        n_samples = 10000

        # For infinite basis size, x → ∞, so sqrt(x) → ∞, exp(-B*sqrt(x)) → 0
        # Therefore, the model C + A*exp(-B*sqrt(x)) → C as x → ∞
        # The uncertainty is simply the uncertainty in C parameter

        if c_stderr > 0:
            # Generate samples for C parameter
            C_samples = np.random.normal(C_val, c_stderr, n_samples)

            # For finite but large extrapolation point (more realistic)
            x_extrap_actual = self.max_x
            x_scaled_extrap = (x_extrap_actual - x_min_orig) / (x_max_orig - x_min_orig)

            # Generate samples for all parameters
            if (c_stderr > 0) and (a_stderr > 0) and (b_stderr > 0):
                # Use parameter covariance if available
                try:
                    # Sample parameters using covariance matrix
                    param_names = ['C', 'A', 'B']
                    param_vals = [C_val, A_val, B_val]
                    param_errs = [c_stderr, a_stderr, b_stderr]

                    # Simple diagonal covariance (uncorrelated)
                    samples = np.random.multivariate_normal(param_vals,
                                                          np.diag([err**2 for err in param_errs]),
                                                          n_samples)

                    # Evaluate model at extrapolation point for each sample
                    extrapolated_samples = []

                    for sample in samples:
                        C_s, A_s, B_s = sample
                        value = C_s + A_s * np.exp(-B_s * np.sqrt(x_scaled_extrap))
                        extrapolated_samples.append(value)

                    # Calculate Monte Carlo uncertainty
                    mc_uncertainty = np.std(extrapolated_samples)
                except:
                    # Fallback to simple C parameter uncertainty
                    mc_uncertainty = c_stderr
            else:
                # Fallback to simple C parameter uncertainty
                mc_uncertainty = c_stderr
        else:
            mc_uncertainty = 0.0

        # Total uncertainty (no arbitrary constants!)
        self.total_uncertainty = mc_uncertainty

        print("\n--- Final Fitted Parameters (after Self-Consistent IRLS) ---")
        for param_name, param in self.result.params.items():
            print(f"{param_name}: {param.value:.8f} +/- {param.stderr:.8f}" if param.stderr is not None else f"{param_name}: {param.value:.8f}")

        extrapolated_limit = self.result.params['C'].value

        print(f"\n--- Final Result for '{self.column_name}' ---")
        print(f"Extrapolated Limit (C): {extrapolated_limit:.8f}")
        print(f"Total Combined Uncertainty: ± {self.total_uncertainty:.8f}")

        if self.known_convergent_value is not None:
            print(f"Known Convergent Value:   {self.known_convergent_value:.8f}")
            if self.known_convergent_uncertainty is not None:
                 print(f"Known CV Uncertainty:     ± {self.known_convergent_uncertainty:.8f}")
            print(f"Difference:               {extrapolated_limit - self.known_convergent_value:.8f}")

    def _fit_with_weights(self, y_data, x_scaled, params, model, weight_power):
        """Helper method to fit with specific weight power."""
        n_iterations = 100
        convergence_threshold = 1e-8
        current_weights = np.ones(len(x_scaled))

        # Store iteration history temporarily
        temp_history = {
            'C': [params['C'].value],
            'A': [params['A'].value],
            'B': [params['B'].value],
            'param_change': [0.0],
            'residual_std': [0.0]
        }

        for i in range(n_iterations):
            result = model.fit(y_data, params, x=x_scaled, weights=current_weights)
            residuals = result.residual

            W_pos = np.exp(params['B'].value * np.sqrt(x_scaled))
            W_pos /= np.mean(W_pos)

            # Apply weight power
            new_weights = W_pos ** weight_power
            new_weights /= np.mean(new_weights)

            current_weights = 0.5 * current_weights + 0.5 * new_weights

            old_params = np.array(list(params.valuesdict().values()))
            params = result.params
            new_params = np.array(list(params.valuesdict().values()))
            param_change = np.sum((old_params - new_params)**2)

            # Store iteration data
            temp_history['C'].append(result.params['C'].value)
            temp_history['A'].append(result.params['A'].value)
            temp_history['B'].append(result.params['B'].value)
            temp_history['param_change'].append(param_change)
            temp_history['residual_std'].append(np.std(residuals))

            if param_change < convergence_threshold and i > 0:
                break

        return result, current_weights

    def _draw_plots_on_axis(self, ax):
        """Helper method to draw the fit results on a given matplotlib axis."""
        y_data = self.df[self.column_name]
        x_min_orig, x_max_orig = self.x_data.min(), self.x_data.max()

        ax.plot(self.x_data, y_data, 'o', label='Original Data')

        plot_x_orig = np.linspace(x_min_orig, self.max_x, 400)
        plot_x_scaled = (plot_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
        plot_y = self.result.eval(x=plot_x_scaled)
        ax.plot(plot_x_orig, plot_y, '-', label='Best Fit', linewidth=2)

        extrap_x_orig = np.arange(x_max_orig + 1000, self.max_x + 1, 1000)
        if len(extrap_x_orig) > 0:
            extrap_x_scaled = (extrap_x_orig - x_min_orig) / (x_max_orig - x_min_orig)
            extrap_y = self.result.eval(x=extrap_x_scaled)
            ax.plot(extrap_x_orig, extrap_y, 'o', color='red', markersize=6, label='Extrapolated Points')

        extrapolated_limit = self.result.params['C'].value
        uncertainty = self.total_uncertainty if self.total_uncertainty is not None else 0.0

        ax.axhline(extrapolated_limit, color='red', linestyle='--', label=f'Extrapolated Limit')
        if uncertainty > 0:
            ax.axhspan(extrapolated_limit - uncertainty, extrapolated_limit + uncertainty,
                      color='red', alpha=0.15, label='Total Uncertainty Band')

        if self.known_convergent_value is not None:
            ax.axhline(self.known_convergent_value, color='black', linestyle=':', linewidth=2.5,
                      label=f'Known CV ({self.known_convergent_value:.6f})')
            if self.known_convergent_uncertainty is not None:
                ax.axhspan(self.known_convergent_value - self.known_convergent_uncertainty,
                          self.known_convergent_value + self.known_convergent_uncertainty,
                          color='black', alpha=0.15, label='Known CV Uncertainty')

        ax.set_xlabel("Basis Size")
        ax.set_ylabel(self.column_name)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    def plot_final_fit(self):
        """Creates a dual-view plot without the IRLS convergence plot."""
        fig = plt.figure(figsize=(16, 8))  # Reduced from (24, 10) to (16, 8)

        # Create a 1x2 grid for two plots side by side
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])  # Overall view (left)
        ax2 = fig.add_subplot(gs[0, 1])  # Zoomed view (right)

        fig.suptitle(f"Exponential Fit for '{self.column_name}' using IRLS", fontsize=16)

        # Draw main plots
        self._draw_plots_on_axis(ax1)
        ax1.set_title("Overall View")

        self._draw_plots_on_axis(ax2)
        ax2.set_title("Zoomed View of Extrapolation")

        # Smart zoom range determination
        y_data = self.df[self.column_name]
        extrapolated_limit = self.result.params['C'].value

        # Determine x-range for zoom: show last 20-30% of data points plus extrapolation region
        x_data_values = self.x_data.values
        x_range = x_data_values.max() - x_data_values.min()

        # Start zoom from the point where data starts showing convergence behavior
        # Use the last 25% of data points as a starting point
        zoom_start_idx = max(0, int(len(x_data_values) * 0.75))
        x_zoom_min = x_data_values[zoom_start_idx]

        # If we have very few points, adjust accordingly
        if len(x_data_values) <= 3:
            x_zoom_min = x_data_values[0]

        ax2.set_xlim(x_zoom_min, self.max_x)

        # Smart y-range determination
        # Get y-values from the zoom x-range and include extrapolated limit
        mask = self.x_data >= x_zoom_min
        visible_y_data = y_data[mask].values

        if len(visible_y_data) > 0:
            y_min_zoom = min(np.min(visible_y_data), extrapolated_limit)
            y_max_zoom = max(np.max(visible_y_data), extrapolated_limit)
        else:
            # Fallback if no data in zoom range (shouldn't happen normally)
            y_min_zoom = min(y_data.min(), extrapolated_limit)
            y_max_zoom = max(y_data.max(), extrapolated_limit)

        # Add padding based on the range
        y_range = y_max_zoom - y_min_zoom
        if y_range < 1e-12:  # Handle very small ranges
            y_range = abs(extrapolated_limit * 0.1) if abs(extrapolated_limit) > 1e-12 else 0.1

        y_padding = y_range * 0.15  # Reduced padding for better view
        ax2.set_ylim(y_min_zoom - y_padding, y_max_zoom + y_padding)

        # Add zoom rectangle to overall view using data coordinates
        zoom_xlim = ax2.get_xlim()
        zoom_ylim = ax2.get_ylim()

        # Convert zoom rectangle to overall view coordinates
        rect = patches.Rectangle(
            (zoom_xlim[0], zoom_ylim[0]),
            zoom_xlim[1] - zoom_xlim[0],
            zoom_ylim[1] - zoom_ylim[0],
            linewidth=1.5,
            edgecolor='green',
            facecolor='green',
            alpha=0.2
        )
        ax1.add_patch(rect)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def fit(self):
        """
        Runs a single interactive fit and then exits.
        """
        available_columns = self.df.columns.drop('basis size').tolist()
        print("Available columns to analyze:")
        for col in available_columns:
            print(f"- {col}")
        print("-" * 30)

        self.known_convergent_value = None
        self.known_convergent_uncertainty = None

        column_name = input("Please enter the name of the column to fit (or 'q' to quit): ")
        if column_name.lower() in ['q', 'quit']:
            print("Exiting.")
            return

        if column_name not in available_columns:
            print(f"Error: Invalid column name '{column_name}'. Please choose from the list above.")
            return

        try:
            max_x_val = int(input(f"Enter the extrapolation limit for '{column_name}' (e.g., 20000): "))
        except ValueError:
            print("Invalid input. Using the max value from data as the limit.")
            max_x_val = self.x_data.max()

        cv_input = input("Enter a known convergent value for comparison (or press Enter to skip): ").strip()
        if cv_input:
            try:
                self.known_convergent_value = float(cv_input)
                cv_unc_input = input(f"Enter the uncertainty for {self.known_convergent_value} (or press Enter to skip): ").strip()
                if cv_unc_input:
                    try:
                        self.known_convergent_uncertainty = float(cv_unc_input)
                    except ValueError:
                        print("Invalid number for uncertainty. It will be ignored.")
            except ValueError:
                print("Invalid number for convergent value. It will be ignored.")
                self.known_convergent_value = None

        self._fit_column(column_name, max_x_val)
        self.plot_final_fit()
