
import matplotlib.patches as patches
from lmfit import Model

# --- 1. Define the Mathematical Model Function ---
def exponential_decay(x, C, A, B):
    """A 3-parameter exponential decay function."""
    return C + A * np.exp(-B * (x ** (0.6)) )

# --- 2. Create the Enhanced Fitter Class ---
class fit:
    """
    A class to interactively fit an exponential model using a
    Self-Consistent Iteratively Reweighted Least Squares (IRLS) approach
    and calculate a comprehensive uncertainty for the extrapolated limit.
    """
    def __init__(self, dataframe):
        if 'Basis Size' not in dataframe.columns:
            raise ValueError("Input DataFrame must contain a 'Basis Size' column.")
        self.df = dataframe
        self.x_data = self.df['Basis Size']
        self.result = None
        self.total_uncertainty = None
        self.column_name = None
        self.max_x = None
        self.known_convergent_value = None
        self.known_convergent_uncertainty = None

    def _fit_column(self, column_name, max_x):
        """
        Performs a self-consistent, data-driven exponential fit using IRLS.
        """
        self.column_name = column_name
        self.max_x = max_x
        y_data = self.df[self.column_name].values

        x_min_orig, x_max_orig = self.x_data.min(), self.x_data.max()
        x_scaled = (self.x_data.values - x_min_orig) / (x_max_orig - x_min_orig) 

        exp_model = Model(exponential_decay)
        params = exp_model.make_params()

        y_last, y_first = y_data[-1], y_data[0]
        params['C'].set(value=y_last)
        params['A'].set(value=y_first - y_last)
        if y_last < y_first:
            params['A'].set(min=0.0)
        else:
            params['A'].set(max=0.0)

        try:
            half_life_y = y_last + (y_first - y_last) / 2.0
            half_life_x = x_scaled[np.argmin(np.abs(y_data - half_life_y))]
            params['B'].set(value=np.log(2) / np.sqrt(half_life_x) if half_life_x > 1e-6 else 0.1, min=1e-6)
        except:
            params['B'].set(value=0.1, min=1e-6)

        print("\n--- Starting Self-Consistent IRLS ---")

        n_iterations = 50
        convergence_threshold = 1e-8
        current_weights = np.exp(x_scaled / x_scaled.max()) 
        current_weights /= np.mean(current_weights)

        for i in range(n_iterations):
            result = exp_model.fit(y_data, params, x=x_scaled, weights=current_weights)
            residuals = result.residual

            W_pos = np.exp(x_scaled / x_scaled.max())
            eps_res = np.median(np.abs(residuals)) * 0.1 + 1e-9
            W_res = 1.0 / (np.abs(residuals) + eps_res)
            rolling_std_res = pd.Series(residuals).rolling(window=7, center=True).std().bfill().ffill().values
            eps_stab = np.median(rolling_std_res[np.isfinite(rolling_std_res)]) * 0.1 + 1e-9
            W_stab = 1.0 / (rolling_std_res + eps_stab)

            new_weights = W_pos * W_res * W_stab
            new_weights /= np.mean(new_weights)
            current_weights = 0.1 * current_weights + 0.9 * new_weights

            old_params = np.array(list(params.valuesdict().values()))
            new_params = np.array(list(result.params.valuesdict().values()))
            param_change = np.sum((old_params - new_params)**2)

            print(f"Iteration {i+1}: Parameter change = {param_change:.2e}")

            if param_change < convergence_threshold and i > 0:
                print(f"Convergence reached after {i+1} iterations.")
                break

            params = result.params

        self.result = result
        final_weights = current_weights

        c_stderr = self.result.params['C'].stderr if self.result.params['C'].stderr is not None else 0.0
        squared_residuals = result.residual**2
        weighted_variance = final_weights * squared_residuals
        data_uncertainty_contrib = np.mean(np.sqrt(weighted_variance))
        self.total_uncertainty = np.sqrt(c_stderr**2 + data_uncertainty_contrib**2)

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
        """Creates the dual-view plot with a static, auto-zoomed panel."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
        fig.suptitle(f"Exponential Fit for '{self.column_name}' using IRLS", fontsize=18)
        self._draw_plots_on_axis(ax1)
        ax1.set_title("Overall View")
        self._draw_plots_on_axis(ax2)
        ax2.set_title("Zoomed View of Extrapolation")

        y_data = self.df[self.column_name]
        num_points_to_show = 4
        x_zoom_min = self.x_data.iloc[-num_points_to_show] if len(self.x_data) > num_points_to_show else self.x_data.iloc[0]
        ax2.set_xlim(x_zoom_min, self.max_x)

        extrapolated_limit = self.result.params['C'].value
        y_range_at_end = abs(y_data.iloc[-num_points_to_show] - extrapolated_limit)
        if y_range_at_end < 1e-9:
            y_range_at_end = abs(extrapolated_limit * 0.001) if abs(extrapolated_limit) > 1e-9 else 0.01

        y_padding = y_range_at_end * 1.5
        ax2.set_ylim(extrapolated_limit - y_padding, extrapolated_limit + y_padding)

        zoom_xlim = ax2.get_xlim()
        zoom_ylim = ax2.get_ylim()
        rect = patches.Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1] - zoom_xlim[0], zoom_ylim[1] - zoom_ylim[0],
                                 linewidth=1.5, edgecolor='green', facecolor='green', alpha=0.2)
        ax1.add_patch(rect)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def fit(self):
        """
        Runs a single interactive fit and then exits.
        """
        available_columns = self.df.columns.drop('Basis Size').tolist()
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

