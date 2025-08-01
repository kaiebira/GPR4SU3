import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# latqcdtools imports
from latqcdtools.base.readWrite import readTable
from latqcdtools.physics.lattice_params import latticeParams
from latqcdtools.base.printErrorBars import get_err_str
import latqcdtools.base.logger as logger

# --- Configuration ---
# Define which Nt values and their corresponding Ns values to load.
DATA_TO_LOAD = {
    6: [18],
    8: [24],
    10: [30],
    12: [18, 24, 36, 48, 60, 72, 84],
    14: [42],
    16: [48],
    18: [54],
    20: [60] # Ns=140 data for Nt=20 is not in this list, so GPR will extrapolate for Ns.
}

# Option to filter beta points (excluding lowest and highest for each dataset)
# Set to False to use all beta points, True to filter.
FILTER_BETA_POINTS = True

# --- 1. Data Loading and Preparation ---

logger.info("Loading and preparing data for GPR model X(T, Ns, Nt)...")

all_T_data, all_Ns_data, all_Nt_data, all_X_data, all_X_err_data = [], [], [], [], []

for current_Nt, Ns_list_for_Nt in DATA_TO_LOAD.items():
    for current_Ns in Ns_list_for_Nt:
        data_path = f'Nt{current_Nt}_Ns{current_Ns}.txt'
        logger.info(f"Attempting to load: {data_path}")
        try:
            data = readTable(data_path)
            beta_full, X_full, X_err_full = data[0], data[3], data[4]
            
            if FILTER_BETA_POINTS:
                if len(beta_full) >= 3:
                    beta = beta_full[1:-1]; X = X_full[1:-1]; X_err = X_err_full[1:-1]
                    logger.info(f"  For (Nt={current_Nt}, Ns={current_Ns}), filtered beta points. Kept {len(beta)} of {len(beta_full)}.")
                else:
                    logger.warning(f"  For (Nt={current_Nt}, Ns={current_Ns}), fewer than 3 beta points. Using all {len(beta_full)} points.")
                    beta, X, X_err = beta_full, X_full, X_err_full
            else:
                beta, X, X_err = beta_full, X_full, X_err_full
                logger.info(f"  For (Nt={current_Nt}, Ns={current_Ns}), using all {len(beta)} beta points.")

            if not beta.size: # Check if beta is empty after slicing
                logger.warning(f"  No data points for (Nt={current_Nt}, Ns={current_Ns}) after potential filtering. Skipping.")
                continue
            
            current_T_values_for_this_set = [latticeParams(current_Ns, current_Nt, b_val, scaleType='r0', scaleYear=2014, paramYear=2015).getT() for b_val in beta]
            all_T_data.extend(current_T_values_for_this_set)
            all_Ns_data.extend([current_Ns] * len(beta))
            all_Nt_data.extend([current_Nt] * len(beta))
            all_X_data.extend(X)
            all_X_err_data.extend(X_err)
            logger.info(f"  Successfully loaded and processed data for (Nt={current_Nt}, Ns={current_Ns}).")

        except FileNotFoundError:
            logger.TBError(f"  Data file not found: {data_path}.")
            data_path_root = f'Nt{current_Nt}_Ns{current_Ns}.txt'
            logger.info(f"  Attempting to load from root: {data_path_root}")
            try:
                data = readTable(data_path_root)
                beta_full, X_full, X_err_full = data[0], data[3], data[4]
                if FILTER_BETA_POINTS:
                    if len(beta_full) >= 3: beta, X, X_err = beta_full[1:-1], X_full[1:-1], X_err_full[1:-1]
                    else: beta, X, X_err = beta_full, X_full, X_err_full
                else: beta, X, X_err = beta_full, X_full, X_err_full
                
                if beta.size > 0:
                    current_T_values_for_this_set = [latticeParams(current_Ns, current_Nt, b_val, scaleType='r0', scaleYear=2014, paramYear=2015).getT() for b_val in beta]
                    all_T_data.extend(current_T_values_for_this_set)
                    all_Ns_data.extend([current_Ns] * len(beta))
                    all_Nt_data.extend([current_Nt] * len(beta))
                    all_X_data.extend(X)
                    all_X_err_data.extend(X_err)
                    logger.info(f"  Successfully loaded and processed data from root for (Nt={current_Nt}, Ns={current_Ns}).")
                else:
                    logger.warning(f"  No data points for (Nt={current_Nt}, Ns={current_Ns}) from root path after potential filtering. Skipping.")
            except FileNotFoundError:
                logger.TBError(f"  Data file also not found at root: {data_path_root}. Skipping this (Nt,Ns) pair.")
            except Exception as e_root:
                 logger.TBError(f"  Error processing file {data_path_root} from root: {e_root}")
        except Exception as e:
            logger.TBError(f"  Error processing file {data_path}: {e}")
            continue

if not all_T_data:
    logger.critical("No data loaded. Please check DATA_TO_LOAD and data file paths. Exiting.")
    exit()

X_train_raw = np.vstack([all_T_data, all_Ns_data, all_Nt_data]).T
y_train = np.array(all_X_data)
y_train_err = np.array(all_X_err_data)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e3)) +
    WhiteKernel(noise_level=np.mean(y_train_err**2), noise_level_bounds=(1e-7, 1e+2))
)

gpr = GaussianProcessRegressor(kernel=kernel, alpha=y_train_err**2, n_restarts_optimizer=25 )

logger.info("Training GPR model on (T, Ns, Nt) data...")
old_err_state = np.seterr(all='ignore')
try:
    gpr.fit(X_train_scaled, y_train)
finally:
    np.seterr(**old_err_state) 
logger.info(f"GPR training complete. Final kernel: {gpr.kernel_}")

N_SAMPLES_ERROR_PROP = 500 # Used for both training data Tc error and surrogate Tc error
T_RESOLUTION_PLOT = 200
final_Tc_results = {}
loaded_Nt_Ns_pairs = sorted(list(set(zip(all_Nt_data, all_Ns_data))))

if not loaded_Nt_Ns_pairs:
    logger.critical("No (Nt, Ns) pairs in training data. Cannot proceed with plotting.")
    exit()
    
for plot_Nt, plot_Ns in loaded_Nt_Ns_pairs:
    logger.info(f"Analyzing and plotting for Nt = {plot_Nt}, Ns = {plot_Ns} (from training data)...")
    mask_for_this_slice = (X_train_raw[:, 1] == plot_Ns) & (X_train_raw[:, 2] == plot_Nt)
    relevant_T_for_plot_slice = X_train_raw[mask_for_this_slice, 0]
    if not relevant_T_for_plot_slice.size: continue
    T_min_plot, T_max_plot = np.min(relevant_T_for_plot_slice), np.max(relevant_T_for_plot_slice)
    T_plot_grid = np.linspace(T_min_plot, T_max_plot, T_RESOLUTION_PLOT) if T_min_plot < T_max_plot else np.array([T_min_plot])
    
    Ns_for_pred_grid = np.full_like(T_plot_grid, plot_Ns)
    Nt_for_pred_grid = np.full_like(T_plot_grid, plot_Nt)
    X_pred_grid_raw = np.vstack([T_plot_grid, Ns_for_pred_grid, Nt_for_pred_grid]).T
    X_pred_grid_scaled = scaler.transform(X_pred_grid_raw)

    y_mean_pred = gpr.predict(X_pred_grid_scaled)
    Tc_central_val = T_plot_grid[np.argmax(y_mean_pred)] if y_mean_pred.size > 0 else T_min_plot

    if y_mean_pred.size > 0 :
        y_samples_pred = gpr.sample_y(X_pred_grid_scaled, N_SAMPLES_ERROR_PROP)
        Tc_samples_list = [T_plot_grid[np.argmax(y_samples_pred[:, i])] for i in range(N_SAMPLES_ERROR_PROP)]
        Tc_mean_val, Tc_err_val = np.mean(Tc_samples_list), np.std(Tc_samples_list)
    else: 
        Tc_mean_val, Tc_err_val = Tc_central_val, np.nan 

    final_Tc_results[(plot_Nt, plot_Ns)] = {'Tc': Tc_mean_val, 'Tc_err': Tc_err_val}
    logger.info(f"  Result for (Nt={plot_Nt}, Ns={plot_Ns}): Tc = {get_err_str(Tc_mean_val, Tc_err_val)} MeV")

    plt.figure(figsize=(10, 7))
    plt.errorbar(X_train_raw[mask_for_this_slice, 0], y_train[mask_for_this_slice], yerr=y_train_err[mask_for_this_slice],fmt='o', capsize=3, label=f'Sim Data (Nt={plot_Nt}, Ns={plot_Ns})', zorder=10)
    y_mean_plot, y_std_plot = gpr.predict(X_pred_grid_scaled, return_std=True)
    plt.plot(T_plot_grid, y_mean_plot, 'r-', label='GPR Mean Prediction')
    plt.fill_between(T_plot_grid, y_mean_plot - 1.96 * y_std_plot, y_mean_plot + 1.96 * y_std_plot, color='red', alpha=0.2, label='95% Confidence Interval')
    plt.axvline(Tc_mean_val, color='k', linestyle='--', label=f'Determined Tc')
    plt.axvspan(Tc_mean_val - Tc_err_val, Tc_mean_val + Tc_err_val, color='gray', alpha=0.4, label='1σ Error on Tc')
    plt.title(f'GPR Full Model Prediction for $N_\\tau={plot_Nt}$, $N_s={plot_Ns}$')
    plt.xlabel('Temperature T [MeV]'); plt.ylabel('Susceptibility $\\chi$'); plt.legend(); plt.grid(True, linestyle=':')
    plt.savefig(f'GPR_FullModel_Nt{plot_Nt}_Ns{plot_Ns}.pdf'); plt.show()

logger.info(f"--- Final Results (Tc for each (Nt, Ns) pair from training data) ---")
for (nt_val, ns_val), result in final_Tc_results.items():
    logger.info(f"Nt={nt_val}, Ns={ns_val}: Tc = {get_err_str(result['Tc'], result['Tc_err'])} MeV")


logger.info("--- Performing surrogate simulation for Nt=20, Ns=140 (with Tc uncertainty) ---")

predict_Nt = 20
predict_Ns = 140 
literature_critical_beta_Nt20 = 6.7132 # Used as the center for beta scan
beta_center = literature_critical_beta_Nt20
beta_range_width = 0.2 
num_beta_points_fine = 101

beta_fine_array = np.linspace(beta_center - beta_range_width / 2, 
                              beta_center + beta_range_width / 2, 
                              num_beta_points_fine)

T_fine_array = np.array([latticeParams(predict_Ns, predict_Nt, b_val, scaleType='r0', scaleYear=2014, paramYear=2015).getT() for b_val in beta_fine_array])

Ns_for_surrogate_pred = np.full_like(T_fine_array, predict_Ns)
Nt_for_surrogate_pred = np.full_like(T_fine_array, predict_Nt)

X_surrogate_pred_raw = np.vstack([T_fine_array, Ns_for_surrogate_pred, Nt_for_surrogate_pred]).T
X_surrogate_pred_scaled = scaler.transform(X_surrogate_pred_raw)

# Make predictions (mean and std for plotting the curve)
y_surrogate_mean, y_surrogate_std = gpr.predict(X_surrogate_pred_scaled, return_std=True)

# --- Calculate Tc and its uncertainty for the surrogate model ---
# Sample functions from the GPR posterior for the surrogate prediction points
y_surrogate_samples = gpr.sample_y(X_surrogate_pred_scaled, N_SAMPLES_ERROR_PROP)

# Find the peak temperature for each sample curve
Tc_surrogate_samples_list = [T_fine_array[np.argmax(y_surrogate_samples[:, i])] for i in range(N_SAMPLES_ERROR_PROP)]

# Calculate the mean and standard deviation of these peak temperatures
Tc_surrogate_mean = np.mean(Tc_surrogate_samples_list)
Tc_surrogate_err = np.std(Tc_surrogate_samples_list)
beta_peak_surrogate_from_mean_Tc = beta_fine_array[np.argmin(np.abs(T_fine_array - Tc_surrogate_mean))] # Beta closest to mean Tc

logger.info(f"Surrogate model results for (Nt={predict_Nt}, Ns={predict_Ns}):")
logger.info(f"  Predicted Tc = {get_err_str(Tc_surrogate_mean, Tc_surrogate_err)} MeV")
logger.info(f"  Corresponding critical beta (approx.) = {beta_peak_surrogate_from_mean_Tc:.5f}")
# Peak susceptibility from the mean prediction curve
peak_sus_mean_curve = np.max(y_surrogate_mean)
peak_sus_std_at_mean_peak = y_surrogate_std[np.argmax(y_surrogate_mean)]
logger.info(f"  Predicted peak susceptibility (from mean curve) = {peak_sus_mean_curve:.4e} +/- {peak_sus_std_at_mean_peak:.4e}")

# Plot the surrogate simulation results
plt.figure(figsize=(10, 7))
plt.plot(T_fine_array, y_surrogate_mean, 'b-', label='GPR Surrogate Prediction (Mean)')
plt.fill_between(
    T_fine_array, y_surrogate_mean - 1.96 * y_surrogate_std, y_surrogate_mean + 1.96 * y_surrogate_std,
    color='blue', alpha=0.2, label='95% Confidence Interval (Mean Prediction)'
)
# Plot the determined Tc and its error band for the surrogate
plt.axvline(Tc_surrogate_mean, color='green', linestyle=':', label=f'Predicted Tc = {Tc_surrogate_mean:.2f} $\\pm$ {Tc_surrogate_err:.2f} MeV')
plt.axvspan(Tc_surrogate_mean - Tc_surrogate_err, Tc_surrogate_mean + Tc_surrogate_err, color='lightgreen', alpha=0.4, label='1σ Error on Predicted Tc')

plt.title(f'GPR Surrogate Simulation for $N_\\tau={predict_Nt}$, $N_s={predict_Ns}$')
plt.xlabel('Temperature T [MeV]')
plt.ylabel('Predicted Susceptibility $\\chi$')
plt.legend(loc='best')
plt.grid(True, linestyle=':')
plt.savefig(f'GPR_Surrogate_Nt{predict_Nt}_Ns{predict_Ns}_with_Tc_Error.pdf')
plt.show()

logger.info("Surrogate simulation complete.")
