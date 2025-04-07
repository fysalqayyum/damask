import numpy as np
import subprocess
import shutil
import os
import json
import re
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)

class Config:
    """
    Configuration class with all user-adjustable parameters at the top.
    """
    # =====================
    #  SIMULATION SETTINGS
    # =====================
    MAX_PARALLEL_JOBS = 3
    CORES_PER_JOB = 30
    MAX_ITER = 100
    
    # =====================
    #   STRAIN SEGMENTS
    # =====================
    # We define two strain segments for both stress–strain and hardening error.
    # For example: 0–SPLIT_STRAIN is "low strain", SPLIT_STRAIN–10% is "high strain."
    SPLIT_STRAIN = 3.0  # in percent
    
    # Weights for the two segments in the stress–strain error
    WEIGHT_STRESS_LOW = 0.2
    WEIGHT_STRESS_HIGH = 0.8
    
    # Weights for the two segments in the hardening derivative error
    WEIGHT_HARD_LOW = 0.3
    WEIGHT_HARD_HIGH = 0.7
    
    # ===========================
    #   SIGNED BIAS PENALTY
    # ===========================
    # If you want to penalize systematic upward/downward offsets in stress, set LAMBDA_BIAS > 0.
    LAMBDA_BIAS = 5.0  # scale factor for the absolute mean difference (bias)
    
    # ===========================
    #  COMBINED OBJECTIVE WEIGHTS
    # ===========================
    STRESS_ERROR_WEIGHT = 0.5
    HARDENING_ERROR_WEIGHT = 0.5
    
    # =============================
    #  HARDENING PARAMETERS
    # =============================
    HARDENING_SMOOTHING_WINDOW = 60
    EXP_HARDENING_STEP = 80
    SIM_HARDENING_STEP = 1
    
    # For plotting re-scaling
    STRAIN_THRESHOLD = 0.5
    
    # =============================
    #    FILES / DIRECTORIES
    # =============================
    BASE_DIR = Path(os.getcwd())
    RESULTS_DIR = BASE_DIR / "optimization_results"
    TEMPLATE_DIR = BASE_DIR / "templates"
    PLOTS_DIR = RESULTS_DIR / "plots"
    
    MATERIAL_TEMPLATE = BASE_DIR / "material.config.template"
    GEOM_FILE = next(BASE_DIR.glob("*.geom"))
    LOAD_FILE = next(BASE_DIR.glob("*.load"))
    EXP_DATA_FILE = BASE_DIR / "pureNb.txt"
    
    # =============================
    #   PARAMETER SPACE
    # =============================
    PARAM_SPACE = [
        Real(name='gdot0_slip',  low=0.0001, high=0.01),
        Real(name='n_slip',      low=2, high=50),
        Real(name='tau0_slip',   low=1.0e7, high=1.0e9),
        Real(name='tausat_slip', low=1.0e7, high=1.0e10),
        Real(name='a_slip',      low=0.2,   high=5.0),
        Real(name='h0_slipslip', low=1.0e7, high=1.0e9),
    ]
    
    @classmethod
    def setup(cls):
        """Create directories and initialize the optimization history file."""
        cls.RESULTS_DIR.mkdir(exist_ok=True)
        cls.TEMPLATE_DIR.mkdir(exist_ok=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True)
        
        history_file = cls.RESULTS_DIR / "optimization_history.json"
        if not history_file.exists():
            with open(history_file, 'w') as f:
                json.dump({"iterations": []}, f)

# ==================================
#       I/O AND PREPARATION
# ==================================
def read_experimental_data():
    """Load experimental stress–strain data from CSV or TXT"""
    logging.info("Reading experimental data...")
    data = np.genfromtxt(Config.EXP_DATA_FILE, delimiter='\t', skip_header=1)
    # Convert strain to percent, keep stress in MPa
    strain = data[:, 1]
    stress = data[:, 0]
    return {'stress_strain': np.column_stack((strain, stress))}

def update_material_config(params, run_dir):
    material_file = run_dir / "material.config"
    logging.info(f"Updating material config in {run_dir} with parameters:")
    for name, value in params.items():
        logging.info(f"  {name}: {value:.6e}")

    with open(Config.MATERIAL_TEMPLATE, 'r') as f:
        content = f.read()

    for pname, pvalue in params.items():
        # Pattern to match the parameter name followed by its numeric value.
        pattern = rf'(?P<paramName>{pname}\s+)(?P<paramValue>\d+\.?\d*e?-?\d*)'
        
        # Use a lambda to safely build the replacement string.
        content = re.sub(
            pattern,
            lambda m: m.group("paramName") + f'{pvalue:.6e}',
            content
        )

    with open(material_file, 'w') as f:
        f.write(content)

    return material_file


def run_damask_simulation(run_dir):
    """
    Run DAMASK in run_dir, copying the geometry and load files, writing and executing run_simulation.sh.
    """
    logging.info(f"Starting DAMASK simulation in {run_dir}")
    try:
        shutil.copy(Config.GEOM_FILE, run_dir)
        shutil.copy(Config.LOAD_FILE, run_dir)
        
        geom_name = Config.GEOM_FILE.stem
        load_name = Config.LOAD_FILE.stem
        
        script_content = f"""#!/bin/bash
export DAMASK_NUM_THREADS={Config.CORES_PER_JOB}

DAMASK_spectral -l {load_name}.load -g {geom_name}.geom

# Post-processing
postResults --cr f,p {geom_name}_{load_name}.spectralOut
addCauchy ./postProc/{geom_name}_{load_name}.txt
addStrainTensors -0 -v ./postProc/{geom_name}_{load_name}.txt
addMises -s Cauchy ./postProc/{geom_name}_{load_name}.txt
addMises -e 'ln(V)' ./postProc/{geom_name}_{load_name}.txt

# Add calculations
addCalculation --label Mises_stress_MPa --formula '#Mises(Cauchy)#/(1e6)' ./postProc/{geom_name}_{load_name}.txt
addCalculation --label Mises_strain_percent --formula '#Mises(ln(V))#*100' ./postProc/{geom_name}_{load_name}.txt

# Create clean output
cp ./postProc/{geom_name}_{load_name}.txt ./postProc/{geom_name}_{load_name}_clean.txt
filterTable -w inc,Mises_stress_MPa,Mises_strain_percent ./postProc/{geom_name}_{load_name}_clean.txt

rm {geom_name}_{load_name}.spectralOut
"""
        script_file = run_dir / "run_simulation.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        os.chmod(script_file, 0o755)
        
        result = subprocess.run(
            ["./run_simulation.sh"],
            cwd=run_dir,
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode == 0:
            logging.info(f"Simulation completed successfully in {run_dir}")
            return True
        else:
            logging.error(f"Simulation failed in {run_dir} with return code {result.returncode}")
            return False
    except Exception as e:
        logging.error(f"Error running simulation in {run_dir}: {str(e)}")
        return False

# ==================================
#      STRESS–STRAIN CALCULATIONS
# ==================================
def compute_seg_masks(strain_array, split_strain):
    """
    Return two boolean masks: 
      low_mask for strain < split_strain, 
      high_mask for strain >= split_strain.
    """
    low_mask = (strain_array >= 0) & (strain_array < split_strain)
    high_mask = (strain_array >= split_strain)
    return low_mask, high_mask

def calculate_segmented_stress_error(sim_data, exp_data):
    """
    Compute two-segment stress error:
    - Segment 1: 0–SPLIT_STRAIN
    - Segment 2: SPLIT_STRAIN–max
    Then we do a weighted combination, returning an RMSE.
    """
    exp_strain = exp_data['stress_strain'][:, 0]
    exp_stress = exp_data['stress_strain'][:, 1]
    sim_stress_interp = np.interp(exp_strain, sim_data[:, 1], sim_data[:, 0])
    
    diff = sim_stress_interp - exp_stress
    
    low_mask, high_mask = compute_seg_masks(exp_strain, Config.SPLIT_STRAIN)
    
    # MSE in each segment
    mse_low = np.mean(diff[low_mask]**2) if np.any(low_mask) else 0
    mse_high = np.mean(diff[high_mask]**2) if np.any(high_mask) else 0
    
    # Weighted sum
    wlow = Config.WEIGHT_STRESS_LOW
    whigh = Config.WEIGHT_STRESS_HIGH
    combined_mse = wlow * mse_low + whigh * mse_high
    rmse = np.sqrt(combined_mse)
    
    logging.info(f"Segmented stress error: {rmse:.3f} "
                 f"(low={mse_low:.3f}, high={mse_high:.3f})")
    return rmse

def compute_stress_bias(sim_data, exp_data):
    """
    Compute the average signed difference (bias) across the entire strain range
    or you could also do partial range if desired.
    We'll do 0–max of experimental strain.
    """
    exp_strain = exp_data['stress_strain'][:, 0]
    exp_stress = exp_data['stress_strain'][:, 1]
    
    sim_stress_interp = np.interp(exp_strain, sim_data[:, 1], sim_data[:, 0])
    differences = sim_stress_interp - exp_stress
    
    mean_diff = np.mean(differences)  # bias
    logging.info(f"Signed bias: {mean_diff:.3f} MPa")
    return mean_diff

# ==================================
#    HARDENING CALCULATIONS
# ==================================
def calculate_hardening_curve(strain, stress, step):
    """
    Compute (delta_stress / delta_strain) over 'step' intervals, returning midpoints + derivative.
    """
    n = len(strain)
    if n <= step:
        return np.array([]), np.array([])
    ds = stress[step:] - stress[:-step]
    dE = strain[step:] - strain[:-step]
    with np.errstate(divide='ignore', invalid='ignore'):
        hard = ds / dE
    mids = 0.5*(strain[:-step] + strain[step:])
    return mids, hard

def compute_segmented_hardening_error(sim_midpoints, sim_hard, exp_midpoints, exp_hard):
    """
    Similar 2-segment approach for hardening:
    - Segment 1: 0–SPLIT_STRAIN
    - Segment 2: SPLIT_STRAIN–end
    Weighted by Config.WEIGHT_HARD_LOW and Config.WEIGHT_HARD_HIGH.
    We'll compute MSE over whichever points exist in each segment, then combine.
    """
    # Interpolate sim_hard onto exp_midpoints is presumably done already
    # We'll just do segmented MSE from the difference.
    
    diff = exp_hard - sim_hard
    
    low_mask, high_mask = compute_seg_masks(exp_midpoints, Config.SPLIT_STRAIN)
    mse_low = np.mean(diff[low_mask]**2) if np.any(low_mask) else 0
    mse_high = np.mean(diff[high_mask]**2) if np.any(high_mask) else 0
    
    wlow = Config.WEIGHT_HARD_LOW
    whigh = Config.WEIGHT_HARD_HIGH
    combined_mse = wlow*mse_low + whigh*mse_high
    rmse = np.sqrt(combined_mse)
    
    logging.info(f"Segmented hardening error: {rmse:.3f} "
                 f"(low={mse_low:.3f}, high={mse_high:.3f})")
    return rmse

def calculate_hardening_error(sim_data, exp_data):
    """
    Compute segmented hardening error:
      1. Build derivative for exp & sim using a finite difference approach
      2. Smooth the exp derivative
      3. Interpolate sim onto exp midpoints
      4. Segment the domain (0–SPLIT_STRAIN vs SPLIT_STRAIN–end), compute weighted MSE, then RMSE
    """
    exp_strain = exp_data['stress_strain'][:, 0]
    exp_stress = exp_data['stress_strain'][:, 1]
    
    # Experimental
    exp_mid, exp_hard = calculate_hardening_curve(exp_strain, exp_stress, Config.EXP_HARDENING_STEP)
    if exp_hard.size == 0:
        logging.info("No exp hardening data.")
        return 1e6, np.array([]), np.array([]), np.array([])
    window = Config.HARDENING_SMOOTHING_WINDOW
    exp_hard_smooth = np.convolve(exp_hard, np.ones(window)/window, mode='same')
    
    # Simulation
    sim_strain = sim_data[:, 1]
    sim_stress = sim_data[:, 0]
    sim_mid, sim_hard = calculate_hardening_curve(sim_strain, sim_stress, Config.SIM_HARDENING_STEP)
    if sim_hard.size == 0:
        logging.info("No sim hardening data.")
        return 1e6, exp_mid, exp_hard_smooth, np.full_like(exp_mid, np.nan)
    
    # Interpolate sim -> exp
    sim_hard_interp = np.interp(exp_mid, sim_mid, sim_hard)
    
    # Restrict to domain
    max_sim = sim_mid.max()
    valid_range = exp_mid <= max_sim
    exp_mid = exp_mid[valid_range]
    exp_hard_smooth = exp_hard_smooth[valid_range]
    sim_hard_interp = sim_hard_interp[valid_range]
    
    # Segment & combine
    if len(exp_mid) < 2:
        return 1e6, exp_mid, exp_hard_smooth, sim_hard_interp
    
    # Evaluate segmented MSE
    error = compute_segmented_hardening_error(
        sim_midpoints=exp_mid,   # after interpolation
        sim_hard=sim_hard_interp,
        exp_midpoints=exp_mid,
        exp_hard=exp_hard_smooth
    )
    return error, exp_mid, exp_hard_smooth, sim_hard_interp

# ==================================
#      PLOTTING AND HISTORY
# ==================================
def plot_current_results(sim_data, exp_data, params, run_id,
                         exp_hard_mid, exp_hard_smooth, sim_hard_interp):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot stress–strain
    ax1.plot(exp_data['stress_strain'][:, 0], exp_data['stress_strain'][:, 1],
             'b-', label='Experimental Stress–Strain')
    ax1.plot(sim_data[:, 1], sim_data[:, 0],
             'r--', label=f'Simulation (Run {run_id})')
    ax1.set_xlabel("Strain (%)")
    ax1.set_ylabel("Stress (MPa)")
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Hardening on secondary y-axis
    ax2 = ax1.twinx()
    if exp_hard_mid.size > 0:
        ax2.plot(exp_hard_mid, exp_hard_smooth, 'g-', label='Experimental Hardening Rate')
        ax2.plot(exp_hard_mid, sim_hard_interp, 'm--', label=f'Simulation Hardening Rate (Run {run_id})')
    ax2.set_ylabel("Hardening Rate (MPa/%)")
    ax2.legend(loc='upper right')
    
    # Rescale for plateau if any
    plateau_idx = (exp_hard_mid >= Config.STRAIN_THRESHOLD) & \
                  np.isfinite(exp_hard_smooth) & np.isfinite(sim_hard_interp)
    if np.any(plateau_idx):
        vals = np.concatenate([exp_hard_smooth[plateau_idx], sim_hard_interp[plateau_idx]])
        if vals.size and np.all(np.isfinite(vals)):
            vmin, vmax = vals.min(), vals.max()
            margin = 0.1 * (vmax - vmin) if vmax != vmin else 1.0
            ax2.set_ylim(vmin - margin, vmax + margin)
    
    # Extra space for parameter text
    plt.subplots_adjust(right=0.75)
    param_text = "Parameters:\n" + "\n".join(f"{k}: {v:.3e}" for k, v in params.items())
    plt.figtext(0.77, 0.5, param_text, fontsize=10, va='center')
    
    plt.tight_layout()
    out_file = Config.PLOTS_DIR / f"comparison_run_{run_id}.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

def update_optimization_history(params, stress_error, hardening_error, total_error, run_id):
    """Append iteration data to the optimization_history.json."""
    history_file = Config.RESULTS_DIR / "optimization_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    iteration_data = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": params,
        "stress_error": float(stress_error),
        "hardening_error": float(hardening_error),
        "total_error": float(total_error)
    }
    history["iterations"].append(iteration_data)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

def plot_optimization_progress():
    """Plot how errors and parameters evolve over iterations."""
    history_file = Config.RESULTS_DIR / "optimization_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)
    iterations = history["iterations"]
    
    stress_errors = [it["stress_error"] for it in iterations]
    hardening_errors = [it["hardening_error"] for it in iterations]
    
    plt.figure(figsize=(15, 10))
    
    # Error evolution
    plt.subplot(2, 1, 1)
    plt.plot(stress_errors, 'b-o', label="Stress–Strain Error")
    plt.plot(hardening_errors, 'r-o', label="Hardening Error")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Error Evolution")
    plt.legend()
    plt.grid(True)
    
    # Parameter evolution
    plt.subplot(2, 1, 2)
    param_values = {p.name: [] for p in Config.PARAM_SPACE}
    for it in iterations:
        for p in Config.PARAM_SPACE:
            param_values[p.name].append(it["parameters"][p.name])
    for name, vals in param_values.items():
        plt.plot(vals, '-o', label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Evolution")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(Config.PLOTS_DIR / 'optimization_progress.png')
    plt.close()

# ==================================
#   OBJECTIVE + EVALUATION
# ==================================
def run_single_evaluation(params, exp_data, run_id):
    """One iteration: update config, run DAMASK, compute errors, save/plot results."""
    logging.info(f"\nStarting evaluation {run_id}")
    for k, v in params.items():
        logging.info(f"  {k}: {v:.6e}")
    
    run_dir = Config.RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)
    
    # Make new material.config
    update_material_config(params, run_dir)
    
    # Run DAMASK
    if not run_damask_simulation(run_dir):
        logging.error(f"Evaluation {run_id} failed to run simulation.")
        return 1e6
    
    try:
        geom_name = Config.GEOM_FILE.stem
        load_name = Config.LOAD_FILE.stem
        results_file = run_dir / "postProc" / f"{geom_name}_{load_name}_clean.txt"
        sim_data = np.loadtxt(results_file, skiprows=10, usecols=(1, 2))
        
        # 1) Stress–strain error (segmented)
        stress_error = calculate_segmented_stress_error(sim_data, exp_data)
        
        # 2) Signed bias penalty
        bias = compute_stress_bias(sim_data, exp_data)
        bias_penalty = Config.LAMBDA_BIAS * abs(bias)
        
        # 3) Hardening error (segmented)
        hard_error, exp_mid, exp_hard_smooth, sim_hard_interp = calculate_hardening_error(sim_data, exp_data)
        
        # Weighted combination
        total_error = (Config.STRESS_ERROR_WEIGHT * (stress_error + bias_penalty) +
                       Config.HARDENING_ERROR_WEIGHT * hard_error)
        
        # Plot
        plot_current_results(sim_data, exp_data, params, run_id, exp_mid, exp_hard_smooth, sim_hard_interp)
        update_optimization_history(params, stress_error, hard_error, total_error, run_id)
        
        logging.info(
            f"Evaluation {run_id} done: stress_err={stress_error:.3f}, bias_pen={bias_penalty:.3f}, "
            f"hard_err={hard_error:.3f}, total={total_error:.3f}"
        )
        return total_error
    except Exception as e:
        logging.error(f"Error processing results in {run_dir}: {str(e)}")
        return 1e6

@use_named_args(Config.PARAM_SPACE)
def objective(**params):
    """The scikit-optimize objective function."""
    exp_data = read_experimental_data()
    run_id = len(list(Config.RESULTS_DIR.glob("run_*"))) + 1
    
    total_error = run_single_evaluation(params, exp_data, run_id)
    plot_optimization_progress()
    return total_error

def main():
    logging.info("Starting parameter optimization with segmented error + signed bias penalty.")
    Config.setup()
    
    res = gp_minimize(
        func=objective,
        dimensions=Config.PARAM_SPACE,
        n_calls=Config.MAX_ITER,
        n_random_starts=10,
        noise=0.01,
        verbose=True,
        n_jobs=Config.MAX_PARALLEL_JOBS
    )
    
    logging.info("Optimization completed!")
    logging.info(f"Best parameters: {res.x}")
    logging.info(f"Minimum total error: {res.fun}")
    
    final_params = {p.name: val for p, val in zip(Config.PARAM_SPACE, res.x)}
    with open(Config.RESULTS_DIR / 'best_parameters.json', 'w') as f:
        json.dump(final_params, f, indent=2)
    
    # Convergence plot
    plt.figure(figsize=(10, 6))
    plot_convergence(res)
    plt.savefig(Config.PLOTS_DIR / 'final_convergence.png')
    plt.close()

if __name__ == "__main__":
    main()

