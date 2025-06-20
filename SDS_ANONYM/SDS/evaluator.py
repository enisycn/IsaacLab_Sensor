import re
import os
import numpy as np
import subprocess
from matplotlib import pyplot as plt
import pickle
import warnings
import glob
from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)

eval_iter = 10
eval_steps = 7500
ROOT_DIR = os.path.join(os.getcwd(),"..")

def find_isaac_lab_root():
    """Find Isaac Lab root directory by looking for isaaclab.sh."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "isaaclab.sh").exists():
            return str(parent)
    return None

def get_evaluation_paths():
    """Get evaluation paths based on available framework (Isaac Lab preferred)."""
    # Try Isaac Lab first
    isaac_lab_root = find_isaac_lab_root()
    if isaac_lab_root:
        return {
            "framework": "isaac_lab",
            "isaac_lab_root": isaac_lab_root,
            "logs_dir": os.path.join(isaac_lab_root, "logs", "rsl_rl"),
            "play_script": None,  # Will use isaaclab.sh wrapper
            "eval_path": os.path.join(isaac_lab_root, "logs", "rsl_rl", "evaluation_results")
        }
    else:
        raise FileNotFoundError("No supported framework found. Please ensure Isaac Lab is properly installed.")

# Get paths based on available framework
PATHS = get_evaluation_paths()

def do_plot(metrics:dict,eval_cfg,chkpt_dir):
    num_eval_steps = eval_cfg["num_eval_steps"]
    dt = eval_cfg["dt"]

    fig, axs = plt.subplots(len(metrics)-1, 1, figsize=(12, 10))
    
    i = 0
    for metric in metrics.keys():
        if metric == "Resets":
            continue
        metric_dic = metrics[metric]
        data = np.array([iter["data"] for iter in metric_dic])
        y_label = metric_dic[0]["y_label"]
        
        mean_values = np.mean(data, axis=0)
        std_values = np.std(data, axis=0)
        
        if len(metrics)-1 == 1:
            axs.plot(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values, color='black', linestyle="-", label="Measured")
            axs.fill_between(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values - std_values, mean_values + std_values, color='blue', alpha=0.2, label='Std Dev')
            axs.legend()
            axs.set_title(metric)
            axs.set_xlabel("Time (s)")
            axs.set_ylabel(y_label)
        else:
            axs[i].plot(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values, color='black', linestyle="-", label="Measured")
            axs[i].fill_between(np.linspace(0, num_eval_steps * dt, num_eval_steps), mean_values - std_values, mean_values + std_values, color='blue', alpha=0.2, label='Std Dev')
            axs[i].legend()
            axs[i].set_title(metric)
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel(y_label)
        
        i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(chkpt_dir, "eval_plot.png"))

def do_evaluation_isaac_lab(experiment_dir, iter=eval_iter):
    """Evaluation for Isaac Lab experiments."""
    metrics = {}
    isaac_lab_root = PATHS["isaac_lab_root"]
    
    # Find the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(experiment_dir, "model_*.pt"))
    if not checkpoint_files:
        print(f"No checkpoints found in {experiment_dir}")
        return 0
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Evaluating checkpoint: {latest_checkpoint}")
    
    # Create evaluation directory
    eval_dir = os.path.join(experiment_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    for iteration in range(iter):
        print(f"Evaluation iteration {iteration + 1}/{iter}")
        
        # Run Isaac Lab evaluation with contact plotting
        eval_script = f"{isaac_lab_root}/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_with_contact_plotting.py --task=Isaac-SDS-Velocity-Flat-Unitree-Go1-Play-v0 --num_envs=1 --checkpoint={latest_checkpoint} --plot_steps={eval_steps} --contact_threshold=5.0 --warmup_steps=50 --headless"
        
        try:
            result = subprocess.run(eval_script.split(" "), cwd=isaac_lab_root, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Evaluation iteration {iteration} failed: {result.stderr}")
                continue
                
            # For Isaac Lab, we'll extract metrics from contact analysis if available
            contact_analysis_dir = os.path.join(experiment_dir, "contact_analysis")
            if os.path.exists(contact_analysis_dir):
                # Load contact data if available
                contact_data_file = os.path.join(contact_analysis_dir, "contact_data.npy")
                if os.path.exists(contact_data_file):
                    contact_data = np.load(contact_data_file)
                    # Calculate contact metrics
                    contact_percentage = np.mean(contact_data) * 100
                    
                    # Create metrics in expected format
                    if "Contact_Percentage" not in metrics:
                        metrics["Contact_Percentage"] = []
                    
                    metrics["Contact_Percentage"].append({
                        "data": np.full(eval_steps, contact_percentage),
                        "y_label": "Contact Percentage (%)"
                    })
                    
        except Exception as e:
            print(f"Error in evaluation iteration {iteration}: {e}")
            continue
    
    # Save evaluation results
    eval_data_file = os.path.join(eval_dir, "final_eval_data.pkl")
    with open(eval_data_file, 'wb') as file:
        pickle.dump(metrics, file)
    
    # Create evaluation config
    eval_cfg = {
        "num_eval_steps": eval_steps,
        "dt": 0.02  # Assume 50Hz control frequency
    }
    
    eval_cfg_file = os.path.join(eval_dir, "eval_config.npz")
    np.savez(eval_cfg_file, **eval_cfg)
    
    # Generate plots if we have metrics
    if metrics:
        do_plot(metrics, eval_cfg, eval_dir)
    
    # Calculate resets (for Isaac Lab, we'll use a placeholder)
    # In Isaac Lab, we don't have direct access to reset counts during evaluation
    # This would need to be implemented differently if reset tracking is crucial
    resets = 0  # Placeholder
    
    return resets

def do_evaluation_isaacgym(chkpt, iter=eval_iter):
    """Original evaluation for IsaacGym experiments."""
    metrics = {}
    run_chkpt_dir = os.path.join(PATHS["eval_path"], f"{chkpt}")
    eval_data_dir = os.path.join(run_chkpt_dir, "eval_data.pkl")
    eval_cfg_dir = os.path.join(run_chkpt_dir, "eval_config.npz")
    play_script = f"python -u {PATHS['play_script']} --run {run_chkpt_dir} --dr-config sds --headless --save_contact --iterations {eval_steps} --evaluation"
    
    for _ in range(iter):
        subprocess.run(play_script.split(" "))
        with open(eval_data_dir, 'rb') as file:
            eval_data = pickle.load(file)
            
        for metric in eval_data.keys():
            if metric not in metrics:
                metrics[metric] = [eval_data[metric]]
            else:
                metrics[metric].append(eval_data[metric])
                
    eval_cfg = np.load(eval_cfg_dir)
    resets = np.mean([iter["data"] for iter in metrics["Resets"]])
    
    with open(os.path.join(run_chkpt_dir, "final_eval_data.pkl"), 'wb') as file:
        pickle.dump(metrics, file)
    
    do_plot(metrics, eval_cfg, run_chkpt_dir)
    
    return resets

def do_evaluation(experiment_identifier, iter=eval_iter):
    """Main evaluation function that chooses the appropriate method."""
    if PATHS["framework"] == "isaac_lab":
        return do_evaluation_isaac_lab(experiment_identifier, iter)
    else:
        return do_evaluation_isaacgym(experiment_identifier, iter)

def get_experiments_list():
    """Get list of experiments based on the framework."""
    if PATHS["framework"] == "isaac_lab":
        # Get Isaac Lab experiments
        logs_dir = PATHS["logs_dir"]
        experiment_dirs = []
        
        # Look for SDS-related experiments
        sds_experiment_dirs = glob.glob(os.path.join(logs_dir, "unitree_go1_flat", "*"))
        experiment_dirs.extend(sds_experiment_dirs)
        
        # Filter to only include directories with checkpoints
        valid_experiments = []
        for exp_dir in experiment_dirs:
            if os.path.isdir(exp_dir):
                checkpoint_files = glob.glob(os.path.join(exp_dir, "model_*.pt"))
                if checkpoint_files:
                    valid_experiments.append(exp_dir)
        
        return valid_experiments
    else:
        # Get IsaacGym experiments
        eval_path = PATHS["eval_path"]
        if os.path.exists(eval_path):
            return [chkpt for chkpt in os.listdir(eval_path) if os.path.isdir(os.path.join(eval_path, chkpt))]
        else:
            return []

if __name__ == "__main__":
    print(f"Using framework: {PATHS['framework']}")
    
    all_resets = []
    experiment_names = []
    experiments = get_experiments_list()
    
    if not experiments:
        print("No experiments found for evaluation!")
        exit(1)
    
    print(f"Found {len(experiments)} experiments to evaluate")
    
    for experiment in experiments:
        experiment_name = os.path.basename(experiment) if PATHS["framework"] == "isaac_lab" else experiment
        experiment_names.append(experiment_name)
        print(f"Evaluating: {experiment_name}")
        
        resets = do_evaluation(experiment)
        all_resets.append(resets)
        print(f"Completed evaluation for {experiment_name}, resets: {resets}")
        
    # Create comparison plot
    if PATHS["framework"] == "isaac_lab":
        output_dir = os.path.join(PATHS["logs_dir"], "evaluation_comparison")
    else:
        output_dir = os.path.join(PATHS["eval_path"], '../')
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(experiment_names)), all_resets, label='Resets per Experiment')
    ax.set_title('Resets per Experiment')
    ax.set_ylabel('Number of Resets')
    ax.set_xlabel('Experiment')
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resets_per_experiment_chart.png'))
    plt.close()
    
    print(f"Evaluation complete. Results saved to: {output_dir}")
