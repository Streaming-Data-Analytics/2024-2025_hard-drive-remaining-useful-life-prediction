"""
Model Definitions and Training Utilities
==========================================
This module contains all model configurations and training functions for the 
streaming RUL prediction pipeline.
"""

from river import stream, linear_model, tree, ensemble, metrics, preprocessing, compose
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import config
import os
from preprocessing import apply_feature_transformation


def create_baseline_model(use_scaler: bool = True) -> compose.Pipeline:
    """
    Create a Linear Regression baseline model.
    
    This is the simplest streaming model - it learns a linear relationship
    between SMART features and RUL.
    
    Args:
        use_scaler: Whether to use StandardScaler preprocessing
        
    Returns:
        River Pipeline with the baseline model
    """
    if use_scaler:
        return compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LinearRegression()
        )
    else:
        return linear_model.LinearRegression()


def create_hoeffding_tree(grace_period: int = 50,
                         leaf_prediction: str = "mean",
                         model_selector_decay: float = 0.9,
                         use_scaler: bool = True) -> compose.Pipeline:
    """
    Create a Hoeffding Tree Regressor (incremental decision tree).
    
    The Hoeffding Tree is a streaming decision tree that makes split decisions
    based on statistical bounds (Hoeffding bounds) rather than scanning all data.
    
    Args:
        grace_period: Number of instances between split attempts
        leaf_prediction: Prediction method at leaves ("mean", "model", "adaptive")
        model_selector_decay: Decay factor for model selection
        use_scaler: Whether to use StandardScaler preprocessing
        
    Returns:
        River Pipeline with Hoeffding Tree
    """
    tree_model = tree.HoeffdingTreeRegressor(
        grace_period=grace_period,
        leaf_prediction=leaf_prediction,
        model_selector_decay=model_selector_decay
    )
    
    if use_scaler:
        return compose.Pipeline(
            preprocessing.StandardScaler(),
            tree_model
        )
    else:
        return tree_model


def create_srp_ensemble(n_models: int = 10,
                       grace_period: int = 50,
                       leaf_prediction: str = "mean",
                       seed: int = 42,
                       use_scaler: bool = True) -> compose.Pipeline:
    """
    Create a Streaming Random Patches (SRP) ensemble.
    
    SRP is like a "Random Forest for streams" - it trains multiple Hoeffding Trees
    in parallel, each on a random subset of features.
    
    Args:
        n_models: Number of trees in the ensemble
        grace_period: Number of instances between split attempts
        leaf_prediction: Prediction method at leaves
        seed: Random seed for reproducibility
        use_scaler: Whether to use StandardScaler preprocessing
        
    Returns:
        River Pipeline with SRP ensemble
    """
    base_tree = tree.HoeffdingTreeRegressor(
        grace_period=grace_period,
        leaf_prediction=leaf_prediction
    )
    
    srp_model = ensemble.SRPRegressor(
        model=base_tree,
        n_models=n_models,
        seed=seed
    )
    
    if use_scaler:
        return compose.Pipeline(
            preprocessing.StandardScaler(),
            srp_model
        )
    else:
        return srp_model


def get_model_by_name(model_name: str, custom_params: dict = None):
    """
    Factory function to create models by name with optional custom parameters.
    
    Args:
        model_name: Name of the model ("baseline", "hoeffding", "srp")
        custom_params: Dictionary of custom parameters to override defaults
                      Can include 'model_type' to override the model_name
        
    Returns:
        Configured model pipeline
    """
    # Allow custom_params to override model_name via 'model_type'
    if custom_params and 'model_type' in custom_params:
        actual_model = custom_params['model_type']
        params = {k: v for k, v in custom_params.items() if k != 'model_type'}
    else:
        actual_model = model_name
        params = custom_params or {}
    
    if actual_model.lower() == "baseline":
        base_params = config.BASELINE_CONFIG.copy()
        base_params.update(params)
        return create_baseline_model(use_scaler=base_params.get("use_scaler", True))
    
    elif actual_model.lower() == "hoeffding" or actual_model.lower() == "tree":
        base_params = config.HOEFFDING_TREE_CONFIG.copy()
        base_params.update(params)
        return create_hoeffding_tree(
            grace_period=base_params.get("grace_period", 50),
            leaf_prediction=base_params.get("leaf_prediction", "mean"),
            model_selector_decay=base_params.get("model_selector_decay", 0.9),
            use_scaler=base_params.get("use_scaler", True)
        )
    
    elif actual_model.lower() == "srp" or actual_model.lower() == "ensemble":
        base_params = config.SRP_CONFIG.copy()
        base_params.update(params)
        return create_srp_ensemble(
            n_models=base_params.get("n_models", 10),
            grace_period=base_params.get("grace_period", 50),
            leaf_prediction=base_params.get("leaf_prediction", "mean"),
            seed=base_params.get("seed", 42),
            use_scaler=base_params.get("use_scaler", True)
        )
    
    else:
        raise ValueError(f"Unknown model name: {actual_model} (from key: {model_name})")


def train_model(model_name: str,
               data_file: str = None,
               feature_transform: str = "log",
               custom_params: dict = None,
               report_frequency: int = None,
               verbose: bool = True) -> Tuple[object, metrics.MAE, List, List]:
    """
    Train a single model on streaming data with prequential evaluation.
    
    Prequential evaluation = Test-Then-Train: For each instance, we first test
    (predict), then update the metric, then train (learn).
    
    Args:
        model_name: Name of the model to train
        data_file: Path to preprocessed CSV file
        feature_transform: Transformation method ("raw", "log", "normalized")
        custom_params: Custom model parameters
        report_frequency: How often to print progress
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_model, final_metric, error_history, instance_history)
    """
    if data_file is None:
        data_file = config.PREPROCESSED_FILE
    if report_frequency is None:
        report_frequency = config.REPORT_FREQUENCY
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {model_name.upper()}")
        print(f"{'='*60}")
        print(f"Data: {data_file}")
        print(f"Feature Transform: {feature_transform}")
        print(f"{'='*60}\n")
    
    # Create model
    model = get_model_by_name(model_name, custom_params)
    
    # Create metric
    metric = metrics.MAE()
    
    # Tracking for visualization
    error_history = []
    instance_history = []
    
    # Create stream
    stream_data = stream.iter_csv(
        data_file,
        target="RUL",
        converters={"RUL": float},
        drop=config.COLUMNS_TO_DROP
    )
    
    # Training loop (Prequential evaluation)
    for i, (x, y) in enumerate(stream_data):
        try:
            # Apply feature transformation
            x = apply_feature_transformation(x, method=feature_transform)
            
            # Test (Predict)
            y_pred = model.predict_one(x)
            
            # Evaluate
            metric.update(y_true=y, y_pred=y_pred)
            
            # Train (Learn)
            model.learn_one(x, y)
            
            # Track for visualization
            if i % config.PLOT_FREQUENCY == 0:
                error_history.append(metric.get())
                instance_history.append(i)
            
            # Report progress
            if verbose and i % report_frequency == 0 and i > 0:
                print(f"Instance {i:,} | Current MAE: {metric.get():.2f} days")
                
        except (ValueError, TypeError) as e:
            if verbose and i < 10:  # Only print first few errors
                print(f"   [WARNING] Error at instance {i}: {e}")
            continue
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Training Complete!")
        print(f"Final MAE: {metric.get():.2f} days")
        print(f"{'='*60}\n")
    
    return model, metric, error_history, instance_history


def compare_models(model_configs: Dict[str, dict],
                  data_file: str = None,
                  feature_transform: str = "log",
                  report_frequency: int = None,
                  save_plot: bool = True,
                  plot_filename: str = "model_comparison.png") -> Dict:
    """
    Compare multiple models on the same streaming data.
    
    This function trains multiple models simultaneously on the same stream,
    allowing for direct performance comparison.
    
    Args:
        model_configs: Dict mapping model names to their custom parameters
                      Example: {"baseline": {}, "hoeffding": {"grace_period": 100}}
        data_file: Path to preprocessed CSV file
        feature_transform: Transformation method
        report_frequency: How often to print progress
        save_plot: Whether to save comparison plot
        plot_filename: Filename for the comparison plot
        
    Returns:
        Dictionary with results for each model
    """
    if data_file is None:
        data_file = config.PREPROCESSED_FILE
    if report_frequency is None:
        report_frequency = config.REPORT_FREQUENCY
    
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Models to compare: {', '.join(model_configs.keys())}")
    print(f"Feature Transform: {feature_transform}")
    print(f"{'='*70}\n")
    
    # Initialize models and metrics
    models = {}
    metrics_dict = {}
    histories = {}
    
    for name, params in model_configs.items():
        models[name] = get_model_by_name(name, params)
        metrics_dict[name] = metrics.MAE()
        histories[name] = {"errors": [], "instances": []}
    
    # Create stream
    stream_data = stream.iter_csv(
        data_file,
        target="RUL",
        converters={"RUL": float},
        drop=config.COLUMNS_TO_DROP
    )
    
    # Training loop
    print("Training all models simultaneously...\n")
    for i, (x, y) in enumerate(stream_data):
        try:
            # Apply feature transformation
            x = apply_feature_transformation(x, method=feature_transform)
            
            # Train each model
            for name in model_configs.keys():
                # Test
                y_pred = models[name].predict_one(x)
                
                # Evaluate
                metrics_dict[name].update(y_true=y, y_pred=y_pred)
                
                # Train
                models[name].learn_one(x, y)
            
            # Track for visualization
            if i % config.PLOT_FREQUENCY == 0:
                for name in model_configs.keys():
                    histories[name]["errors"].append(metrics_dict[name].get())
                    histories[name]["instances"].append(i)
            
            # Report progress
            if i % report_frequency == 0 and i > 0:
                status = " | ".join([f"{name}: {metrics_dict[name].get():.2f}" 
                                    for name in model_configs.keys()])
                print(f"Instance {i:,} | {status}")
                
        except (ValueError, TypeError):
            continue
    
    # Final results
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    
    results = {}
    for name in model_configs.keys():
        final_mae = metrics_dict[name].get()
        results[name] = {
            "model": models[name],
            "mae": final_mae,
            "history": histories[name]
        }
        print(f"{name:20s} | MAE: {final_mae:.2f} days")
    
    print(f"{'='*70}\n")
    
    # Plot comparison
    if save_plot:
        plot_model_comparison(histories, plot_filename)
    
    return results


def plot_model_comparison(histories: Dict, filename: str = "model_comparison.png"):
    """
    Create a comparison plot of model performance over time.
    
    Args:
        histories: Dictionary mapping model names to their error histories
        filename: Output filename for the plot
    """
    plt.figure(figsize=(12, 6))
    
    for name, history in histories.items():
        plt.plot(history["instances"], history["errors"], 
                label=name.upper(), linewidth=2, alpha=0.8)
    
    plt.xlabel("Instances Processed", fontsize=12)
    plt.ylabel("MAE (Days)", fontsize=12)
    plt.title("Model Comparison: Streaming RUL Prediction", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(config.PLOTS_DIR, filename)
    plt.savefig(output_path, dpi=150)
    print(f"✓ Comparison plot saved to: {output_path}\n")
    plt.close()


def plot_single_model_performance(error_history: List, instance_history: List,
                                  model_name: str, filename: str = None):
    """
    Plot the performance of a single model over time.
    
    Args:
        error_history: List of MAE values over time
        instance_history: List of instance counts
        model_name: Name of the model
        filename: Output filename (auto-generated if None)
    """
    if filename is None:
        filename = f"{model_name}_performance.png"
    
    plt.figure(figsize=(10, 6))
    plt.plot(instance_history, error_history, linewidth=2, color='steelblue')
    plt.xlabel("Instances Processed", fontsize=12)
    plt.ylabel("MAE (Days)", fontsize=12)
    plt.title(f"{model_name.upper()} - Learning Curve", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(config.PLOTS_DIR, filename)
    plt.savefig(output_path, dpi=150)
    print(f"✓ Performance plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    """
    Stand-alone execution: Train a single model
    """
    import sys
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "hoeffding"
    
    model, metric, errors, instances = train_model(
        model_name=model_name,
        feature_transform="log",
        verbose=True
    )
    
    plot_single_model_performance(errors, instances, model_name)
