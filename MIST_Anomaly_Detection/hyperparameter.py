"""
Improved Hyperparameter Tuning Script for SparseTSF Classification
With better error handling, progress tracking, and visualization
"""

import subprocess
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class HyperparameterTuner:
    def __init__(self, base_args):
        self.base_args = base_args
        self.results = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = f"tuning_results_{self.timestamp}.csv"
        self.log_file = f"tuning_log_{self.timestamp}.txt"
        self.total_experiments = 0
        self.completed_experiments = 0
        
    def run_experiment(self, config, experiment_name):
        """Run a single experiment with given configuration"""
        self.total_experiments += 1
        
        print("\n" + "="*80)
        print(f"EXPERIMENT {self.completed_experiments + 1}/{self.total_experiments}: {experiment_name}")
        print("="*80)
        print(f"Configuration: {config}")
        print("="*80 + "\n")
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write(f"Config: {config}\n")
            f.write(f"{'='*80}\n")
        
        # Build command
        cmd = ["python", "run_classification.py"]
        
        # Add base arguments
        for key, value in self.base_args.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Add experiment-specific arguments
        for key, value in config.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Update model_id and scaler_path for this experiment
        cmd.extend(["--model_id", f"tune_{experiment_name}"])
        cmd.extend(["--scaler_path", f"./scalers/veremi_scaler_{experiment_name}.pkl"])
        
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run experiment
        start_time = datetime.now()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results from output
            output = result.stdout
            stderr = result.stderr
            
            # Save full output to log
            with open(self.log_file, 'a') as f:
                f.write(f"\nSTDOUT:\n{output}\n")
                if stderr:
                    f.write(f"\nSTDERR:\n{stderr}\n")
                f.write(f"\nCompleted: {end_time}\n")
                f.write(f"Duration: {duration:.2f}s\n")
            
            # Extract metrics
            val_acc = self._extract_metric(output, "Best validation accuracy:")
            val_f1 = self._extract_metric(output, "Best validation F1:")
            test_acc = self._extract_metric(output, "Test Accuracy:")
            test_f1 = self._extract_metric(output, "Test F1-Score:")
            inference_time = self._extract_metric(output, "Time per Sample:")
            
            # Check if extraction was successful
            if val_acc is None or test_acc is None:
                print(f"⚠️ Could not extract metrics from output")
                with open(self.log_file, 'a') as f:
                    f.write(f"WARNING: Could not extract metrics\n")
                return None
            
            # Store results
            result_dict = {
                'experiment': experiment_name,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'inference_time_ms': inference_time,
                'duration_seconds': duration,
                'timestamp': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                **config
            }
            
            self.results.append(result_dict)
            self.completed_experiments += 1
            self._save_results()
            
            print(f"\n✅ Experiment completed successfully")
            print(f"  Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            print(f"  Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
            print(f"  Inference: {inference_time:.2f} ms/sample" if inference_time else "")
            print(f"  Duration: {duration:.2f}s")
            
            return result_dict
            
        except subprocess.TimeoutExpired:
            print(f"❌ Experiment timed out (>2 hours)")
            with open(self.log_file, 'a') as f:
                f.write(f"ERROR: Timeout\n")
            return None
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            with open(self.log_file, 'a') as f:
                f.write(f"ERROR: {e}\n")
            return None
    
    def _extract_metric(self, output, metric_name):
        """Extract metric value from output string"""
        try:
            for line in output.split('\n'):
                if metric_name in line:
                    parts = line.split(metric_name)
                    if len(parts) > 1:
                        value_str = parts[1].strip().split()[0]
                        return float(value_str)
        except:
            pass
        return None
    
    def _save_results(self):
        """Save results to CSV"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        print(f"📊 Results saved to: {self.results_file}")
    
    def count_experiments(self, configs_list):
        """Count total experiments to be run"""
        return sum(len(configs) for configs in configs_list)
    
    def tune_sequence_length(self):
        """Tune sequence length and period length"""
        print("\n" + "🔍 PHASE 1: TUNING SEQUENCE LENGTH AND PERIOD LENGTH")
        
        configs = [
            # (seq_len, period_len) - must be divisible
            (48, 12),
            (48, 16),
            (64, 16),
            (96, 12),
            (96, 16),
            (96, 24),
            (96, 32),
            (128, 16),
            (128, 32),
            (192, 24),
            (192, 48),
        ]
        
        print(f"Testing {len(configs)} configurations...")
        
        for seq_len, period_len in configs:
            config = {
                'seq_len': seq_len,
                'period_len': period_len
            }
            self.run_experiment(config, f"seq{seq_len}_per{period_len}")
    
    def tune_learning_rate(self):
        """Tune learning rate"""
        print("\n" + "🔍 PHASE 2: TUNING LEARNING RATE")
        
        learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
        
        print(f"Testing {len(learning_rates)} learning rates...")
        
        for lr in learning_rates:
            config = {'learning_rate': lr}
            self.run_experiment(config, f"lr{lr}")
    
    def tune_batch_size(self):
        """Tune batch size"""
        print("\n" + "🔍 PHASE 3: TUNING BATCH SIZE")
        
        batch_sizes = [64, 128, 256, 512]
        
        print(f"Testing {len(batch_sizes)} batch sizes...")
        
        for bs in batch_sizes:
            config = {'batch_size': bs}
            self.run_experiment(config, f"bs{bs}")
    
    def tune_combined(self):
        """Tune multiple parameters together"""
        print("\n" + "🔍 PHASE 4: COMBINED TUNING (TOP CONFIGURATIONS)")
        
        # Based on initial results, test best combinations
        configs = [
            {'seq_len': 128, 'period_len': 32, 'learning_rate': 0.0005, 'batch_size': 128},
            {'seq_len': 96, 'period_len': 24, 'learning_rate': 0.0003, 'batch_size': 256},
            {'seq_len': 128, 'period_len': 32, 'learning_rate': 0.001, 'batch_size': 256},
            {'seq_len': 96, 'period_len': 16, 'learning_rate': 0.0005, 'batch_size': 128},
        ]
        
        print(f"Testing {len(configs)} combined configurations...")
        
        for i, config in enumerate(configs):
            self.run_experiment(config, f"combined_{i+1}")
    
    def plot_results(self):
        """Generate visualization plots"""
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Test Accuracy by Experiment
        ax1 = fig.add_subplot(gs[0, :])
        x = range(len(df))
        ax1.bar(x, df['test_accuracy'], alpha=0.7, color='steelblue')
        ax1.axhline(y=df['test_accuracy'].mean(), color='r', linestyle='--', label='Mean')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Test Accuracy Across All Experiments', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['experiment'], rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Seq Length vs Accuracy
        ax2 = fig.add_subplot(gs[1, 0])
        if 'seq_len' in df.columns:
            seq_grouped = df.groupby('seq_len')['test_accuracy'].mean()
            ax2.bar(seq_grouped.index, seq_grouped.values, alpha=0.7, color='green')
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Mean Test Accuracy')
            ax2.set_title('Sequence Length Impact')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate vs Accuracy
        ax3 = fig.add_subplot(gs[1, 1])
        if 'learning_rate' in df.columns:
            lr_data = df.dropna(subset=['learning_rate'])
            if len(lr_data) > 0:
                ax3.scatter(lr_data['learning_rate'], lr_data['test_accuracy'], s=100, alpha=0.6)
                ax3.set_xlabel('Learning Rate')
                ax3.set_ylabel('Test Accuracy')
                ax3.set_title('Learning Rate Impact')
                ax3.set_xscale('log')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Batch Size vs Accuracy
        ax4 = fig.add_subplot(gs[1, 2])
        if 'batch_size' in df.columns:
            bs_grouped = df.groupby('batch_size')['test_accuracy'].mean()
            ax4.bar(bs_grouped.index, bs_grouped.values, alpha=0.7, color='orange')
            ax4.set_xlabel('Batch Size')
            ax4.set_ylabel('Mean Test Accuracy')
            ax4.set_title('Batch Size Impact')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Accuracy vs Inference Time
        ax5 = fig.add_subplot(gs[2, 0])
        valid_data = df.dropna(subset=['inference_time_ms', 'test_accuracy'])
        if len(valid_data) > 0:
            scatter = ax5.scatter(valid_data['inference_time_ms'], valid_data['test_accuracy'],
                                 s=100, alpha=0.6, c=range(len(valid_data)), cmap='viridis')
            ax5.set_xlabel('Inference Time (ms/sample)')
            ax5.set_ylabel('Test Accuracy')
            ax5.set_title('Accuracy vs Speed Trade-off')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: F1 Score Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(df['test_f1'].dropna(), bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(x=df['test_f1'].mean(), color='r', linestyle='--', label=f'Mean: {df["test_f1"].mean():.4f}')
        ax6.set_xlabel('Test F1 Score')
        ax6.set_ylabel('Frequency')
        ax6.set_title('F1 Score Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Top 5 Configurations
        ax7 = fig.add_subplot(gs[2, 2])
        top5 = df.nlargest(5, 'test_accuracy')
        ax7.barh(range(len(top5)), top5['test_accuracy'], alpha=0.7, color='gold')
        ax7.set_yticks(range(len(top5)))
        ax7.set_yticklabels(top5['experiment'], fontsize=8)
        ax7.set_xlabel('Test Accuracy')
        ax7.set_title('Top 5 Configurations')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle(f'Hyperparameter Tuning Results - {self.timestamp}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plot_file = f'tuning_plots_{self.timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n📈 Plots saved to: {plot_file}")
        plt.close()
    
    def get_best_config(self):
        """Get best configuration based on validation accuracy"""
        if not self.results:
            print("No results available")
            return None
        
        df = pd.DataFrame(self.results)
        best_idx = df['val_accuracy'].idxmax()
        best_config = df.loc[best_idx]
        
        print("\n" + "="*80)
        print("🏆 BEST CONFIGURATION (by validation accuracy)")
        print("="*80)
        print(f"Experiment: {best_config['experiment']}")
        print(f"Val Accuracy: {best_config['val_accuracy']:.4f}")
        print(f"Val F1: {best_config['val_f1']:.4f}")
        print(f"Test Accuracy: {best_config['test_accuracy']:.4f}")
        print(f"Test F1: {best_config['test_f1']:.4f}")
        if best_config['inference_time_ms']:
            print(f"Inference Time: {best_config['inference_time_ms']:.2f} ms/sample")
        print(f"\nConfiguration:")
        for key, value in best_config.items():
            if key not in ['experiment', 'val_accuracy', 'val_f1', 'test_accuracy', 
                          'test_f1', 'inference_time_ms', 'duration_seconds', 'timestamp']:
                print(f"  {key}: {value}")
        print("="*80)
        
        return best_config
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results available")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("📊 TUNING SUMMARY STATISTICS")
        print("="*80)
        print(f"Total experiments: {len(df)}")
        print(f"Successful: {self.completed_experiments}")
        print(f"\nTest Accuracy:")
        print(f"  Mean: {df['test_accuracy'].mean():.4f}")
        print(f"  Std: {df['test_accuracy'].std():.4f}")
        print(f"  Min: {df['test_accuracy'].min():.4f}")
        print(f"  Max: {df['test_accuracy'].max():.4f}")
        print(f"\nTest F1:")
        print(f"  Mean: {df['test_f1'].mean():.4f}")
        print(f"  Std: {df['test_f1'].std():.4f}")
        print(f"  Min: {df['test_f1'].min():.4f}")
        print(f"  Max: {df['test_f1'].max():.4f}")
        print(f"\nImprovement from baseline:")
        baseline_acc = df.iloc[0]['test_accuracy'] if len(df) > 0 else 0
        best_acc = df['test_accuracy'].max()
        improvement = ((best_acc - baseline_acc) / baseline_acc) * 100
        print(f"  Baseline: {baseline_acc:.4f}")
        print(f"  Best: {best_acc:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print("="*80)


# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    # Base arguments (these remain constant)
    base_args = {
        'task_name': 'classification',
        'is_training': 1,
        'model': 'SparseTSF',
        'data': 'VeReMi',
        'root_path': '/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Data/',
        'data_path': 'veremi_extension_cleaned.csv',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'enc_in': 8,
        'num_classes': 2,
        'num_workers': 20,
        'train_epochs': 50,  # Reduce for faster tuning
        'use_gpu': True,  # Set based on availability
        'seed': 2021
    }
    
    print("="*80)
    print("🚀 HYPERPARAMETER TUNING FOR SPARSET SF")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base configuration: {base_args}")
    print("="*80 + "\n")
    
    # verify if this script use GPU
    if base_args['use_gpu']:
        print("Using GPU for experiments\n")
    else:
        print("Using CPU for experiments\n")


    tuner = HyperparameterTuner(base_args)
    
    print(f"Results will be saved to: {tuner.results_file}")
    print(f"Logs will be saved to: {tuner.log_file}\n")
    
    # Run tuning experiments
    # UNCOMMENT THE PHASES YOU WANT TO RUN:
    
    # Phase 1: Sequence length tuning 
    #tuner.tune_sequence_length()
    
    # Phase 2: Learning rate tuning 
    tuner.tune_learning_rate()
    
    # Phase 3: Batch size tuning 
    # tuner.tune_batch_size()
    
    # Phase 4: Combined tuning 
    # tuner.tune_combined()
    
    # Generate plots and summary
    tuner.plot_results()
    tuner.print_summary()
    tuner.get_best_config()
    
    print("\n" + "="*80)
    print("✅ HYPERPARAMETER TUNING COMPLETED")
    print("="*80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results: {tuner.results_file}")
    print(f"Plots: tuning_plots_{tuner.timestamp}.png")
    print(f"Logs: {tuner.log_file}")
    print("="*80)