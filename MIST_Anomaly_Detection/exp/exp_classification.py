import torch
import torch.nn as nn
from torch import optim
import os
import time
from datetime import datetime
import numpy as np
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models import (Informer, Autoformer, Transformer, DLinear, Linear, PatchTST, Mist, 
                    FEDformer, Film, Nonstationary_Transformer, SparseTSF)
from data_provider.data_factory import data_provider
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from utils.track import ResourceTracker


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

        # Initialize TensorBoard writer 
        log_name = f"{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.period_len}"
        log_dir = os.path.join('./logs/', log_name)
        os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"Run 'tensorboard --logdir=./logs' to view")

        # Initialize Resource Tracker
        self.resource_tracker = ResourceTracker()
        
    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'MIST': Mist,
            'SparseTSF': SparseTSF,
            'FEDformer': FEDformer,
            'Film': Film,
            'Nonstationary_Transformer': Nonstationary_Transformer
        }
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def save_checkpoint(self, epoch, model_optim, best_val_acc, global_step, early_stopping, path):
        """Save checkpoint for resuming training"""
        checkpoint = {
            'epoch': epoch + 1,  # Next epoch to start from
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'best_val_acc': best_val_acc,
            'global_step': global_step,
            'early_stopping_counter': getattr(early_stopping, 'counter', 0),
            'best_score': getattr(early_stopping, 'best_score', None),
            'best_val_loss': getattr(early_stopping, 'val_loss_min', float('inf')),  # Changed attribute name
            'best_accuracy': getattr(early_stopping, 'best_accuracy', 0),
            'best_epoch': getattr(early_stopping, 'best_epoch', 0),
            'best_metrics': getattr(early_stopping, 'best_metrics', {})
        }
        checkpoint_path = os.path.join(path, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Print dataset information
        print("\n" + "="*80)
        print("DATASET INFORMATION")
        print("="*80)
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(vali_data)}")
        print(f"Test samples: {len(test_data)}")
        
        checkpoint_name = f"{self.args.model_id}_{self.args.model}_{self.args.data}"
        path = os.path.join(self.args.checkpoints, checkpoint_name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        best_val_acc = 0
        global_step = 0
        start_epoch = 0

        print(f"Starting training for {self.args.train_epochs} epochs...")
        print(f"Total batches per epoch: {train_steps}\n")

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=100,
            verbose=True,
            delta=0.0001,
            path=os.path.join(path, 'checkpoint.pth')
        )

        # ============ RESUME LOGIC - CHECKS BOTH CHECKPOINT FILES ============
        latest_checkpoint_path = os.path.join(path, 'latest_checkpoint.pth')
        best_checkpoint_path = os.path.join(path, 'checkpoint.pth')
        
        resume_enabled = hasattr(self.args, 'resume') and self.args.resume
        
        if resume_enabled:
            # Try latest_checkpoint.pth first (has full training state)
            if os.path.exists(latest_checkpoint_path):
                print(f"\n{'='*80}")
                print(f"🔄 RESUMING TRAINING FROM LATEST CHECKPOINT")
                print(f"{'='*80}")
                
                try:
                    checkpoint = torch.load(latest_checkpoint_path, map_location=self.device)
                    
                    # Load full state
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch']
                    best_val_acc = checkpoint['best_val_acc']
                    global_step = checkpoint['global_step']
                    early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
                    early_stopping.best_score = checkpoint.get('best_score', None)
                    early_stopping.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    early_stopping.best_accuracy = checkpoint.get('best_accuracy', 0)
                    early_stopping.best_epoch = checkpoint.get('best_epoch', 0)
                    early_stopping.best_metrics = checkpoint.get('best_metrics', {})
                    
                    print(f"✓ Successfully resumed from epoch {start_epoch}")
                    print(f"✓ Best validation accuracy: {best_val_acc:.4f}")
                    print(f"✓ Best validation loss: {early_stopping.best_val_loss:.6f}")
                    print(f"✓ Early stopping counter: {early_stopping.counter}/{early_stopping.patience}")
                    print(f"✓ Global step: {global_step}")
                    print(f"{'='*80}\n")
                    
                except Exception as e:
                    print(f"❌ Error loading latest checkpoint: {e}")
                    print(f"Starting from scratch...\n")
                    start_epoch = 0
            
            # If latest_checkpoint doesn't exist, try best checkpoint (old format)
            elif os.path.exists(best_checkpoint_path):
                print(f"\n{'='*80}")
                print(f"🔄 RESUMING FROM BEST MODEL CHECKPOINT")
                print(f"{'='*80}")
                
                try:
                    checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
                    
                    # Check format
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        # New format
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                        start_epoch = checkpoint.get('epoch', 0)
                        best_val_acc = checkpoint.get('best_val_acc', 0)
                        print(f"✓ Loaded new-format checkpoint")
                    else:
                        # Old format (just weights)
                        self.model.load_state_dict(checkpoint)
                        print(f"✓ Loaded model weights from old checkpoint")
                        print(f"⚠️  Optimizer state not available - will be reinitialized")
                        start_epoch = 0  # Can't determine epoch from old format
                    
                    print(f"✓ Resuming from epoch {start_epoch}")
                    print(f"{'='*80}\n")
                    
                except Exception as e:
                    print(f"❌ Error loading checkpoint: {e}")
                    print(f"Starting from scratch...\n")
                    start_epoch = 0
            
            else:
                print(f"\n⚠️  --resume flag set but no checkpoint found")
                print(f"Checked: {latest_checkpoint_path}")
                print(f"Checked: {best_checkpoint_path}")
                print(f"Starting training from scratch...\n")

        # start tracking resources for the training process
        self.resource_tracker.start()
        
        # ============ TRAINING LOOP - STARTS FROM start_epoch ============
        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_preds = []
            train_trues = []
            
            self.model.train()
            epoch_time = time.time()

            # Progress bar for training
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.train_epochs} [Train]")
            
            for i, (batch_x, batch_y) in enumerate(pbar):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                outputs = self.model(batch_x)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # Store predictions and true labels for metrics
                pred = torch.argmax(outputs, dim=1)
                train_preds.extend(pred.cpu().numpy())
                train_trues.extend(batch_y.cpu().numpy())

                loss.backward()
                model_optim.step()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log to TensorBoard
                if i % 10 == 0:
                    self.writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
                
                global_step += 1
            
            # Update resource usage at the end of the epoch
            self.resource_tracker.update()
            
            # Calculate training metrics
            train_loss_avg = np.average(train_loss)
            train_acc = accuracy_score(train_trues, train_preds)
            
            # Validation
            vali_loss, vali_acc, vali_precision, vali_recall, vali_f1 = self.vali(
                vali_data, vali_loader, criterion, epoch
            )
            
            # Epoch time
            epoch_duration = time.time() - epoch_time

            # Print epoch results
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.args.train_epochs} Summary:")
            print(f"{'='*80}")
            print(f"Time: {epoch_duration:.2f}s")
            print(f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {vali_loss:.4f} | Val Acc: {vali_acc:.4f}")
            print(f"Val Precision: {vali_precision:.4f} | Val Recall: {vali_recall:.4f} | Val F1: {vali_f1:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar('Train/Loss_Epoch', train_loss_avg, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', vali_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', vali_acc, epoch)
            self.writer.add_scalar('Val/Precision', vali_precision, epoch)
            self.writer.add_scalar('Val/Recall', vali_recall, epoch)
            self.writer.add_scalar('Val/F1', vali_f1, epoch)
            
            # Early stopping check
            metrics = {
                'precision': vali_precision,
                'recall': vali_recall,
                'f1': vali_f1
            }
            early_stopping(vali_loss, vali_acc, self.model, epoch+1, metrics)
            
            # Update best val acc for display
            if vali_acc > best_val_acc:
                best_val_acc = vali_acc
            
            # ============ SAVE CHECKPOINT AFTER EACH EPOCH ============
            self.save_checkpoint(epoch, model_optim, best_val_acc, global_step, early_stopping, path)
            
            print(f"{'='*80}\n")
            
            # Check if early stopping triggered
            if early_stopping.early_stop:
                print(f"\n{'='*80}")
                print(f"🛑 EARLY STOPPING TRIGGERED")
                print(f"{'='*80}")
                best = early_stopping.get_best_metrics()
                print(f"Training stopped at epoch {epoch+1}")
                print(f"Best model was at epoch {best['best_epoch']}")
                print(f"Best validation loss: {best['best_val_loss']:.6f}")
                print(f"Best validation accuracy: {best['accuracy']:.4f}")
                print(f"Best validation F1: {best.get('f1', 0):.4f}")
                print(f"{'='*80}\n")
                break
        
        # Get training resource summary
        train_summary = self.resource_tracker.get_summary()
        
        # Print training resource usage
        print(f"\n{'='*80}")
        print(f"TRAINING RESOURCE USAGE")
        print(f"{'='*80}")
        print(f"Duration: {train_summary['duration_seconds']:.2f} seconds")
        print(f"RAM Usage - Max: {train_summary['ram_max_gb']:.4f} GB, Mean: {train_summary['ram_mean_gb']:.4f} GB")
        print(f"GPU Usage - Max: {train_summary['gpu_max_gb']:.4f} GB, Mean: {train_summary['gpu_mean_gb']:.4f} GB")
        print(f"{'='*80}\n")
        
        # Save training resource logs
        model_name = f"{self.args.model_id}_{self.args.model}"
        log_file = self.resource_tracker.save_log(model_name, "training")
        plot_file = self.resource_tracker.plot_usage(model_name, "training")
        print(f"✓ Training resource log saved: {log_file}")
        print(f"✓ Training resource plot saved: {plot_file}\n")

        # Training summary
        if not early_stopping.early_stop:
            print(f"\n✓ Completed all {self.args.train_epochs} epochs")
        
        best = early_stopping.get_best_metrics()
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Best epoch: {best['best_epoch']}/{self.args.train_epochs}")
        print(f"Best validation loss: {best['best_val_loss']:.6f}")
        print(f"Best validation accuracy: {best['accuracy']:.4f}")
        print(f"Best validation precision: {best.get('precision', 0):.4f}")
        print(f"Best validation recall: {best.get('recall', 0):.4f}")
        print(f"Best validation F1: {best.get('f1', 0):.4f}")
        print(f"{'='*80}\n")
        
        # Load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        print(f"✓ Loaded best model from epoch {best['best_epoch']}\n")
        
        return self.model
    
    def vali(self, vali_data, vali_loader, criterion, epoch=None):
        total_loss = []
        preds = []
        trues = []
        
        self.model.eval()

        desc = f"Epoch {epoch+1} [Val]" if epoch is not None else "Validation"
        pbar = tqdm(vali_loader, desc=desc, leave=False)

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pbar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                outputs = self.model(batch_x)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
                
                preds.append(pred.numpy())
                trues.append(true.numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        total_loss = np.average(total_loss)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        probs = torch.softmax(torch.tensor(preds), dim=1).numpy()
        predictions = np.argmax(probs, axis=1)
        
        accuracy = accuracy_score(trues, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(trues, predictions, average='weighted', zero_division=0)
        
        self.model.train()
        return total_loss, accuracy, precision, recall, f1
    
    def test(self, setting, test=0, flag='test'):
        test_data, test_loader = self._get_data(flag=flag)

        print("\n" + "="*80)
        print("TESTING")
        print("="*80)
        print(f"Test samples: {len(test_data)}")
        
        # Get sample to check shapes
        sample_x, sample_y = next(iter(test_loader))
        print(f"Test batch shapes:")
        print(f"  Input (X): {sample_x.shape}")
        print(f"  Labels (Y): {sample_y.shape}")
        print("="*80 + "\n")
        
        if test:
            print('loading model') 
            checkpoint_name = f"{self.args.model_id}_{self.args.model}_{self.args.data}"
            checkpoint_path = os.path.join(self.args.checkpoints, checkpoint_name, 'checkpoint.pth')
            #self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
            print("model loaded successfully!\n")
        
        preds = []
        trues = []
        probs_list = []
        
        self.model.eval()

        pbar = tqdm(test_loader, desc=f"Testing ({flag})")

        # Start tracking inference
        inference_tracker = ResourceTracker()
        inference_tracker.start()
        inference_start = time.time()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pbar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                outputs = self.model(batch_x)
                
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

                # Update inference resource usage
                if i % 5 == 0:
                    inference_tracker.update()

        inference_tracker.update()  # Final update after inference
        inference_time = time.time() - inference_start

        inference_summary = inference_tracker.get_summary()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        probs = torch.softmax(torch.tensor(preds), dim=1).numpy()
        predictions = np.argmax(probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(trues, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(trues, predictions, average='weighted', zero_division=0)

        # Calculate per-sample inference time
        num_samples = len(test_data)
        time_per_sample = inference_time / num_samples * 1000  # in milliseconds
        throughput = num_samples / inference_time
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Test Accuracy:  {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall:    {recall:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")
        print("="*80)

        print("\n" + "="*80)
        print("INFERENCE TIME")
        print("="*80)
        print(f"Total Inference Time: {inference_time:.4f} seconds")
        print(f"Total Samples: {num_samples}")
        print(f"Time per Sample: {time_per_sample:.4f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print("="*80)
        
        print("\n" + "="*80)
        print("INFERENCE RESOURCE USAGE")
        print("="*80)
        print(f"Duration: {inference_summary['duration_seconds']:.2f} seconds")
        print(f"RAM Usage - Max: {inference_summary['ram_max_gb']:.4f} GB, Mean: {inference_summary['ram_mean_gb']:.4f} GB")
        print(f"GPU Usage - Max: {inference_summary['gpu_max_gb']:.4f} GB, Mean: {inference_summary['gpu_mean_gb']:.4f} GB")
        print("="*80 + "\n")
        
        # Save inference resource logs
        model_name = f"{self.args.model_id}_{self.args.model}"
        test_set_name = self.args.data
        log_file = inference_tracker.save_log(model_name, test_set_name)
        plot_file = inference_tracker.plot_usage(model_name, test_set_name)
        print(f"✓ Inference resource log saved: {log_file}")
        print(f"✓ Inference resource plot saved: {plot_file}\n")
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(trues, predictions, average=None, zero_division=0)
        
        print("\nPer-Class Metrics:")
        print("-"*80)
        for i in range(len(precision_per_class)):
            print(f"Class {i}:")
            print(f"  Precision: {precision_per_class[i]:.4f}")
            print(f"  Recall:    {recall_per_class[i]:.4f}")
            print(f"  F1-Score:  {f1_per_class[i]:.4f}")
            print(f"  Support:   {support[i]}")
        
        # Confusion matrix
        cm = confusion_matrix(trues, predictions)
        print("\n" + "-"*80)
        print("Confusion Matrix:")
        print("-"*80)
        print(cm)
        print("="*80 + "\n")

        # Log to TensorBoard
        self.writer.add_scalar('Test/Accuracy', accuracy, 0)
        self.writer.add_scalar('Test/Precision', precision, 0)
        self.writer.add_scalar('Test/Recall', recall, 0)
        self.writer.add_scalar('Test/F1', f1, 0)

        self.writer.add_scalar('Test/Inference_Time_Total', inference_time, 0)
        self.writer.add_scalar('Test/Inference_Time_Per_Sample_ms', time_per_sample, 0)
        self.writer.add_scalar('Test/Throughput_samples_per_sec', throughput, 0)
        self.writer.add_scalar('Test/RAM_Max_GB', inference_summary['ram_max_gb'], 0)
        self.writer.add_scalar('Test/RAM_Mean_GB', inference_summary['ram_mean_gb'], 0)
        self.writer.add_scalar('Test/GPU_Max_GB', inference_summary['gpu_max_gb'], 0)
        self.writer.add_scalar('Test/GPU_Mean_GB', inference_summary['gpu_mean_gb'], 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        self.writer.add_figure('Test/Confusion_Matrix', fig, 0)
        plt.close()
        
        # Save results
        results_name = f"{self.args.model_id}_{self.args.model}_{self.args.data}"
        folder_path = os.path.join('./results/', results_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save as TXT files
        print(f"Saving results to {folder_path}...")
        
        # Save predictions
        with open(folder_path + 'predictions.txt', 'w') as f:
            f.write("Sample_ID,Predicted_Class,True_Class,Probability_Class_0,Probability_Class_1\n")
            for idx, (pred, true) in enumerate(zip(predictions, trues)):
                prob_str = ','.join([f'{p:.6f}' for p in probs[idx]])
                f.write(f"{idx},{pred},{true},{prob_str}\n")
        
        # Save summary metrics
        with open(folder_path + 'test_metrics.txt', 'w') as f:
            f.write("TEST METRICS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  Accuracy:  {accuracy:.6f}\n")
            f.write(f"  Precision: {precision:.6f}\n")
            f.write(f"  Recall:    {recall:.6f}\n")
            f.write(f"  F1-Score:  {f1:.6f}\n\n")

            f.write(f"Inference Time:\n")
            f.write(f"  Total Time: {inference_time:.6f} seconds\n")
            f.write(f"  Time per Sample: {time_per_sample:.6f} ms\n")
            f.write(f"  Throughput: {throughput:.2f} samples/sec\n\n")
            
            f.write(f"Resource Usage:\n")
            f.write(f"  Duration: {inference_summary['duration_seconds']:.2f} seconds\n")
            f.write(f"  RAM Max: {inference_summary['ram_max_gb']:.4f} GB\n")
            f.write(f"  RAM Mean: {inference_summary['ram_mean_gb']:.4f} GB\n")
            f.write(f"  GPU Max: {inference_summary['gpu_max_gb']:.4f} GB\n")
            f.write(f"  GPU Mean: {inference_summary['gpu_mean_gb']:.4f} GB\n\n")
            
            f.write(f"Per-Class Metrics:\n")
            for i in range(len(precision_per_class)):
                f.write(f"\nClass {i}:\n")
                f.write(f"  Precision: {precision_per_class[i]:.6f}\n")
                f.write(f"  Recall:    {recall_per_class[i]:.6f}\n")
                f.write(f"  F1-Score:  {f1_per_class[i]:.6f}\n")
                f.write(f"  Support:   {support[i]}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(str(cm))
        
        # Save confusion matrix
        with open(folder_path + 'confusion_matrix.txt', 'w') as f:
            f.write("Confusion Matrix\n")
            f.write("="*50 + "\n")
            f.write("Rows: True labels\n")
            f.write("Columns: Predicted labels\n\n")
            np.savetxt(f, cm, fmt='%d')
        
        print(f"✓ predictions.txt saved ({len(predictions)} samples)")
        print(f"✓ test_metrics.txt saved")
        print(f"✓ confusion_matrix.txt saved")
        print(f"\nAll results saved to: {folder_path}\n")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return accuracy, precision, recall, f1