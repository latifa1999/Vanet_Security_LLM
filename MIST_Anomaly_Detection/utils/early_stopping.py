import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
        # Additional tracking
        self.best_epoch = 0
        self.best_metrics = {}
        
    def __call__(self, val_loss, val_acc, model, epoch, metrics=None):
        """
        Call this after each epoch to check if we should stop early.
        
        Args:
            val_loss: Current validation loss
            val_acc: Current validation accuracy
            model: PyTorch model to save
            epoch: Current epoch number
            metrics: Dictionary of additional metrics to track
        """
        score = -val_loss  # We want to maximize this (minimize loss)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model, epoch, metrics)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'⚠️  EarlyStopping counter: {self.counter} out of {self.patience}')
                self.trace_func(f'   No improvement in validation loss for {self.counter} epoch(s)')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func(f'\n🛑 Early stopping triggered!')
                    self.trace_func(f'   Best epoch was: {self.best_epoch}')
                    self.trace_func(f'   Best validation loss: {self.val_loss_min:.6f}')
                    if self.best_metrics:
                        self.trace_func(f'   Best validation accuracy: {self.best_metrics.get("accuracy", 0):.4f}')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model, epoch, metrics)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, val_acc, model, epoch, metrics):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.val_loss_min == np.Inf:
                self.trace_func(f'✓ Validation loss initialized: {val_loss:.6f}. Saving model...')
            else:
                improvement = self.val_loss_min - val_loss
                self.trace_func(f'✓ Validation loss improved: {self.val_loss_min:.6f} → {val_loss:.6f} '
                              f'(↓ {improvement:.6f}). Saving model...')
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_epoch = epoch
        
        # Store best metrics
        if metrics:
            self.best_metrics = metrics.copy()
        self.best_metrics['accuracy'] = val_acc
        self.best_metrics['loss'] = val_loss
        
    def get_best_metrics(self):
        """Returns dictionary of best metrics achieved"""
        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.val_loss_min,
            **self.best_metrics
        }