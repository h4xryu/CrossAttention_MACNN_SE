"""
Trainer Class

학습 과정을 관리하는 통합 Trainer 클래스입니다.

사용 예시:
    from src.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        exp_dir=exp_dir
    )

    trainer.fit(epochs=100)
    metrics = trainer.evaluate(test_loader)
"""

import os
os.environ["TENSORBOARD_NO_TF"] = "1"
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)


class Trainer:
    """
    ECG Classification Trainer

    Args:
        model: PyTorch model
        train_loader: Training data loader
        valid_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: LR scheduler (optional)
        device: torch.device
        exp_dir: Experiment output directory
        gradient_clip: Gradient clipping value (optional)
        save_every: Save checkpoint every N epochs (0 = don't save)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device = None,
        exp_dir: str = "./results",
        gradient_clip: Optional[float] = None,
        save_every: int = 0,
        class_names: List[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cpu')
        self.exp_dir = exp_dir
        self.gradient_clip = gradient_clip
        self.save_every = save_every
        self.class_names = class_names or ['N', 'S', 'V', 'F']

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_metrics = {
            'macro_auprc': {'value': 0.0, 'epoch': 0},
            'macro_auroc': {'value': 0.0, 'epoch': 0},
            'macro_recall': {'value': 0.0, 'epoch': 0},
        }
        self.history = {'train': [], 'valid': []}

        # Directories
        self.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        self.best_model_dir = os.path.join(exp_dir, 'best_models')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

        # TensorBoard Logger
        self.log_dir = os.path.join(exp_dir, 'runs')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train_one_epoch(self) -> Tuple[float, Dict[str, Any]]:
        """
        Train for one epoch.

        Returns:
            (loss, metrics_dict) - metrics는 간소화된 버전 (acc만)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Timing
        data_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        batch_start = time.time()

        for i, batch in enumerate(self.train_loader):
            data_time += time.time() - batch_start
            batch_start = time.time()
            # Unpack batch: (ecg, label, rr, pid, sid)
            ecg, labels, rr_features, *_ = batch
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)
            rr_features = rr_features.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            forward_start = time.time()
            logits, _ = self.model(ecg, rr_features)
            forward_time += time.time() - forward_start

            # Loss
            loss = self.criterion(logits, labels)

            # Backward
            backward_start = time.time()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            # Step scheduler (if per-iteration)
            if self.scheduler is not None:
                self.scheduler.step()
            backward_time += time.time() - backward_start

            # Simple accuracy (no sklearn overhead)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()
            
            batch_start = time.time()
            
            # Print timing breakdown every 20 batches (more frequent for debugging)
            if (i + 1) % 20 == 0:
                avg_data = data_time / (i + 1)
                avg_forward = forward_time / (i + 1)
                avg_backward = backward_time / (i + 1)
                total_batch_time = avg_data + avg_forward + avg_backward
                print(f"  [Batch {i+1}/{len(self.train_loader)}] "
                      f"Data: {avg_data*1000:.1f}ms, "
                      f"Forward: {avg_forward*1000:.1f}ms, "
                      f"Backward: {avg_backward*1000:.1f}ms, "
                      f"Total: {total_batch_time*1000:.1f}ms")

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total

        # 간소화된 metrics (train은 loss/acc만 필요)
        metrics = {
            'acc': acc,
            'macro_f1': 0.0,  # placeholder
            'macro_auprc': 0.0,
            'macro_auroc': 0.0,
        }

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Tuple[float, Dict[str, Any]]:
        """
        Validate on given loader - calculates accuracy, AUPRC, and AUROC.

        Args:
            loader: DataLoader (default: self.valid_loader)

        Returns:
            (loss, metrics_dict)
        """
        loader = loader or self.valid_loader
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_probs = []

        for batch in loader:
            ecg, labels, rr_features, *_ = batch
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)
            rr_features = rr_features.to(self.device)

            logits, _ = self.model(ecg, rr_features)
            loss = self.criterion(logits, labels)

            # Calculate probabilities for AUPRC/AUROC
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Collect for metrics calculation
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            total_loss += loss.item()

        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(loader)

        # Calculate AUPRC and AUROC
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs, axis=0)
        macro_auprc, macro_auroc = self._calculate_auc_metrics(all_labels, all_probs, all_probs.shape[1])
        
        # Calculate macro_recall for best model selection
        all_preds = all_probs.argmax(axis=1)
        _, macro_recall, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        metrics = {
            'acc': acc,
            'macro_f1': 0.0,  # placeholder (not needed for best model selection)
            'macro_auprc': macro_auprc,
            'macro_auroc': macro_auroc,
            'macro_recall': macro_recall,
        }

        return avg_loss, metrics

    def fit(
        self,
        epochs: int,
        early_stopping: bool = False,
        patience: int = 20,
        monitor: str = 'macro_auprc'
    ):
        """
        Full training loop.

        Args:
            epochs: Number of epochs
            early_stopping: Enable early stopping
            patience: Patience for early stopping
            monitor: Metric to monitor for early stopping
        """
        print("\n" + "=" * 60)
        print(f"Training for {epochs} epochs")
        print("=" * 60)

        no_improve = 0
        best_monitor_value = 0.0

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_start = time.time()
            train_loss, train_metrics = self.train_one_epoch()
            train_time = time.time() - train_start
            self.history['train'].append({'loss': train_loss, **train_metrics})

            # Validate
            valid_start = time.time()
            valid_loss, valid_metrics = self.validate()
            valid_time = time.time() - valid_start
            self.history['valid'].append({'loss': valid_loss, **valid_metrics})

            epoch_time = time.time() - epoch_start

            # Print epoch summary with timing breakdown
            self._print_epoch_summary(
                epoch, epochs,
                train_loss, train_metrics,
                valid_loss, valid_metrics,
                epoch_time, train_time, valid_time
            )

            # TensorBoard logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self._log_epoch(epoch, train_loss, train_metrics, current_lr, phase='train')
            self._log_epoch(epoch, valid_loss, valid_metrics, current_lr, phase='valid')

            # Update best models
            self._update_best_models(valid_metrics)

            # Early stopping check
            if early_stopping:
                current_value = valid_metrics.get(monitor, 0)
                if current_value > best_monitor_value:
                    best_monitor_value = current_value
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

            # Save checkpoint
            if self.save_every > 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, valid_metrics)

        # Save final model
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        self._save_checkpoint(epoch, valid_metrics, final_path)

        # Close TensorBoard writer
        self.writer.close()

        print("\n" + "=" * 60)
        print("Training Complete!")
        self._print_best_models()
        print("=" * 60)

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set with full metrics.

        Args:
            loader: Test data loader

        Returns:
            Full metrics dictionary including per-class, confusion matrix
        """
        self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                ecg, labels, rr_features, *_ = batch
                ecg = ecg.to(self.device)
                labels = labels.to(self.device)
                rr_features = rr_features.to(self.device)

                logits, _ = self.model(ecg, rr_features)
                loss = self.criterion(logits, labels)

                probs = torch.softmax(logits, dim=1)

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Concatenate all results
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs, axis=0)
        y_pred = y_prob.argmax(axis=1)

        # Calculate full metrics using _calculate_metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        metrics['loss'] = avg_loss

        return metrics

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        num_classes = y_prob.shape[1]

        # Basic metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

        # Per-class accuracy (TP + TN) / Total
        per_class_acc = np.zeros(num_classes)
        for i in range(num_classes):
            tp = cm[i, i]
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            per_class_acc[i] = (tp + tn) / np.sum(cm)

        # Per-class specificity: TN / (TN + FP)
        per_class_specificity = np.zeros(num_classes)
        for i in range(num_classes):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            per_class_specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision, Recall, F1 (per-class and macro)
        prec, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        macro_prec, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # Weighted metrics
        weighted_prec, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Class weights for weighted averaging
        class_counts = np.bincount(y_true, minlength=num_classes)
        class_weights = class_counts / len(y_true)

        # Macro and weighted specificity/accuracy
        macro_specificity = np.mean(per_class_specificity)
        weighted_specificity = np.sum(per_class_specificity * class_weights)
        macro_accuracy = np.mean(per_class_acc)
        weighted_accuracy = np.sum(per_class_acc * class_weights)

        # AUPRC and AUROC
        macro_auprc, macro_auroc = self._calculate_auc_metrics(y_true, y_prob, num_classes)

        return {
            'acc': acc,
            # Per-class metrics
            'per_class_acc': per_class_acc,
            'per_class_precision': prec,
            'per_class_recall': recall,
            'per_class_specificity': per_class_specificity,
            'per_class_f1': f1,
            # Macro metrics
            'macro_accuracy': macro_accuracy,
            'macro_precision': macro_prec,
            'macro_recall': macro_recall,
            'macro_specificity': macro_specificity,
            'macro_f1': macro_f1,
            'macro_auprc': macro_auprc,
            'macro_auroc': macro_auroc,
            # Weighted metrics
            'weighted_accuracy': weighted_accuracy,
            'weighted_precision': weighted_prec,
            'weighted_recall': weighted_recall,
            'weighted_specificity': weighted_specificity,
            'weighted_f1': weighted_f1,
            # Confusion matrix
            'confusion_matrix': cm,
        }

    def _calculate_auc_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_classes: int
    ) -> Tuple[float, float]:
        """Calculate AUPRC and AUROC."""
        auprc_list, auroc_list = [], []

        for c in range(num_classes):
            binary_true = (y_true == c).astype(int)

            if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
                try:
                    auprc = average_precision_score(binary_true, y_prob[:, c])
                    auroc = roc_auc_score(binary_true, y_prob[:, c])
                    auprc_list.append(auprc)
                    auroc_list.append(auroc)
                except Exception:
                    pass

        macro_auprc = np.mean(auprc_list) if auprc_list else 0.0
        macro_auroc = np.mean(auroc_list) if auroc_list else 0.0

        return macro_auprc, macro_auroc

    def _update_best_models(self, metrics: Dict[str, Any]):
        """Update and save best models."""
        metrics_to_check = ['macro_auprc', 'macro_auroc', 'macro_recall']

        for metric_name in metrics_to_check:
            current_value = metrics.get(metric_name, 0)

            if current_value > self.best_metrics[metric_name]['value']:
                self.best_metrics[metric_name] = {
                    'value': current_value,
                    'epoch': self.current_epoch
                }
                # Save best model
                path = os.path.join(self.best_model_dir, f'best_{metric_name}.pth')
                self._save_checkpoint(self.current_epoch, metrics, path)

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        path: str = None
    ):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')

        # 디렉토리 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metrics': self.best_metrics,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def load_best_model(self, metric: str = 'macro_auprc') -> bool:
        """Load best model for given metric.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        path = os.path.join(self.best_model_dir, f'best_{metric}.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model (best {metric})")
            return True
        else:
            print(f"Best model for {metric} not found")
            return False

    def _log_epoch(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, Any],
        lr: float,
        phase: str = 'train'
    ):
        """Log metrics to TensorBoard."""
        prefix = phase.capitalize()

        # Loss and LR
        self.writer.add_scalar(f'{prefix}/Loss', loss, epoch)
        self.writer.add_scalar(f'{prefix}/LR', lr, epoch)

        # Accuracy
        self.writer.add_scalar(f'{prefix}/Accuracy', metrics.get('acc', 0), epoch)

        # AUPRC / AUROC (if available)
        if metrics.get('macro_auprc', 0) > 0:
            self.writer.add_scalar(f'{prefix}/AUPRC/macro', metrics['macro_auprc'], epoch)
        if metrics.get('macro_auroc', 0) > 0:
            self.writer.add_scalar(f'{prefix}/AUROC/macro', metrics['macro_auroc'], epoch)

        # Macro metrics (if available)
        if metrics.get('macro_f1', 0) > 0:
            self.writer.add_scalar(f'{prefix}/Macro/f1', metrics['macro_f1'], epoch)
        if metrics.get('macro_recall', 0) > 0:
            self.writer.add_scalar(f'{prefix}/Macro/recall', metrics['macro_recall'], epoch)
        if metrics.get('macro_precision', 0) > 0:
            self.writer.add_scalar(f'{prefix}/Macro/precision', metrics['macro_precision'], epoch)

        # Weighted metrics (if available)
        if metrics.get('weighted_f1', 0) > 0:
            self.writer.add_scalar(f'{prefix}/Weighted/f1', metrics['weighted_f1'], epoch)
        if metrics.get('weighted_auprc', 0) > 0:
            self.writer.add_scalar(f'{prefix}/AUPRC/weighted', metrics['weighted_auprc'], epoch)
        if metrics.get('weighted_auroc', 0) > 0:
            self.writer.add_scalar(f'{prefix}/AUROC/weighted', metrics['weighted_auroc'], epoch)

        # Per-class metrics (if available)
        if 'per_class_acc' in metrics:
            for i, name in enumerate(self.class_names):
                if i < len(metrics['per_class_acc']):
                    self.writer.add_scalar(f'{prefix}/PerClass/{name}/accuracy', metrics['per_class_acc'][i], epoch)
        if 'per_class_f1' in metrics:
            for i, name in enumerate(self.class_names):
                if i < len(metrics['per_class_f1']):
                    self.writer.add_scalar(f'{prefix}/PerClass/{name}/f1', metrics['per_class_f1'][i], epoch)

    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_metrics: Dict,
        valid_loss: float,
        valid_metrics: Dict,
        elapsed: float,
        train_time: float = None,
        valid_time: float = None
    ):
        """Print epoch summary."""
        lr = self.optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{total_epochs} ({elapsed:.1f}s) | LR: {lr:.6f}")
        if train_time is not None and valid_time is not None:
            print(f"  [Timing] Train: {train_time:.1f}s, Valid: {valid_time:.1f}s, Other: {elapsed-train_time-valid_time:.1f}s")
        print("-" * 60)
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f}")
        print(f"  Valid | Loss: {valid_loss:.4f} | Acc: {valid_metrics['acc']:.4f} | "
              f"AUPRC: {valid_metrics['macro_auprc']:.4f} | AUROC: {valid_metrics['macro_auroc']:.4f}")

    def _print_best_models(self):
        """Print best model summary."""
        print("\nBest Models:")
        for metric, info in self.best_metrics.items():
            print(f"  {metric}: {info['value']:.4f} (epoch {info['epoch']})")

    def print_metrics(self, metrics: Dict[str, Any], title: str = "Metrics"):
        """Print detailed metrics."""
        print(f"\n{title}")
        print("=" * 60)
        print(f"Accuracy: {metrics['acc']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro AUPRC: {metrics['macro_auprc']:.4f}")
        print(f"Macro AUROC: {metrics['macro_auroc']:.4f}")

        print("\nPer-class metrics:")
        print("-" * 60)
        print(f"{'Class':<8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 60)

        for i, name in enumerate(self.class_names):
            print(f"{name:<8} "
                  f"{metrics['per_class_acc'][i]:>8.4f} "
                  f"{metrics['per_class_precision'][i]:>8.4f} "
                  f"{metrics['per_class_recall'][i]:>8.4f} "
                  f"{metrics['per_class_f1'][i]:>8.4f}")

        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
