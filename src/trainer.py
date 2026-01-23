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
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

        for batch in self.train_loader:
            # Unpack batch: (ecg, label, rr, pid, sid)
            ecg, labels, rr_features, *_ = batch
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)
            rr_features = rr_features.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits, _ = self.model(ecg, rr_features)

            # Loss
            loss = self.criterion(logits, labels)

            # Backward
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            # Step scheduler (if per-iteration)
            if self.scheduler is not None:
                self.scheduler.step()

            # Simple accuracy (no sklearn overhead)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

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
        Validate on given loader.

        Args:
            loader: DataLoader (default: self.valid_loader)

        Returns:
            (loss, metrics_dict)
        """
        loader = loader or self.valid_loader
        self.model.eval()

        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch in loader:
            ecg, labels, rr_features, *_ = batch
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)
            rr_features = rr_features.to(self.device)

            logits, _ = self.model(ecg, rr_features)
            loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            total_loss += loss.item()

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.concatenate(all_probs, axis=0)

        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        avg_loss = total_loss / len(loader)

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
            train_loss, train_metrics = self.train_one_epoch()
            self.history['train'].append({'loss': train_loss, **train_metrics})

            # Validate
            valid_loss, valid_metrics = self.validate()
            self.history['valid'].append({'loss': valid_loss, **valid_metrics})

            # Print epoch summary
            self._print_epoch_summary(
                epoch, epochs,
                train_loss, train_metrics,
                valid_loss, valid_metrics,
                time.time() - epoch_start
            )

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

        print("\n" + "=" * 60)
        print("Training Complete!")
        self._print_best_models()
        print("=" * 60)

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            loader: Test data loader

        Returns:
            Metrics dictionary
        """
        loss, metrics = self.validate(loader)
        metrics['loss'] = loss
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

        # Per-class accuracy
        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)

        # Precision, Recall, F1
        prec, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        macro_prec, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # AUPRC and AUROC
        macro_auprc, macro_auroc = self._calculate_auc_metrics(y_true, y_prob, num_classes)

        return {
            'acc': acc,
            'per_class_acc': per_class_acc,
            'per_class_precision': prec,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'macro_precision': macro_prec,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'macro_auprc': macro_auprc,
            'macro_auroc': macro_auroc,
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
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def load_best_model(self, metric: str = 'macro_auprc'):
        """Load best model for given metric."""
        path = os.path.join(self.best_model_dir, f'best_{metric}.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model (best {metric})")
        else:
            print(f"Best model for {metric} not found")

    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_metrics: Dict,
        valid_loss: float,
        valid_metrics: Dict,
        elapsed: float
    ):
        """Print epoch summary."""
        lr = self.optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{total_epochs} ({elapsed:.1f}s) | LR: {lr:.6f}")
        print("-" * 60)
        print(f"  Train | Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f}")
        print(f"  Valid | Loss: {valid_loss:.4f} | Acc: {valid_metrics['acc']:.4f} | "
              f"F1: {valid_metrics['macro_f1']:.4f} | AUPRC: {valid_metrics['macro_auprc']:.4f} | AUROC: {valid_metrics['macro_auroc']:.4f}")

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
