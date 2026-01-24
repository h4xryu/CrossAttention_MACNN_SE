# train.py에 validation 함수 추가

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch(model: nn.Module, train_loader, num_epoch: int,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    scheduler=None, class_weights=None) -> tuple:
    """
    Train one epoch with optional scheduler step per iteration (DAEAC style).

    Args:
        scheduler: If provided, scheduler.step() is called after each iteration
        class_weights: Optional tensor of class weights for weighted CE loss
    """
    model.train()
    total_loss = 0.0
    y_pred, y_true = [], []
    y_probs_all = []  # For AUPRC/AUROC



    for batch in tqdm(train_loader, desc=f"Training Epoch {num_epoch} ", leave=True):
        ecg_inputs, labels, rr_features, pids, _ = batch
        ecg_inputs = ecg_inputs.to(device)
        rr_features = rr_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(ecg_inputs, rr_features)
        probs = torch.softmax(logits, dim=1)

        # Store predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_probs_all.append(probs.detach().cpu().numpy())

        # Loss calculation: weighted cross-entropy (DAEAC uses weighted CE)
        if class_weights is not None:
            ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='mean')
        else:
            ce_loss = F.cross_entropy(logits, labels, reduction='mean')

        loss = ce_loss

        loss.backward()
        optimizer.step()

        # DAEAC style: LR decay every iteration
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs_all = np.concatenate(y_probs_all, axis=0)
    
    accuracy = accuracy_score(y_true, y_pred)
    

    
    
    metrics = {
        'acc': accuracy,
    }
    


    return (total_loss / len(train_loader), metrics)


def validate(model: nn.Module, valid_loader, device: torch.device) -> tuple:
    """Validation function - simplified to only calculate accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in valid_loader:
            ecg_inputs, labels, rr_features, pids, _ = batch
            ecg_inputs = ecg_inputs.to(device)
            rr_features = rr_features.to(device)
            labels = labels.to(device)
            
            logits, _ = model(ecg_inputs, rr_features)
            
            # Loss calculation
            ce_loss = F.cross_entropy(logits, labels, reduction='mean')
            total_loss += ce_loss.item()
            
            # Simple accuracy calculation
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        'acc': accuracy,
        'macro_f1': 0.0,  # placeholder
        'macro_auprc': 0.0,
        'macro_auroc': 0.0,
    }
    
    return (total_loss / len(valid_loader), metrics)


def save_model(model: nn.Module, 
               optimizer: torch.optim.Optimizer,
               epoch: int, 
               metrics: dict, 
               save_path: str) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved: {save_path}")


def load_model(model: nn.Module, 
               load_path: str, 
               optimizer: torch.optim.Optimizer = None,
               device: torch.device = None) -> tuple:
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Model loaded from: {load_path}")
    return model, optimizer, epoch, metrics