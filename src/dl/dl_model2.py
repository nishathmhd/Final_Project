import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from tqdm import tqdm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

dev = get_device()
print(f"Using device: {dev}")

# Dataset class to load images from organized folders
class MammogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def load_all_data(data_dir, img_size=300):
    """Load all images into a single dataset (no train/val/test split yet)"""
    image_paths = []
    labels = []
    
    # Collect benign images
    benign_dir = os.path.join(data_dir, "benign")
    for img_file in os.listdir(benign_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(benign_dir, img_file))
            labels.append(0)  # 0 for benign
    
    # Collect malignant images
    malignant_dir = os.path.join(data_dir, "malignant")
    for img_file in os.listdir(malignant_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(malignant_dir, img_file))
            labels.append(1)  # 1 for malignant
    
    print(f"Total images: {len(image_paths)}")
    print(f"Benign: {labels.count(0)}, Malignant: {labels.count(1)}")
    
    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    return image_paths, labels, train_transform, val_transform

from .efficientnet_model import EfficientNetB3Mammo

def build_model():
    model = EfficientNetB3Mammo(pretrained=True)
    
    # Freeze early layers (already handled in EfficientNetB3Mammo)
    # Fine-tune the last layers
    for param in model.base_model.features[-5:].parameters():
        param.requires_grad = True
    
    model = model.to(dev)
    return model

def train_fold(model, train_loader, val_loader, fold, epochs=10, save_path="breast_cancer_model_fold{}.pth"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_loss = float('inf')
    fold_results = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Fold {fold+1}, Epoch {epoch+1} (Train)"):
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        fold_results['train_loss'].append(avg_train_loss)
        
        # Validation phase - Model Evaluation Metrics
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        true_pos, false_pos = 0, 0
        true_neg, false_neg = 0, 0
        all_probs = []
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Fold {fold+1}, Epoch {epoch+1} (Val)"):
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = model(inputs).squeeze(1)
                val_loss += criterion(outputs, labels).item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                # Update counts
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                true_pos += ((preds == 1) & (labels == 1)).sum().item()
                false_pos += ((preds == 1) & (labels == 0)).sum().item()
                true_neg += ((preds == 0) & (labels == 0)).sum().item()
                false_neg += ((preds == 0) & (labels == 1)).sum().item()
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        precision = true_pos / (true_pos + false_pos + 1e-10)
        recall = true_pos / (true_pos + false_neg + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Calculate AUC if sklearn is available
        try:
            from sklearn.metrics import roc_auc_score, confusion_matrix
            auc = roc_auc_score(all_labels, all_probs)
            conf_matrix = confusion_matrix(all_labels, all_preds)
        except:
            auc = 0.0
            conf_matrix = None
            
        # Store results
        fold_results['val_loss'].append(avg_val_loss)
        fold_results['val_accuracy'].append(accuracy)
        fold_results['val_auc'].append(auc)
        fold_results['val_precision'].append(precision)
        fold_results['val_recall'].append(recall)
        fold_results['val_f1'].append(f1_score)
        fold_results['val_confusion_matrix'].append(conf_matrix)
        
        scheduler.step()
        
        print(f"Fold {fold+1}, Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = save_path.format(fold+1)
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved to {model_save_path}")
    
    return fold_results

def cross_validation(image_paths, labels, train_transform, val_transform, n_folds=5, batch_size=16, epochs=10):
    """Perform k-fold cross-validation"""
    # Use stratified k-fold to maintain class distribution
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Convert labels to numpy for sklearn
    labels_np = np.array(labels)
    
    # Initialize results tracking
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels_np)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")
        print(f"Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
        
        # Create datasets for this fold
        train_dataset = MammogramDataset(
            [image_paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            transform=train_transform
        )
        
        val_dataset = MammogramDataset(
            [image_paths[i] for i in val_idx],
            [labels[i] for i in val_idx],
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
        
        # Initialize a fresh model for each fold
        model = build_model()
        
        # Train model on this fold
        fold_results = train_fold(
            model, 
            train_loader, 
            val_loader, 
            fold, 
            epochs=epochs,
            save_path=f"breast_cancer_model_fold{fold+1}.pth"
        )
        
        all_fold_results.append(fold_results)
    
    # Calculate average performance across folds
    avg_val_loss = np.mean([results['val_loss'][-1] for results in all_fold_results])
    avg_val_acc = np.mean([results['val_accuracy'][-1] for results in all_fold_results])
    avg_val_auc = np.mean([results['val_auc'][-1] for results in all_fold_results])
    
    print("\n" + "="*50)
    print("Cross-validation complete!")
    print(f"Average validation loss: {avg_val_loss:.4f}")
    print(f"Average validation accuracy: {avg_val_acc:.4f}")
    print(f"Average validation AUC: {avg_val_auc:.4f}")
    
    # Save results for visualization
    results_path = "breast_cancer_project/results/cv_results.pkl"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(all_fold_results, f)
    print(f"Cross-validation results saved to {results_path}")
    
    return all_fold_results

def train_final_model(image_paths, labels, transform, best_fold, epochs=15, batch_size=16):
    """Train final model on all data using the best hyperparameters"""
    print("\n" + "="*50)
    print("Training final model on all data")
    
    # Create dataset from all data
    full_dataset = MammogramDataset(image_paths, labels, transform=transform)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize fresh model
    model = build_model()
    
    # Train model
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(full_loader, desc=f"Final Model, Epoch {epoch+1}"):
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(full_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "breast_cancer_final_model.pth")
    print("Final model saved to breast_cancer_final_model.pth")
    
    return model

def predict(model, image):
    """Make prediction on a single image"""
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = transform(image.convert('RGB')).unsqueeze(0).to(dev)
    with torch.no_grad():
        output = model(image)
        proba = torch.sigmoid(output).item()
        pred = 1 if proba > 0.5 else 0
    return pred, proba

if __name__ == "__main__":
    # Path to mammogram images
    data_dir = "breast_cancer_project/data/images"
    
    # Load all data
    image_paths, labels, train_transform, val_transform = load_all_data(data_dir)
    
    # Perform 5-fold cross-validation
    results = cross_validation(
        image_paths, 
        labels, 
        train_transform, 
        val_transform, 
        n_folds=5, 
        batch_size=16, 
        epochs=10
    )
    
    # Find the best performing fold
    best_fold = np.argmin([r['val_loss'][-1] for r in results])
    print(f"Best performing fold: {best_fold+1}")
    
    # Train final model on all data
    final_model = train_final_model(
        image_paths, 
        labels, 
        train_transform, 
        best_fold, 
        epochs=15, 
        batch_size=16
    )


