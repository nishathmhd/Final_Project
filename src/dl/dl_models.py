import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

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

def load_data(data_dir, img_size=224):
    image_paths = []
    labels = []
    
    for split in ['train', 'validation', 'test']:
        for label, class_name in enumerate(['benign', 'malignant']):
            class_dir = os.path.join(data_dir, "images", split, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(label)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = MammogramDataset(train_paths, train_labels, transform)
    val_dataset = MammogramDataset(val_paths, val_labels, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    return train_loader, val_loader

def build_model():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    # Fine-tune the last layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(dev)
    return model

def predict_dl(model, image):
    """Make prediction on a single image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    image = transform(image.convert('RGB')).unsqueeze(0).to(dev)
    with torch.no_grad():
        output = model(image)
        proba = torch.sigmoid(output).item()
        pred = 1 if proba > 0.5 else 0
    return pred, proba

def train_model(model, train_loader, val_loader, epochs=10, save_path="breast_cancer_model.pth"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = model(inputs).squeeze(1)
                val_loss += criterion(outputs, labels).item()
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with Val Loss: {val_loss:.4f}")

def load_dl_model(model_path="/Users/nishathmhd/Desktop/Nishathmhd/breast_cancer_project/breast_cancer_model.pth"):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

if __name__ == "__main__":
    data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(__file__)
            )
        ),
        "data"
    )
    train_loader, val_loader = load_data(data_dir)
    model = build_model()
    train_model(model, train_loader, val_loader, epochs=15)
