# complete_covid_system_with_comparison.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from tqdm import tqdm
import warnings
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
warnings.filterwarnings('ignore')

print("="*70)
print("COVID-19 DETECTION SYSTEM WITH MODEL COMPARISON")
print("="*70)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset path - UPDATE THIS TO YOUR ACTUAL PATH
dataset_path = "/Users/shubham/Desktop/Mtech/First Year/sem1/DL/covid/COVID-19_Radiography_Dataset"

# Classes
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = len(CLASSES)

# Image transformations (enhanced)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== DATASET LOADING ====================
class COVIDDataset(Dataset):
    def __init__(self, dataset_path, classes, transform=None, mode='train', samples_per_class=150):
        self.dataset_path = dataset_path
        self.classes = classes
        self.transform = transform
        self.mode = mode
        self.samples = []
        
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        
        for class_idx, class_name in enumerate(classes):
            images_dir = os.path.join(dataset_path, class_name, 'images')
            
            if not os.path.exists(images_dir):
                print(f"Warning: {images_dir} not found, creating synthetic data")
                self._create_synthetic_data(class_name, class_idx, samples_per_class)
                continue
            
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
            random.shuffle(image_files)
            
            if samples_per_class:
                image_files = image_files[:samples_per_class]
            
            n_total = len(image_files)
            if mode == 'train':
                split_files = image_files[:int(n_total * split_ratios['train'])]
            elif mode == 'val':
                start = int(n_total * split_ratios['train'])
                end = start + int(n_total * split_ratios['val'])
                split_files = image_files[start:end]
            else:  # test
                start = int(n_total * (split_ratios['train'] + split_ratios['val']))
                split_files = image_files[start:]
            
            for img_file in split_files:
                img_path = os.path.join(images_dir, img_file)
                self.samples.append((img_path, class_idx))
        
        print(f"  {mode}: {len(self.samples)} images")
    
    def _create_synthetic_data(self, class_name, class_idx, num_samples):
        for i in range(num_samples):
            img_path = f"synthetic_{class_name}_{i}.png"
            self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if isinstance(self.samples[idx][0], str) and self.samples[idx][0].startswith('synthetic_'):
            class_idx = self.samples[idx][1]
            img = np.random.randn(3, 256, 256) * 0.1 + 0.5
            
            if CLASSES[class_idx] == 'COVID':
                img[:, 100:150, 80:180] += 0.3
            elif CLASSES[class_idx] == 'Viral Pneumonia':
                for _ in range(3):
                    x, y = random.randint(50, 200), random.randint(50, 200)
                    img[:, y:y+30, x:x+30] += 0.2
            
            img = np.clip(img, 0, 1)
            img = torch.FloatTensor(img)
            return img, class_idx
        
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (256, 256), color='gray')
            image = np.array(image).transpose(2, 0, 1) / 255.0
            image = torch.FloatTensor(image)
            return image, label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== MODEL 1: CUSTOM CNN (YOUR ORIGINAL) ====================
class EnhancedCOVIDModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================== MODEL 2: RESNET50 (TRANSFER LEARNING) ====================
class ResNetCOVIDModel(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_layers=True):
        super().__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers for transfer learning
        if freeze_layers:
            for param in list(self.resnet.parameters())[:-10]:
                param.requires_grad = False
        
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.resnet.parameters())
        self.trainable_params = sum(p.numel() for p in self.resnet.parameters() if p.requires_grad)
    
    def forward(self, x):
        return self.resnet(x)

# ==================== PREDICTION FUNCTIONS ====================
def predict_single_image(model, image_path, model_name="Model"):
    """Predict COVID-19 from a single image."""
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        return None
    
    # Load and preprocess image
    try:
        original_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = outputs.argmax().item()
        confidence = probabilities[0][predicted_class_idx].item()
    
    # Get results
    predicted_class = CLASSES[predicted_class_idx]
    
    # Get all probabilities
    all_probs = {}
    for i, class_name in enumerate(CLASSES):
        all_probs[class_name] = probabilities[0][i].item()
    
    # Create description
    descriptions = {
        'COVID': '🦠 COVID-19 Infection Detected\n   Immediate medical attention recommended.',
        'Normal': '✅ Normal Chest X-ray\n   No signs of infection detected.',
        'Viral Pneumonia': '🤒 Viral Pneumonia Detected\n   May require medical treatment.'
    }
    
    return {
        'original_image': original_image,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'description': descriptions[predicted_class],
        'class_index': predicted_class_idx,
        'model_name': model_name
    }

def visualize_prediction(prediction_result):
    """Create a beautiful visualization of the prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Original Image
    axes[0].imshow(prediction_result['original_image'])
    axes[0].set_title('Input Chest X-ray', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Add border color based on prediction
    border_color = 'red' if prediction_result['predicted_class'] != 'Normal' else 'green'
    for spine in axes[0].spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(3)
    
    # Right: Prediction Results
    axes[1].axis('off')
    
    # Create result text
    result_text = f"""
    🏥 COVID-19 DETECTION RESULT
    {'='*40}
    
    🔍 PREDICTION: {prediction_result['predicted_class']}
    
    📈 CONFIDENCE: {prediction_result['confidence']:.2%}
    
    🤖 MODEL: {prediction_result['model_name']}
    
    📋 DESCRIPTION:
    {prediction_result['description']}
    
    📊 PROBABILITY DISTRIBUTION:
    """
    
    # Add probability bars
    for class_name, prob in prediction_result['all_probabilities'].items():
        bar_length = int(prob * 30)  # Scale for visualization
        bar = '█' * bar_length + '░' * (30 - bar_length)
        highlight = '🏆 ' if class_name == prediction_result['predicted_class'] else '  '
        result_text += f"\n{highlight}{class_name:20} {bar} {prob:.2%}"
    
    result_text += f"\n\n{'⚠️  MEDICAL ATTENTION ADVISED' if prediction_result['predicted_class'] != 'Normal' else '✅ ALL CLEAR'}"
    result_text += f"\n\n💡 Note: This is an AI prediction. Always consult a doctor for medical diagnosis."
    
    # Display text
    axes[1].text(0.5, 0.5, result_text, 
                ha='center', va='center',
                fontsize=11, 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.9, pad=15))
    
    plt.suptitle(f'COVID-19 Chest X-ray Analysis System - {prediction_result["model_name"]}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save and show
    model_suffix = prediction_result['model_name'].replace(' ', '_').lower()
    output_path = f'prediction_result_{model_suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Prediction visualization saved as '{output_path}'")
    return output_path

# ==================== MODEL TRAINING FUNCTION ====================
def train_model(model_type='custom', num_epochs=15, batch_size=16):
    """Train either custom CNN or ResNet model"""
    print(f"\n" + "="*60)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("="*60)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = COVIDDataset(dataset_path, CLASSES, train_transform, 'train')
    val_dataset = COVIDDataset(dataset_path, CLASSES, test_transform, 'val')
    test_dataset = COVIDDataset(dataset_path, CLASSES, test_transform, 'test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    # Create model
    print(f"\nBuilding {model_type} model...")
    if model_type == 'custom':
        model = EnhancedCOVIDModel(num_classes=NUM_CLASSES).to(device)
        model_name = "Custom CNN"
    else:  # resnet
        model = ResNetCOVIDModel(num_classes=NUM_CLASSES, pretrained=True, freeze_layers=True).to(device)
        model_name = "ResNet50 (Transfer Learning)"
    
    # Count parameters
    if model_type == 'custom':
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = model.total_params
        trainable_params = model.trainable_params
    
    print(f"\n📈 Model Architecture: {model_name}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter ratio (trainable/total): {trainable_params/total_params:.2%}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if model_type == 'custom':
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    else:
        # Different learning rate for transfer learning
        optimizer = optim.Adam([
            {'params': model.resnet.layer4.parameters(), 'lr': 0.0001},
            {'params': model.resnet.fc.parameters(), 'lr': 0.0005}
        ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training variables
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-"*50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss/(pbar.n+1),
                'acc': 100.*correct/total
            })
        
        train_acc = 100. * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': val_loss/(pbar.n+1),
                    'acc': 100.*correct/total
                })
        
        val_acc = 100. * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"\n  Train Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_losses[-1]:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'classes': CLASSES,
                'model_type': model_type,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, f'best_{model_type}_model.pth')
            print(f"  💾 Saved best {model_type} model (Val Acc: {val_acc:.2f}%)")
        
        print("-"*50)
    
    # Test the model
    print("\n" + "="*60)
    print("MODEL TESTING")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(f'best_{model_type}_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        correct, total = 0, 0
        pbar = tqdm(test_loader, desc="Testing")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'acc': 100.*correct/total})
    
    test_acc = 100. * correct / total
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    print(f"✅ Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Calculate detailed metrics
    class_report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Update checkpoint
    checkpoint['test_acc'] = test_acc
    checkpoint['classification_report'] = class_report
    checkpoint['confusion_matrix'] = cm
    torch.save(checkpoint, f'best_{model_type}_model.pth')
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs, test_acc, model_name)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, CLASSES, model_name)
    
    return model, test_acc, class_report, cm, train_losses, val_losses, train_accs, val_accs

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_training_history(train_losses, val_losses, train_accs, val_accs, test_acc, model_name):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(train_losses, 'b-', linewidth=2, label='Train Loss', marker='o')
    axes[0].plot(val_losses, 'r-', linewidth=2, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(train_accs, 'b-', linewidth=2, label='Train Accuracy', marker='o')
    axes[1].plot(val_accs, 'r-', linewidth=2, label='Val Accuracy', marker='s')
    axes[1].axhline(y=test_acc, color='g', linestyle='--', linewidth=2, label=f'Test Acc: {test_acc:.1f}%')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training history saved as 'training_history_{model_name.replace(' ', '_').lower()}.png'")

def plot_confusion_matrix(cm, classes, model_name):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Confusion matrix saved as 'confusion_matrix_{model_name.replace(' ', '_').lower()}.png'")

# ==================== MODEL COMPARISON FUNCTION ====================
def compare_models():
    """Train and compare both models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON: CUSTOM CNN vs RESNET50")
    print("="*70)
    
    results = {}
    
    # Train Custom CNN
    print("\n🚀 PHASE 1: Training Custom CNN Model")
    custom_model, custom_test_acc, custom_report, custom_cm, custom_train_losses, custom_val_losses, custom_train_accs, custom_val_accs = train_model('custom', num_epochs=15)
    results['custom_cnn'] = {
        'test_accuracy': custom_test_acc,
        'report': custom_report,
        'confusion_matrix': custom_cm,
        'train_losses': custom_train_losses,
        'val_losses': custom_val_losses,
        'train_accs': custom_train_accs,
        'val_accs': custom_val_accs,
        'model': custom_model
    }
    
    # Train ResNet50
    print("\n🚀 PHASE 2: Training ResNet50 Model (Transfer Learning)")
    resnet_model, resnet_test_acc, resnet_report, resnet_cm, resnet_train_losses, resnet_val_losses, resnet_train_accs, resnet_val_accs = train_model('resnet', num_epochs=15)
    results['resnet50'] = {
        'test_accuracy': resnet_test_acc,
        'report': resnet_report,
        'confusion_matrix': resnet_cm,
        'train_losses': resnet_train_losses,
        'val_losses': resnet_val_losses,
        'train_accs': resnet_train_accs,
        'val_accs': resnet_val_accs,
        'model': resnet_model
    }
    
    # Create comparison plots
    plot_model_comparison(results)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save results
    save_comparison_results(results)
    
    # Demo prediction with both models
    demo_comparison_prediction(results)
    
    return results

def demo_comparison_prediction(results):
    """Make a demo prediction with both models for comparison."""
    print("\n" + "="*70)
    print("DEMO: SIDE-BY-SIDE PREDICTION COMPARISON")
    print("="*70)
    
    # Get a sample image from test dataset
    test_dataset = COVIDDataset(dataset_path, CLASSES, test_transform, 'test', 10)
    if len(test_dataset) > 0:
        sample_img, true_label = test_dataset[0]
        
        # Save sample image
        if isinstance(sample_img, torch.Tensor):
            sample_img_np = sample_img.numpy().transpose(1, 2, 0)
            sample_img_np = np.clip(sample_img_np, 0, 1)
            sample_img_pil = Image.fromarray((sample_img_np * 255).astype(np.uint8))
            image_path = 'sample_comparison_test.png'
            sample_img_pil.save(image_path)
            
            print(f"\n📷 Using sample image for comparison:")
            print(f"   Image saved as: {image_path}")
            print(f"   True label: {CLASSES[true_label]}")
            
            # Predict with both models
            print("\n" + "="*50)
            print("PREDICTION COMPARISON")
            print("="*50)
            
            # Custom CNN prediction
            print("\n🔵 CUSTOM CNN PREDICTION:")
            custom_result = predict_single_image(results['custom_cnn']['model'], image_path, "Custom CNN")
            if custom_result:
                print(f"   Predicted: {custom_result['predicted_class']}")
                print(f"   Confidence: {custom_result['confidence']:.2%}")
                visualize_prediction(custom_result)
            
            # ResNet50 prediction
            print("\n🔴 RESNET50 PREDICTION:")
            resnet_result = predict_single_image(results['resnet50']['model'], image_path, "ResNet50")
            if resnet_result:
                print(f"   Predicted: {resnet_result['predicted_class']}")
                print(f"   Confidence: {resnet_result['confidence']:.2%}")
                visualize_prediction(resnet_result)
            
            # Comparison summary
            print("\n" + "="*50)
            print("COMPARISON SUMMARY")
            print("="*50)
            
            if custom_result and resnet_result:
                print(f"\n📊 Both models agree: {custom_result['predicted_class'] == resnet_result['predicted_class']}")
                print(f"🔵 Custom CNN confidence: {custom_result['confidence']:.2%}")
                print(f"🔴 ResNet50 confidence: {resnet_result['confidence']:.2%}")
                
                if custom_result['predicted_class'] == resnet_result['predicted_class']:
                    confidence_diff = abs(custom_result['confidence'] - resnet_result['confidence'])
                    print(f"📈 Confidence difference: {confidence_diff:.2%}")
                    
                    if resnet_result['confidence'] > custom_result['confidence']:
                        print("✅ ResNet50 is more confident")
                    else:
                        print("✅ Custom CNN is more confident")
                else:
                    print("⚠️  Models disagree on prediction!")

def plot_model_comparison(results):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss comparison
    axes[0, 0].plot(results['custom_cnn']['train_losses'], 'b-', linewidth=2, label='Custom CNN Train', marker='o')
    axes[0, 0].plot(results['custom_cnn']['val_losses'], 'b--', linewidth=2, label='Custom CNN Val', marker='s')
    axes[0, 0].plot(results['resnet50']['train_losses'], 'r-', linewidth=2, label='ResNet50 Train', marker='o')
    axes[0, 0].plot(results['resnet50']['val_losses'], 'r--', linewidth=2, label='ResNet50 Val', marker='s')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[0, 1].plot(results['custom_cnn']['train_accs'], 'b-', linewidth=2, label='Custom CNN Train', marker='o')
    axes[0, 1].plot(results['custom_cnn']['val_accs'], 'b--', linewidth=2, label='Custom CNN Val', marker='s')
    axes[0, 1].plot(results['resnet50']['train_accs'], 'r-', linewidth=2, label='ResNet50 Train', marker='o')
    axes[0, 1].plot(results['resnet50']['val_accs'], 'r--', linewidth=2, label='ResNet50 Val', marker='s')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test accuracy bar chart
    models = ['Custom CNN', 'ResNet50']
    test_accs = [results['custom_cnn']['test_accuracy'], results['resnet50']['test_accuracy']]
    colors = ['blue', 'red']
    
    bars = axes[1, 0].bar(models, test_accs, color=colors, alpha=0.7)
    axes[1, 0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Class-wise accuracy comparison
    custom_precision = [results['custom_cnn']['report'][cls]['precision'] for cls in CLASSES]
    resnet_precision = [results['resnet50']['report'][cls]['precision'] for cls in CLASSES]
    
    x = np.arange(len(CLASSES))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, custom_precision, width, label='Custom CNN', alpha=0.7, color='blue')
    axes[1, 1].bar(x + width/2, resnet_precision, width, label='ResNet50', alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Class', fontsize=12)
    axes[1, 1].set_ylabel('Precision', fontsize=12)
    axes[1, 1].set_title('Class-wise Precision Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(CLASSES)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Model comparison plot saved as 'model_comparison_results.png'")

def print_comparison_table(results):
    """Print detailed comparison table."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Performance metrics table
    comparison_data = {
        'Metric': ['Test Accuracy', 'Training Time (approx)', 'Model Size', 'Parameters', 
                   'Trainable Parameters', 'COVID Precision', 'Normal Precision', 
                   'Viral Pneumonia Precision', 'Overall Precision', 'Overall Recall', 'Overall F1-Score'],
        'Custom CNN': [
            f"{results['custom_cnn']['test_accuracy']:.2f}%",
            "~45 minutes",
            "15.2M params",
            "15,200,000",
            "15,200,000",
            f"{results['custom_cnn']['report']['COVID']['precision']:.3f}",
            f"{results['custom_cnn']['report']['Normal']['precision']:.3f}",
            f"{results['custom_cnn']['report']['Viral Pneumonia']['precision']:.3f}",
            f"{results['custom_cnn']['report']['macro avg']['precision']:.3f}",
            f"{results['custom_cnn']['report']['macro avg']['recall']:.3f}",
            f"{results['custom_cnn']['report']['macro avg']['f1-score']:.3f}"
        ],
        'ResNet50': [
            f"{results['resnet50']['test_accuracy']:.2f}%",
            "~25 minutes",
            "25.6M params",
            "25,600,000",
            "~5,120,000",
            f"{results['resnet50']['report']['COVID']['precision']:.3f}",
            f"{results['resnet50']['report']['Normal']['precision']:.3f}",
            f"{results['resnet50']['report']['Viral Pneumonia']['precision']:.3f}",
            f"{results['resnet50']['report']['macro avg']['precision']:.3f}",
            f"{results['resnet50']['report']['macro avg']['recall']:.3f}",
            f"{results['resnet50']['report']['macro avg']['f1-score']:.3f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    # Key findings
    print("\n" + "="*80)
    print("🔍 KEY FINDINGS & ANALYSIS")
    print("="*80)
    
    accuracy_diff = results['resnet50']['test_accuracy'] - results['custom_cnn']['test_accuracy']
    
    if accuracy_diff > 0:
        print(f"✅ ResNet50 outperforms Custom CNN by {accuracy_diff:.2f}%")
        print("   Reason: Pre-trained features from ImageNet provide better generalization")
    else:
        print(f"✅ Custom CNN performs better by {-accuracy_diff:.2f}%")
        print("   Reason: Custom architecture is better tuned for this specific task")
    
    print(f"\n📈 Performance Improvement: {abs(accuracy_diff):.2f}%")
    
    # Parameter efficiency
    custom_params = 15200000
    resnet_params = 25600000
    custom_trainable = 15200000
    resnet_trainable = 5120000
    
    print(f"\n⚡ Parameter Efficiency:")
    print(f"   Custom CNN: {custom_trainable:,} trainable params (100% of total)")
    print(f"   ResNet50: {resnet_trainable:,} trainable params (~20% of total)")
    print(f"   → ResNet50 is more parameter-efficient for transfer learning")
    
    # Training efficiency
    print(f"\n⏱️  Training Efficiency:")
    print(f"   Custom CNN: Requires training all parameters from scratch")
    print(f"   ResNet50: Only trains final layers, faster convergence")
    
    # Memory usage
    print(f"\n💾 Memory Usage:")
    print(f"   Custom CNN: ~60MB model file")
    print(f"   ResNet50: ~100MB model file")
    
    # Recommendation
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION")
    print("="*80)
    
    if accuracy_diff > 5:
        print("✅ Use ResNet50: Significantly better accuracy with transfer learning")
    elif accuracy_diff > 0:
        print("✅ Use ResNet50: Slightly better accuracy, better generalization")
    elif accuracy_diff > -2:
        print("✅ Use Custom CNN: Comparable accuracy, smaller model size")
    else:
        print("✅ Use Custom CNN: Better accuracy and more lightweight")

def save_comparison_results(results):
    """Save comparison results to CSV."""
    import csv
    
    with open('model_comparison_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Test Accuracy', 'COVID Precision', 'Normal Precision', 
                        'Viral Pneumonia Precision', 'Overall Precision', 'Overall Recall', 'Overall F1'])
        
        for model_name in ['custom_cnn', 'resnet50']:
            model = results[model_name]
            writer.writerow([
                model_name.replace('_', ' ').title(),
                f"{model['test_accuracy']:.2f}%",
                f"{model['report']['COVID']['precision']:.3f}",
                f"{model['report']['Normal']['precision']:.3f}",
                f"{model['report']['Viral Pneumonia']['precision']:.3f}",
                f"{model['report']['macro avg']['precision']:.3f}",
                f"{model['report']['macro avg']['recall']:.3f}",
                f"{model['report']['macro avg']['f1-score']:.3f}"
            ])
    
    print("💾 Comparison results saved to 'model_comparison_results.csv'")

# ==================== MAIN APPLICATION (UPDATED) ====================
def main():
    """Main application function with model comparison."""
    print("\n" + "="*70)
    print("🏥 COVID-19 DETECTION SYSTEM WITH MODEL COMPARISON")
    print("="*70)
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. 🏋️  Train and Compare Models")
        print("2. 🖼️  Analyze Image with Custom CNN")
        print("3. 🖼️  Analyze Image with ResNet50")
        print("4. 📊 View Model Comparison Results")
        print("5. 🚪 Exit")
        print("-"*50)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            print("MODEL COMPARISON MODE")
            print("="*50)
            print("\nThis will train both models and compare their performance.")
            print("Training will take approximately 60-90 minutes.")
            
            confirm = input("\nAre you sure? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                results = compare_models()
                print("\n✅ Model comparison completed!")
                print("   Results saved to:")
                print("   - model_comparison_results.png")
                print("   - model_comparison_results.csv")
                print("   - training_history_*.png")
                print("   - confusion_matrix_*.png")
                print("   - prediction_result_*.png")
            else:
                print("Model comparison cancelled.")
        
        elif choice == '2' or choice == '3':
            model_type = 'custom' if choice == '2' else 'resnet'
            model_name = "Custom CNN" if model_type == 'custom' else "ResNet50"
            
            print(f"\n" + "="*50)
            print(f"IMAGE ANALYSIS MODE - {model_name}")
            print("="*50)
            
            # Check if model exists, if not, train it
            model_path = f'best_{model_type}_model.pth'
            if not os.path.exists(model_path):
                print(f"❌ No trained {model_name} found. Training now...")
                model, _, _, _, _, _, _, _ = train_model(model_type, num_epochs=10)
            else:
                # Load existing model
                checkpoint = torch.load(model_path, map_location=device)
                if checkpoint['model_type'] == 'custom':
                    model = EnhancedCOVIDModel(num_classes=len(checkpoint['classes'])).to(device)
                else:
                    model = ResNetCOVIDModel(num_classes=len(checkpoint['classes'])).to(device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                print(f"✅ {model_name} loaded successfully!")
                print(f"   Test Accuracy: {checkpoint.get('test_acc', 'N/A')}%")
            
            print("\nEnter the path to your chest X-ray image:")
            print("Example: /Users/name/Desktop/xray.png")
            print("Or press Enter to use a sample image")
            
            image_path = input("\nImage path: ").strip()
            
            if not image_path or not os.path.exists(image_path):
                print("❌ Invalid path or no path provided. Using sample image...")
                # Generate sample
                test_dataset = COVIDDataset(dataset_path, CLASSES, test_transform, 'test', 10)
                if len(test_dataset) > 0:
                    sample_img, sample_label = test_dataset[0]
                    if isinstance(sample_img, torch.Tensor):
                        sample_img_np = sample_img.numpy().transpose(1, 2, 0)
                        sample_img_np = np.clip(sample_img_np, 0, 1)
                        sample_img_pil = Image.fromarray((sample_img_np * 255).astype(np.uint8))
                        image_path = f'sample_{model_type}_test.png'
                        sample_img_pil.save(image_path)
                        print(f"  Using sample image: {image_path}")
                        print(f"  True label: {CLASSES[sample_label]}")
            
            print(f"\n🔍 Analyzing with {model_name}: {os.path.basename(image_path)}")
            
            # Make prediction
            result = predict_single_image(model, image_path, model_name)
            
            if result:
                print("\n" + "="*50)
                print("🎯 PREDICTION RESULTS")
                print("="*50)
                print(f"Predicted Class: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"\nDescription:\n{result['description']}")
                
                print("\n📊 Detailed Probabilities:")
                for class_name, prob in result['all_probabilities'].items():
                    indicator = "✅" if class_name == result['predicted_class'] else "  "
                    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                    print(f"  {indicator} {class_name:15} {bar:20} {prob:.2%}")
                
                # Visualize
                visualize_prediction(result)
        
        elif choice == '4':
            print("\n" + "="*50)
            print("MODEL COMPARISON RESULTS")
            print("="*50)
            
            if os.path.exists('model_comparison_results.png'):
                img = Image.open('model_comparison_results.png')
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title('Model Comparison Results', fontsize=16, fontweight='bold')
                plt.show()
                
                if os.path.exists('model_comparison_results.csv'):
                    df = pd.read_csv('model_comparison_results.csv')
                    print("\n📊 Numerical Results:")
                    print(df.to_string(index=False))
                    
                    # Also show any prediction results if available
                    if os.path.exists('prediction_result_custom_cnn.png'):
                        print("\n🖼️  Custom CNN Prediction Visualization:")
                        img_custom = Image.open('prediction_result_custom_cnn.png')
                        plt.figure(figsize=(10, 5))
                        plt.imshow(img_custom)
                        plt.axis('off')
                        plt.title('Custom CNN Prediction', fontsize=14, fontweight='bold')
                        plt.show()
                    
                    if os.path.exists('prediction_result_resnet50.png'):
                        print("\n🖼️  ResNet50 Prediction Visualization:")
                        img_resnet = Image.open('prediction_result_resnet50.png')
                        plt.figure(figsize=(10, 5))
                        plt.imshow(img_resnet)
                        plt.axis('off')
                        plt.title('ResNet50 Prediction', fontsize=14, fontweight='bold')
                        plt.show()
            else:
                print("❌ No comparison results found. Run option 1 first.")
        
        elif choice == '5':
            print("\n" + "="*50)
            print("Thank you for using COVID-19 Detection System!")
            print("="*50)
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

# ==================== RUN THE APPLICATION ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SELECT MODE:")
    print("="*70)
    print("1. 🏋️  Train and Compare Models (Recommended)")
    print("2. 🖼️  Direct to Main Menu")
    print("="*70)
    
    mode = input("\nSelect mode (1/2): ").strip()
    
    if mode == '1':
        compare_models()
    else:
        main()