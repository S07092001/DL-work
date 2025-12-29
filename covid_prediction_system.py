# covid_prediction_system.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COVID-19 IMAGE PREDICTION SYSTEM")
print("="*70)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Classes
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = len(CLASSES)

# Image transformation (same as training)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== MODEL ARCHITECTURES ====================
class EnhancedCOVIDModel(nn.Module):
    """Your original custom CNN model"""
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
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
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNetCOVIDModel(nn.Module):
    """ResNet50 model for comparison"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet50(pretrained=False)
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
    
    def forward(self, x):
        return self.resnet(x)

# ==================== PREDICTION FUNCTIONS ====================
def load_model(model_type='custom'):
    """Load your pre-trained model"""
    model_path = 'best_covid_model.pth'  # Your existing model
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please make sure 'best_covid_model.pth' is in the current directory.")
        return None
    
    print(f"Loading pre-trained model from '{model_path}'...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine model type
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        else:
            # Try to infer from model structure
            model_type = 'custom'  # Default to your custom model
        
        # Create model
        if model_type == 'custom':
            model = EnhancedCOVIDModel(num_classes=len(checkpoint['classes'])).to(device)
        else:
            model = ResNetCOVIDModel(num_classes=len(checkpoint['classes'])).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model Type: {model_type.upper()}")
        print(f"   Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        print(f"   Test Accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")
        print(f"   Classes: {', '.join(checkpoint.get('classes', CLASSES))}")
        
        return model, checkpoint
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_single_image(model, image_path):
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
    
    # Store original size for display
    original_size = original_image.size
    
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
        'original_size': original_size
    }

def visualize_prediction(prediction_result, model_name="Pre-trained Model"):
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
    
    # Add image info
    img_info = f"Size: {prediction_result['original_size'][0]}×{prediction_result['original_size'][1]}"
    axes[0].text(0.5, -0.05, img_info, ha='center', va='top', 
                 transform=axes[0].transAxes, fontsize=10, style='italic')
    
    # Right: Prediction Results
    axes[1].axis('off')
    
    # Create result text
    result_text = f"""
    🏥 COVID-19 DETECTION RESULT
    {'='*40}
    
    🔍 PREDICTION: {prediction_result['predicted_class']}
    
    📈 CONFIDENCE: {prediction_result['confidence']:.2%}
    
    🤖 MODEL: {model_name}
    
    📋 DESCRIPTION:
    {prediction_result['description']}
    
    📊 PROBABILITY DISTRIBUTION:
    """
    
    # Add probability bars with better visualization
    for class_name, prob in prediction_result['all_probabilities'].items():
        bar_length = int(prob * 30)  # Scale for visualization
        bar = '█' * bar_length + '░' * (30 - bar_length)
        highlight = '🏆 ' if class_name == prediction_result['predicted_class'] else '  '
        result_text += f"\n{highlight}{class_name:20} {bar} {prob:.2%}"
    
    # Add medical advice
    if prediction_result['predicted_class'] == 'COVID':
        result_text += f"\n\n{'⚠️  URGENT: MEDICAL ATTENTION REQUIRED'}"
    elif prediction_result['predicted_class'] == 'Viral Pneumonia':
        result_text += f"\n\n{'⚠️  MEDICAL ATTENTION ADVISED'}"
    else:
        result_text += f"\n\n{'✅ ALL CLEAR - No immediate concerns'}"
    
    result_text += f"\n\n💡 Note: This is an AI prediction. Always consult a doctor for medical diagnosis."
    
    # Display text
    axes[1].text(0.5, 0.5, result_text, 
                ha='center', va='center',
                fontsize=11, 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.9, pad=15))
    
    plt.suptitle('COVID-19 Chest X-ray Analysis System', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save and show
    output_path = 'prediction_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Prediction visualization saved as '{output_path}'")
    return output_path

def analyze_folder(folder_path, model):
    """Analyze all images in a folder."""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found!")
        return
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("❌ No image files found in the folder.")
        return
    
    print(f"\nFound {len(image_files)} image(s) in '{folder_path}'")
    print("="*60)
    
    results = []
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f"\n[{i}/{len(image_files)}] Analyzing: {img_file}")
        
        result = predict_single_image(model, img_path)
        if result:
            results.append({
                'filename': img_file,
                'prediction': result['predicted_class'],
                'confidence': result['confidence'],
                'description': result['description'].split('\n')[0]
            })
            
            print(f"   → Prediction: {result['predicted_class']} ({result['confidence']:.2%})")
    
    # Save results to CSV
    if results:
        import csv
        
        output_file = 'batch_predictions.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'prediction', 'confidence', 'description'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✅ Batch predictions saved to '{output_file}'")
        
        # Summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        print("="*40)
        
        for cls in CLASSES:
            count = sum(1 for r in results if r['prediction'] == cls)
            if count > 0:
                confidences = [r['confidence'] for r in results if r['prediction'] == cls]
                avg_conf = np.mean(confidences)
                max_conf = np.max(confidences)
                min_conf = np.min(confidences)
                
                print(f"\n  {cls}:")
                print(f"    Count: {count} image(s)")
                print(f"    Avg Confidence: {avg_conf:.2%}")
                print(f"    Max Confidence: {max_conf:.2%}")
                print(f"    Min Confidence: {min_conf:.2%}")
        
        # Overall statistics
        print("\n📈 OVERALL STATISTICS:")
        print("="*40)
        print(f"  Total Images Analyzed: {len(results)}")
        print(f"  Most Common Prediction: {max(set(r['prediction'] for r in results), key=[r['prediction'] for r in results].count)}")
        
        # Count COVID cases
        covid_count = sum(1 for r in results if r['prediction'] == 'COVID')
        if covid_count > 0:
            print(f"\n⚠️  ALERT: {covid_count} potential COVID-19 case(s) detected!")
            print("   Please ensure these patients receive immediate medical attention.")

# ==================== MAIN APPLICATION ====================
def main():
    """Main application function."""
    print("\n" + "="*70)
    print("🏥 COVID-19 CHEST X-RAY DETECTION SYSTEM")
    print("="*70)
    
    # Load pre-trained model
    model_result = load_model()
    if not model_result:
        print("\n❌ Failed to load model. Exiting...")
        return
    
    model, checkpoint = model_result
    
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. 🖼️  Analyze a single chest X-ray image")
        print("2. 📁 Analyze all images in a folder")
        print("3. 📊 View model performance")
        print("4. 🚪 Exit")
        print("-"*50)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            print("SINGLE IMAGE ANALYSIS")
            print("="*50)
            print("\nEnter the path to your chest X-ray image.")
            print("Supported formats: PNG, JPG, JPEG, BMP, TIFF")
            print("\nExample: /Users/name/Desktop/xray.png")
            print("Or drag and drop your image here:")
            
            image_path = input("\nImage path: ").strip()
            
            # Remove quotes if user drag-drops
            image_path = image_path.strip('"').strip("'")
            
            if not image_path or not os.path.exists(image_path):
                print("❌ Invalid path or file not found.")
                print("Please check the path and try again.")
                continue
            
            print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
            print("-"*50)
            
            # Predict
            result = predict_single_image(model, image_path)
            
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
                
                # Ask if user wants visualization
                visualize = input("\nGenerate visualization? (yes/no): ").strip().lower()
                if visualize in ['yes', 'y', '']:
                    # Visualize
                    model_name = checkpoint.get('model_type', 'custom').upper()
                    if model_name == 'CUSTOM':
                        model_name = "Custom CNN Model"
                    else:
                        model_name = "ResNet50 Model"
                    
                    visualize_prediction(result, model_name)
                else:
                    print("Visualization skipped.")
        
        elif choice == '2':
            print("\n" + "="*50)
            print("BATCH FOLDER ANALYSIS")
            print("="*50)
            print("\nEnter the folder path containing chest X-ray images:")
            print("Example: /Users/name/Desktop/xray_folder/")
            
            folder_path = input("\nFolder path: ").strip()
            folder_path = folder_path.strip('"').strip("'")
            
            if not folder_path or not os.path.exists(folder_path):
                print("❌ Invalid folder path.")
                continue
            
            analyze_folder(folder_path, model)
        
        elif choice == '3':
            print("\n" + "="*50)
            print("MODEL PERFORMANCE")
            print("="*50)
            
            if checkpoint:
                print(f"\n📈 PERFORMANCE METRICS:")
                print(f"   Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
                print(f"   Test Accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")
                print(f"   Training Accuracy: {checkpoint.get('train_acc', 'N/A'):.2f}%")
                print(f"   Trained for: {checkpoint.get('epoch', 'N/A') + 1 if 'epoch' in checkpoint else 'N/A'} epochs")
                print(f"   Classes: {', '.join(checkpoint.get('classes', CLASSES))}")
                print(f"   Model Type: {checkpoint.get('model_type', 'custom').upper()}")
                
                # Show training history plot if available
                if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
                    print("\n📊 Training History Available")
                    show_history = input("Show training history plot? (yes/no): ").strip().lower()
                    if show_history in ['yes', 'y']:
                        # Plot training history
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        
                        # Loss plot
                        axes[0].plot(checkpoint['train_losses'], 'b-', linewidth=2, label='Train Loss', marker='o')
                        axes[0].plot(checkpoint['val_losses'], 'r-', linewidth=2, label='Val Loss', marker='s')
                        axes[0].set_xlabel('Epoch', fontsize=12)
                        axes[0].set_ylabel('Loss', fontsize=12)
                        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
                        axes[0].legend(fontsize=11)
                        axes[0].grid(True, alpha=0.3)
                        
                        # Accuracy plot
                        axes[1].plot(checkpoint['train_accs'], 'b-', linewidth=2, label='Train Accuracy', marker='o')
                        axes[1].plot(checkpoint['val_accs'], 'r-', linewidth=2, label='Val Accuracy', marker='s')
                        axes[1].set_xlabel('Epoch', fontsize=12)
                        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
                        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
                        axes[1].legend(fontsize=11)
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.show()
                
                # Show sample prediction if available
                if os.path.exists('prediction_result.png'):
                    print("\n🖼️  Sample Prediction Visualization:")
                    img = Image.open('prediction_result.png')
                    plt.figure(figsize=(10, 5))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title('Sample Prediction Result', fontsize=14, fontweight='bold')
                    plt.show()
        
        elif choice == '4':
            print("\n" + "="*50)
            print("Thank you for using COVID-19 Detection System!")
            print("Stay safe and healthy! 🩺")
            print("="*50)
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

# ==================== QUICK PREDICTION MODE ====================
def quick_predict(image_path):
    """Quick prediction mode for command line usage."""
    print("\n" + "="*70)
    print("QUICK PREDICTION MODE")
    print("="*70)
    
    # Load model
    model_result = load_model()
    if not model_result:
        return
    
    model, checkpoint = model_result
    
    # Make prediction
    result = predict_single_image(model, image_path)
    
    if result:
        print(f"\n📊 PREDICTION RESULTS for '{os.path.basename(image_path)}':")
        print("="*50)
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nDescription: {result['description'].split('   ')[0]}")
        
        print("\nDetailed Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            indicator = "✅" if class_name == result['predicted_class'] else "  "
            print(f"  {indicator} {class_name:20} {prob:.2%}")
        
        # Automatically create visualization
        model_name = checkpoint.get('model_type', 'custom').upper()
        if model_name == 'CUSTOM':
            model_name = "Custom CNN Model"
        else:
            model_name = "ResNet50 Model"
        
        visualize_prediction(result, model_name)

# ==================== RUN THE APPLICATION ====================
if __name__ == "__main__":
    import sys
    
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            quick_predict(image_path)
        else:
            print(f"❌ Error: File '{image_path}' not found!")
            print("Starting interactive mode...\n")
            main()
    else:
        # Start interactive mode
        main()