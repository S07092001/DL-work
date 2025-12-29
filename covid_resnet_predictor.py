# covid_resnet_predictor.py
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COVID-19 RESNET50 PREDICTION SYSTEM")
print("="*70)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Classes
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']
NUM_CLASSES = len(CLASSES)

# Image transformation for ResNet50
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # ResNet typically uses 224x224, but we'll use 256
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== RESNET50 MODEL ====================
class COVIDResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Load ResNet50
        self.resnet = models.resnet50(pretrained=False)
        
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
    
    def forward(self, x):
        return self.resnet(x)

# ==================== MODEL LOADING ====================
def load_resnet_model(model_path='best_resnet_model.pth'):
    """Load pre-trained ResNet50 model"""
    
    if not os.path.exists(model_path):
        # Try alternative names
        alternative_paths = [
            'resnet_covid_model.pth',
            'resnet_model.pth',
            'best_covid_model.pth',
            'model_resnet.pth'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Found model at: {model_path}")
                break
        else:
            print(f"❌ Error: Model file not found!")
            print("Please make sure you have one of these files:")
            print("  - best_resnet_model.pth")
            print("  - resnet_covid_model.pth")
            print("  - best_covid_model.pth")
            return None
    
    print(f"Loading ResNet50 model from '{model_path}'...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        model = COVIDResNet50(num_classes=NUM_CLASSES).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ ResNet50 model loaded successfully!")
        
        # Display model info
        if 'val_acc' in checkpoint:
            print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        if 'test_acc' in checkpoint:
            print(f"   Test Accuracy: {checkpoint['test_acc']:.2f}%")
        if 'train_acc' in checkpoint:
            print(f"   Training Accuracy: {checkpoint['train_acc']:.2f}%")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        
        return model, checkpoint
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTrying to load with alternative method...")
        
        try:
            # Try loading just the state dict directly
            model = COVIDResNet50(num_classes=NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("✅ Model loaded successfully (direct state dict)")
            return model, {}
        except:
            print("❌ Failed to load model.")
            return None

# ==================== PREDICTION FUNCTIONS ====================
def predict_with_resnet(model, image_path):
    """Predict COVID-19 from a single image using ResNet50."""
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        return None
    
    # Load and preprocess image
    try:
        original_image = Image.open(image_path).convert('RGB')
        print(f"✅ Image loaded: {image_path}")
        print(f"   Original size: {original_image.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    # Apply transformations
    input_tensor = test_transform(original_image).unsqueeze(0).to(device)
    
    # Make prediction
    print("🔍 Making prediction with ResNet50...")
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
    
    print(f"✅ Prediction complete!")
    
    return {
        'original_image': original_image,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'description': descriptions[predicted_class],
        'class_index': predicted_class_idx,
        'original_size': original_image.size,
        'model_name': 'ResNet50'
    }

def create_resnet_visualization(prediction_result):
    """Create beautiful visualization for ResNet50 prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Original Image
    axes[0].imshow(prediction_result['original_image'], cmap='gray')
    axes[0].set_title('Input Chest X-ray', fontsize=16, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    # Add border color based on prediction
    if prediction_result['predicted_class'] == 'COVID':
        border_color = '#FF6B6B'  # Red for COVID
        severity = 'HIGH RISK'
    elif prediction_result['predicted_class'] == 'Viral Pneumonia':
        border_color = '#FFA726'  # Orange for Pneumonia
        severity = 'MEDIUM RISK'
    else:
        border_color = '#4CAF50'  # Green for Normal
        severity = 'LOW RISK'
    
    for spine in axes[0].spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(4)
    
    # Add image info
    img_info = f"Size: {prediction_result['original_size'][0]}×{prediction_result['original_size'][1]}\nSeverity: {severity}"
    axes[0].text(0.5, -0.05, img_info, ha='center', va='top', 
                 transform=axes[0].transAxes, fontsize=11, 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Right: Prediction Results
    axes[1].axis('off')
    
    # Create result text with ResNet50 specific info
    result_text = f"""
    🏥 COVID-19 DETECTION RESULT
    {'='*45}
    
    🔬 MODEL: ResNet50 (Transfer Learning)
    ⭐ PRETRAINED: ImageNet + COVID-19 Fine-tuning
    
    🎯 PREDICTION: {prediction_result['predicted_class']}
    
    📊 CONFIDENCE: {prediction_result['confidence']:.2%}
    
    📋 CLINICAL ASSESSMENT:
    {prediction_result['description']}
    
    📈 PROBABILITY DISTRIBUTION:
    """
    
    # Add probability bars with color coding
    colors = {'COVID': '#FF6B6B', 'Normal': '#4CAF50', 'Viral Pneumonia': '#FFA726'}
    
    for class_name, prob in prediction_result['all_probabilities'].items():
        bar_length = int(prob * 35)  # Scale for visualization
        bar = '█' * bar_length + '░' * (35 - bar_length)
        highlight = '🏆 ' if class_name == prediction_result['predicted_class'] else '  '
        
        # Add color to class name
        colored_name = f"\033[38;2;{int(colors[class_name][1:3], 16)};{int(colors[class_name][3:5], 16)};{int(colors[class_name][5:7], 16)}m{class_name}\033[0m"
        result_text += f"\n{highlight}{class_name:20} {bar} {prob:.2%}"
    
    # Add confidence meter
    confidence_level = prediction_result['confidence'] * 100
    if confidence_level >= 90:
        confidence_emoji = '🎯'  # Bullseye
        confidence_text = 'VERY HIGH CONFIDENCE'
    elif confidence_level >= 70:
        confidence_emoji = '✅'  # Check mark
        confidence_text = 'HIGH CONFIDENCE'
    elif confidence_level >= 50:
        confidence_emoji = '⚠️'   # Warning
        confidence_text = 'MODERATE CONFIDENCE'
    else:
        confidence_emoji = '❓'   # Question mark
        confidence_text = 'LOW CONFIDENCE'
    
    result_text += f"\n\n{confidence_emoji} CONFIDENCE LEVEL: {confidence_text}"
    
    # Add medical advice
    if prediction_result['predicted_class'] == 'COVID':
        result_text += f"\n\n⚠️  URGENT MEDICAL ATTENTION REQUIRED"
        result_text += f"\n   • Isolate immediately"
        result_text += f"\n   • Contact healthcare provider"
        result_text += f"\n   • Monitor for symptoms"
    elif prediction_result['predicted_class'] == 'Viral Pneumonia':
        result_text += f"\n\n⚠️  MEDICAL CONSULTATION ADVISED"
        result_text += f"\n   • Schedule doctor appointment"
        result_text += f"\n   • Rest and stay hydrated"
        result_text += f"\n   • Monitor fever"
    else:
        result_text += f"\n\n✅ NO IMMEDIATE CONCERNS"
        result_text += f"\n   • Continue regular checkups"
        result_text += f"\n   • Maintain healthy lifestyle"
    
    result_text += f"\n\n💡 AI DISCLAIMER:"
    result_text += f"\n   This is an AI-assisted prediction using ResNet50."
    result_text += f"\n   Always consult with a qualified healthcare professional"
    result_text += f"\n   for medical diagnosis and treatment."
    
    # Display text with better formatting
    axes[1].text(0.5, 0.5, result_text, 
                ha='center', va='center',
                fontsize=11.5, 
                fontfamily='DejaVu Sans Mono',
                linespacing=1.4,
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                         edgecolor='#dee2e6', alpha=0.95, pad=20))
    
    plt.suptitle('COVID-19 Chest X-ray Analysis - ResNet50 Deep Learning Model', 
                 fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')
    plt.tight_layout()
    
    # Save with high quality
    timestamp = np.datetime64('now').astype('datetime64[s]').astype(str).replace(':', '')
    output_path = f'resnet_prediction_{timestamp}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✅ ResNet50 prediction visualization saved as '{output_path}'")
    return output_path

def print_terminal_results(prediction_result):
    """Print prediction results in terminal with colors."""
    print("\n" + "="*70)
    print("🎯 RESNET50 PREDICTION RESULTS")
    print("="*70)
    
    # Print prediction with color
    if prediction_result['predicted_class'] == 'COVID':
        prediction_color = '\033[91m'  # Red
    elif prediction_result['predicted_class'] == 'Viral Pneumonia':
        prediction_color = '\033[93m'  # Yellow
    else:
        prediction_color = '\033[92m'  # Green
    
    print(f"\n{prediction_color}PREDICTED CLASS: {prediction_result['predicted_class']}\033[0m")
    print(f"CONFIDENCE: \033[1m{prediction_result['confidence']:.2%}\033[0m")
    
    print(f"\n\033[94mDESCRIPTION:\033[0m")
    for line in prediction_result['description'].split('\n'):
        print(f"  {line}")
    
    print(f"\n\033[94mPROBABILITY DISTRIBUTION:\033[0m")
    for class_name, prob in prediction_result['all_probabilities'].items():
        bar_length = int(prob * 40)
        bar = '█' * bar_length
        
        if class_name == 'COVID':
            color_code = '\033[91m'  # Red
        elif class_name == 'Normal':
            color_code = '\033[92m'  # Green
        else:
            color_code = '\033[93m'  # Yellow
        
        if class_name == prediction_result['predicted_class']:
            indicator = '✅'
        else:
            indicator = '  '
        
        print(f"  {indicator} {color_code}{class_name:18}\033[0m {bar:40} {prob:6.2%}")
    
    print("\n" + "="*70)

# ==================== BATCH PROCESSING ====================
def batch_process_folder(model, folder_path):
    """Process all images in a folder."""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found!")
        return
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.jfif', '.webp']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("❌ No image files found in the folder.")
        return
    
    print(f"\n📁 Found {len(image_files)} image(s) in '{folder_path}'")
    print("="*70)
    
    results = []
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, img_file)
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file}")
        
        result = predict_with_resnet(model, img_path)
        if result:
            results.append({
                'filename': img_file,
                'prediction': result['predicted_class'],
                'confidence': result['confidence'],
                'description': result['description'].split('\n')[0]
            })
            
            # Print brief result
            if result['predicted_class'] == 'COVID':
                print(f"   🚨 → \033[91mCOVID-19\033[0m ({result['confidence']:.2%})")
            elif result['predicted_class'] == 'Viral Pneumonia':
                print(f"   ⚠️  → \033[93mViral Pneumonia\033[0m ({result['confidence']:.2%})")
            else:
                print(f"   ✅ → \033[92mNormal\033[0m ({result['confidence']:.2%})")
    
    # Save results to CSV
    if results:
        import csv
        
        output_file = 'resnet_batch_results.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'prediction', 'confidence', 'description'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✅ Batch predictions saved to '{output_file}'")
        
        # Generate summary
        print("\n" + "="*70)
        print("📊 RESNET50 BATCH ANALYSIS SUMMARY")
        print("="*70)
        
        for cls in CLASSES:
            count = sum(1 for r in results if r['prediction'] == cls)
            if count > 0:
                confidences = [r['confidence'] for r in results if r['prediction'] == cls]
                avg_conf = np.mean(confidences)
                
                if cls == 'COVID':
                    print(f"\n🚨 \033[91mCOVID-19 CASES: {count}\033[0m")
                    print(f"   Average Confidence: {avg_conf:.2%}")
                    if count > 0:
                        print(f"   \033[91m⚠️  URGENT: These cases require immediate attention!\033[0m")
                elif cls == 'Viral Pneumonia':
                    print(f"\n⚠️  \033[93mVIRAL PNEUMONIA CASES: {count}\033[0m")
                    print(f"   Average Confidence: {avg_conf:.2%}")
                else:
                    print(f"\n✅ \033[92mNORMAL CASES: {count}\033[0m")
                    print(f"   Average Confidence: {avg_conf:.2%}")
        
        print(f"\n📈 TOTAL IMAGES ANALYZED: {len(results)}")
        print("="*70)

# ==================== MAIN APPLICATION ====================
def main():
    """Main interactive application."""
    print("\n" + "="*70)
    print("🏥 COVID-19 RESNET50 ANALYSIS SYSTEM")
    print("="*70)
    
    # Load model
    print("\n📥 Loading ResNet50 model...")
    model_result = load_resnet_model()
    
    if not model_result:
        print("\n❌ Failed to load ResNet50 model.")
        print("Please ensure you have a trained ResNet50 model file.")
        return
    
    model, checkpoint = model_result
    
    while True:
        print("\n" + "="*50)
        print("RESNET50 MAIN MENU")
        print("="*50)
        print("1. 🖼️  Analyze Single Chest X-ray")
        print("2. 📁 Batch Process Folder")
        print("3. 📊 View Model Information")
        print("4. 🎨 Create Sample Visualization")
        print("5. 🚪 Exit")
        print("-"*50)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            print("SINGLE IMAGE ANALYSIS - RESNET50")
            print("="*50)
            print("\nEnter the path to your chest X-ray image.")
            print("You can also drag and drop the image file here.")
            print("\nExample: /Users/name/Desktop/chest_xray.png")
            
            image_path = input("\n📁 Image path: ").strip()
            
            # Clean up path (remove quotes from drag-drop)
            image_path = image_path.strip('"').strip("'")
            
            if not image_path or not os.path.exists(image_path):
                print("❌ Invalid path or file not found.")
                continue
            
            print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
            print("-"*50)
            
            # Make prediction
            result = predict_with_resnet(model, image_path)
            
            if result:
                # Print terminal results
                print_terminal_results(result)
                
                # Ask for visualization
                visualize = input("\nGenerate visualization image? (yes/no): ").strip().lower()
                if visualize in ['yes', 'y', '']:
                    create_resnet_visualization(result)
        
        elif choice == '2':
            print("\n" + "="*50)
            print("BATCH FOLDER PROCESSING - RESNET50")
            print("="*50)
            print("\nEnter the folder path containing chest X-ray images.")
            print("All images in the folder will be analyzed.")
            
            folder_path = input("\n📁 Folder path: ").strip()
            folder_path = folder_path.strip('"').strip("'")
            
            if not folder_path or not os.path.exists(folder_path):
                print("❌ Invalid folder path.")
                continue
            
            batch_process_folder(model, folder_path)
        
        elif choice == '3':
            print("\n" + "="*50)
            print("RESNET50 MODEL INFORMATION")
            print("="*50)
            
            print(f"\n🔬 Model Architecture: ResNet50")
            print(f"   • Depth: 50 layers")
            print(f"   • Pretrained: ImageNet")
            print(f"   • Fine-tuned: COVID-19 Detection")
            print(f"   • Input size: 256×256 pixels")
            print(f"   • Output classes: {NUM_CLASSES}")
            
            if checkpoint:
                print(f"\n📊 Training Performance:")
                if 'val_acc' in checkpoint:
                    print(f"   • Validation Accuracy: {checkpoint['val_acc']:.2f}%")
                if 'test_acc' in checkpoint:
                    print(f"   • Test Accuracy: {checkpoint['test_acc']:.2f}%")
                if 'train_acc' in checkpoint:
                    print(f"   • Training Accuracy: {checkpoint['train_acc']:.2f}%")
                if 'epoch' in checkpoint:
                    print(f"   • Training Epochs: {checkpoint['epoch'] + 1}")
            
            # Show sample architecture
            print(f"\n🔄 Model Layers:")
            print(f"   • Initial Conv (7×7)")
            print(f"   • Max Pooling")
            print(f"   • 4 ResNet Blocks")
            print(f"   • Average Pooling")
            print(f"   • Custom Classifier Head")
            print(f"   • Softmax Output")
        
        elif choice == '4':
            print("\n" + "="*50)
            print("SAMPLE VISUALIZATION - RESNET50")
            print("="*50)
            
            # Create a sample visualization
            print("\nCreating a sample visualization of ResNet50 prediction...")
            
            # Create a sample image (gray X-ray like)
            sample_image = Image.new('RGB', (512, 512), color='gray')
            
            # Create sample prediction result
            sample_result = {
                'original_image': sample_image,
                'predicted_class': 'COVID',
                'confidence': 0.8519,
                'all_probabilities': {'COVID': 0.8519, 'Normal': 0.1323, 'Viral Pneumonia': 0.0158},
                'description': '🦠 COVID-19 Infection Detected\n   Immediate medical attention recommended.',
                'class_index': 0,
                'original_size': (512, 512),
                'model_name': 'ResNet50'
            }
            
            print("\n📊 Sample Prediction:")
            print(f"   Predicted: COVID-19")
            print(f"   Confidence: 85.19%")
            print(f"   Model: ResNet50")
            
            create_resnet_visualization(sample_result)
        
        elif choice == '5':
            print("\n" + "="*50)
            print("Thank you for using ResNet50 COVID-19 Detection System!")
            print("Stay safe and healthy! 🩺")
            print("="*50)
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

# ==================== COMMAND LINE INTERFACE ====================
def command_line_predict(image_path):
    """Command line interface for quick predictions."""
    print("\n" + "="*70)
    print("🚀 RESNET50 QUICK PREDICTION MODE")
    print("="*70)
    
    # Load model
    model_result = load_resnet_model()
    if not model_result:
        return
    
    model, _ = model_result
    
    # Make prediction
    result = predict_with_resnet(model, image_path)
    
    if result:
        print_terminal_results(result)
        
        # Automatically create visualization
        print("\n📸 Generating visualization...")
        create_resnet_visualization(result)

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Check if it's a file or folder
        if os.path.isfile(image_path):
            # Single file prediction
            command_line_predict(image_path)
        elif os.path.isdir(image_path):
            # Batch processing
            model_result = load_resnet_model()
            if model_result:
                model, _ = model_result
                batch_process_folder(model, image_path)
        else:
            print(f"❌ Error: '{image_path}' not found!")
            print("\nStarting interactive mode...")
            main()
    else:
        # Start interactive mode
        main()