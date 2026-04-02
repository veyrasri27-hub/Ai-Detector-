# detector.py - Our first Deepfake Image Detector
# Run with: python detector.py

import sys
import os

# === CHECK & INSTALL MISSING PACKAGES ===
def check_and_install_dependencies():
    """Check if required packages are installed, install if missing."""
    required_packages = {
        'transformers': 'transformers',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'PIL': 'pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy'
    }
    
    missing = []
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {pip_name} is installed")
        except ImportError:
            print(f"✗ {pip_name} is MISSING")
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠️ Installing missing packages: {', '.join(missing)}")
        import subprocess
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print("✓ Installation complete!\n")

# Run the check at startup
check_and_install_dependencies()

# === SAFE IMPORTS WITH ERROR HANDLING ===
try:
    from transformers import ViTImageProcessor, ViTForImageClassification
    print("✓ Transformers imported successfully")
except ImportError as e:
    print(f"✗ ERROR: Could not import transformers: {e}")
    print("  → Run: pip install transformers")
    sys.exit(1)

try:
    from PIL import Image
    print("✓ Pillow (PIL) imported successfully")
except ImportError:
    print("✗ ERROR: Could not import PIL. Run: pip install pillow")
    sys.exit(1)

try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError:
    print("✗ ERROR: Could not import torch. Run: pip install torch")
    sys.exit(1)

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError:
    print("✗ ERROR: Could not import numpy. Run: pip install numpy")
    sys.exit(1)

# === Configuration ===
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

# Change this to your actual image path!
# Example: r"C:\Users\saich\Pictures\my_selfie.jpg"  (use r"" for Windows paths with backslashes)
IMAGE_PATH = r"C:\Users\saich\OneDrive\Documents\Ai detector\Ai detector\image.png" # ← EDIT THIS LINE

# === Load Model & Processor with Error Handling ===
print("\n" + "="*70)
print("Loading model... (first time may take 1-2 minutes)")
print("="*70)

try:
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    print("✓ Processor loaded successfully")
except Exception as e:
    print(f"✗ ERROR loading processor: {e}")
    print("  → Check your internet connection")
    print("  → Model URL may be invalid")
    sys.exit(1)

try:
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    print("  → Check your internet connection")
    print("  → You may need ~500MB disk space")
    sys.exit(1)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✓ Model moved to {device.upper()}")
print("="*70 + "\n")

# === Function to Detect ===
def detect_deepfake(image_path):
    try:
        # Open and prepare image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Send to GPU/CPU

        # Run inference (no gradients needed)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Get predicted class and probabilities
        predicted_idx = torch.argmax(logits, dim=1).item()
        label = model.config.id2label[predicted_idx]  # "Realism" or "Deepfake"
        prob_real = probabilities[model.config.label2id["Realism"]] if "Realism" in model.config.label2id else 0
        prob_fake = probabilities[model.config.label2id["Deepfake"]] if "Deepfake" in model.config.label2id else 0

        # Output results
        print("\n" + "="*50)
        print(f"Image: {image_path}")
        print(f"Prediction: {label.upper()}")
        print(f"Confidence:")
        print(f"  Real:    {prob_real:.2%}")
        print(f"  Fake:    {prob_fake:.2%}")
        print("="*50 + "\n")

    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure the file exists and is a valid image (jpg/png/etc.)")

# === Run the detection ===
if __name__ == "__main__":
    if IMAGE_PATH == r"path\to\your\test_image.jpg":
        print("ERROR: Please edit the IMAGE_PATH variable in the code first!")
        sys.exit(1)
    
    detect_deepfake(IMAGE_PATH)