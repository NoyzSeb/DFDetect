import os
import glob
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image, write_jpeg
from tqdm import tqdm

RAW_DIR = Path("data/raw/FF32_extractedFrames")
PROCESSED_DIR = Path("data/processed/FF32_extractedFrames")
TARGET_SIZE = (480, 480)

FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
REAL_METHOD = "Original"

def get_corresponding_real_path(fake_img_path):
    """
    In this specific FF32_extractedFrames extraction, fake videos are directly in the 
    technique folder, named like: 000_003_f0.jpg
    (TargetVideo_SourceVideo_FrameNumber.jpg)
    
    The original (real) equivalent is in Original/000_f0.jpg
    """
    filename = fake_img_path.name # e.g. "000_003_f0.jpg"
    parts = filename.split('_')
    
    if len(parts) >= 3:
        target_id = parts[0] # "000"
        frame_id = parts[-1] # "f0.jpg"
        
        real_filename = f"{target_id}_{frame_id}"
        real_path = RAW_DIR / REAL_METHOD / real_filename
        return real_path
        
    return None

def pad_or_crop(img_tensor, target_size=TARGET_SIZE):
    """
    Center crops if image is larger than target_size.
    Pads with zeros if image is smaller than target_size.
    """
    _, h, w = img_tensor.shape
    th, tw = target_size
    
    # 1. Crop if larger
    if h > th or w > tw:
        img_tensor = TF.center_crop(img_tensor, output_size=(min(h, th), min(w, tw)))
        
    _, h, w = img_tensor.shape
    
    # 2. Pad if smaller
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h > 0 or pad_w > 0:
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        img_tensor = TF.pad(img_tensor, padding)
        
    return img_tensor

def compute_fft(img_tensor):
    """
    Computes 2D FFT and shifts the zero-frequency component to the center.
    Input must be float tensor.
    """
    # Convert to float [0, 1] if not already
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float() / 255.0
        
    fft_tensor = torch.fft.fft2(img_tensor, norm="ortho")
    fft_shifted = torch.fft.fftshift(fft_tensor)
    return fft_shifted

def generate_and_save_masks(real_tensor, fake_tensor, save_dir, base_name):
    """
    Calculates spatial and frequency difference masks and saves them.
    Assumes inputs are processed 480x480 tensors.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to float for math
    real_f = real_tensor.float() / 255.0
    fake_f = fake_tensor.float() / 255.0
    
    # 1. Spatial Mask: |Real - Fake|
    spatial_mask = torch.abs(real_f - fake_f)
    
    # 2. Frequency Mask: |FFT(Real) - FFT(Fake)|
    real_fft = compute_fft(real_f)
    fake_fft = compute_fft(fake_f)
    freq_mask = torch.abs(real_fft - fake_fft) # Magnitude of the complex difference
    
    # Save artifacts as .pt files
    # PyTorch serialization can sometimes encounter transient I/O bounds when writing 
    # many large tensors rapidly explicitly over Windows filesystems. Wrapping in try/except 
    # ensures safe fallbacks or retry loops.
    try:
        torch.save(spatial_mask.clone(), save_dir / f"{base_name}_spatial_mask.pt")
        torch.save(freq_mask.clone(), save_dir / f"{base_name}_freq_mask.pt")
        torch.save(fake_fft.clone(), save_dir / f"{base_name}_fake_fft.pt")
    except Exception as e:
        print(f"Error saving tensors for {base_name}: {e}")

def process_single_image(img_path, output_dir):
    """
    Processes a single image (crop/pad, FFT) and saves to processed folder.
    Returns processed spatial tensor for paired mask generation if needed.
    """
    img_path = Path(img_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        img_tensor = read_image(str(img_path))
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None
        
    # Crop / Pad
    processed_img = pad_or_crop(img_tensor)
    
    # Save processed spatial image
    write_jpeg(processed_img, str(output_dir / f"{img_path.stem}.jpg"))
    
    # Also compute & save the FFT representation for all independently processed images
    try:
        fft_tensor = compute_fft(processed_img)
        torch.save(fft_tensor.clone(), output_dir / f"{img_path.stem}_fft.pt")
    except Exception as e:
        print(f"Error saving FFT for {img_path.stem}: {e}")
    
    return processed_img

if __name__ == "__main__":
    if not RAW_DIR.exists():
        print(f"Directory {RAW_DIR} does not exist!")
        exit(1)

    # 1. Process all original (Real) images first
    print(f"Processing '{REAL_METHOD}' Images...")
    real_images = glob.glob(str(RAW_DIR / REAL_METHOD / "**" / "*.*"), recursive=True)
    real_images = [Path(p) for p in real_images if Path(p).is_file() and Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    for real_path in tqdm(real_images, desc="Originals"):
        rel_path = real_path.relative_to(RAW_DIR)
        out_dir = PROCESSED_DIR / rel_path.parent
        
        # Skip if already fully processed to save time
        if (out_dir / f"{real_path.stem}_fft.pt").exists():
            continue
            
        # This will crop/pad and save to formatted correctly for training usage
        process_single_image(real_path, out_dir)

    # 2. Process all Fake images, and calculate the masks
    print("Processing Fake Images and Generating Ground Truth Masks...")
    fake_images = []
    for method in FAKE_METHODS:
        images = glob.glob(str(RAW_DIR / method / "**" / "*.*"), recursive=True)
        fake_images.extend([Path(p) for p in images if Path(p).is_file() and Path(p).suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
    for fake_path in tqdm(fake_images, desc="Fakes"):
        fake_rel_path = fake_path.relative_to(RAW_DIR)
        fake_out_dir = PROCESSED_DIR / fake_rel_path.parent
        
        # Skip if already fully processed to save time
        if (fake_out_dir / f"{fake_path.stem}_fake_fft.pt").exists() and (fake_out_dir / f"{fake_path.stem}.jpg").exists():
            continue
            
        # 1. Find corresponding real image
        real_path = get_corresponding_real_path(fake_path)
        
        if not real_path or not real_path.exists():
            # If the original frame wasn't found for whatever reason, just process the fake itself
            process_single_image(fake_path, fake_out_dir)
            continue
            
        real_rel_path = real_path.relative_to(RAW_DIR)
        real_out_dir = PROCESSED_DIR / real_rel_path.parent
        
        # 2. Load the raw arrays 
        try:
            fake_tensor_raw = read_image(str(fake_path))
            real_tensor_raw = read_image(str(real_path))
        except Exception as e:
            continue
            
        # 3. Standardize dimensions (480x480)
        fake_tensor_proc = pad_or_crop(fake_tensor_raw)
        real_tensor_proc = pad_or_crop(real_tensor_raw)
        
        # 4. Save processed spatial output for the Fake image
        fake_out_dir.mkdir(parents=True, exist_ok=True)
        write_jpeg(fake_tensor_proc, str(fake_out_dir / f"{fake_path.stem}.jpg"))

        # 5. Math! Generate Diff Constraints and Save
        generate_and_save_masks(real_tensor_proc, fake_tensor_proc, fake_out_dir, fake_path.stem)
