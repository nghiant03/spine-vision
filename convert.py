import os
import shutil
import SimpleITK as sitk
import json
import numpy as np

# --- CONFIGURATION ---
# Path where you downloaded the SPIDER data
INPUT_IMAGES_DIR = "./SPIDER/images" 
INPUT_MASKS_DIR = "./SPIDER/masks"

# Path to your nnUNet_raw folder defined in Step 1
OUTPUT_DIR = "./Spider_nnUNet/nnUNet_raw/Dataset501_Spider"

# --- LABEL MAPPING ---
# We must map SPIDER labels to contiguous integers (0, 1, 2...)
# 0 is always background.
# Example mapping strategy (Customize as needed):
# 1-25 -> Vertebrae (Keep as is)
# 100 -> Spinal Canal (Map to 26)
# 201-225 -> Discs (Map to 27-51)

def get_label_mapping():
    mapping = {0: 0}
    new_id = 1
    
    label_names = {"0": "background"}

    # Map Vertebrae (1-25)
    for i in range(1, 26):
        mapping[i] = new_id
        label_names[str(new_id)] = f"Vertebra_{i}"
        new_id += 1
        
    # Map Spinal Canal (100)
    mapping[100] = new_id
    label_names[str(new_id)] = "Spinal_Canal"
    new_id += 1
    
    # Map Discs (201-225)
    for i in range(201, 226):
        mapping[i] = new_id
        label_names[str(new_id)] = f"Disc_{i}"
        new_id += 1
        
    return mapping, label_names

def convert_data():
    # Create folders
    imagesTr = os.path.join(OUTPUT_DIR, "imagesTr")
    labelsTr = os.path.join(OUTPUT_DIR, "labelsTr")
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    mapping, label_names = get_label_mapping()
    
    files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith('.mha')]
    files.sort()
    
    print(f"Found {len(files)} files. Starting conversion...")

    for filename in files:
        case_id = filename.replace(".mha", "")
        
        # 1. Process Image
        img_path = os.path.join(INPUT_IMAGES_DIR, filename)
        img_itk = sitk.ReadImage(img_path)
        
        # Save as .nii.gz
        # nnUNet requires _0000 suffix for channel 0
        new_img_name = f"{case_id}_0000.nii.gz"
        sitk.WriteImage(img_itk, os.path.join(imagesTr, new_img_name))

        # 2. Process Mask
        mask_path = os.path.join(INPUT_MASKS_DIR, filename)
        if os.path.exists(mask_path):
            mask_itk = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_itk)
            
            # REMAP LABELS FAST
            new_mask_arr = np.zeros_like(mask_arr)
            for old_lbl, new_lbl in mapping.items():
                new_mask_arr[mask_arr == old_lbl] = new_lbl
            
            new_mask_itk = sitk.GetImageFromArray(new_mask_arr)
            new_mask_itk.CopyInformation(mask_itk)
            
            sitk.WriteImage(new_mask_itk, os.path.join(labelsTr, f"{case_id}.nii.gz"))
        else:
            print(f"Warning: No mask found for {filename}")

    # 3. Create dataset.json
    generate_json(len(files), label_names)

def generate_json(num_training, label_names):
    json_dict = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": label_names,
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4)
    print("dataset.json created.")

if __name__ == "__main__":
    convert_data()
