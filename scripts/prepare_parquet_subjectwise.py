import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import gc
import random

def resample_to_inplane_1mm(image, is_mask=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_spacing = (1.0, 1.0, original_spacing[2])
    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        original_size[2]
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    
    return resample.Execute(image)

def process_medical_3d(path, is_mask=False):
    try:
        img = sitk.ReadImage(path)
        img = sitk.DICOMOrient(img, 'LPI')
        img = resample_to_inplane_1mm(img, is_mask=is_mask)
        data_bytes = sitk.GetArrayFromImage(img).astype(np.float32).tobytes()
        header_info = {
            "spacing": img.GetSpacing(),
            "direction": img.GetDirection(),
            "size": img.GetSize()
        }
        del img
        return data_bytes, json.dumps(header_info)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, None

# 配置路径
data_path = r'../../data/RadGenome-Brain_MRI/all_data_Reg'
out_path = r'../../data/RadGenome-Brain_MRI/all_parquet_Reg'
modality_wise_finding = r'../../data/RadGenome-Brain_MRI/combined_modal_wise_finding.json'
impression = r'../../data/RadGenome-Brain_MRI/combined_impression.json'
global_finding = r'../../data/RadGenome-Brain_MRI/combined_global_finding.json'
meta_output_path = r'../../data/RadGenome-Brain_MRI/dataset_info_Reg.json'

os.makedirs(out_path, exist_ok=True)

with open(modality_wise_finding, 'r') as f:
    modality_finding_dict = json.load(f)

with open(impression, 'r') as f:
    impression_dict = json.load(f)

with open(global_finding, 'r') as f:
    global_finding_dict = json.load(f)

# --- 参数设置 ---
SAMPLES_PER_CHUNK = 5
ROW_GROUP_SIZE = 1     

all_rows = []
chunk_count = 0
chunk_metadata = {}

def save_chunk(rows, count):
    df = pd.DataFrame(rows)
    schema = pa.schema([
        pa.field("subject_id", pa.string()),
        pa.field("image_list", pa.list_(pa.binary())),
        pa.field("modality_names", pa.list_(pa.string())),
        pa.field("modality_findings", pa.list_(pa.string())),
        pa.field("impression", pa.string()),
        pa.field("disease", pa.list_(pa.string())),
        pa.field("global_finding", pa.string()),
        pa.field("header_metadata", pa.string()),
        pa.field("mask", pa.binary(), nullable=True)
    ])
    table = pa.Table.from_pandas(df, schema=schema)
    filename = f"chunk_{count}.parquet"
    full_path = os.path.join(out_path, filename)
    
    # 写入文件
    pq.write_table(table, full_path, compression='SNAPPY', row_group_size=ROW_GROUP_SIZE)
    
    meta = pq.read_metadata(full_path)
    chunk_metadata[full_path] = {
        "num_row_groups": meta.num_row_groups,
        "num_row": meta.num_rows,
        'subject_list': df['subject_id'].tolist()
    }

    del table
    del df
    gc.collect()
    
    print(f"Saved {filename}: {meta.num_rows} rows, {meta.num_row_groups} row groups.")

subjects = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]
random.shuffle(subjects)

for subject in subjects:
    print(f"Processing subject: {subject}")
    subject_dir = os.path.join(data_path, subject)
    
    # 1. Initialize subject-level containers
    subject_images = []
    subject_instructions = []
    subject_modality_names = [] # Optional: to track which image is which
    
    # Load shared mask for this subject
    mask_path = os.path.join(subject_dir, f"{subject}-seg.nii.gz")
    mask_bytes, header_json = process_medical_3d(mask_path, is_mask=True) if os.path.exists(mask_path) else (None, None)

    # 2. Iterate through all modality files for THIS subject
    for modality_file in os.listdir(subject_dir):
        if modality_file.endswith('.nii.gz') and not modality_file.endswith('-seg.nii.gz'):
            modality_key = modality_file.replace('.nii.gz', '')
            
            img_path = os.path.join(subject_dir, modality_file)
            raw_bytes, _ = process_medical_3d(img_path) # Header handling might need logic if different per modality
            if modality_key in modality_finding_dict:
                subject_images.append(raw_bytes)
                subject_instructions.append(modality_finding_dict[modality_key])
                subject_modality_names.append(modality_key)
            else:
                subject_images.append(raw_bytes)
                subject_instructions.append("")
                subject_modality_names.append(modality_key)

    # 3. Only append to all_rows if we found at least one modality
    if subject_images:
        all_rows.append({
            "subject_id": subject,
            "image_list": subject_images,             # List of binary blobs
            "modality_names": subject_modality_names, # Helps the VLM know which is T1/T2
            "modality_findings": subject_instructions, # List of strings per modality
            "impression": impression_dict.get(subject, {}).get('impression', ""),
            "disease": impression_dict.get(subject, {}).get('disease', []),
            "global_finding": global_finding_dict.get(subject, ""),
            "header_metadata": header_json,
            "mask": mask_bytes
        })

    # 4. Chunking logic
    if len(all_rows) >= SAMPLES_PER_CHUNK:
        save_chunk(all_rows, chunk_count)
        all_rows = []
        chunk_count += 1

if all_rows:
    save_chunk(all_rows, chunk_count)

with open(meta_output_path, 'w') as f:
    json.dump(chunk_metadata, f, indent=4)

print(f"\nProcessing finished. Metadata saved to: {meta_output_path}")