import io
import random
import json
import torch
import numpy as np
from PIL import Image
from .edit_dataset import UnifiedEditIterableDataset
from ..data_utils import pil_img2rgb
from ..mask_util import random_bbox, bbox2mask, mask2bbox, get_irregular_mask, brush_stroke_mask
from .utils import pad_to_target, translation_question_list, get_impression_question, get_diagnosis_question, translation_think_list, convert_list_to_string, rescale_intensity, change_size_description, segmentation_question_list

import torch.distributed as dist

from data.consts import get_recon_prompt_list
prompt_templates = get_recon_prompt_list()

def read_buffer_as_array(buffer, shape, dtype=np.float32):
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)

class ReconDataset(UnifiedEditIterableDataset):
    """
    Modified dataset for Medical VLM Grounded generation (inpainting).
    Converts 3D NIfTI volumes into 2D slices on-the-fly.
    Target (edited image) is the original image.
    """

    def shared_randomness(self, random_range):
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            random_value = torch.randint(random_range[0], random_range[1], (1,), dtype=torch.long, device='cuda')
        else:
            random_value = torch.empty(1, dtype=torch.long, device='cuda')

        dist.broadcast(random_value, src=0)  # Broadcast the random value to all ranks
        return random_value.item() if isinstance(random_value, torch.Tensor) else random_value
    
    def parse_row(self, row):
        # 1. Load the 3D Image and 3D Mask from raw bytes
        # We assume the first item in image_list is the main 3D volume
        shape = json.loads(row["header_metadata"])["size"]
        shape = [shape[2], shape[1], shape[0]]  # NIfTI to (H, W, D) ordering
        
        # Reconstruct 3D arrays (H, W, D)
        subject_images = row["image_list"]
        subject_modality_names = row["modality_names"]
        mask_3d = read_buffer_as_array(row["mask"], shape)

        data = self._init_data()

        modality_idx = random.randint(0, len(subject_images) - 1)
        vol_3d = read_buffer_as_array(subject_images[modality_idx], shape)

        # 2. Select a random slice index (random axis)
        # We ensure the slice actually contains some mask data to make training effective
        axis = random.choice([0, 1, 2])
        if axis == 0:
            vol_3d = np.transpose(vol_3d, (1, 2, 0))
            mask_3d = np.transpose(mask_3d, (1, 2, 0))
        elif axis == 1:
            vol_3d = np.transpose(vol_3d, (0, 2, 1))
            mask_3d = np.transpose(mask_3d, (0, 2, 1))
        
        mask_indices = np.where(np.any(mask_3d > 0, axis=(0, 1)))[0]
        if len(mask_indices) > 0: # 80% chance to pick a slice with a mask
            slice_idx = random.choice(mask_indices)
            random_mask_tag = False
            mask_slice = mask_3d[:, :, slice_idx][..., None].astype(np.uint8)
            mask_slice[mask_slice > 0] = 1
        else:
            slice_idx = random.choice(range(vol_3d.shape[2]))
            random_mask_tag = True

        # 3. Extract 2D Slices
        img_slice = vol_3d[:, :, slice_idx]
        img_slice = self._normalize_to_255(img_slice)
        # convert to HWC 3-channel
        img_slice = np.stack([img_slice]*3, axis=-1)
        
        # random rotate
        rotate_times = random.choice([0, 1, 2, 3])
        img_slice = np.rot90(img_slice, k=rotate_times)
        
        pil_img = Image.fromarray(img_slice).convert("RGB")
        
        # Add Input Image (Condition)
        data = self._add_image(
            data, 
            pil_img2rgb(pil_img),
            need_loss=False, 
            need_vae=False, 
            need_vit=True, 
        )

        data = self._add_text(data, random.choice(prompt_templates), need_loss=False)

        data = self._add_image(
            data, 
            pil_img2rgb(pil_img),
            need_loss=True, 
            need_vae=False, 
            need_vit=False, 
        )
        return data

    def _normalize_to_255(self, array):
        """Helper to normalize medical floats to 8-bit visual range."""
        amin, amax = array.min(), array.max()
        if amax - amin > 0:
            array = (array - amin) / (amax - amin) * 255
        return array.astype(np.uint8)