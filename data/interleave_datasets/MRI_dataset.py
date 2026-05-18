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

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

global_gen = 1
global_und = 1
segment = 0

def read_buffer_as_array(buffer, shape, dtype=np.float32):
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)

class MRIDataset(UnifiedEditIterableDataset):
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
        modality_names = [x.split('-')[-1] for x in subject_modality_names]
        modality_findings = row["modality_findings"]
        impression = row["impression"]
        diseases = row["disease"]
        global_findings = row["global_finding"]
        mask_3d = read_buffer_as_array(row["mask"], shape)

        data = self._init_data()

        mask_indices = np.where(np.any(mask_3d > 0, axis=(1, 2)))[0]
        slice_idx = random.choice(mask_indices) if len(mask_indices) > 0 else random.randint(0, shape[0]-1)
        # interleaved modality translation task
        start_modality_idx = random.randint(0, len(subject_images) - 1)
        input_modalities = modality_names[start_modality_idx:start_modality_idx+1]
        target_modalities = modality_names[:start_modality_idx] + modality_names[start_modality_idx+1:]
        vol_3d = read_buffer_as_array(subject_images[start_modality_idx], shape)
        vol_3d = rescale_intensity(vol_3d, norm=False).astype(np.uint8)
        input_image_slice = vol_3d[slice_idx, :, :]
        input_image_slice = pad_to_target(input_image_slice, (256, 256))
        input_image_slice = np.stack([input_image_slice]*3, axis=-1)
        pil_input_img = Image.fromarray(input_image_slice).convert("RGB")

        current_mask = mask_3d[slice_idx, :, :]
        current_mask = pad_to_target(current_mask, (256, 256))
        pixel_count = np.sum(current_mask > 0)

        h_spacing = w_spacing = 1 # normalized spacing
        
        if pixel_count > 0:
            # 计算当前切片最大径 (mm)
            top, left, height, width = mask2bbox(current_mask[..., None])
            h_mm = height * h_spacing
            w_mm = width * w_spacing
        else:
            h_mm = w_mm = 0
        
        # Add Input Image (Condition)
        instruction_idx = random.choice(range(4)) # gen_only, impression, diagnosis, impression+diagnosis
        if (instruction_idx > 0 and random.random() < 0.2) or not global_gen:
            gen = False
        else:
            gen = True
        # gen = True
        
        # Add Thinking Process
        include_think = random.random() < 0.5
        if gen and include_think:
            data = self._add_text(data, GEN_THINK_SYSTEM_PROMPT, need_loss=False)

        data = self._add_image(
            data, 
            pil_img2rgb(pil_input_img),
            need_loss=False, 
            need_vae=gen, 
            need_vit=True, 
        )
        
        target_modalities = random.sample(target_modalities, random.randint(0, len(target_modalities)))

        for idx, target_modality in enumerate(target_modalities):
            if gen:
                instruction = translation_question_list(input_modalities, [target_modality], drop_input_modality_prob=0.5)
                # if random.random() < 0.2:
                #     describe_generated = True
                #     instruction += ' Describe the generated image.'
                # else:
                #     describe_generated = False
                data = self._add_text(data, instruction.rstrip(), need_loss=False)
            # Extract 2D slice for target modality
            
            target_idx = modality_names.index(target_modality)
            target_vol_3d = read_buffer_as_array(subject_images[target_idx], shape)
            
            target_vol_3d = rescale_intensity(target_vol_3d, norm=False).astype(np.uint8)
            target_img_slice = target_vol_3d[slice_idx, :, :]
            target_img_slice = pad_to_target(target_img_slice, (256, 256))
            target_img_slice = np.stack([target_img_slice]*3, axis=-1)

            pil_target_img = Image.fromarray(target_img_slice).convert("RGB")

            if include_think:
                cur_modality_findings = ''
                for input_modality in input_modalities:
                    input_idx = modality_names.index(input_modality)
                    modality_finding = change_size_description(modality_findings[input_idx], (w_mm, h_mm))
                    cur_modality_findings += ' ' + modality_finding
                thinking_text = '<think>' + translation_think_list(input_modalities, [target_modality], task_idx=None) + cur_modality_findings + '</think>'
                data = self._add_text(data, thinking_text, need_loss=True)

            # Add Target Image (Result)
            if idx != len(target_modalities) - 1 and gen:
                data = self._add_image(
                    data, 
                    pil_img2rgb(pil_target_img),
                    need_loss=True, 
                    need_vae=True, 
                    need_vit=True,
                )
            else:
                data = self._add_image(
                    data, 
                    pil_img2rgb(pil_target_img),
                    need_loss=gen,
                    need_vae=False, 
                    need_vit=True,#  if instruction_idx!=0 else False, 
                )

            input_modalities += [target_modality]

        if segment and random.random() < 0.5:
            instruction = segmentation_question_list()
            data = self._add_text(data, instruction.rstrip(), need_loss=False)
            # Add Segmentation Mask as the last target
            mask_rgb = np.zeros_like(input_image_slice)
            mask_rgb[current_mask > 0] = [255, 255, 255]  # Black color for masked region
            pil_mask_img = Image.fromarray(mask_rgb).convert("RGB")
            data = self._add_image(
                data, 
                pil_img2rgb(pil_mask_img),
                need_loss=True, 
                need_vae=False, 
                need_vit=True,
            )
        
        if global_und:
            if instruction_idx == 1:
                data = self._add_text(data, get_impression_question(), need_loss=False)
                data = self._add_text(data, 'Impression: ' + impression, need_loss=True)
            elif instruction_idx == 2:
                data = self._add_text(data, get_diagnosis_question(), need_loss=False)
                data = self._add_text(data, 'Final Diagnosis: ' + convert_list_to_string(diseases), need_loss=True)
            elif instruction_idx == 3:
                data = self._add_text(data, get_impression_question() + ' ' + get_diagnosis_question(), need_loss=False)
                data = self._add_text(data, 'Impression: ' + impression + '\nFinal Diagnosis: ' + convert_list_to_string(diseases), need_loss=True)

        return data

    def _normalize_to_255(self, array):
        """Helper to normalize medical floats to 8-bit visual range."""
        amin, amax = array.min(), array.max()
        if amax - amin > 0:
            array = (array - amin) / (amax - amin) * 255
        return array.astype(np.uint8)