import argparse
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import SimpleITK as sitk
from copy import deepcopy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import necessary modules from your existing codebase
from inferencer import InterleaveInferencer
from modeling.bagel import BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.autoencoder import load_ae
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens, pil_img2rgb
from data.interleave_datasets.utils import rescale_intensity, pad_to_target, crop_to_target, convert_list_to_string
from data.transforms import ImageTransform
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import pyarrow.parquet as pq
from data.parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

enable_taylorseer = False

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

diagnosis_options = {
                'A': 'Gli',
                'B': 'Men',
                'C': 'Meta',
                'D': 'Infarct',
                'E': 'Degeneration',
                'F': 'Normal'
}

def get_multiple_choice_prompt(question, choices):
    choices = [str(choice) for choice in choices]
    options = "\n".join(choices)

    prompt = f"""
    Question: {question}
    Options: 
    {options}"""
    prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly." 

    return prompt

# --- Metric Helper Functions ---
def calculate_metrics(generated_img, gt_img):
    """
    Calculates PSNR and SSIM between generated and ground truth images.
    Images should be PIL Images or numpy arrays (H, W, C).
    """
    # Convert PIL to numpy
    if isinstance(generated_img, Image.Image):
        generated_img = np.array(generated_img)
    if isinstance(gt_img, Image.Image):
        gt_img = np.array(gt_img)

    # Ensure grayscale for medical evaluation if needed, or calculate per channel
    # Here we calculate multichannel SSIM
    
    # # Check dimensions
    # if generated_img.shape != gt_img.shape:
    #     # Resize generated to match GT if there's a slight mismatch due to padding
    #     generated_img = np.array(Image.fromarray(generated_img).resize(gt_img.shape[:2][::-1]))

    val_psnr = psnr(gt_img / 255., generated_img / 255., data_range=1)
    val_ssim = ssim(gt_img / 255., generated_img / 255., data_range=1)
        
    return val_psnr, val_ssim

# --- Data Processing (Adapted from MRI_dataset.py) ---
def process_eval_row(row, task="generation", modalities_in=None, modalities_out=None, mode='2d', replaced_by=None):
    """
    Deterministically processes a data row for evaluation.
    Returns:
        input_list: List of [Image, text, Image, text...] for the inferencer
        gt: The Ground Truth for current task
    """
    # Parse dimensions and reshaping (Same as your MRI_dataset.py)
    shape = json.loads(row["header_metadata"])["size"]
    shape = [shape[2], shape[1], shape[0]]
    
    subject_images = row["image_list"]
    modality_findings = row["modality_findings"]
    subject_modality_names = row["modality_names"]
    modality_names = [x.split('-')[-1] for x in subject_modality_names]
    
    # Helper to read buffer
    def read_buffer(bytes, s):
        raw = np.frombuffer(bytes, dtype=np.float32).reshape(s)
        # Normalize to 0-255 uint8
        raw = rescale_intensity(raw, norm=False).astype(np.uint8)
        return raw.astype(np.uint8)

    # Define tasks
    input_mod_names = []
    target_mod_names = []
    
    if task == 'generation' or task == 'unified':
        # Use provided modalities
        if modalities_in is not None and modalities_out is not None:
            if all(m in modality_names for m in modalities_in) and all(m in modality_names for m in modalities_out):
                input_mod_names = modalities_in
                target_mod_names = modalities_out
            else:
                return None, None, None, None, None
        else:
            return None, None, None, None, None
    
    elif task == 'diagnosis':
        required = modalities_in
        if all(m in modality_names for m in required) and "t1n" in modality_names:
            input_mod_names = required
            target_mod_names = None
        else:
            return None, None, None, None, None
    else:
        raise ValueError(f"Unknown task: {task}")

    # Select a deterministic slice
    
    if mode == '2d':
        mask_3d = np.frombuffer(row['mask'], dtype=np.float32).reshape(shape)
        mask_indices = np.where(np.any(mask_3d > 0, axis=(1, 2)))[0]
        selected_idxs = [mask_indices[len(mask_indices)//2] if len(mask_indices) > 0 else shape[0] // 2]
    else:
        exist_3d = np.frombuffer(subject_images[modality_names.index(modalities_in[0])], dtype=np.float32).reshape(shape)
        exist_3d = exist_3d > 0
        selected_idxs = np.where(np.any(exist_3d, axis=(1, 2)))[0] if np.any(exist_3d) else np.arange(shape[0])
    
    # Construct Inputs
    
    # 3. Add Instruction Text
    # Construct standard question prompt
    if task == 'diagnosis':
        question = 'Based on the provided information, what is the most likely clinical diagnosis?'
        instruction = question
    elif task == 'generation' or task == 'unified':
        target_string = target_mod_names[0].replace('t2f','T2-FLAIR')
        instruction = f"Translate the attached MRI into its {target_string} image."

    
    # 2. Add Input Images
    input_lists = []
    for selected_idx in selected_idxs:
        input_list = []
        for idx, mod in enumerate(input_mod_names):
            idx = modality_names.index(mod)
            if replaced_by is None:
                vol = read_buffer(subject_images[idx], shape)
                # Extract Slice (Assuming Axial axis 0 based on your code's transpose logic check)
                # Note: In your code axis=0 was D, H, W. You transposed. 
                # We assume standard orientation here; adjust slice_idx extraction if needed.
                img_slice = vol[selected_idx, :, :] 
            else:
                img = Image.open(os.path.join(replaced_by, f"{row['subject_id']}-{mod}.png")).convert('L')
                img_slice = np.array(img)
            
            # Pad to 256x256 (from your pad_to_target logic)
            target_h, target_w = 256, 256
            img_slice = pad_to_target(img_slice, (target_h, target_w))
            
            img_pil = Image.fromarray(np.stack([img_slice]*3, axis=-1))
            input_list.append(img_pil)
            # if task == 'diagnosis':
            #     input_list.append(modality_findings[idx])
        
        input_list.append(instruction)
        input_lists.append(input_list)

    subject_name = '-'.join(subject_modality_names[0].split('-')[:-1])
    if '-GLI-' in subject_name:
        correct_answer = 'A'
    elif '-MEN-' in subject_name:
        correct_answer = 'B'
    elif '-MET-' in subject_name:
        correct_answer = 'C'
    elif 'strokecase' in subject_name:
        correct_answer = 'D'
    else:
        correct_answer = 'E'
    if task == 'diagnosis':
        return input_list, None, diagnosis_options[correct_answer], None, None
    elif task == 'generation' or task == 'unified':
        # 4. Prepare Ground Truth
        
        all_gt_pils = []
        for selected_idx in selected_idxs:
            all_gt_pil = []
            for target_mod_name in target_mod_names:
                tgt_idx = modality_names.index(target_mod_name)
                tgt_vol = read_buffer(subject_images[tgt_idx], shape)
                tgt_slice = tgt_vol[selected_idx, :, :]
                
                # padded_tgt = pad_to_target(tgt_slice, (target_h, target_w)) # no need to pad gt
                gt_pil = Image.fromarray(tgt_slice)
                all_gt_pil.append(gt_pil)
            all_gt_pils.append(all_gt_pil)

        return input_lists, all_gt_pils, diagnosis_options[correct_answer] if task == 'unified' else None, shape, selected_idxs

def run_understanding_inference(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, task="diagnosis", modalities_in=None, row_groups=None, replaced_by=None, device='cuda'):
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, device=device)
    # --- 3. Evaluation Loop ---
    correct_count = 0
    total_count = 0
    results_log = []

    for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(row_groups):
        row_start_id = 0
        fs = init_arrow_pf_fs(parquet_file_path)
        with fs.open_input_file(parquet_file_path) as f:
            fr = pq.ParquetFile(f)
            df = fr.read_row_group(row_group_id).to_pandas()
            df = df.iloc[row_start_id:]
        for row_idx, row in df.iterrows():
            # Prepare Data
            input_list, _ , gt_label, _ , _ = process_eval_row(row, task=task, modalities_in=modalities_in, replaced_by=replaced_by)
            
            # Skip if missing modalities or disease not in options
            if input_list is None or gt_label is None:
                continue 

            try:
                # Inference
                output_list = inferencer.interleave_inference(
                    input_lists=input_list, # Special arg for your InterleaveInferencer
                    think=False,
                    understanding_output=True, # Critical for text generation
                    do_sample=False, # Deterministic greedy decoding
                    max_think_token_n=512,
                    text_temperature=0.0,
                    enable_taylorseer=enable_taylorseer,
                )
                
                pred_label = output_list[-1]
                
                # Scoring
                is_correct = gt_label.lower() in pred_label.lower()  # (pred_label == gt_label)
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                results_log.append({
                    "id": row_idx,
                    "gt": gt_label,
                    "pred": pred_label,
                    "correct": is_correct,
                    "raw_output": output_list
                })
                
                print(f"Sample {row_idx}: GT={gt_label}, Pred={pred_label}, Correct={is_correct}")

            except Exception as e:
                print(f"Error on sample {row_idx}: {e}")
                continue

    return correct_count, total_count, results_log

def run_generation_inference(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, task="generation", modalities_in=["t1n", "t2w"], modalities_out=["flair"], save_path=None, row_groups=None, mode='2d', device='cuda'):
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, device=device)
    
    # 3. Evaluation Loop
    psnr_list, ssim_list = [], []
    
    for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(row_groups):
        row_start_id = 0
        fs = init_arrow_pf_fs(parquet_file_path)
        with fs.open_input_file(parquet_file_path) as f:
            fr = pq.ParquetFile(f)
            df = fr.read_row_group(row_group_id).to_pandas()
            df = df.iloc[row_start_id:]
        
        subject_psnr_list, subject_ssim_list = [], []
        for row_idx, row in df.iterrows():
            # Prepare Data
            input_lists, all_gt_imgs, _, ori_size, selected_idxs = process_eval_row(row, task=task, modalities_in=modalities_in, modalities_out=modalities_out, mode=mode)
            if input_lists is None: 
                continue # Skip subjects missing required modalities
            
            generated_imgs = np.zeros((ori_size), dtype=np.uint8) # Assuming all GTs have the same size after cropping
            for slice_idx, (input_list, all_gt_img) in enumerate(zip(input_lists, all_gt_imgs)):
                gt_img = all_gt_img[-1] # only process the last modality
                subject_modality_names = row["modality_names"]
                subject_name = '-'.join(subject_modality_names[0].split('-')[:-1])
                
                # Inference
                # We enable thinking because your model expects it, but we only care about the final image
                
                original_width, original_height = input_list[0].size
                target_size = inferencer._calculate_target_size_with_aspect_ratio(
                    original_width, original_height
                )
                output_list = inferencer.interleave_inference(
                    input_lists=input_list,
                    think=False, 
                    understanding_output=False,
                    do_sample=False, # Deterministic for evaluation
                    max_think_token_n=800,
                    image_shapes=target_size,
                    cfg_text_scale = 3,
                    cfg_img_scale = 1.2,
                    cfg_interval=[0.4, 1.0], 
                    num_timesteps=10,
                    # timestep_shift=1,
                    # cfg_text_scale = 2,
                    # cfg_img_scale = 0,
                    # cfg_interval=[0.4, 1.0], 
                    # num_timesteps=50,
                    # timestep_shift=3,
                    # cfg_text_scale = 2,
                    # cfg_img_scale = 1.2,
                    # cfg_interval=[0.0, 1.0], 
                    cfg_renorm_type='text_channel',
                    enable_taylorseer=enable_taylorseer,
                )
                
                # The output list will be [Thinking_Text, Generated_Image]
                generated_img = None
                for item in output_list:
                    if isinstance(item, Image.Image):
                        generated_img = item
                        break
                if generated_img is None:
                    print(f"Sample {subject_name}: Failed to generate image.")
                    continue

                # Calculate Metrics
                generated_img = crop_to_target(np.mean(np.array(generated_img), axis=-1).astype('uint8'), ori_size[1:]) # Convert to grayscale
                generated_imgs[selected_idxs[slice_idx], :, :] = generated_img # Place the generated slice back into the 3D volume (if 3D)
                # generated_img = rescale_intensity(generated_img, norm=False).astype('uint8')
                generated_img = Image.fromarray(generated_img)
                p, s = calculate_metrics(generated_img, gt_img)
                subject_psnr_list.append(p)
                subject_ssim_list.append(s)
                
                psnr_list.append(p)
                ssim_list.append(s)

            if save_path is not None:
                # Optional: Save images
                print(f"Sample {subject_name} {ori_size}: PSNR={np.mean(subject_psnr_list):.4f}, SSIM={np.mean(subject_ssim_list):.4f}")
                os.makedirs(save_path, exist_ok=True)
                with open(f"{save_path}/eval_metrics.txt", 'a') as f:
                    f.write(f"{subject_name}\tPSNR: {np.mean(subject_psnr_list):.4f}\tSSIM: {np.mean(subject_ssim_list):.4f}\n")
                if mode == '2d':
                    generated_img.save(f"{save_path}/{subject_name}_gen.png")
                    gt_img.save(f"{save_path}/{subject_name}_gt.png")
                else:
                    generated_sitk = sitk.GetImageFromArray(generated_imgs)
                    gt_img_3d = np.zeros((ori_size), dtype=np.uint8)
                    for idx, selected_idx in enumerate(selected_idxs):
                        gt_img_3d[selected_idx, :, :] = np.array(all_gt_imgs[idx][-1]) # only process the last modality
                    gt_sitk = sitk.GetImageFromArray(gt_img_3d)
                    sitk.WriteImage(generated_sitk, f"{save_path}/{subject_name}_gen.nii.gz")
                    sitk.WriteImage(gt_sitk, f"{save_path}/{subject_name}_gt.nii.gz")


    return psnr_list, ssim_list

def run_unified_inference(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, task="unified", modalities_in=["t1n", "t2w"], modalities_out=["flair","t1c"], save_path=None, row_groups=None, sample_time=1, device='cuda'):
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, device=device)

    max_think_token_n=800
    do_sample=False
    text_temperature=0.0
    num_timesteps = 10
    timestep_shift = 1
    cfg_text_scale = 3
    cfg_img_scale = 1.2
    cfg_interval = [0.4, 1.0]
    cfg_renorm_type = 'text_channel'
    
    # 3. Evaluation Loop
    correct_count = 0
    total_count = 0
    psnr_dict = {mod: [] for mod in modalities_out}
    ssim_dict = {mod: [] for mod in modalities_out}
    
    for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(row_groups):
        row_start_id = 0
        fs = init_arrow_pf_fs(parquet_file_path)
        with fs.open_input_file(parquet_file_path) as f:
            fr = pq.ParquetFile(f)
            df = fr.read_row_group(row_group_id).to_pandas()
            df = df.iloc[row_start_id:]
        for row_idx, row in df.iterrows():
            # Prepare Data
            input_lists, all_gt_imgs, gt_label, ori_size, _ = process_eval_row(row, task=task, modalities_in=modalities_in, modalities_out=modalities_out)
            subject_modality_names = row["modality_names"]
            subject_name = '-'.join(subject_modality_names[0].split('-')[:-1])

            if input_lists is None: 
                continue # Skip subjects missing required modalities

            input_list = input_lists[0] # For unified we only have one set of inputs, so take the first one
            all_gt_img = all_gt_imgs[0]


            gen_context = inferencer.init_gen_context()
            cfg_text_context = deepcopy(gen_context)
            cfg_img_context = deepcopy(gen_context)

            with torch.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
                system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = inferencer.update_context_text(system_prompt, gen_context)
                cfg_img_context = inferencer.update_context_text(system_prompt, cfg_img_context)
                for input_term in input_list:
                    if isinstance(input_term, str):
                        cfg_text_context = deepcopy(gen_context)
                        gen_context = inferencer.update_context_text(input_term, gen_context)
                        cfg_img_context = inferencer.update_context_text(input_term, cfg_img_context)

                    elif isinstance(input_term, Image.Image):
                        input_term = inferencer.vae_transform.resize_transform(pil_img2rgb(input_term))
                        gen_context = inferencer.update_context_image(input_term, gen_context, vae=True)

                        image_shapes = input_term.size[::-1]
                        cfg_text_context = deepcopy(gen_context)

                    else:
                        raise ValueError(f"Unsupported input type: {type(input_term)}")

                for out_idx, current_modality_out in enumerate(modalities_out):
                    
                    original_width, original_height = input_list[0].size
                    target_size = inferencer._calculate_target_size_with_aspect_ratio(
                        original_width, original_height
                    )
                    gen_text = inferencer.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = inferencer.update_context_text(gen_text, gen_context)
                    think_content = gen_text
                    
                    generated_list = []
                    for _ in range(sample_time):
                        generated_img = inferencer.gen_image(
                            target_size, 
                            gen_context, 
                            cfg_text_precontext=cfg_text_context, 
                            cfg_img_precontext=cfg_img_context,
                            cfg_text_scale=cfg_text_scale, 
                            cfg_img_scale=cfg_img_scale, 
                            cfg_interval=cfg_interval, 
                            timestep_shift=timestep_shift, 
                            num_timesteps=num_timesteps,
                            cfg_renorm_min=0.0,
                            cfg_renorm_type=cfg_renorm_type,
                            enable_taylorseer=enable_taylorseer,
                        )
                        generated_list.append(np.array(generated_img))
                    generated_img = np.mean(generated_list, axis=0).astype(np.uint8) # Simple averaging for multiple samples; can be replaced with more sophisticated ensembling if needed
                    generated_img = Image.fromarray(generated_img)
                    # input_list.extend(output_list) # Append previous outputs for next generation
                    
                    # prepare next context for metric calculation (same as cfg_img_context since we want to evaluate the generated image in the same context)
                    cfg_img_context = deepcopy(gen_context)
                    input_term = inferencer.vae_transform.resize_transform(pil_img2rgb(generated_img))
                    gen_context = inferencer.update_context_image(input_term, gen_context, vae=True if out_idx < len(modalities_out) - 1 else False)
                    cfg_text_context = deepcopy(gen_context)
                    
                    # prepare next input if there are more generations to do
                    if out_idx < len(modalities_out) - 1:
                        instruction = f"Translate the attached MRI images into {convert_list_to_string(modalities_out[out_idx+1:out_idx+2])} image."
                        cfg_text_context = deepcopy(gen_context)
                        gen_context = inferencer.update_context_text(instruction, gen_context)
                        cfg_img_context = inferencer.update_context_text(instruction, cfg_img_context)

                    # Calculate Metrics
                    generated_img = crop_to_target(np.mean(np.array(generated_img), axis=-1).astype('uint8'), ori_size[1:]) # Convert to grayscale
                    generated_img = Image.fromarray(generated_img)
                    gt_img = all_gt_img[out_idx]
                    p, s = calculate_metrics(generated_img, gt_img)
                    psnr_dict[current_modality_out].append(p)
                    ssim_dict[current_modality_out].append(s)

                    if save_path is not None:
                        # Optional: Save images
                        print(f"Sample {subject_name} {modalities_in}_to_{current_modality_out} {ori_size}: {think_content}, PSNR={p:.4f}, SSIM={s:.4f}")
                        os.makedirs(save_path, exist_ok=True)
                        with open(f"{save_path}/eval_metrics.txt", 'a') as f:
                            f.write(f"Sample {subject_name} {modalities_in}_to_{current_modality_out} {ori_size}: {think_content}, PSNR={p:.4f}, SSIM={s:.4f} \n")
                        generated_img.save(f"{save_path}/{subject_name}_{current_modality_out}_gen.png")
                        gt_img.save(f"{save_path}/{subject_name}_{current_modality_out}_gt.png")


            final_instruction = 'Based on the provided information, what is the most likely clinical diagnosis?'
            gen_context = inferencer.update_context_text(final_instruction, gen_context)
            pred_label = inferencer.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
            is_correct = gt_label.lower() in pred_label.lower()  # (pred_label == gt_label)
            if is_correct:
                correct_count += 1
            total_count += 1
            print(f"Sample {subject_name}: GT={gt_label}, Pred={pred_label}, Correct={is_correct}")
            with open(f"{save_path}/eval_metrics.txt", 'a') as f:
                f.write(f"Sample {subject_name}: GT={gt_label}, Pred={pred_label}, Correct={is_correct}\n")

    return correct_count, total_count, psnr_dict, ssim_dict



# --- Main Evaluation Loop ---
def main(args):
    # 1. Model Initialization (Copied/Adapted from app.py)
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(os.path.join(args.model_path, "llm_config.json")):
        reference_path = os.path.abspath(r'models/BAGEL-7B-MoT')
        os.symlink(os.path.join(reference_path, "llm_config.json"), os.path.join(args.model_path, "llm_config.json"))
        os.symlink(os.path.join(reference_path, "vit_config.json"), os.path.join(args.model_path, "vit_config.json"))
        os.symlink(os.path.join(reference_path, "tokenizer.json"), os.path.join(args.model_path, "tokenizer.json"))
        os.symlink(os.path.join(reference_path, "vocab.json"), os.path.join(args.model_path, "vocab.json"))
        os.symlink(os.path.join(reference_path, "merges.txt"), os.path.join(args.model_path, "merges.txt"))
        os.symlink(os.path.join(reference_path, "ae.safetensors"), os.path.join(args.model_path, "ae.safetensors"))

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config, vit_config=vit_config,
        vae_config=vae_config, vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    vae_transform = ImageTransform(512, 256, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # Device Mapping & Loading
    device_map = infer_auto_device_map(model, max_memory={0: "80GiB"}, no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"])
    
    # Fix device mapping for tied weights
    same_device_modules = ['language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed']
    for k in same_device_modules:
        device_map[k] = "cuda:0" # Simplify to single GPU for eval script

    vae_model = vae_model.to("cuda:0" )
    
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(args.model_path, "ema.safetensors") if os.path.exists(os.path.join(args.model_path, "ema.safetensors")) else os.path.join(args.model_path, "model.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True
    ).eval()

    # Example: If you have a .tar or .parquet file, load it here.
    # For demonstration, I'm assuming 'data_rows' is a list of your raw data dicts.
    dataroot = '../../data/RadGenome-Brain_MRI/all_parquet_Reg'
    with open('../../data/RadGenome-Brain_MRI/dataset_info_Reg.json', 'r') as f:
        parquet_info = json.load(f)
    with open('../../data/RadGenome-Brain_MRI/train_val_test_split_subject.json', 'r') as f:
        split_info = json.load(f)
    data_paths = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
    row_groups = []
    for data_path in data_paths:
        if data_path in parquet_info.keys():
            num_row_groups = parquet_info[data_path]['num_row_groups']
            for rg_idx in range(num_row_groups):
                if split_info is not None and parquet_info[data_path]['subject_list'][rg_idx] not in split_info['test']:
                    continue
                else:
                    row_groups.append((data_path, rg_idx))
    
    if args.task == 'generation':
        modality_in_str = '_'.join(args.modalities_in)
        modality_out_str = '_'.join(args.modalities_out)
        save_path = f"results/eval_samples/{args.task}_{modality_in_str}__{modality_out_str}"
        psnr_list, ssim_list = run_generation_inference(
            model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids,
            task=args.task, modalities_in=args.modalities_in, modalities_out=args.modalities_out, save_path=save_path, row_groups=row_groups, mode=args.mode
        )
        print(f"Evaluation Results for {args.task}:")
        print(f"Average PSNR: {np.mean(psnr_list):.4f}")
        print(f"Average SSIM: {np.mean(ssim_list):.4f}")
        with open(save_path + "/eval_metrics.txt", 'a') as f:
            f.write(f"Average PSNR: {np.mean(psnr_list):.4f}\n")
            f.write(f"Average SSIM: {np.mean(ssim_list):.4f}\n")
    elif args.task == 'diagnosis':
        correct_count, total_count, results_log = run_understanding_inference(
            model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids,
            task=args.task, modalities_in=args.modalities_in, row_groups=row_groups, replaced_by=args.replaced_by
        )
        print(f"Evaluation Results for {args.task}:")
        print(f"Correct Count: {correct_count}")
        print(f"Total Count: {total_count}")
        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"Accuracy: {accuracy:.2%}")
        else:
            print("No valid samples evaluated.")
    elif args.task == 'unified':
        modality_in_str = '_'.join(args.modalities_in)
        modality_out_str = '_'.join(args.modalities_out)
        save_path = f"results/eval_samples/{args.task}_{modality_in_str}__{modality_out_str}"
        correct_count, total_count, psnr_dict, ssim_dict = run_unified_inference(
            model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids,
            task=args.task, modalities_in=args.modalities_in, modalities_out=args.modalities_out, save_path=save_path, row_groups=row_groups, sample_time=args.sample_time
        )
        print(f"Evaluation Results for {args.task}:")
        print(f"Correct Count: {correct_count}")
        print(f"Total Count: {total_count}")
        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"Accuracy: {accuracy:.2%}")
            with open(save_path + "/eval_metrics.txt", 'a') as f:
                f.write(f"Correct Count: {correct_count}\n")
                f.write(f"Total Count: {total_count}\n")
                f.write(f"Accuracy: {accuracy:.2%}\n")
        else:
            print("No valid samples evaluated.")
        for mod in args.modalities_out:
            if len(psnr_dict[mod]) > 0:
                print(f"Modality {mod} - Average PSNR: {np.mean(psnr_dict[mod]):.4f}, Average SSIM: {np.mean(ssim_dict[mod]):.4f}")
                with open(save_path + f"/eval_metrics.txt", 'a') as f:
                    f.write(f"Modality {mod} - Average PSNR: {np.mean(psnr_dict[mod]):.4f}, Average SSIM: {np.mean(ssim_dict[mod]):.4f}\n")
            else:
                print(f"Modality {mod} - No valid samples evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/BAGEL-7B-MoT", help="Path to model")
    parser.add_argument("--task", type=str, default="unified", choices=["generation", "diagnosis", "unified"], help="Evaluation task")
    parser.add_argument("--mode", type=str, default='2d', choices=['2d', '3d'], help="Whether to evaluate on 2D slices or 3D volumes (if applicable)")
    parser.add_argument("--modalities_in", type=str, nargs='+', default=["t1n", ], help="Input modalities")
    parser.add_argument("--modalities_out", type=str, nargs='+', default=["t2w", "t2f", "t1c"], help="Output modalities for unified task")
    parser.add_argument("--replaced_by", type=str, default=None, help="The replaced path for diagnosis task")
    parser.add_argument("--sample_time", type=int, default=1, help="Number of samples to generate for each input (only for generation task)")

    args = parser.parse_args()
    main(args)