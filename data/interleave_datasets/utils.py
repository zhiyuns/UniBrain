import re
import random
import numpy as np

modality_map = {
    'T1': ['T1-weighted', 't1-weighted', 'T1W', 'T1w', 'T1', 't1', 'T1n', 't1n'],
    't1n': ['T1-weighted', 't1-weighted', 'T1W', 'T1w', 'T1', 't1', 'T1n', 't1n'],
    't1c': ['T1C', 't1c', 'T1-CE',  't1-ce'], # 'T1-post-contrast', 't1-post-contrast', 'T1 Post-Contrast', 't1 Post-Contrast', 
    't2w': ['T2-weighted', 't2-weighted', 'T2W', 'T2w', 'T2', 't2', ],
    't2f': ['T2-FLAIR', 't2-flair', 'T2 FLAIR', 't2 FLAIR', 'FLAIR', 'flair', 't2f'],
    'FLAIR': ['T2-FLAIR', 't2-flair', 'T2 FLAIR', 't2 FLAIR', 'FLAIR', 'flair', 't2f'],
    'adc': ['ADC', 'adc'],
    'dwi': ['DWI', 'dwi'],
}

image_description_list = ['image', 'scan', 'MRI', 'MR image', 'MRI scan', 'MR scan', 'representation']

def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256, norm=False):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    if norm:
        volume = volume.astype(float) / (bins_num - 1)

    return volume

def get_impression_question():
    selected_modality_description = random.choice([x+'s' for x in image_description_list] + ['data', 'information'])
    impression_question_list = [
        f'Based on the provided {selected_modality_description}, please describe the key observations and any notable findings.',
        f'Examine the given {selected_modality_description} and summarize the main features and abnormalities present.',
        f'Analyze the attached {selected_modality_description} and provide a detailed impression of the observed structures and any potential issues.',
        f'Review the supplied {selected_modality_description} and outline the significant observations along with any clinical implications.',
        f'Interpret the provided {selected_modality_description} and convey the essential findings and their relevance to patient care.',
        f'Assess the given {selected_modality_description} and articulate the primary impressions, highlighting any areas of concern.',
        f'Evaluate the attached {selected_modality_description} and summarize the critical observations, including any pathological findings.',
        f'Consider the supplied {selected_modality_description} and describe the main features, noting any abnormalities or significant details.',
        f'Look at the provided {selected_modality_description} and present a comprehensive impression of the observed anatomical structures.',
        f'Study the given {selected_modality_description} and communicate the key findings, emphasizing any noteworthy observations.'
    ]
    return random.choice(impression_question_list)

def get_diagnosis_question():
    selected_modality_description = random.choice([x+'s' for x in image_description_list] + ['data', 'information'])
    
    diagnosis_question_list = [
        f'Based on the {selected_modality_description}, what is the most likely clinical diagnosis?',
        f'Identify the primary pathology or condition demonstrated in these {selected_modality_description}.',
        f'Analyze the {selected_modality_description} and provide a definitive diagnostic conclusion.',
        f'What is the final assessment of the findings observed in the provided {selected_modality_description}?',
        f'Based on the imaging features in the {selected_modality_description}, classify the observed abnormality.',
        f'What pathological condition do these {selected_modality_description} suggest?',
        f'Provide a final diagnosis for the patient based on the attached {selected_modality_description}.',
        f'Determine the most probable nature of the lesion or finding shown in these {selected_modality_description}.',
        f'Synthesize the findings from the {selected_modality_description} into a single final diagnosis.',
        f'Evaluate the {selected_modality_description} and state the specific disease or condition present.'
    ]
    return random.choice(diagnosis_question_list)

def convert_list_to_string(input_list):
    if len(input_list) == 1:
        return input_list[0]
    else:
        output_string = ''
        for i in range(len(input_list)):
            output_string += input_list[i]
            if i != len(input_list) - 1 and len(input_list) > 2:
                output_string += ', '
            if i == len(input_list) - 2 and len(input_list) > 1:
                output_string += 'and ' if output_string.endswith(', ') else ' and '
        return output_string

def translation_question_list(input_modalities, target_modalities, drop_input_modality_prob=0.0):
    input_modalities = [random.choice(modality_map[modality]) for modality in input_modalities]
    target_modalities = [random.choice(modality_map[modality]) for modality in target_modalities]
    
    input_modality_string = convert_list_to_string(input_modalities)
    target_modality_string = convert_list_to_string(target_modalities)
    
    input_description = input_modality_string + ' '
    if random.random() < drop_input_modality_prob:
        input_description = ''
    if len(input_modalities) > 1:
        input_description = input_description + random.choice([x+'s' for x in image_description_list] + ['data', 'information'])
    else:
        input_description = input_description + random.choice(image_description_list + ['data', 'information'])
    target_description = target_modality_string + ' '
    if len(target_modalities) > 1:
        target_description = target_description + random.choice([x+'s' for x in image_description_list] + ['data', 'information'])
    else:
        target_description = target_description + random.choice(image_description_list + ['data', 'information'])

    # choose from one of 10 question templates
    question_list = [   'Translate the given ' + input_description + ' into the corresponding ' + target_description + '.',
                        'Given the provided ' + input_description + ', generate the equivalent ' + target_description + '.',
                        'Transform the following ' + input_description + ' to its ' + target_description + ' counterpart.',
                        'Using the supplied ' + input_description + ', create the matching ' + target_description + '.',
                        'Convert the given ' + input_description + ' into accurate ' + target_description + ' depiction.',
                        'Translate the attached ' + input_description + ' into its ' + target_description + '.',
                        'From the provided ' + input_description + ', synthesize the corresponding ' + target_description + '.',
                        'Generate the ' + target_description + ' based on the given ' + input_description + '.',
                        'Create accurate ' + target_description + ' from the supplied ' + input_description + '.',
                        'Please produce the ' + target_description + ' that corresponds to the provided ' + input_description + '.']
    return random.choice(question_list)

def segmentation_question_list():
    question_list = [
        'Segment the lesion in the provided image.',
        'Identify and delineate the abnormality in the given scan.',
        'Perform segmentation to isolate the region of interest in the attached image.',
        'Outline the pathological area in the supplied scan.',
        'Create a segmentation mask for the lesion shown in the provided image.',
        'Mark the boundaries of the abnormality in the given scan.',
        'Generate a segmentation map to highlight the region of interest in the attached image.',
        'Identify and segment the lesion present in the supplied scan.',
        'Produce a segmentation output that accurately captures the abnormality in the provided image.',
        'Delineate the pathological region by performing segmentation on the given scan.'
    ]
    return random.choice(question_list)

def translation_think_list(input_modalities, target_modalities, task_idx=None):
    input_modalities = [random.choice(modality_map[modality]) for modality in input_modalities]
    target_modalities = [random.choice(modality_map[modality]) for modality in target_modalities]
    
    input_modality_string = convert_list_to_string(input_modalities)
    target_modality_string = convert_list_to_string(target_modalities)
    
    input_description = input_modality_string + ' '
    if len(input_modalities) > 1:
        input_description = input_description + random.choice([x+'s' for x in image_description_list] + ['data', 'information'])
    else:
        input_description = input_description + random.choice(image_description_list + ['data', 'information'])
    target_description = target_modality_string + ' '
    if len(target_modalities) > 1:
        target_description = target_description + random.choice([x+'s' for x in image_description_list] + ['data', 'information'])
    else:
        target_description = target_description + random.choice(image_description_list + ['data', 'information'])

    # choose from one of 10 question templates
    think_list = [  'I need to translate the given ' + input_description + ' into the corresponding ' + target_description + '.',
                    'to generate the equivalent ' + target_description + ', I will use the provided ' + input_description + '.',
                    'I will transform the above ' + input_description + ' to its ' + target_description + ' counterpart.',
                    'using the supplied ' + input_description + ', I can create the matching ' + target_description + '.',
                    'I should convert the given ' + input_description + ' into accurate ' + target_description + ' depiction.',
                    'I will translate the attached ' + input_description + ' into its ' + target_description + '.',
                    'from the provided ' + input_description + ', I can synthesize the corresponding ' + target_description + '.',
                    'I need to generate the ' + target_description + ' based on the given ' + input_description + '.',
                    'I will create accurate ' + target_description + ' from the supplied ' + input_description + '.',
                    'I should produce the ' + target_description + ' that corresponds to the provided ' + input_description + '.',
                    'the current task is to translate ' + input_description + ' into ' + target_description + '.',
                    'my goal is to convert the ' + input_description + ' into ' + target_description + ' while preserving essential details.',
    ]

    think_content = random.choice(think_list)
    if task_idx is not None:
        if task_idx == 1:
            think_content = "First, " + think_content
        elif task_idx == 2:
            think_content = random.choice(["Next, ", "Then, ", "Second, "]) + think_content
        elif task_idx == 3:
            # at most 3 tasks
            think_content = random.choice(["Finally, ", "At last, ", "In the end, "]) + think_content


    return think_content

def change_size_description(ori_description, new_size):
    if 'mm' not in ori_description:
        return ori_description
    num = r'(\d+(?:\.\d+)?)'
    sep = r'\s*(?:\*|by|x|X)\s*'
    unit = r'\s*mm'
    
    pattern = num + sep + num + sep + num + unit
    
    # 使用 re.IGNORECASE 确保匹配不受大小写影响
    match = re.search(pattern, ori_description, re.IGNORECASE)
    replacement = f"{new_size[0]:.1f} * {new_size[1]:.1f} mm"
    return re.sub(pattern, replacement, ori_description, flags=re.IGNORECASE)


def pad_to_target(img, target_size, pad_value=0):
    """ Pad image to the target size.
    Args:
        img: numpy array of shape (H, W) or (C, H, W)
        target_size: tuple of (target_H, target_W)
        pad_value: value to use for padding
    Returns:
        padded_img: numpy array of shape (target_H, target_W) or (C, target_H, target_W)
    """
    if len(img.shape) == 2:
        h, w = img.shape
        c = None
    elif len(img.shape) == 3:
        c, h, w = img.shape
    else:
        raise ValueError("Unsupported image shape: {}".format(img.shape))

    target_h, target_w = target_size
    if target_h <= h or target_w <= h:
        return img
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    h_start = pad_h // 2
    w_start = pad_w // 2
    if c is None:
        padded_img = np.full((target_h, target_w), pad_value, dtype=img.dtype)
        padded_img[h_start:h_start + h, w_start:w_start + w] = img
    else:
        padded_img = np.full((c, target_h, target_w), pad_value, dtype=img.dtype)
        padded_img[:, h_start:h_start + h, w_start:w_start + w] = img

    return padded_img

def crop_to_target(img, target_size):
    """ Crop image to the target size.
    Args:
        img: numpy array of shape (H, W) or (C, H, W)
        target_size: tuple of (target_H, target_W)
    Returns:
        cropped_img: numpy array of shape (target_H, target_W) or (C, target_H, target_W)
    """
    if len(img.shape) == 2:
        h, w = img.shape
        c = None
    elif len(img.shape) == 3:
        c, h, w = img.shape
    else:
        raise ValueError("Unsupported image shape: {}".format(img.shape))

    target_h, target_w = target_size
    if target_h >= h or target_w >= w:
        return img
    h_start = (h - target_h) // 2
    w_start = (w - target_w) // 2
    if c is None:
        cropped_img = img[h_start:h_start + target_h, w_start:w_start + target_w]
    else:
        cropped_img = img[:, h_start:h_start + target_h, w_start:w_start + target_w]

    return cropped_img