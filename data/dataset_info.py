# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, MRIDataset, ReconDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'medical_recon': ReconDataset,
    'medical_edit': MRIDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'example/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'example/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'example/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'example/bagel_example/vlm/images',
			'jsonl_path': 'example/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'medical_recon':{
        'RadGenome-Brain_MRI_SA': {
            'data_dir': '../../data/RadGenome-Brain_MRI/all_parquet_Reg',
            'split_file': '../../data/RadGenome-Brain_MRI/train_val_test_split_brats.json',
            'num_files': 202,
            'num_total_samples': 1007,
            "parquet_info_path": '../../data/RadGenome-Brain_MRI/dataset_info_Reg.json', # information of the parquet files
		},
    },
    'medical_edit':{
        'RadGenome-Brain_MRI': {
            'data_dir': '../../data/RadGenome-Brain_MRI/all_parquet_Reg',
            'split_file': '../../data/RadGenome-Brain_MRI/train_val_test_split_brats.json',
            'num_files': 202,
            'num_total_samples': 1007,
            "parquet_info_path": '../../data/RadGenome-Brain_MRI/dataset_info_Reg.json', # information of the parquet files
		},
    },
}