# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset, MRIDataset, ReconDataset


DATASET_REGISTRY = {
    'medical_recon': ReconDataset,
    'medical_edit': MRIDataset,
}


DATASET_INFO = {
    'medical_recon':{
        'RadGenome-Brain_MRI_SA': {
            'data_dir': '../../data/RadGenome-Brain_MRI/all_parquet_Reg',
            'split_file': '../../data/RadGenome-Brain_MRI/train_val_test_split_subject.json',
            'num_files': 202,
            'num_total_samples': 1007,
            "parquet_info_path": '../../data/RadGenome-Brain_MRI/dataset_info_Reg.json', # information of the parquet files
		},
    },
    'medical_edit':{
        'RadGenome-Brain_MRI': {
            'data_dir': '../../data/RadGenome-Brain_MRI/all_parquet_Reg',
            'split_file': '../../data/RadGenome-Brain_MRI/train_val_test_split_subject.json',
            'num_files': 202,
            'num_total_samples': 1007,
            "parquet_info_path": '../../data/RadGenome-Brain_MRI/dataset_info_Reg.json', # information of the parquet files
		},
    },
}