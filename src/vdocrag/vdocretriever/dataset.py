from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from vdocrag.vdocretriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        
        # 此处为修改的代码
        # 根据dataset_path判断是加载本地文件还是从Hub加载
        if self.data_args.dataset_path and (os.path.exists(self.data_args.dataset_path) or 
                                            os.path.exists(os.path.dirname(self.data_args.dataset_path))):
            # 本地文件路径或目录
            if os.path.isdir(self.data_args.dataset_path):
                # 如果是目录，加载目录下的parquet/json文件
                data_files = os.path.join(self.data_args.dataset_path, '*.parquet')
                if not any(f.endswith('.parquet') for f in os.listdir(self.data_args.dataset_path)):
                    data_files = os.path.join(self.data_args.dataset_path, '*.json')
                self.train_data = load_dataset(
                    'parquet' if '*.parquet' in data_files else 'json',
                    data_files=data_files,
                    split=self.data_args.dataset_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
            else:
                # 单个文件
                self.train_data = load_dataset(
                    'parquet' if self.data_args.dataset_path.endswith('.parquet') else 'json',
                    data_files=self.data_args.dataset_path,
                    split=self.data_args.dataset_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
        else:
            # Hub数据集
            self.train_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
            
        if not self.data_args.pretrain:
            # 此处为修改的代码
            # 根据corpus_path判断是加载本地文件还是从Hub加载
            corpus_dir = self.data_args.corpus_path.replace('/*.parquet', '').replace('*.parquet', '').replace('/*', '').replace('*', '')
            if self.data_args.corpus_path and (os.path.exists(corpus_dir) or os.path.isdir(os.path.dirname(self.data_args.corpus_path))):
                # 本地文件路径（支持通配符）
                if os.path.isdir(corpus_dir):
                    # 如果是目录，添加通配符
                    corpus_files = os.path.join(corpus_dir, '*.parquet')
                else:
                    corpus_files = self.data_args.corpus_path
                    
                self.corpus  = load_dataset(
                    'parquet',
                    data_files=corpus_files,
                    split=self.data_args.corpus_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
            else:
                # Hub数据集
                self.corpus  = load_dataset(
                    self.data_args.corpus_name,
                    self.data_args.corpus_config,
                    data_files=self.data_args.corpus_path,
                    split=self.data_args.corpus_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )

            # 确保图像特征正确设置decode=True
            from datasets.features import Image as ImageFeature
            if 'image' in self.corpus.features:
                new_features = self.corpus.features.copy()
                new_features['image'] = ImageFeature(decode=True)
                self.corpus = self.corpus.cast(new_features)


            self.docid2idx = {}
            if 'doc_id' in self.corpus.features:
                for idx, docid in enumerate(self.corpus['doc_id']):
                    self.docid2idx[str(docid)] = idx
            else:
                for idx in range(len(self.corpus)):
                    self.docid2idx[str(idx)] = idx

        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        # 此处为修改的代码
        # 支持新的数据格式，兼容原有格式和新格式
        # 原格式: {"query": "...", "relevant_doc_ids": [...]}
        # 新格式: {"query_text": "...", "query_image": [...], "query_type": "text/image", "relevant_doc_ids": [...]}
        
        # 判断是新格式还是旧格式
        if 'query_type' in group:
            # 新格式
            query_text = group['query_text']
            query_type = group['query_type']
            query_image_list = group.get('query_image', [])
            # 此处为修改的代码
            # 读取describe字段，如果query_type为image则使用describe，否则为空
            describe = group.get('describe', '') if query_type == 'image' else ''
        else:
            # 旧格式，兼容处理
            query_text = group['query']
            query_type = 'text'  # 默认为text类型
            query_image_list = []
            describe = ''
        
        if self.data_args.pretrain:
            image = group['image']
            # 预训练模式保持原有逻辑
            return query_text, image
        else:
            relevant_docids = group['relevant_doc_ids']

            # 此处为修改的代码
            # 检查relevant_docids是否为空
            if not relevant_docids or len(relevant_docids) == 0:
                relevant_doc_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
            else:
                # 根据采样策略选择relevant document
                if self.data_args.positive_document_no_shuffle or self.data_args.image_sample_strategy == 'first':
                    docid = relevant_docids[0]
                else:
                    docid = relevant_docids[(_hashed_seed + epoch) % len(relevant_docids)]

                try:
                    relevant_doc_image = self.corpus[self.docid2idx[docid]]['image']
                except KeyError:
                    # 创建一个空白的RGB图片 (224x224 白色背景)
                    relevant_doc_image = Image.new('RGB', (224, 224), color=(255, 255, 255))

            # 此处为修改的代码
            # 对于image类型的query，需要选择一个query_image
            query_image = None
            if query_type == 'image' and len(query_image_list) > 0:
                if self.data_args.image_sample_strategy == 'first':
                    query_image_docid = query_image_list[0]
                else:
                    query_image_docid = query_image_list[(_hashed_seed + epoch) % len(query_image_list)]
                
                try:
                    query_image = self.corpus[self.docid2idx[query_image_docid]]['image']
                except KeyError:
                    query_image = Image.new('RGB', (224, 224), color=(255, 255, 255))

        # 此处为修改的代码
        # 返回新的数据格式: (query_text, query_image, relevant_doc_image, query_type, describe)
        return query_text, query_image, relevant_doc_image, query_type, describe


class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        if self.data_args.encode_is_query:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
        else:    
            self.encode_data = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        
    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        data = self.encode_data[item]
        text, image = None, None
        if self.data_args.encode_is_query:
            id = data['query_id']
            text = data['query']
        else:
            id = data['doc_id']
            image = data['image']
        return id, text, image
