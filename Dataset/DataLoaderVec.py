import json
import sys

import torch

from Sqlite.SqliteHelper import SqliteHelper
from torch.utils.data import Dataset
from random import randint, choice as rand_choice
from configs import config
from os.path import exists

class CodeDataset(Dataset):
    def __init__(self, db_file, table_name, data_size, node_size=None):
        # 如果batch_size>1则node_size必须设定
        if not exists(db_file):
            print(f'Error: Database file {db_file} does not exist')
            sys.exit(1)
        self.connection = SqliteHelper(db_file)
        self.table_name = table_name
        self.len = data_size
        self.node_size = node_size

        # 从数据库读取数据
        self.raw_data = self.connection.query(f'select * from {table_name}')
        if not self.raw_data:
            print('Error: Failed to read data, please check if the database file is correct!')
            sys.exit(1)

        # 建立类别字典
        category_dict = {}
        for index, item in enumerate(self.raw_data):
            if item['label'] in category_dict.keys():
                category_dict[item['label']].append(index)
            else:
                category_dict[item['label']] = [index]
        category_set = set(category_dict.keys())

        # 随机创建成对代码索引
        raw_len = len(self.raw_data)
        self.indexes = []
        for i in range(data_size):
            index = randint(0, raw_len-1)
            label = self.raw_data[index]['label']
            other_labels = list(category_set - set([label]))
            if i % 2:
                self.indexes.append((index, rand_choice(category_dict[label]), 1))
                # self.indexes.append((index, index, 1))
            else:
                other_label = rand_choice(other_labels)
                self.indexes.append((index, rand_choice(category_dict[other_label]), 0))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = self.indexes[idx]

        # 获取相似度标签
        similarity = torch.tensor(item[2], dtype=torch.float, requires_grad=True)

        # 获取代码向量
        vec_l = torch.tensor(json.loads(self.raw_data[item[0]]['code_vec']), requires_grad=True)
        vec_r = torch.tensor(json.loads(self.raw_data[item[1]]['code_vec']), requires_grad=True)

        # padding张量，以便批量处理
        # if self.node_size:
        #     vec_l_pad = torch.zeros((self.node_size, config.EMBEDDING_DIM))
        #     vec_r_pad = torch.zeros((self.node_size, config.EMBEDDING_DIM))
        #     rows_l = min(vec_l.shape[0], self.node_size)
        #     rows_r = min(vec_r.shape[0], self.node_size)
        #     vec_l_pad[:rows_l, :] = vec_l[:rows_l, :]
        #     vec_r_pad[:rows_r, :] = vec_r[:rows_r, :]
        # else:
        #     vec_l_pad = vec_l
        #     vec_r_pad = vec_r

        # 获取特征向量
        features = config.FEATURES
        features_vec_l = {}
        features_vec_r = {}
        if 'callee' in features:
            features_vec_l['callee'] = torch.tensor(json.loads(self.raw_data[item[0]]['callee_vec']), requires_grad=True)
            features_vec_r['callee'] = torch.tensor(json.loads(self.raw_data[item[1]]['callee_vec']), requires_grad=True)
        if 'string' in features:
            features_vec_l['string'] = torch.tensor(json.loads(self.raw_data[item[0]]['string_vec']), requires_grad=True)
            features_vec_r['string'] = torch.tensor(json.loads(self.raw_data[item[1]]['string_vec']), requires_grad=True)
        if 'ast' in features:
            features_vec_l['ast'] = torch.tensor(json.loads(self.raw_data[item[0]]['ast_vec']), requires_grad=True)
            features_vec_r['ast'] = torch.tensor(json.loads(self.raw_data[item[1]]['ast_vec']), requires_grad=True)
        if 'block' in features:
            features_vec_l['block'] = torch.tensor(json.loads(self.raw_data[item[0]]['block_vec']), requires_grad=True)
            features_vec_r['block'] = torch.tensor(json.loads(self.raw_data[item[1]]['block_vec']), requires_grad=True)

        # return (vec_l_pad, features_vec_l), (vec_r_pad, features_vec_r), similarity
        return (vec_l, features_vec_l), (vec_r, features_vec_r), similarity
