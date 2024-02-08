import torch

from Sqlite.SqliteHelper import SqliteHelper
from torch.utils.data import Dataset
from random import randint, choice as rand_choice
from Embedding.TreeEmbedding import TreeEmbedding
from Embedding.FeaturesEmbedding import FeaturesEmbedding
from Tree.ParserAST import ParserAST
from configs import config

class CodeDataset(Dataset):
    def __init__(self, db_file, table_name, data_size, fasttext_model, node_size=None):
        # 如果batch_size>1则node_size必须设定
        self.connection = SqliteHelper(db_file)
        self.table_name = table_name
        self.len = data_size
        self.node_size = node_size
        self.embedding = TreeEmbedding(config.VOCAB_SIZE, config.EMBEDDING_DIM, max_node_size=node_size)
        self.feature_embedding = FeaturesEmbedding(fasttext_model)

        # 从数据库读取数据
        self.raw_data = self.connection.query(f'select * from {table_name}')

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
            else:
                other_label = rand_choice(other_labels)
                self.indexes.append((index, rand_choice(category_dict[other_label]), 0))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = self.indexes[idx]
        # 获取源代码
        code_l, code_r = self.raw_data[item[0]]['code'], self.raw_data[item[1]]['code']
        similarity = item[2]

        # 获取AST和特征
        features = tuple(['funcs'] + config.FEATURES)
        parser_l, parser_r = ParserAST(code_l, features), ParserAST(code_r, features)
        res_l, res_r = parser_l.get_all(), parser_r.get_all()
        ast_l, ast_r = res_l['funcs'], res_r['funcs']
        # 将AST转成向量
        vec_l, vec_r = self.embedding(ast_l[0], config.TRAVERSAL_METHOD), self.embedding(ast_r[0], config.TRAVERSAL_METHOD)
        # padding张量，以便批量处理
        if self.node_size:
            vec_l_pad = torch.zeros((self.node_size, config.EMBEDDING_DIM))
            vec_r_pad = torch.zeros((self.node_size, config.EMBEDDING_DIM))
            rows_l = min(vec_l.shape[0], self.node_size)
            rows_r = min(vec_r.shape[0], self.node_size)
            vec_l_pad[:rows_l, :] = vec_l[:rows_l, :]
            vec_r_pad[:rows_r, :] = vec_r[:rows_r, :]
        else:
            vec_l_pad = vec_l
            vec_r_pad = vec_r
        # 将特征转成向量
        features_vec_l = {}
        features_vec_r = {}
        if 'callee' in features:
            features_vec_l['callee'] = self.feature_embedding.get_callee_features_vec(res_l['callee'])[0]
            features_vec_r['callee'] = self.feature_embedding.get_callee_features_vec(res_r['callee'])[0]
        if 'string' in features:
            features_vec_l['string'] = self.feature_embedding.get_string_features_vec(res_l['string'])[0]
            features_vec_r['string'] = self.feature_embedding.get_string_features_vec(res_r['string'])[0]
        if 'ast' in features:
            features_vec_l['ast'] = self.feature_embedding.get_ast_features_vec(ast_l)[0]
            features_vec_r['ast'] = self.feature_embedding.get_ast_features_vec(ast_r)[0]
        if 'block' in features:
            features_vec_l['block'] = self.feature_embedding.get_block_features_vec(res_l['block'])[0]
            features_vec_r['block'] = self.feature_embedding.get_block_features_vec(res_r['block'])[0]

        return (vec_l_pad, features_vec_l), (vec_r_pad, features_vec_r), similarity
