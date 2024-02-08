import torch
import torch.nn as nn
from Embedding.TreePositionalEncoding import TreePositionalEncoding
from queue import Queue
from configs.kind_dict import kinds
from utils.tool_function import normalize_position_encoding

class TreeEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_node_size=0):
        super(TreeEmbedding, self).__init__()

        self.max_node_size = max_node_size  # 最大节点数

        # 位置编码嵌入层
        self.position_embedding = TreePositionalEncoding(embedding_dim)

        # 节点编码嵌入层
        torch.manual_seed(0)  # 设置随机种子
        self.node_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, tree, traversal_method='BFS'):
        if traversal_method == 'BFS':
            node_list, position_list = self.breadth_traversal(tree)
        else:
            node_list, position_list = self.depth_traversal(tree)
        node_vec = self.node_embedding(torch.LongTensor(node_list))
        position_vec = self.position_embedding.get_position_encoding_2D(position_list)
        position_vec = normalize_position_encoding(position_vec)
        return node_vec + position_vec

    def breadth_traversal(self, root):
        kind_list = []  # (seq_len, embedding_dim)
        position_list = []  # (seq_len, embedding_dim)
        queue = Queue()
        queue.put((root, [1]))
        while not queue.empty():
            if 0 < self.max_node_size <= len(kind_list):
                break
            node, position = queue.get()
            # if node.get_depth() > self.max_depth
            kind = kinds[node.kind]
            kind_list.append(kind)
            position_list.append(position)
            for index, child in enumerate(node.get_children()):
                queue.put((child, position + [index+1]))
        return kind_list, position_list

    def depth_traversal(self, root):
        kind_list = []  # (seq_len, embedding_dim)
        position_list = []  # (seq_len, embedding_dim)
        stack = [(root, [1])]
        while stack:
            if 0 < self.max_node_size <= len(kind_list):
                break
            node, position = stack.pop()
            kind = kinds[node.kind]
            kind_list.append(kind)
            position_list.append(position)
            for index, child in reversed(list(enumerate(node.get_children()))):
                stack.append((child, position + [index + 1]))
        return kind_list, position_list
