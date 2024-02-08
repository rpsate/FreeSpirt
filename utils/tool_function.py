import numpy as np
from queue import Queue
from torch.nn.utils.rnn import pad_sequence

import jieba_fast
import re

import torch

def get_archimedean_spiral_encoding(n, d):
    """
    获取阿基米德螺线位置编码

    Parameters:
        d (int): 编码维度
        n (int): 编码序号

    Returns:
        numpy.ndarray: 阿基米德螺线位置编码
    """
    phi = np.sqrt(n / d) * 2 * np.pi
    x = np.cos(phi) * n
    y = np.sin(phi) * n
    return np.array([x, y])

def normalize_position_encoding(position_encoding):
    max_value = torch.max(torch.abs(position_encoding))
    if max_value > 0:
        position_encoding = position_encoding / max_value
    return position_encoding

def assign_id_breadth(root):
    if root is None:
        return
    index = 0
    queue = Queue()
    queue.put(root)
    while not queue.empty():
        node = queue.get()
        node.id = index
        for c in node.get_children():
            queue.put(c)
        index += 1

def assign_id_depth(root):
    if root is None:
        return
    stack = [root]
    index = 0
    while stack:
        node = stack.pop()
        node.id = index
        children = node.get_children()[::-1]
        stack.extend(children)
        index += 1

def print_tree(node, indentation=0):
    """
    遍历打印抽象语法树

    Parameters:
        node (Tree.Node): 抽象语法树根节点
        indentation (int): 初始缩进位置
    """
    print('  ' * indentation, 'id: ' + str(node.id), 'kind:' + node.kind, 'spelling:' + node.spelling)
    for c in node.get_children():
        print_tree(c, indentation + 1)


def sentence2words(sentence):
    words = jieba_fast.cut(sentence)
    words = [word for word in words if re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$', word)]
    return words

def pad_ast_list(ast_list):
    ast_list.sort(key=lambda x: len(x), reverse=True)
    ast_list_len = [len(ast) for ast in ast_list]
    ast_list_pad = pad_sequence(ast_list, batch_first=True)
    return ast_list_pad, ast_list_len