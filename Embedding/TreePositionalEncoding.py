import sys
import torch
from configs.config import ARCHIMEDES_SPIRAL_DIM
from utils.tool_function import get_archimedean_spiral_encoding

class TreePositionalEncoding(object):
    def __init__(self, encoding_len):
        if encoding_len % 2 != 0:
            print('Error: The position encoding length must be an even number!')
            sys.exit(1)
        self.encoding_len = encoding_len

    def get_position_encoding(self, position_list):
        encoding = torch.zeros(self.encoding_len)
        encoding_list = []
        for position in position_list:
            encoding_list.extend(get_archimedean_spiral_encoding(position, ARCHIMEDES_SPIRAL_DIM))
        for i in range(len(encoding_list)):
            index = i % self.encoding_len
            encoding[index] += encoding_list[i]
        return encoding

    def get_position_encoding_2D(self, position_list):
        encoding_list = []
        for position in position_list:
            encoding_list.append(self.get_position_encoding(position))
        return torch.stack(encoding_list)
