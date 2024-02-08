from math import log, log10
import numpy as np
from configs.config import INIT_BLOCK_FEATURES
from collections import Counter

class FeaturesEmbedding(object):
    def __init__(self, fasttext_model):
        self.text_embedding = fasttext_model

    @staticmethod
    def get_block_features_vec(blocks):
        # vector size is 6
        keys = list(INIT_BLOCK_FEATURES.keys())
        keys.sort()
        features_vex = []
        for block in blocks:
            feature = INIT_BLOCK_FEATURES
            feature_counter = Counter(block)
            for key, value in feature_counter.items():
                feature[key] += value
            features_vex.append(np.array([log10(feature[key]+1) for key in keys]))
        return features_vex
    
    def get_string_features_vec(self, strings):
        # vector size is text embedding size
        features_vec = []
        for string_list in strings:
            if string_list:
                vectors = [self.text_embedding.get_sentence_vec(string) for string in string_list]
                sentence_vector = sum(vectors) / len(vectors)
            else:
                sentence_vector = self.text_embedding.get_word_vec('')
            features_vec.append(sentence_vector)
        return features_vec

    def get_callee_features_vec(self, callees):
        # vector size is text embedding size
        features_vec = []
        for callee in callees:
            features_vec.append(self.text_embedding.get_list_vec(callee))
        return features_vec

    @staticmethod
    def get_ast_features_vec(asts):
        # vector size is 3
        features_vec = []
        for ast in asts:
            feature = [
                log10(ast.get_size()+1),
                log(ast.get_width()+1),
                log(ast.get_depth()+1)
            ]
            features_vec.append(np.array(feature))
        return features_vec

