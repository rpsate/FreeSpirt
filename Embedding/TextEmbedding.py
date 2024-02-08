import fasttext
from utils.tool_function import sentence2words
from fasttext.util import reduce_model
from configs.config import FASTTEXT_MODEL_PATH

class TextEmbedding:
    def __init__(self, model_path=FASTTEXT_MODEL_PATH, dim=16):
        fasttext.FastText.eprint = lambda x: None if 'does not return WordVectorModel or SupervisedModel any more' in x else x
        print('Info: Fasttext model loading ······')
        self.model = fasttext.load_model(model_path)
        if dim > 300 or dim < 6:
            print('Warning: The dimension range of text embedding vector is 6-300')
        else:
            reduce_model(self.model, int(dim))
        print('Info: Fasttext model loading completed!')

    def get_word_vec(self, word=''):
        return self.model.get_word_vector(word)

    def get_sentence_vec(self, sentence):
        words = sentence2words(sentence)
        return self.get_list_vec(words)

    def get_list_vec(self, words):
        if len(words) < 1:
            return self.get_word_vec('')
        vectors = [self.model.get_word_vector(word) for word in words]
        sentence_vector = sum(vectors) / len(vectors)
        return sentence_vector
