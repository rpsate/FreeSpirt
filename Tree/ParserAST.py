import clang.cindex
from Tree.Node import Node
from utils.features import get_block_features, get_string_features, get_function_call_features


class ParserAST(object):
    def __init__(self, source_code=None, extract=('funcs', 'string', 'block', 'callee')):
        self.source_code = source_code
        self.funcs = []
        self.block_features = []
        self.string_features = []
        self.callee_features = []
        self.init_features(extract)

    def init_features(self, extract):
        index = clang.cindex.Index.create()
        translation_unit = index.parse('temp.c', unsaved_files=[('temp.c', self.source_code)], args=['-std=c99', '-fparse-function-body-only', '-nostdinc'])
        cursor_root = translation_unit.cursor
        # 将AST转成自定义Tree格式
        if 'funcs' in extract:
            self.funcs.append(self._generate_tree(cursor_root))
        # 获取块状特征
        if 'block' in extract:
            self.block_features.append(get_block_features(cursor_root))
        # 获取字符串特征
        if 'string' in extract:
            self.string_features.append(get_string_features(cursor_root))
        # 获取函数调用特征
        if 'callee' in extract:
            self.callee_features.append(get_function_call_features(cursor_root))

    def _generate_tree(self, cursor, parent=None):
        node = Node(kind=cursor.kind.name, spelling=cursor.spelling, parent=parent)
        for c in cursor.get_children():
            node.add_child(self._generate_tree(cursor=c, parent=node))
        return node

    def get_block_features(self):
        return self.block_features

    def get_string_features(self):
        return self.string_features

    def get_callee_features(self):
        return self.callee_features

    def get_funcs(self):
        return self.funcs

    def get_all(self):
        return {
            'funcs': self.funcs,
            'block': self.block_features,
            'callee': self.callee_features,
            'string': self.string_features
        }

    def get_code(self):
        return self.source_code
