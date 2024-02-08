from clang.cindex import CursorKind
from configs.config import INIT_BLOCK_FEATURES

# 提取代码块中的语句类型
def get_block_features(cursor):
    features = []
    # 循环遍历当前游标的子节点
    for child in cursor.get_children():
        # 如果子节点是语句类型，则将其类型名称添加到 features 列表中
        if child.kind.name in INIT_BLOCK_FEATURES.keys():
            features.append(child.kind.name)
        # 递归调用 get_block_features 函数，处理子节点中的更深层次的代码块
        features.extend(get_block_features(child))
    # 返回 features 列表，其中包含代码块中所有语句类型的名称
    return features

# 提取代码中的字符串文本
def get_string_features(cursor):
    features = []
    # 循环遍历当前游标的所有令牌
    for token in cursor.get_tokens():
        # 如果令牌类型是字符串文本，且以双引号开头，则将其添加到 features 列表中
        if token.kind.name == 'LITERAL' and token.spelling.startswith('"'):
            features.append(token.spelling)
    # 返回 features 列表，其中包含代码中所有以双引号开头的字符串文本
    return features

# 提取函数调用的特征
def get_function_call_features(cursor):
    features = []
    # 如果当前游标是函数调用类型，则将其函数名称添加到 features 列表中
    if cursor.kind == CursorKind.CALL_EXPR and cursor.referenced:
        features.append(cursor.referenced.displayname)
    # 递归调用 get_function_call_features 函数，处理函数调用的所有参数
    for child in cursor.get_children():
        features.extend(get_function_call_features(child))
    # 返回 features 列表，其中包含所有函数调用的名称
    return features
