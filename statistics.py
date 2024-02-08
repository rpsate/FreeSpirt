import argparse
import math
import sys

from tqdm import tqdm

from Sqlite.SqliteHelper import SqliteHelper
from Tree.ParserAST import ParserAST
from os.path import exists

def statistics(conn, table_name):
    print('Info: Start counting training data···')
    train_set = conn.query('SELECT * FROM train')
    for item in tqdm(train_set, desc='Statistical training data:'):
        ast = ParserAST(item['code'], ('funcs',)).get_funcs()[0]
        data = {
            'idx': item['id'],
            'label': item['label'],
            'category': 'train',
            'width': ast.get_width(),
            'depth': ast.get_depth(),
            'size': ast.get_size()
        }
        if not conn.insert(table_name, data):
            sys.exit(1)

    print('Info: Start counting validation data···')
    valid_set = conn.query('SELECT * FROM valid')
    for item in tqdm(valid_set, desc='Statistical validation data:'):
        ast = ParserAST(item['code'], ('funcs',)).get_funcs()[0]
        data = {
            'idx': item['id'],
            'label': item['label'],
            'category': 'valid',
            'width': ast.get_width(),
            'depth': ast.get_depth(),
            'size': ast.get_size()
        }
        if not conn.insert(table_name, data):
            sys.exit(1)

    print('Info: Start counting test data···')
    test_set = conn.query('SELECT * FROM test')
    for item in tqdm(test_set, desc='Statistical test data:'):
        ast = ParserAST(item['code'], ('funcs',)).get_funcs()[0]
        data = {
            'idx': item['id'],
            'label': item['label'],
            'category': 'test',
            'width': ast.get_width(),
            'depth': ast.get_depth(),
            'size': ast.get_size()
        }
        if not conn.insert(table_name, data):
            sys.exit(1)

def print_data(conn, table_name):
    try:
        # 获取信息
        max_size = conn.query(f'SELECT MAX(size) FROM {table_name}')[0][0]
        max_width = conn.query(f'SELECT MAX(width) FROM {table_name}')[0][0]
        max_depth = conn.query(f'SELECT MAX(depth) FROM {table_name}')[0][0]

        min_size = conn.query(f'SELECT MIN(size) FROM {table_name}')[0][0]
        min_width = conn.query(f'SELECT MIN(width) FROM {table_name}')[0][0]
        min_depth = conn.query(f'SELECT MIN(depth) FROM {table_name}')[0][0]

        train_num = conn.query(F'SELECT COUNT(*) FROM {table_name} WHERE category="train"')[0][0]
        valid_num = conn.query(F'SELECT COUNT(*) FROM {table_name} WHERE category="valid"')[0][0]
        test_num = conn.query(F'SELECT COUNT(*) FROM {table_name} WHERE category="test"')[0][0]

        size_list = []
        max_range = math.ceil(max_size / 10) * 10
        for i in range(0, max_range, 10):
            num = conn.query(f'SELECT COUNT(*) FROM {table_name} WHERE size>={i} AND size<{i+10}')[0][0]
            size_list.append(num)

        # 打印信息
        print('-' * 20)
        print('Total number of training data:', train_num)
        print('Total number of validation data:', valid_num)
        print('Total number of test data:', test_num)

        print('-' * 20)
        print('Maximum number of nodes:', max_size)
        print('Maximum breadth:', max_width)
        print('Maximum depth:', max_depth)

        print('-' * 20)
        print('Minimum number of nodes:', min_size)
        print('Minimum breadth:', min_width)
        print('Minimum depth:', min_depth)

        print('-' * 20)
        print('Node size distribution:')
        for index, item in enumerate(size_list):
            if item > 0:
                print('{:3}-{:3}:{:4}'.format(index * 10, index * 10 + 10, item))
    except Exception:
        sys.exit(1)

def clear_data(conn, table_name):
    print(f'Are you sure you want to delete all data in table {table_name}?[Y/N]:', end='')
    select = str(input()).lower()
    if select == 'n':
        sys.exit(0)
    if conn.execute(f'delete from {table_name}'):
        print(f'Info: Successfully deleted table {table_name}!')
    else:
        print(f'Failed to delete Table {table_name}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical Dataset Program.')
    parser.add_argument('--database', type=str, required=True, help='SQLite database file for storing datasets.')
    parser.add_argument('--delete', action='store_true', help='Clear data in the table!')
    args = parser.parse_args()
    if not exists(args.database):
        print(f'Error: Database {args.database} does not exist')
        sys.exit(1)

    # 如果没有表则创建表
    table_name = 'statistics'
    create_table = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY,
        idx INTEGER,
        label INTEGER,
        category TEXT,
        width INTEGER,
        depth INTEGER,
        size INTEGER
    );
    """
    conn = SqliteHelper(args.database)
    if not conn.execute(create_table):
        print(f'Error: The table {table_name} creation failed！')
        sys.exit(1)

    if args.delete:
        clear_data(conn, table_name)
    else:
        # 判断表是否为空
        count = conn.query(f'SELECT COUNT(*) FROM {table_name}')[0][0]
        # 如果表为空则开始统计
        if count == 0:
            statistics(conn, table_name)
        print_data(conn, table_name)