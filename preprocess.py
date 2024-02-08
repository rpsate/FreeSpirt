import os
import sys
from tqdm import tqdm
import argparse
from Sqlite.SqliteHelper import SqliteHelper


def files(path):
    g = os.walk(path)
    file = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            file.append(os.path.join(path, file_name))
    return file

def process(conn, directory, dataset):
    # 预处理
    dirs = os.listdir(directory)
    dirs.sort(key=lambda x: int(x))
    n = len(dirs)
    cut_point1, cut_point2 = int(0.7 * n), int(0.8 * n)
    train_data = dirs[:cut_point1]
    valid_data = dirs[cut_point1:cut_point2]
    test_data = dirs[cut_point2:]

    # 处理数据
    if 'train' in dataset:
        count = 0
        for i in tqdm(train_data, total=len(train_data), desc='Adding training dataset'):
            items = files(os.path.join(directory, str(i)))
            for item in items:
                code_item = {
                    'id': str(count),
                    'label': item.split(os.path.sep)[-2],
                    'code': open(item, encoding='latin-1').read()
                }
                count += 1
                if not conn.insert(table['train'], code_item):
                    print(f'Error: The {count}th training data failed to be inserted into the database')
                    sys.exit(1)

    if 'valid' in dataset:
        count = 0
        for i in tqdm(valid_data, total=len(valid_data), desc='Adding validation dataset'):
            items = files(os.path.join(directory, str(i)))
            for item in items:
                code_item = {
                    'id': str(count),
                    'label': item.split(os.path.sep)[-2],
                    'code': open(item, encoding='latin-1').read()
                }
                count += 1
                if not conn.insert(table['valid'], code_item):
                    print(f'Error: The {count}th validation data failed to be inserted into the database')
                    sys.exit(1)

    if 'test' in dataset:
        count = 0
        for i in tqdm(test_data, total=len(test_data), desc='Adding test dataset'):
            items = files(os.path.join(directory, str(i)))
            for item in items:
                code_item = {
                    'id': str(count),
                    'label': item.split(os.path.sep)[-2],
                    'code': open(item, encoding='latin-1').read()
                }
                count += 1
                if not conn.insert(table['test'], code_item):
                    print(f'Error: The {count}th test data failed to be inserted into the database')
                    sys.exit(1)
    print('Info: Data processing completed!')

def clear_data(conn, table_list):
    table_str = ','.join(table_list)
    print(f'Are you sure you want to delete all data in table {table_str}?[Y/N]:', end='')
    select = str(input()).lower()
    if select == 'n':
        sys.exit(0)
    for item in table_list:
        if conn.execute(f'delete from {item}'):
            print(f'Info: Successfully deleted table {item}!')
        else:
            print(f'Failed to delete Table {item}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing program.')
    parser.add_argument('--dir', type=str, help='Directory where the dataset is located.')
    parser.add_argument('--database', type=str, required=True, help='SQLite database file for storing datasets.')
    parser.add_argument('--add', type=str, default='all', help='Select the dataset to add! (train/valid/test/all)')
    parser.add_argument('--delete', type=str, help='Clear data in the table! (train/valid/test/all)')
    args = parser.parse_args()
    table = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test'
    }

    # 如果没有表则创建表
    create_table = """
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY,
        label INTEGER,
        code TEXT
    );
    """
    sqlite = SqliteHelper(args.database)
    for table_name in table.values():
        if not sqlite.execute(create_table.format(table=table_name)):
            print(f'Error: The table {table_name} creation failed！')
            sys.exit(1)

    # 清除数据库中数据
    if args.delete:
        if args.delete == 'all':
            tables = list(table.values())
        elif args.delete in table.keys():
            table = [table[args.delete]]
        else:
            print(f'Error: This table {args.delete} does not exist! Please select in (train/valid/test/all)')
            sys.exit(1)
        clear_data(sqlite, tables)
    elif not args.dir:
        print('Error: The --dir parameter cannot be empty!')
    else:
        if args.add == 'all':
            dataset = list(table.values())
        elif args.add in table.keys():
            dataset = [table[args.add]]
        else:
            print(f'Error: This dataset {args.add} does not exist! Please select in (train/valid/test/all)')
            sys.exit(1)
        process(sqlite, args.dir, dataset)
