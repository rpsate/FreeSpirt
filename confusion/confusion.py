import os
import random
import re

from tqdm import tqdm

confusion_range = (0.13, 0.5)
output_number = 20
source_dir = './ProgramData/1'
dst_dir = './dataset/1'
confusion_file = './confusion/confusion_code.txt'

# init
confusion_txt = []
with open(confusion_file, 'r', encoding='utf-8') as f:
    confusion_txt = f.readlines()


def get_confusion(line_size):
    random.shuffle(confusion_txt)
    return confusion_txt[:line_size]


def get_extension(file):
    return os.path.splitext(file)[1]


def get_filename(file):
    return os.path.splitext(file)[0]


files = os.listdir(source_dir)
for i, filename_extension in enumerate(tqdm(files)):
    # read file
    try:
        with open(os.path.join(source_dir, filename_extension), 'r', encoding='utf-8') as f:
            source_file = f.readlines()
    except Exception as e:
        continue

    '''remove notes and replace variable'''
    # read source file
    source_file = ''.join(source_file)

    # remove notes
    source_file = re.sub(r'\/\*[\w\W]*?\*\/|\/\/(.*)|\#(.*)', '', source_file)

    # string convert to list
    source_file_list = source_file.split('\n')
    source_file = []
    for item in source_file_list:
        item.strip()
        if item != '':
            source_file.append(item + '\n')

    # filename init
    cur_dir = os.path.join(dst_dir, str(i))
    cur_filename = get_filename(filename_extension)
    cur_extension = get_extension(filename_extension)

    # confusion init
    file_len = len(source_file)
    confusion_per = random.uniform(*confusion_range)
    confusion_len = round(file_len * confusion_per)

    # create dir if not exist
    if not os.path.exists(cur_dir):
        os.mkdir(cur_dir)

    # create copy file
    for j in range(output_number):
        cur_write_filename = os.path.join(cur_dir, cur_filename + '_' + str(j) + '_confusion' + cur_extension)

        # create confusion content
        confusion_line = random.sample(range(file_len), confusion_len)
        confusion_line.sort()
        confusion_line_list = get_confusion(confusion_len)

        # addition confusion to source_code
        write_content = []
        confusion_index = 0
        for k, line in enumerate(source_file):
            if k in confusion_line:
                write_content.append(confusion_line_list[confusion_index])
                confusion_index += 1
            write_content.append(line)

        with open(cur_write_filename, 'w', encoding='utf-8') as f:
            f.writelines(write_content)