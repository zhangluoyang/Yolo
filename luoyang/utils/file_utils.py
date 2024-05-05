import os
import re
import math
import yaml
import json
import shutil
from typing import List, Union


def write_lines(lines: List[Union[str, dict]], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, dict):
                f.write("{0}\n".format(json.dumps(line, ensure_ascii=False)))
            else:
                f.write("{0}\n".format(line))


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        # Add YAML filename to dict and return
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = list(filter(lambda line: len(line) > 0, lines))
        return lines


def read_dict(path: str, split_str: str = " "):
    lines = read_lines(path=path)
    _dict = {}
    for line in lines:
        key, value = line.split(split_str)
        _dict[key] = value
    return _dict


def cal_file_line_num(path_list: Union[str, List[str]]) -> int:
    """
    读取文件行数 (计算回车换行符的数目)
    :param path_list:
    :return:
    """
    if isinstance(path_list, str):
        path_list = [path_list]
    line_nums = 0
    # 分别遍历每一个文件 并读取文件的总行数
    for path in path_list:
        f = open(path, "rb")
        while True:
            line = f.readline()
            if not line:
                break
            line_nums += 1
            if line_nums % 100 == 0:
                print(line)
                print(line_nums)
    return line_nums


def split_file(file_path: str, save_path: str, num: int):
    """
    将一个大的文件切分为多个小文件
    :param file_path:  文件路径列表
    :param save_path:
    :param num: 切分的份数
    :return:
    """

    line_num = cal_file_line_num(path_list=file_path)
    print("文件总数目:{0}".format(line_num))
    # 单个文件的字符行数
    single_line_num = math.ceil(line_num / num)

    read_file = open(file_path, "r", encoding=try_file_encode(file_path))
    # 写文件计数
    write_file_index = 0
    while write_file_index < num:
        write_file = open("{0}/{1}.txt".format(save_path, write_file_index), "w", encoding="utf-8")
        for j in range(single_line_num):
            line = read_file.readline()
            if line is not None:
                write_file.writelines(line)
            else:
                break
        write_file_index += 1
        write_file.close()


def try_file_encode(file_path: str) -> str:
    """
    :param file_path:
    :return:
    """
    try:
        file_read = open(file_path, "r", encoding="utf-8")
        file_read.readline()
        file_read.readline()
        file_read.readline()
        file_read.readline()
        return "utf-8"
    except:
        try:
            file_read = open(file_path, "r", encoding="gbk")
            file_read.readline()
            file_read.readline()
            file_read.readline()
            file_read.readline()
            return "gbk"
        except:
            pass
    return "utf-8"


def merge_file(file_path_list: List[str], save_path: str):
    """
    多个文件进行合并 生成一个文件
    :param file_path_list:
    :param save_path:
    :return:
    """
    with open(save_path, "w", encoding="utf-8") as writer:
        for file_path in file_path_list:
            with open(file_path, "r", encoding=try_file_encode(file_path)) as reader:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)


def delta_path_or_file(path: str):
    """
    删除一个目录下面所有的文件
    :return:
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)


def file_is_exists(path: str):
    return os.path.exists(path)


def copy_file(source_path: str, target_path: str):
    """
    复制文件
    :param source_path:
    :param target_path:
    :return:
    """
    write = open(target_path, "w", encoding="utf-8")
    reader = open(source_path, "r", encoding="utf-8")
    write.write(reader.read())
    reader.close()
    write.close()


def get_file_paths(root_path: str, prefix: str = None, postfix: str = None) -> List[str]:
    """
    获取一个根目录下所有的文件目录
    :param root_path:  根目录
    :param prefix:  前缀
    :param postfix:  后缀
    :return:
    """
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        root = root.replace("\\", "/")  # 兼容linux格式
        for file in files:
            file_name = file.split("/")[-1]
            flag = True
            if prefix is not None:
                if not file_name.startswith(prefix):
                    flag = False
            if postfix is not None:
                if not file_name.endswith(postfix):
                    flag = False
            if flag:
                file_paths.append("{0}/{1}".format(root, file))
    return file_paths


def get_dir_paths(root_path: str) -> List[str]:
    """
    返回目录下所有的文件夹
    :param root_path:
    :return:
    """
    for root, dirs, files in os.walk(root_path):
        return dirs


def get_path_name(path: str) -> Union[None, str]:
    """
    返回名称
    :param path:
    :return:
    """
    if os.path.exists(path) and os.path.isdir(path):
        return path.split(r"/")[-1]
    return None


def get_file_name(path: str) -> Union[None, str]:
    """

    :param path:
    :return:
    """
    if os.path.exists(path):
        return ".".join(path.split(r"/")[-1].split(".")[: -1])
    return None


def parser_file_name(path: str) -> str:
    return ".".join(path.split(r"/")[-1].split(".")[: -1])


def rename(src_path: str, target_path: str):
    """
    修改名字
    :param src_path:
    :param target_path:
    :return:
    """
    if os.path.exists(src_path):
        os.rename(src_path, target_path)


def get_parent_path_name(file_path: str) -> Union[None, str]:
    """
    返回上一级文件夹名称
    :param file_path:
    :return:
    """
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return file_path.split(r"/")[-2]
    return None


def get_grandfather_path_name(file_path: str) -> Union[None, str]:
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return file_path.split(r"/")[-3]
    return None


def mkdir_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def split_train_test(data_path: str, train_path: str, test_path: str):
    """
    分别生成训练数据和测试数据的索引文件
    :return:
    """
    # 第一层类别
    first_dir_names = get_dir_paths(data_path)
    # 第二层
    second_dir_names = []
    for first_dir_name in first_dir_names:
        for name in get_dir_paths("{0}/{1}".format(data_path, first_dir_name)):
            if name not in second_dir_names:
                second_dir_names.append(name)
    train_file = open(train_path, "w", encoding="utf-8")
    test_file = open(test_path, "w", encoding="utf-8")
    for first_name in first_dir_names:
        for second_name in second_dir_names:
            path = "{0}/{1}/{2}".format(data_path, first_name, second_name)
            image_file_paths = get_file_paths(path)
            length = len(image_file_paths)
            test_index = math.ceil(length * 0.1)
            train_paths = image_file_paths[0: length - test_index]
            test_paths = image_file_paths[length - test_index:]
            for p in train_paths:
                train_file.write("{0}\n".format(p))
            for p in test_paths:
                test_file.write("{0}\n".format(p))
    train_file.close()
    test_file.close()


def load_json(file_path: str):
    """
    加载json数据
    :param file_path:
    :return:
    """
    with open(file_path, "r", encoding="utf-8") as reader:
        return json.load(reader)
