# 该部分已完成！
import re
import os


# TODO
#  有一个问题：词性标注有点复杂，比如"[XXxxx]"，需要做三次词性标注："XX", "xxx", "[XXxxx]"
#  这种的词性标注jieba似乎做不到，如果自己实现一个，感觉非常难，做出来不一定准确。
#  另外，这样的形式导致输入之后做处理也比较麻烦。
#  目前的想法是先忽略掉这种的情况，等以后有办法了再来！

# ===========================工具函数===========================
def translate_to_origin(data=None):
    """
    去除数据集中文本的标记，还原成原文本
    :param data: 文本，string类型
    :return: 原文本，string类型
    """
    if data is None:
        return None
    # 去除标注的词性,一个/+任意多的字母+最多一个数字+最多一个的空格
    return re.sub(u"/[a-z]*[0-9]? ?", "", data)


# 期望是得到numpy类型的，但是遇到了一些问题！！！
def translate_to_dataset(data=None):
    """
    将数据集做处理，得到list类型的数据集，便于后面的训练
    :param data: 文本，string类型
    :return: 数据集，list类型
    """
    if data is None:
        return None
    data = data.split(" ")
    data_list = []
    for str_ in data:
        if "/" in str_:
            data_list.append(str_.split("/"))
    return data_list


def collect_files(path):
    """
    在指定目录下统计所有的txt文件，以列表形式返回
    :param path: 目录
    :return: 文件列表，list类型
    """
    file_list = []
    for parent, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".txt"):
                file_list.append(os.path.join(parent, filename))
    return file_list


# ===========================文件读取函数===========================
def read_single(filename=None):
    """
    读取单个文件，做最基本的数据处理工作，目的是去除中括号和对应的词性标注以及换行符"\n"
    :param filename: 文件名，string类型
    :return: 去除中括号和对应词性标注的内容，string类型
    """
    if filename is None:
        return None
    with open(filename) as file:
        data = file.read()
        # 去除换行符
        origin_txt = data.replace("\n", "")
        # 去除左中括号
        origin_txt = re.sub(u"\[", "", origin_txt)
        # 去除右中括号
        return re.sub(u"\]/[a-z]*[0-9]?", "", origin_txt)


def read_all(path=None):
    """
    读取某一个目录下所有的txt文件，包括子文件夹
    :param path: 路径名，string类型
    :return: txt文件处理之后的内容，list类型
    """
    all_txt = []
    files = collect_files(path)
    for file in files:
        all_txt.append([read_single(file)])
    return all_txt


def read_all_origin(path=None):
    """
    读取某一个目录下所有的原文
    :param path: 路径，string类型
    :return: 文本，list类型
    """
    all_txt = read_all(path)
    all_origin = []
    for str_list in all_txt:
        all_origin.append([translate_to_origin(str_list[0])])
    return all_origin


def read_all_dataset(path=None):
    """
    读取某一个目录下所有的数据集
    :param path: 路径，string类型
    :return: 数据集，list类型
    """
    all_txt = read_all(path)
    all_origin = []
    for str_list in all_txt:
        all_origin.append(translate_to_dataset(str_list[0]))
    return all_origin

# ===========================测试===========================
# if __name__ == "__main__":
#     str_ = read_single("./people-2014/test/0123/c1001-24200319.txt")
#     print(str_)
#     print()
#     print(translate_to_dataset(str_))
#     print(collect_files("./people-2014/test"))
#     print(read_all("./people-2014/test"))
#     print(read_all_origin("./people-2014/test"))
#     print(read_all_dataset("./people-2014/test"))
