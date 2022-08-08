import re
import jieba


# TODO 是否去除标点符号呢？？？
#  感觉可以两种都试一试！
def read_no_punctuation(filename=None):
    """
    读取tsv文件，并且进行处理，去除标点符号
    :param filename: 文件名，string类型
    :return: 文件的内容，形式为[{'label' : '0' or '1', 'text' : 'XXX'}, ......]
    """
    if filename is None:
        return None
    with open(filename) as file:
        # 首先去除空格，之后将"\t"替换为空格，并且按行分词
        data_raw = file.read().replace(" ", "").replace("\t", " ").split("\n")
        data_raw.pop(0)  # 去除第一行的标签
        data = []
        for line_data in data_raw:
            if line_data is None or " " not in line_data:  # 去除一些异常情况
                continue
            line_data = re.sub(r'[^\w\s]', '', line_data)
            temp = line_data.split(" ")
            data.append({"label": temp[0], "text": temp[1]})
        return data


def read_punctuation(filename=None):
    """
    读取tsv文件，并且进行处理，不去除标点符号
    :param filename: 文件名，string类型
    :return: 文件的内容，形式为[{'label' : '0' or '1', 'text' : 'XXX'}, ......]
    """
    if filename is None:
        return None
    with open(filename) as file:
        # 首先去除空格，之后将"\t"替换为空格，并且按行分词
        data_raw = file.read().replace(" ", "").replace("\t", " ").split("\n")
        data_raw.pop(0)  # 去除第一行的标签
        data = []
        for line_data in data_raw:
            if line_data is None or " " not in line_data:  # 去除一些异常情况
                continue
            temp = line_data.split(" ")
            data.append({"label": temp[0], "text": temp[1]})
        return data


def cut_train(data_raw=None):
    """
    使用jieba进行分词处理
    :param data_raw: 文件的内容，形式为[{'label' : '0' or '1', 'text' : 'XXX'}, ......]
    :return: 分词的结果,形式为[{'label' : '0' or '1', 'text' : 'XX X XXX xxx'}, ......]
    """
    bad_result = []  # 差评
    good_result = []  # 好评
    for line_data in data_raw:
        seg_list = list(jieba.cut(line_data["text"]))
        # TODO 这里只有二元语法的时候才可以用？？？？
        seg_list.insert(0, "<BOS>")
        seg_list.append("<EOS>")
        if line_data["label"] == "0":
            bad_result.append({"label": line_data["label"], "text": seg_list})
        else:
            good_result.append({"label": line_data["label"], "text": seg_list})
    return bad_result, good_result


def cut_test(data_raw=None):
    """
    使用jieba进行分词处理
    :param data_raw: 文件的内容，形式为[{'label' : '0' or '1', 'text' : 'XXX'}, ......]
    :return: 分词的结果,形式为[{'label' : '0' or '1', 'text' : 'XX X XXX xxx'}, ......]
    """
    result = []
    for line_data in data_raw:
        seg_list = list(jieba.cut(line_data["text"]))
        # TODO 这里只有二元语法的时候才可以用？？？？
        seg_list.insert(0, "<BOS>")
        seg_list.append("<EOS>")
        result.append({"label": line_data["label"], "text": seg_list})
    return result


# 进行完整的读取、分词处理操作
def read_dataset(path, train=False, punctuation=False, user_dict="data/vocab.txt"):
    """
    将上面的流程整合成一个函数（方便后面调用）
    :param path: 文件路径
    :param train: 是否读取为训练集的格式
    :param punctuation: 是否包含标点符号
    :param user_dict: 词典的路径
    :return: 处理好的数据，形式为[{'label' : '0' or '1', 'text' : 'XX X XXX xxx'}, ......]
    """
    jieba.load_userdict(user_dict)
    if train:
        if punctuation:
            return cut_train(read_punctuation(path))
        return cut_train(read_no_punctuation(path))
    else:
        if punctuation:
            return cut_test(read_punctuation(path))
        return cut_test(read_no_punctuation(path))


# def cut_(data_raw=None):
#     """
#     使用jieba进行分词处理
#     :param data_raw: 文件的内容，形式为[{'label' : '0' or '1', 'text' : 'XXX'}, ......]
#     :return: 分词的结果,形式为[{'label' : '0' or '1', 'text' : 'XX X XXX xxx'}, ......]
#     """
#     result = []
#     for line_data in data_raw:
#         seg_list = list(jieba.cut(line_data["text"]))
#         seg_list.insert(0, "<BOS>")
#         seg_list.append("<EOS>")
#         result.append({"label": line_data["label"], "text": seg_list})
#     return result
#
#
# # 进行完整的读取、分词处理操作
# def read_dataset(path, punctuation=False, user_dict="data/vocab.txt"):
#     """
#     将上面的流程整合成一个函数（方便后面调用）
#     :param path: 文件路径
#     :param punctuation: 是否包含标点符号
#     :param user_dict: 词典的路径
#     :return: 处理好的数据，形式为[{'label' : '0' or '1', 'text' : 'XX X XXX xxx'}, ......]
#     """
#     jieba.load_userdict(user_dict)
#     if punctuation:
#         return cut_(read_punctuation(path))
#     return cut_(read_no_punctuation(path))


if __name__ == "__main__":
    bad_, good_ = read_dataset("data/little_test.tsv", train=True, punctuation=True)
    for line_ in bad_:
        print(line_)
    for line_ in good_:
        print(line_)

    data_ = read_dataset("data/little_test.tsv", punctuation=True)
    for line_ in data_:
        print(line_)
