import math
from collections import defaultdict
from read import read_data


def TFIDF(document_list, show=False):
    """
    计算文档列表中每一个词的TFIDF值
    :param document_list: 文档列表
    :param show: 是否展示计算结果
    :return: tfidf值
    """
    # ==================获取单词集合==================
    unique_words = []
    for word_list in document_list:
        for i in word_list:
            unique_words.append(i)
    unique_words = list(set(unique_words))

    # ==================计算每一个单词的TF==================
    word_tf = {}
    for t in unique_words:
        for d in range(len(document_list)):
            if t not in document_list[d]:
                word_tf[(t, d)] = 0
            else:
                word_tf[(t, d)] = 1 + math.log(document_list[d].count(t))

    # ==================计算每一个单词的IDF==================
    word_idf = {}  # 存储每个词的idf值
    word_df = defaultdict(int)  # 存储包含该词的文档数
    for t in unique_words:
        for d in document_list:
            if t in d:
                word_df[t] += 1
        word_idf[t] = math.log(len(document_list) / (word_df[t]))  # 单词是从文档里提取出来的，所以不担心分母为0

    # ==================计算每一个单词的TFIDF==================
    word_tfidf = {}  # 存储每个词的idf值
    for t in unique_words:
        for d in range(len(document_list)):
            word_tfidf[(t, d)] = word_tf[(t, d)] * word_idf[t]

    if show:
        format_rule = "{:<20}"

        print(format_rule.format("单词/文档"), end="")
        for i in range(len(document_list)):
            print(format_rule.format(i), end="")
        print()
        for t in unique_words:
            print(format_rule.format(t), end="")
            for d in range(len(document_list)):
                print(format_rule.format(word_tfidf[(t, d)]), end="")
            print()

    return word_tfidf


if __name__ == '__main__':
    data_list = read_data("data/test.txt")  # 加载数据
    TFIDF(data_list, show=True)  # 所有词的TF-IDF值
