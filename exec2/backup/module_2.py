# TODO 根据module_1.py修改过来的，可能是有问题的！！！
#  module_1还存在问题，主要是不清楚语言模型如何建立，建立之后如何使用。。。
# 二元模型
import math
import random

from read_process import read_dataset

# ==========================训练阶段==========================
# 读取数据集
data_raw = read_dataset("data/train.tsv", punctuation=True)

# 提前定义一系列变量
data_bad = []  # 差评单词列表
data_bad_tuple = []  # 差评单词列表
data_good = []  # 好评单词列表
data_good_tuple = []  # 好评单词列表
word_count_bad = {}  # 差评单词词频统计
word_count_bad_tuple = {}  # 差评单词词频统计
word_count_good = {}  # 好评单词词频统计
word_count_good_tuple = {}  # 好评单词词频统计

# 分别构建差评和好评的语言模型
for line_data in data_raw:
    if line_data["label"] == "0":  # 如果是差评
        # 开头特殊处理一下
        data_bad.append("<BOS>")
        word_count_bad.setdefault("<BOS>", 0)
        word_count_bad["<BOS>"] += 1
        for i in range(len(line_data["text"]) - 1):
            current_word = line_data["text"][i + 1]
            forward_word = line_data["text"][i]
            # 记录单个单词
            data_bad.append(current_word)
            word_count_bad.setdefault(current_word, 0)
            word_count_bad[current_word] += 1
            # 记录两个单词在一起
            data_bad_tuple.append((current_word, forward_word))
            word_count_bad_tuple.setdefault((current_word, forward_word), 0)
            word_count_bad_tuple[(current_word, forward_word)] += 1
            # 更新当前单词
            forward_word = current_word
    elif line_data["label"] == "1":  # 如果是好评
        # 开头特殊处理一下
        data_good.append("<BOS>")
        word_count_good.setdefault("<BOS>", 0)
        word_count_good["<BOS>"] += 1
        for i in range(len(line_data["text"]) - 1):
            current_word = line_data["text"][i + 1]
            forward_word = line_data["text"][i]
            # 记录单个单词
            data_good.append(current_word)
            word_count_good.setdefault(current_word, 0)
            word_count_good[current_word] += 1
            # 记录两个单词在一起
            data_good_tuple.append((current_word, forward_word))
            word_count_good_tuple.setdefault((current_word, forward_word), 0)
            word_count_good_tuple[(current_word, forward_word)] += 1
            # 更新当前单词
            forward_word = current_word
# 统计非重复词
unique_words_bad = list(set(data_bad))
unique_words_good = list(set(data_good))

unique_words_bad_tuple = list(set(data_bad_tuple))
unique_words_good_tuple = list(set(data_good_tuple))

# ==========================测试阶段==========================
# 读取测试集
test_data = read_dataset("data/test.tsv", punctuation=True)

correct_num = 0
for line_data in test_data:
    possibility_bad = 0
    possibility_good = 0
    for i in range(len(line_data["text"]) - 1):
        current_word = line_data["text"][i + 1]
        forward_word = line_data["text"][i]

        # TODO 分母需要统计某个单词的次数！
        # 计算各单词概率，取对数后相加，使用了拉普拉斯平滑
        if (current_word, forward_word) in word_count_bad_tuple:
            possibility_bad += math.log(
                (word_count_bad_tuple[(current_word, forward_word)] + 1) / (
                        word_count_bad[forward_word] + (len(unique_words_bad) + len(unique_words_good))))
        elif forward_word in word_count_bad:
            possibility_bad += math.log(
                1 / (word_count_bad[forward_word] + (len(unique_words_bad) + len(unique_words_good))))
        else:
            possibility_bad += math.log(1 / (len(unique_words_bad) + len(unique_words_good)))

        # 计算各单词概率，取对数后相加，使用了拉普拉斯平滑
        if (current_word, forward_word) in word_count_good_tuple:
            possibility_good += math.log(
                (word_count_good_tuple[(current_word, forward_word)] + 1) / (
                        word_count_good[forward_word] + (len(unique_words_bad) + len(unique_words_good))))
        elif forward_word in word_count_good:
            possibility_good += math.log(
                1 / (word_count_good[forward_word] + (len(unique_words_bad) + len(unique_words_good))))
        else:
            possibility_good += math.log(1 / (len(unique_words_bad) + len(unique_words_good)))

    # 最后加上该分类概率的对数
    possibility_bad += math.log(len(data_bad) / (len(data_bad) + len(data_good)))
    possibility_good += math.log(len(data_good) / (len(data_bad) + len(data_good)))

    # 统计正确的个数
    predict = "0"
    # if random.randint(0, 100) > 50:
    if possibility_good > possibility_bad:
        predict = "1"
    if predict == line_data["label"]:
        correct_num += 1

print("正确率：", correct_num / len(test_data))
