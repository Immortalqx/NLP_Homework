# TODO 貌似没有建立出一元模型！
#  并且朴素贝叶斯分类模型好像也有问题？
# 一元模型
from read_process import read_dataset
import math

# ==========================训练阶段==========================
# 读取数据集
data_raw = read_dataset("data/train.tsv", punctuation=True)

# 提前定义一系列变量
data_bad = []  # 差评单词列表
data_good = []  # 好评单词列表
word_count_bad = {}  # 差评单词词频统计
word_count_good = {}  # 好评单词词频统计

# 分别构建差评和好评的语言模型
for line_data in data_raw:
    if line_data["label"] == "0":  # 如果是差评
        for word in line_data["text"]:
            data_bad.append(word)
            # 统计单词出现次数
            word_count_bad.setdefault(word, 0)
            word_count_bad[word] += 1
    elif line_data["label"] == "1":  # 如果是好评
        for word in line_data["text"]:
            data_good.append(word)
            # 统计单词出现次数
            word_count_good.setdefault(word, 0)
            word_count_good[word] += 1

# 统计非重复词
unique_words_bad = list(set(data_bad))
unique_words_good = list(set(data_good))

# ==========================测试阶段==========================
# 读取测试集
test_data = read_dataset("data/test.tsv", punctuation=True)

correct_num = 0
for line_data in test_data:
    possibility_bad = 0
    possibility_good = 0
    for word in line_data["text"]:
        # 计算各单词概率，取对数后相加，使用了拉普拉斯平滑
        if word in word_count_bad:
            possibility_bad += math.log(
                (word_count_bad[word] + 1) / (len(data_bad) + len(unique_words_bad) + len(unique_words_good)))
        else:
            possibility_bad += math.log(1 / (len(data_bad) + len(unique_words_bad) + len(unique_words_good)))
        # 计算各单词概率，取对数后相加，使用了拉普拉斯平滑
        if word in word_count_good:
            possibility_good += math.log(
                (word_count_good[word] + 1) / (len(data_good) + len(unique_words_bad) + len(unique_words_good)))
        else:
            possibility_good += math.log(1 / (len(data_good) + len(unique_words_bad) + len(unique_words_good)))
    # TODO 为什么这里要加上该分类概率的对数？？？[把贝叶斯公式再看一遍。。。]
    # 最后加上该分类概率的对数
    possibility_bad += math.log(len(data_bad) / (len(data_bad) + len(data_good)))
    possibility_good += math.log(len(data_good) / (len(data_bad) + len(data_good)))

    # 统计正确的个数
    predict = "0"
    if possibility_good > possibility_bad:
        predict = "1"
    if predict == line_data["label"]:
        correct_num += 1

print("正确率：", correct_num / len(test_data))

# 测试结果记录：
#  忽略标点符号的正确率：0.8425
#  不忽略标点符号的正确率：0.8375
