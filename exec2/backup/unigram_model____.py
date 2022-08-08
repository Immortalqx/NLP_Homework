import math


class UnigramModel:
    def __init__(self):
        # 差评
        self.data_bad = []  # 单词列表
        self.word_count_bad = {}  # 单词词频统计
        self.unique_words_bad = None  # 非重复词
        # 好评
        self.data_good = []  # 单词列表
        self.word_count_good = {}  # 单词词频统计
        self.unique_words_good = None  # 非重复词

    def train_model(self, train_data):
        """
        训练一元语法模型
        :param train_data:训练集
        """
        # 分别构建差评和好评的语言模型
        for line_data in train_data:
            if line_data["label"] == "0":  # 如果是差评
                for word in line_data["text"]:
                    self.data_bad.append(word)
                    # 统计单词出现次数
                    self.word_count_bad.setdefault(word, 0)
                    self.word_count_bad[word] += 1
            elif line_data["label"] == "1":  # 如果是好评
                for word in line_data["text"]:
                    self.data_good.append(word)
                    # 统计单词出现次数
                    self.word_count_good.setdefault(word, 0)
                    self.word_count_good[word] += 1
        # 统计非重复词
        self.unique_words_bad = list(set(self.data_bad))
        self.unique_words_good = list(set(self.data_good))

    def get_possibility(self, line_data):
        """
        获取一个句子是差评、好评的概率的对数
        :param line_data: 句子
        :return: 差评、好评的概率
        """
        possibility_bad = 0
        possibility_good = 0
        for word in line_data["text"]:
            # 计算各单词概率，取对数后相加，使用了拉普拉斯平滑
            if word in self.word_count_bad:
                possibility_bad += \
                    math.log((self.word_count_bad[word] + 1) /
                             (len(self.data_bad) + len(self.unique_words_bad) + len(self.unique_words_good)))
            else:
                possibility_bad += \
                    math.log(1 /
                             (len(self.data_bad) + len(self.unique_words_bad) + len(self.unique_words_good)))
            if word in self.word_count_good:
                possibility_good += \
                    math.log((self.word_count_good[word] + 1) /
                             (len(self.data_good) + len(self.unique_words_good) + len(self.unique_words_good)))
            else:
                possibility_good += \
                    math.log(1 /
                             (len(self.data_good) + len(self.unique_words_good) + len(self.unique_words_good)))
        # 最后加上该分类概率的对数
        possibility_bad += math.log(len(self.data_bad) / (len(self.data_bad) + len(self.data_good)))
        possibility_good += math.log(len(self.data_good) / (len(self.data_bad) + len(self.data_good)))

        return possibility_bad, possibility_good

    # TODO
    def get_perplexity(self, line_data):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        pass

# if __name__ == "__main__":
#     import math
#     from read_process import read_dataset
#
#     bad_data, good_data = read_dataset("data/little_test.tsv", train=True, punctuation=True)
#
#     bad_model = UnigramModel()
#     good_model = UnigramModel()
#
#     bad_model.train_model(bad_data)
#     good_model.train_model(good_data)
#
#     test_data = read_dataset("data/test.tsv", train=False, punctuation=True)
#     correct_num = 0
#     for line_data in test_data:
#         possibility_bad ,possibility_good = pass
#
#         # 统计正确的个数
#         predict = "0"
#         if possibility_good > possibility_bad:
#             predict = "1"
#         if predict == line_data["label"]:
#             correct_num += 1
#
#     print("正确率：", correct_num / len(test_data))
