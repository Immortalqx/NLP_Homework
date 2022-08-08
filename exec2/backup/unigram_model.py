class UnigramModel:
    def __init__(self):
        self.data_ = []  # 单词列表
        self.word_count_ = {}  # 单词词频统计
        self.unique_words_ = None  # 非重复词

    def train_model(self, train_data):
        """
        训练一元语法模型
        :param train_data:训练集
        """
        # 分别构建差评和好评的语言模型
        for line_data in train_data:
            for word in line_data["text"]:
                self.data_.append(word)
                # 统计单词出现次数
                self.word_count_.setdefault(word, 0)
                self.word_count_[word] += 1
        # 统计非重复词
        self.unique_words_ = list(set(self.data_))

    def get_possibility(self, word):
        """
        获取某一个单词的概率
        :param word: 单词
        :return: 该单词的概率
        """
        if word in self.word_count_:
            return (self.word_count_[word] + 1) / \
                   (len(self.data_) + len(self.unique_words_))

        return 1 / (len(self.data_) + len(self.unique_words_))

    # TODO
    def get_perplexity(self, sentence):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        pass


if __name__ == "__main__":
    import math
    from read_process import read_dataset

    bad_data, good_data = read_dataset("../data/train.tsv", train=True, punctuation=True)

    bad_model = UnigramModel()
    good_model = UnigramModel()

    bad_model.train_model(bad_data)
    good_model.train_model(good_data)

    test_data = read_dataset("../data/test.tsv", train=False, punctuation=True)
    correct_num = 0
    for line_data in test_data:
        possibility_bad = 0
        possibility_good = 0
        for word in line_data["text"]:
            possibility_bad += math.log(bad_model.get_possibility(word))
            possibility_good += math.log(good_model.get_possibility(word))
        # TODO 为什么这里要加上该分类概率的对数？？？[把贝叶斯公式再看一遍。。。]
        # 最后加上该分类概率的对数
        possibility_bad += math.log(len(bad_model.data_) / (len(bad_model.data_) + len(good_model.data_)))
        possibility_good += math.log(len(good_model.data_) / (len(bad_model.data_) + len(good_model.data_)))

        # 统计正确的个数
        predict = "0"
        if possibility_good > possibility_bad:
            predict = "1"
        if predict == line_data["label"]:
            correct_num += 1

    print("正确率：", correct_num / len(test_data))
