import math


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

    def get_poss(self, word):
        """
        获取某一个单词的概率的对数
        :param word: 单词
        :return: 该单词的概率
        """
        if word in self.word_count_:
            return math.log((self.word_count_[word] + 1) / \
                            (len(self.data_) + len(self.unique_words_)))

        return math.log(1 / (len(self.data_) + len(self.unique_words_)))

    def get_perp(self, sentence):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        perplexity = 0
        for word in sentence:
            perplexity += self.get_poss(word)  # 对数相加
        # return math.exp(perplexity * (-1 / len(sentence)))  # 乘以-1/N，最后还原
        return perplexity * (-1 / len(sentence))  # 乘以-1/N


# 之前写了个散装的语法模型，效果比上面封装的好，这里封装了实现一下
class MixUnigramModel:
    def __init__(self):
        # 差评
        self.data_bad = []  # 单词列表
        self.word_count_bad = {}  # 单词词频统计
        self.unique_words_bad = None  # 非重复词
        # 好评
        self.data_good = []  # 单词列表
        self.word_count_good = {}  # 单词词频统计
        self.unique_words_good = None  # 非重复词

    def train_bad_model(self, train_data):
        """
        训练一元语法模型
        :param train_data:训练集
        """
        # 分别构建差评和好评的语言模型
        for line_data in train_data:
            for word in line_data["text"]:
                self.data_bad.append(word)
                # 统计单词出现次数
                self.word_count_bad.setdefault(word, 0)
                self.word_count_bad[word] += 1
        # 统计非重复词
        self.unique_words_bad = list(set(self.data_bad))

    def train_good_model(self, train_data):
        """
        训练一元语法模型
        :param train_data:训练集
        """
        # 分别构建差评和好评的语言模型
        for line_data in train_data:
            for word in line_data["text"]:
                self.data_good.append(word)
                # 统计单词出现次数
                self.word_count_good.setdefault(word, 0)
                self.word_count_good[word] += 1
        # 统计非重复词
        self.unique_words_good = list(set(self.data_good))

    def get_badword_poss(self, word):
        """
        获取某一个单词的概率的对数
        :param word: 单词
        :return: 该单词的概率
        """
        if word in self.word_count_bad:
            return math.log((self.word_count_bad[word] + 1) / \
                            (len(self.data_bad) + (len(self.unique_words_bad) + len(self.unique_words_good))))
        return math.log(1 / (len(self.data_bad) + (len(self.unique_words_bad) + len(self.unique_words_good))))

    def get_goodword_poss(self, word):
        """
        获取某一个单词的概率的对数
        :param word: 单词
        :return: 该单词的概率
        """
        if word in self.word_count_good:
            return math.log((self.word_count_good[word] + 1) / \
                            (len(self.data_good) + (len(self.unique_words_bad) + len(self.unique_words_good))))
        return math.log(1 / (len(self.data_good) + (len(self.unique_words_bad) + len(self.unique_words_good))))

    def get_bad_poss(self, sentence):
        """
        获取一句话是差评的概率
        :param sentence: 一句话
        :return: 差评的概率
        """
        possibility_bad = 0
        for current_word in sentence:
            possibility_bad += self.get_badword_poss(current_word)
        # 最后加上该分类概率的对数
        possibility_bad += math.log(len(self.data_bad) / (len(self.data_bad) + len(self.data_good)))
        return possibility_bad

    def get_good_poss(self, sentence):
        """
        获取一句话是好评的概率
        :param sentence: 一句话
        :return: 好评的概率
        """
        possibility_good = 0
        for current_word in sentence:
            possibility_good += self.get_goodword_poss(current_word)
        # 最后加上该分类概率的对数
        possibility_good += math.log(len(self.data_good) / (len(self.data_bad) + len(self.data_good)))
        return possibility_good

    def get_bad_perp(self, sentence):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        perplexity = 0
        for word in sentence:
            perplexity += self.get_badword_poss(word)  # 对数相加
        # return math.exp(perplexity * (-1 / len(sentence)))  # 乘以-1/N，最后还原
        return perplexity * (-1 / len(sentence))  # 乘以-1/N

    def get_good_perp(self, sentence):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        perplexity = 0
        for word in sentence:
            perplexity += self.get_goodword_poss(word)  # 对数相加
        # return math.exp(perplexity * (-1 / len(sentence)))  # 乘以-1/N，最后还原
        return perplexity * (-1 / len(sentence))  # 乘以-1/N
