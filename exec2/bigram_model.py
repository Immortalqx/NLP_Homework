import math


class BigramModel:
    def __init__(self):
        self.data_ = []  # 单词列表
        self.word_count_ = {}  # 单词词频统计
        self.unique_words_ = None  # 非重复词

        self.data_tuple_ = []  # 单词列表
        self.word_count_tuple_ = {}  # 单词词频统计
        self.unique_words_tuple_ = None  # 非重复词

    def train_model(self, train_data):
        """
        训练一元语法模型
        :param train_data:训练集
        """
        for line_data in train_data:
            # 开头特殊处理一下
            self.data_.append("<BOS>")
            self.word_count_.setdefault("<BOS>", 0)
            self.word_count_["<BOS>"] += 1
            for i in range(len(line_data["text"]) - 1):
                current_word = line_data["text"][i + 1]
                forward_word = line_data["text"][i]
                # 记录单个单词
                self.data_.append(current_word)
                self.word_count_.setdefault(current_word, 0)
                self.word_count_[current_word] += 1
                # 记录两个单词在一起
                self.data_tuple_.append((current_word, forward_word))
                self.word_count_tuple_.setdefault((current_word, forward_word), 0)
                self.word_count_tuple_[(current_word, forward_word)] += 1
        # 统计非重复词
        self.unique_words_ = list(set(self.data_))
        self.unique_words_tuple_ = list(set(self.data_tuple_))

    def get_poss(self, current_word, forward_word):
        """
        获取某一个单词的概率
        :param current_word: 当前单词
        :param forward_word: 前一个单词
        :return: 该单词的概率
        """
        if (current_word, forward_word) in self.word_count_tuple_:
            return (self.word_count_tuple_[(current_word, forward_word)] + 1) / (
                    self.word_count_[forward_word] + len(self.unique_words_))
        elif forward_word in self.word_count_:
            return 1 / (self.word_count_[forward_word] + len(self.unique_words_))
        else:
            return 1 / (len(self.unique_words_))

    def get_perp(self, sentence):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        perplexity = 0
        for i in range(len(sentence) - 1):
            current_word = sentence[i + 1]
            forward_word = sentence[i]
            perplexity += math.log(self.get_poss(current_word, forward_word))  # 取对数相加
        # return math.exp(perplexity * (-1 / len(sentence)))  # 乘以-1/N，最后还原
        return perplexity * (-1 / len(sentence))  # 乘以-1/N


# 之前写了个散装的语法模型，效果比上面封装的好，这里封装了实现一下
class MixBigramModel:
    def __init__(self):
        # ===============差评===============
        self.data_bad = []  # 单词列表
        self.word_count_bad = {}  # 单词词频统计
        self.unique_words_bad = None  # 非重复词

        self.data_tuple_bad = []  # 单词列表
        self.word_count_tuple_bad = {}  # 单词词频统计
        self.unique_words_tuple_bad = None  # 非重复词
        # ===============好评===============
        self.data_good = []  # 单词列表
        self.word_count_good = {}  # 单词词频统计
        self.unique_words_good = None  # 非重复词

        self.data_tuple_good = []  # 单词列表
        self.word_count_tuple_good = {}  # 单词词频统计
        self.unique_words_tuple_good = None  # 非重复词

    def train_bad_model(self, train_data):
        """
        训练一元语法模型
        :param train_data:训练集
        """
        for line_data in train_data:
            # 开头特殊处理一下
            self.data_bad.append("<BOS>")
            self.word_count_bad.setdefault("<BOS>", 0)
            self.word_count_bad["<BOS>"] += 1
            for i in range(len(line_data["text"]) - 1):
                current_word = line_data["text"][i + 1]
                forward_word = line_data["text"][i]
                # 记录单个单词
                self.data_bad.append(current_word)
                self.word_count_bad.setdefault(current_word, 0)
                self.word_count_bad[current_word] += 1
                # 记录两个单词在一起
                self.data_tuple_bad.append((current_word, forward_word))
                self.word_count_tuple_bad.setdefault((current_word, forward_word), 0)
                self.word_count_tuple_bad[(current_word, forward_word)] += 1
        # 统计非重复词
        self.unique_words_bad = list(set(self.data_bad))
        self.unique_words_tuple_bad = list(set(self.data_tuple_bad))

    def train_good_model(self, train_data):
        """
        训练一元语法模型
        :param train_data:训练集
        """
        for line_data in train_data:
            # 开头特殊处理一下
            self.data_good.append("<BOS>")
            self.word_count_good.setdefault("<BOS>", 0)
            self.word_count_good["<BOS>"] += 1
            for i in range(len(line_data["text"]) - 1):
                current_word = line_data["text"][i + 1]
                forward_word = line_data["text"][i]
                # 记录单个单词
                self.data_good.append(current_word)
                self.word_count_good.setdefault(current_word, 0)
                self.word_count_good[current_word] += 1
                # 记录两个单词在一起
                self.data_tuple_good.append((current_word, forward_word))
                self.word_count_tuple_good.setdefault((current_word, forward_word), 0)
                self.word_count_tuple_good[(current_word, forward_word)] += 1
        # 统计非重复词
        self.unique_words_good = list(set(self.data_good))
        self.unique_words_tuple_good = list(set(self.data_tuple_good))

    def get_badword_poss(self, current_word, forward_word):
        """
        获取某一个单词的概率
        :param current_word: 当前单词
        :param forward_word: 前一个单词
        :return: 该单词的概率
        """
        if (current_word, forward_word) in self.word_count_tuple_bad:
            return (self.word_count_tuple_bad[(current_word, forward_word)] + 1) / (
                    self.word_count_bad[forward_word] + (len(self.unique_words_bad) + len(self.unique_words_good)))
        elif forward_word in self.word_count_bad:
            return 1 / (self.word_count_bad[forward_word] + (len(self.unique_words_bad) + len(self.unique_words_good)))
        else:
            return 1 / (len(self.unique_words_bad) + len(self.unique_words_good))

    def get_goodword_poss(self, current_word, forward_word):
        """
        获取某一个单词的概率
        :param current_word: 当前单词
        :param forward_word: 前一个单词
        :return: 该单词的概率
        """
        if (current_word, forward_word) in self.word_count_tuple_good:
            return (self.word_count_tuple_good[(current_word, forward_word)] + 1) / (
                    self.word_count_good[forward_word] + (len(self.unique_words_bad) + len(self.unique_words_good)))
        elif forward_word in self.word_count_good:
            return 1 / (self.word_count_good[forward_word] + (len(self.unique_words_bad) + len(self.unique_words_good)))
        else:
            return 1 / (len(self.unique_words_bad) + len(self.unique_words_good))

    def get_bad_poss(self, sentence):
        """
        获取一句话是差评的概率
        :param sentence: 一句话
        :return: 差评的概率
        """
        possibility_bad = 0
        for i in range(len(sentence) - 1):
            current_word = sentence[i + 1]
            forward_word = sentence[i]
            possibility_bad += math.log(self.get_badword_poss(current_word, forward_word))
        # 最后加上该分类概率的对数
        possibility_bad += math.log(len(self.data_bad) / (len(self.data_good) + len(self.data_bad)))
        return possibility_bad

    def get_good_poss(self, sentence):
        """
        获取一句话是好评的概率
        :param sentence: 一句话
        :return: 好评的概率
        """
        possibility_good = 0
        for i in range(len(sentence) - 1):
            current_word = sentence[i + 1]
            forward_word = sentence[i]
            possibility_good += math.log(self.get_goodword_poss(current_word, forward_word))
        # 最后加上该分类概率的对数
        possibility_good += math.log(len(self.data_good) / (len(self.data_good) + len(self.data_bad)))
        return possibility_good

    def get_bad_perp(self, sentence):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        perplexity = 0
        for i in range(len(sentence) - 1):
            current_word = sentence[i + 1]
            forward_word = sentence[i]
            perplexity += math.log(self.get_badword_poss(current_word, forward_word))  # 取对数相加
        # return math.exp(perplexity * (-1 / len(sentence)))  # 乘以-1/N，最后还原
        return perplexity * (-1 / len(sentence))  # 乘以-1/N

    def get_good_perp(self, sentence):
        """
        计算语言模型的困惑度
        :param sentence: 句子
        :return: 语言模型对于该句子的困惑度
        """
        perplexity = 0
        for i in range(len(sentence) - 1):
            current_word = sentence[i + 1]
            forward_word = sentence[i]
            perplexity += math.log(self.get_goodword_poss(current_word, forward_word))  # 取对数相加
        # return math.exp(perplexity * (-1 / len(sentence)))  # 乘以-1/N，最后还原
        return perplexity * (-1 / len(sentence))  # 乘以-1/N
