from read_process import read_dataset
from unigram_model import UnigramModel, MixUnigramModel
from bigram_model import BigramModel, MixBigramModel
import math


def test_unigram_model():
    bad_data, good_data = read_dataset("./data/train.tsv", train=True, punctuation=True)

    bad_model = UnigramModel()
    good_model = UnigramModel()

    bad_model.train_model(bad_data)
    good_model.train_model(good_data)

    test_data = read_dataset("./data/test.tsv", train=False, punctuation=True)

    correct_num = 0
    tp = fp = fn = tn = 0
    good_perplexity = 0
    bad_perplexity = 0

    for line_data in test_data:
        possibility_bad = 0
        possibility_good = 0
        for current_word in line_data["text"]:
            possibility_bad += bad_model.get_poss(current_word)
            possibility_good += good_model.get_poss(current_word)
        # TODO 为什么这里要加上该分类概率的对数？？？[把贝叶斯公式再看一遍。。。]
        # 最后加上该分类概率的对数
        possibility_bad += math.log(len(bad_model.data_) / (len(bad_model.data_) + len(good_model.data_)))
        possibility_good += math.log(len(good_model.data_) / (len(bad_model.data_) + len(good_model.data_)))

        # print("===============================")
        # print("好评概率：", possibility_good)
        # print("好评困惑度：", good_model.get_perp(line_data["text"]))
        # print("差评概率：", possibility_bad)
        # print("差评困惑度：", bad_model.get_perp(line_data["text"]))

        good_perplexity += good_model.get_perp(line_data["text"])
        bad_perplexity += bad_model.get_perp(line_data["text"])

        predict = "0"
        if possibility_good > possibility_bad:
            predict = "1"

        if line_data["label"] == "1":
            if predict == "1":
                tp += 1
            else:
                fn += 1
        else:
            if predict == "1":
                fp += 1
            else:
                tn += 1
        # 统计正确的个数
        if predict == line_data["label"]:
            correct_num += 1

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    print("===============一元语言模型测试（好评模型与差评模型）===============")
    print("好评平均困惑度：", math.exp(good_perplexity / len(test_data)))
    print("差评平均困惑度：", math.exp(bad_perplexity / len(test_data)))
    print("精确率：", P)
    print("召回率：", R)
    print("F1度量：", F)
    print()
    print("正确率：", correct_num / len(test_data))


def test_bigram_model():
    bad_data, good_data = read_dataset("./data/train.tsv", train=True, punctuation=True)

    bad_model = BigramModel()
    good_model = BigramModel()

    bad_model.train_model(bad_data)
    good_model.train_model(good_data)

    test_data = read_dataset("./data/test.tsv", train=False, punctuation=True)

    correct_num = 0
    tp = fp = fn = tn = 0
    good_perplexity = 0
    bad_perplexity = 0

    for line_data in test_data:
        possibility_bad = 0
        possibility_good = 0
        for i in range(len(line_data["text"]) - 1):
            current_word = line_data["text"][i + 1]
            forward_word = line_data["text"][i]
            possibility_bad += math.log(bad_model.get_poss(current_word, forward_word))
            possibility_good += math.log(good_model.get_poss(current_word, forward_word))
        # TODO 为什么这里要加上该分类概率的对数？？？[把贝叶斯公式再看一遍。。。]
        # 最后加上该分类概率的对数
        possibility_bad += math.log(len(bad_model.data_) / (len(bad_model.data_) + len(good_model.data_)))
        possibility_good += math.log(len(good_model.data_) / (len(bad_model.data_) + len(good_model.data_)))

        # print("===============================")
        # print("好评概率：", possibility_good)
        # print("好评困惑度：", good_model.get_perp(line_data["text"]))
        # print("差评概率：", possibility_bad)
        # print("差评困惑度：", bad_model.get_perp(line_data["text"]))

        good_perplexity += good_model.get_perp(line_data["text"])
        bad_perplexity += bad_model.get_perp(line_data["text"])

        predict = "0"
        if possibility_good > possibility_bad:
            predict = "1"

        if line_data["label"] == "1":
            if predict == "1":
                tp += 1
            else:
                fn += 1
        else:
            if predict == "1":
                fp += 1
            else:
                tn += 1
        # 统计正确的个数
        if predict == line_data["label"]:
            correct_num += 1

    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    print("===============二元语言模型测试（好评模型与差评模型）===============")
    print("好评平均困惑度：", math.exp(good_perplexity / len(test_data)))
    print("差评平均困惑度：", math.exp(bad_perplexity / len(test_data)))
    print("精确率：", P)
    print("召回率：", R)
    print("F1度量：", F)
    print()
    print("正确率：", correct_num / len(test_data))


def test_mix_model(model):
    bad_data, good_data = read_dataset("./data/train.tsv", train=True, punctuation=True)

    model.train_bad_model(bad_data)
    model.train_good_model(good_data)

    test_data = read_dataset("./data/test.tsv", train=False, punctuation=True)

    correct_num = 0
    tp = fp = fn = tn = 0
    good_perplexity = 0
    bad_perplexity = 0

    for line_data in test_data:
        # 计算差评和好评的对数概率
        possibility_bad = model.get_bad_poss(line_data["text"])
        possibility_good = model.get_good_poss(line_data["text"])
        # 计算差评和好评的困惑度
        bad_perplexity += model.get_bad_perp(line_data["text"])
        good_perplexity += model.get_good_perp(line_data["text"])
        # 比较对数概率大小得到预测结果
        predict = "0"
        if possibility_good > possibility_bad:
            predict = "1"
        # 计算混淆矩阵（非常朴素的方式）
        if line_data["label"] == "1":
            if predict == "1":
                tp += 1
            else:
                fn += 1
        else:
            if predict == "1":
                fp += 1
            else:
                tn += 1
        # 统计正确的个数
        if predict == line_data["label"]:
            correct_num += 1
    # 计算各种性能度量
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    # 输出最终结果
    print("好评平均困惑度：", math.exp(good_perplexity / len(test_data)))
    print("差评平均困惑度：", math.exp(bad_perplexity / len(test_data)))
    print("精确率：", P)
    print("召回率：", R)
    print("F1度量：", F)
    print()
    print("正确率：", correct_num / len(test_data))


if __name__ == "__main__":
    # # 对一元语法模型进行测试
    # test_unigram_model()
    # # 对二元语法模型进行测试
    # test_bigram_model()
    # 测试一元和二元的混合模型
    # print("===============一元语言模型测试===============")
    # test_mix_model(MixUnigramModel())
    # print("===============二元语言模型测试===============")
    # test_mix_model(MixBigramModel())
    import jieba

    result=list(jieba.cut("早上好呀今天要做什么呢我要学习我不想学习啊这是什么东西啊手机真好玩待会儿吃什么不行什么都没做要做事到时间吃午饭了吃什么不行不能吃胖死算了要健康吃完一定好好学埃买杯咖啡吧啊美好充实的下午开始了手机真好玩不准玩要干活儿埃这有张便利贴我写点儿东西不算偷懒想看小说了不想学工我的人生就这么被毁了真羡慕搞文学的人啊不想不想学习画图不会画这垃圾软件我怎么还没死啊要不写点儿文艺的文案发不不不不可以矫情快干活干活我真喜欢学习上学给人一种锒铛入狱的感觉埃押韵了想一下其实人活着就是为了死亡说的好像也没错啊啊啊再不学我就要死了好想去旅游啊好想回家啊好想死啊咻不是鱼露是鲈鱼哦怎么回事怎么八点了这天还不黑呢太影响我夜黑风高去觅食了不对晚上进食胖十斤不能吃想吃水果要不我去竞标一下大超水果摊吧ddl快到了我怎么还不学没有手机我可怎么活啊我不理解我大为震惊吼让我一套女拳打醒ta们算了跟我没关系我还是回去敲木鱼吧你又走神到哪里去了学学学学太晚了我应该休息了没有功劳也有苦劳嗯剩下的事留给明天吧明天一定能好好学这么早睡觉我不亏了人生怎么能浪费在睡觉上面霍这人怎么越听越像郭德纲不行不行要早睡呜呜呜这首歌太好听了原唱可以直接退休了早睡早睡大家晚安我要当一个高冷健康的早睡人mud怎么睡不着不准失眠失眠失眠"))
    print(result)