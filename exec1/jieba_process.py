import read_in
import jieba
import jieba.posseg as pseg


#  预期的步骤：
#   1. 读取train下所有的文件
#   2. 将文件内容进行统计
#   3. 按照jieba词典的格式保存最后的结果
def make_dictionary(path=None):
    """
    创建用户词典
    :param path: 训练集的目录
    """
    # 1. 读取train下所有的文件
    all_text = read_in.read_all(path)
    if all_text is None:
        return

    # 2.将文件内容进行统计
    # 2.1 统计文件中每一个单词以及对应词性的频率（这样做是为了方面后面处理）
    frequency = {}
    for single_text in all_text:
        for word in single_text[0].split():
            if word not in frequency:
                frequency[word] = 1
            else:
                frequency[word] += 1
    # 2.2 将统计结果转化为jieba词典需要的格式
    dictionary = ""
    for word in frequency:
        if '/' not in word:
            continue

        result = word.split('/')
        if result[0] == "" or result[1] == "":
            continue

        dictionary += (result[0] + " " + str(frequency[word]) + " " + result[1] + "\n")

    # 3. 保存最后的结果
    file = open("normal_dictionary.txt", 'w')
    file.write(dictionary)
    file.close()


# 发现问题：即使使用用户词典，但是系统词典优先级更高。
#  jieba分词系统词典中词语词频的最大值是883634，因此将自定义词典中词语的词频设置为大于883634的数，则用户词典就可以绝对优先于系统词典
#  但实际这么做的时候，没看到明显的变化
# 参考：https://www.cnblogs.com/linkcxt/p/14696496.html
def load_user_dict(dictionary_file=None):
    """
    加载用户词典
    :param dictionary_file: 用户词典路径
    """
    if dictionary_file is None:
        print("dictionary is not exist!!!")
        return
    jieba.load_userdict(dictionary_file)


# 尝试通过这个函数解决上面的问题
# 新的问题：设置词库不报错，但进行分词的时候报错了，没有找到具体的原因。
#  之后根据网上的教程直接去替换了jieba目录下的词库，但是效果不好，介于不使用用户词库和使用用户词库之间，很奇怪
def set_dictionary(dictionary_file=None):
    """
    设置用户词库为主词库
    :param dictionary_file: 用户词库路径
    """
    if dictionary_file is None:
        print("dictionary is not exist!!!")
        return
    jieba.initialize()
    jieba.set_dictionary(dictionary_file)


def process(origin_list=None):
    """
    使用jieba进行分词和词性标注
    :param origin_list: 原文列表
    :return: 分词和词性标注的结果,list类型
    """
    result = []
    for single_list in origin_list:
        single_result = []
        seg_list = pseg.cut(single_list[0])
        for w in seg_list:
            single_result.append([w.word, w.flag])
        result.append(single_result)
    return result


# def people_tag_jieba(tag_1, tag_2):
#     if tag_1 == tag_2:
#         return True
#     if tag_1 == "bg" and tag_2 == "b":
#         return True
#     if tag_1 == "mg" and tag_2 == "m":
#         return True
#     if tag_1 == "nx" and tag_2 == "n":
#         return True
#     if tag_1 == "qg" and tag_2 == "q":
#         return True
#     if tag_1 == "rg" and tag_2 == "r":
#         return True
#     if tag_1 == "ug" and tag_2 == "u":
#         return True
#     if tag_1 == "yg" and tag_2 == "y":
#         return True
#     return False


#  问题：
#    对测试集分别进行分词和词性标注性能评估，评估指标至少包括准确率，召回率，F-测度；
#  想法：
#    关键思路 字符的个数是固定的。
#    那么从第一个词开始，分词正确，分词正确数+1，在分词正确的基础上词性标注正确，则词性标注正确数+1
#    如果分词不正确，就需要对两边的序号进行调整：
#       比如List A此时是第a_i个词，第a_j个字符，List B此时是第b_i个词，第b_j个字符。
#       如果a_j<b_j，那么a_i+1，否则b_j+1，然后重新进行匹配
#    本质上就是两个集合求交集的问题，但是这里的集合不是分词结果的集合，而是划分位置的集合。（位置是唯一的）
#
#  参考：https://blog.csdn.net/u012297539/article/details/111864251
def calculate_single_prf(truth, pred):
    """
    计算单个序列的A，B，分词TP_1，词性TP_2
    :param truth: 真值
    :param pred: 预测值
    :return: [A,B,TP_1,TP_2]
    """
    A = len(truth)
    B = len(pred)
    TP_1 = 0
    TP_2 = 0
    a_i = a_j = 0
    b_i = b_j = 0

    while a_i < A and b_i < B:
        # print(truth[a_i], pred[b_i])
        if truth[a_i][0] == pred[b_i][0]:  # 分词正确
            TP_1 += 1
            # if people_tag_jieba(truth[a_i][1], pred[b_i][1]):
            if truth[a_i][1] == pred[b_i][1]:  # 词性标注正确
                TP_2 += 1
            a_j += len(truth[a_i][0])
            b_j += len(pred[b_i][0])
            a_i += 1
            b_i += 1
        elif a_j < b_j:  # 如果pred序列在前
            a_j += len(truth[a_i][0])
            a_i += 1
        else:  # 如果truth在前
            b_j += len(pred[b_i][0])
            b_i += 1
    return A, B, TP_1, TP_2


def calculate_prf(truth=None, pred=None):
    """
     计算P、R、F1
     :param truth: 标准答案
     :param pred: 分词与词性标注结果
     :return: 分词的(P, R, F1), 词性标注的(P ,R, F1)
    """
    list_len = len(truth)
    if list_len != len(pred):
        print("ERROR!")
        exit(0)

    A = B = TP_1 = TP_2 = 0
    for i in range(list_len):
        a, b, tp_1, tp_2 = calculate_single_prf(truth[i], pred[i])
        A += a
        B += b
        TP_1 += tp_1
        TP_2 += tp_2
    P_1 = TP_1 / B
    R_1 = TP_1 / A
    F1_1 = 2 * P_1 * R_1 / (P_1 + R_1)

    P_2 = TP_2 / B
    R_2 = TP_2 / A
    F1_2 = 2 * P_2 * R_2 / (P_2 + R_2)
    return P_1, R_1, F1_1, P_2, R_2, F1_2

# if __name__ == "__main__":
#     make_dictionary("./people-2014/train")
#     load_user_dict("normal_dictionary.txt")
#     set_dictionary("normal_dictionary.txt")
#     for single_result in process(read_in.read_all_origin("./people-2014/test")):
#         print(single_result)
