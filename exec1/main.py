import read_in
import jieba_process

# 设置数据集路径
dat_path = "./people-2014/test"
# 获取真值
truth = read_in.read_all_dataset(dat_path)

# 在未使用用户词典的情况下，用jieba进行预测
predict = jieba_process.process(read_in.read_all_origin(dat_path))
# 计算分词的正确率、召回率和F1，词性标注的正确率、召回率和F1
P_1, R_1, F1_1, P_2, R_2, F1_2 = jieba_process.calculate_prf(truth, predict)
print("在未使用用户词典的情况下")
print("分词的精准率：", P_1)
print("分词的召回率：", R_1)
print("分词的F1值", F1_1)
print("词性标注的精准率：", P_2)
print("词性标注的召回率：", R_2)
print("词性标注的F1值", F1_2)
print()
# 在使用用户词典的情况下，用jieba进行预测
jieba_process.set_dictionary("normal_dictionary.txt")
jieba_process.load_user_dict("normal_dictionary.txt")
predict = jieba_process.process(read_in.read_all_origin(dat_path))
# 计算分词的正确率、召回率和F1，词性标注的正确率、召回率和F1
P_1, R_1, F1_1, P_2, R_2, F1_2 = jieba_process.calculate_prf(truth, predict)
print("在使用用户词典的情况下")
print("分词的精准率：", P_1)
print("分词的召回率：", R_1)
print("分词的F1值", F1_1)
print("词性标注的精准率：", P_2)
print("词性标注的召回率：", R_2)
print("词性标注的F1值", F1_2)
print()
