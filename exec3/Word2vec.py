from gensim.models import word2vec

# 训练模型
sentences = word2vec.LineSentence("data/input.txt")
model = word2vec.Word2Vec(sentences, window=2, min_count=1, sg=1)

# 设定10个词
word_list = ["国家", "主权", "普京", "因特网", "国务卿",
             "总统", "人民", "联合国", "贡献", "主席"]

# 查找Top5相似词
for word in word_list:
    print("\"" + word + "\"" + " 的Top5相似词为：")

    req_count = 6
    for key in model.wv.similar_by_word(word, topn=100):
        # 计数
        req_count -= 1
        if req_count == 0:
            break
        # 忽略长度为1的词
        if len(key[0]) <= 1:
            continue

        print("\t", end="")
        print(key[0], key[1])
    print()
