# ===================================
# 基于朴素贝叶斯的文本分类起
# ===================================

# 1.导入必要的库
import re
import jieba
import math
import sqlite3
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 2.加载数据
def load_and_clean_data():
    print("正在读取数据")
    conn = sqlite3.connect('douban.db')
    query = """
    SELECT m.NAME AS movie_name, c.CONTENT AS comment_text, c.RATING AS rating FROM comment c
    LEFT JOIN movie m ON c.MOVIEID = m.ID
    WHERE m.NAME IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # 数据清晰和构建标签
    df = df[df['rating'] != 3].copy()
    df['label'] = df['rating'].apply(lambda x: 0 if x <= 2 else 1)  #1,2星差评为0，4，5星好评为1

    df = df.dropna(subset=['comment_text']).drop_duplicates(subset=['movie_name', 'comment_text'])


    # 采样，每部电影最多40条评论，防止热门电影主导
    # 1. 先把所有数据随机打乱
    df = df.sample(frac=1, random_state=42)
    # 2. 每部电影取前 40 条 (head(40))
    df = df.groupby('movie_name').head(40)
    # 3. 重置索引，此时绝对不会丢失 movie_name 列
    df = df.reset_index(drop=True)

    # 打印确认（你可以保留这行调试）
    print("采样后的列名:", df.columns.tolist())
    print(f"采样后的总数据量: {len(df)}条")

    # 文本清洗和分词
    def clean_and_cut(text):
        text = re.sub(r'<.*?>', '', text) # 去html
        text = re.sub(r'http\S+', '', text) # URL
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]',' ', text) #保留中英数字
        text = re.sub(r'\s+', ' ', text).strip()
        return jieba.lcut(text)

    df['words'] = df['comment_text'] .apply(clean_and_cut)

    # 按电影名划分，防止同一部电影的相似评论同时出现在训练集和测试集
    unique_movies = df['movie_name'].unique().tolist()
    train_movies, test_movies = train_test_split(unique_movies, test_size=0.2, random_state = 42)

    train_df = df[df['movie_name'].isin(train_movies)]
    test_df = df[df['movie_name'].isin(test_movies)]

    # 转换为朴素贝叶斯需要的格式：[(标签，[词列表])， ...]
    train_data = [(row['label'], row['words']) for _, row in train_df.iterrows()]
    test_data = [(row['label'], row['words']) for _, row in test_df.iterrows()]

    print(f"训练集：{len(train_data)} 条， 测试集：{len(test_data)} 条")
    return train_data, test_data


# 3.朴素贝叶斯类（NB）
class NaiveBayes:
    def __init__(self):
        self.class_priors = {} # 先验概率 logP(c)
        self.word_probs = {}   # 条件概率 logP(w|c)
        self.classes = [0, 1]
        self.vocab_size = 0
        self.class_word_counts = {} #统计每个类别的总词数

    def fit(self, train_data):
        # 1.统计每个类别的文档数
        class_doc_counts = {c: 0 for c in self.classes}
        class_word_freq = {c: {} for c in self.classes}
        class_total_words = {c: 0 for c in self.classes}
        vocab = set()

        for label, words in train_data:
            class_doc_counts[label] += 1
            for word in words:
                vocab.add(word)
                class_word_freq[label][word] = class_word_freq[label].get(word, 0)+1
                class_total_words[label] += 1

        self.vocab_size = len(vocab)
        total_docs = len(train_data)

        # 2.计算先验概率 P(c)并取对数
        for c in self.classes:
            self.class_priors[c] = math.log(class_doc_counts[c] / total_docs)
            self.class_word_counts[c] = class_total_words[c]

        # 3.计算条件概率 P(w|c)并取对数，引入拉普拉斯平滑
        for c in self.classes:
            self.word_probs[c] = {}
            for word in vocab:
                count = class_word_freq[c].get(word,0)

                prob = (count + 1)/ (class_total_words[c] + self.vocab_size)
                self.word_probs[c][word] = math.log(prob)

    def predict(self, words):
        # 计算每个类别的后验概率得分（对数相加）
        scores = {}
        for c in self.classes:
            score = self.class_priors[c]
            for word in words:
                # 如果测试集出现了训练集里没有的词，直接跳过（相当于概率为1）
                if word in self.word_probs[c]:
                    score += self.word_probs[c][word]
            scores[c] = score
        # 返回得分最高的类别
        return max(scores, key=scores.get)
    

# 4.实例化模型
nb_model = NaiveBayes()

# 5.模型训练
train_data, test_data = load_and_clean_data()
print('开始训练NB模型...')
nb_model.fit(train_data)
print('训练完成')

# 6.模型测试
print('开始预测测试集...')
y_true = []
y_pred = []
for label, words in test_data:
    y_true.append(label)
    y_pred.append(nb_model.predict(words))


# 7.各类评价指标
print("=== 朴素贝叶斯模型评估 ===")
print(f"准确率 (Accuracy) : {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, target_names=['差评(0)','好评(1)']))
