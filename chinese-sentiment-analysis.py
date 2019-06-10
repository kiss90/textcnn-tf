
# coding: utf-8

# # 中文情感分析
# 
# - 数据分析
# - 预处理
# - 词向量
# - TextCNN

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from collections import Counter
import tqdm
import jieba
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tf.__version__


# ## 数据分析

# In[3]:


os.listdir('./data/')


# In[4]:


with open('./data/pos.txt', 'r') as f:
    pos_text = f.readlines()
with open('./data/neg.txt', 'r') as f:
    neg_text = f.readlines()


# In[5]:


pos_text[:5]


# In[6]:


neg_text[:5]


# In[7]:


pos_segs = [" ".join(jieba.cut(x)) for x in pos_text]
neg_segs = [" ".join(jieba.cut(x)) for x in neg_text]


# In[8]:


pos_segs[:5]


# In[9]:


neg_segs[:5]


# ### 数据分布

# In[10]:


print('-' * 20 + ' POSITIVE TEXT ' + '-' * 20)
print("Total number: {}".format(len(pos_segs)))
print("Average length: {}".format(np.mean([len(sentence.split()) for sentence in pos_segs])))
print("Max length: {}".format(np.max([len(sentence.split()) for sentence in pos_segs])))
print("Min length: {}".format(np.min([len(sentence.split()) for sentence in pos_segs])))
pos_text_seg = " ".join(pos_segs)
c = Counter(pos_text_seg.split()).most_common(100)
print("Most common words : \n{} \n".format(c))

print("-" * 20 + " NEGATIVE TEXT " + "-" * 20)
print("Total number: {}".format(len(neg_segs)))
print("Average length: {}".format(np.mean([len(sentence.split()) for sentence in neg_segs])))
print("Max length: {}".format(np.max([len(sentence.split()) for sentence in neg_segs])))
print("Min length: {}".format(np.min([len(sentence.split()) for sentence in neg_segs])))
neg_text_seg = " ".join(neg_segs)
c = Counter(neg_text_seg.split()).most_common(100)
print("Most common words : \n{} \n".format(c))


# 从中可以看出高频词中有很多是无意义的停用词，因此我们进一步对数据进行去停用词操作。

# ## 预处理

# In[11]:


stopword_path = '/Users/huih/pinduoduo/stopwords/chinese_stopwords.txt'


# 创建停用词列表
def get_stop_words():
    stopwords = [line.strip() for line in open(stopword_path, encoding='UTF-8').readlines()]
    stopwords += [",", "'"]
    return stopwords
stopwords = get_stop_words()

pos_segs_without_stop = []
for sentence in pos_segs:
    sentWords = [x.strip() for x in sentence.split(' ') if x.strip() and x.strip() not in stopwords]
    pos_segs_without_stop.append(' '.join(sentWords))

pos_segs_without_stop[:10]


# In[12]:


neg_segs_without_stop = []
for sentence in neg_segs:
    sentWords = [x.strip() for x in sentence.split(' ') if x.strip() and x.strip() not in stopwords]
    neg_segs_without_stop.append(' '.join(sentWords))

neg_segs_without_stop[:10]


# In[13]:


# 对去停用词后对数据进行统计分析
pos_segs = pos_segs_without_stop
neg_segs = neg_segs_without_stop

print('-' * 20 + ' POSITIVE TEXT ' + '-' * 20)
print("Total number: {}".format(len(pos_segs)))
print("Average length: {}".format(np.mean([len(sentence.split()) for sentence in pos_segs])))
print("Max length: {}".format(np.max([len(sentence.split()) for sentence in pos_segs])))
print("Min length: {}".format(np.min([len(sentence.split()) for sentence in pos_segs])))
pos_text_seg = " ".join(pos_segs)
c = Counter(pos_text_seg.split()).most_common(100)
print("Most common words : \n{} \n".format(c))

print("-" * 20 + " NEGATIVE TEXT " + "-" * 20)
print("Total number: {}".format(len(neg_segs)))
print("Average length: {}".format(np.mean([len(sentence.split()) for sentence in neg_segs])))
print("Max length: {}".format(np.max([len(sentence.split()) for sentence in neg_segs])))
print("Min length: {}".format(np.min([len(sentence.split()) for sentence in neg_segs])))
neg_text_seg = " ".join(neg_segs)
c = Counter(neg_text_seg.split()).most_common(100)
print("Most common words : \n{} \n".format(c))


# 从上可以看出，去停用词后句子的平均长度由32和34变成了17，最大长度由804和1280变成了427和607(说明差评的平均长度要长于好评)。
# 最小长度有2和1变成了0，说明数据中有些无效样本可以进一步删除。

# In[14]:


while '' in pos_segs:
    pos_segs.remove('')
while '' in neg_segs:
    neg_segs.remove('')
print("Min length for pos text: {}".format(np.min([len(sentence.split()) for sentence in pos_segs])))
print("Min length for neg text: {}".format(np.min([len(sentence.split()) for sentence in neg_segs])))


# In[15]:


print('-' * 20 + ' POSITIVE TEXT ' + '-' * 20)
print("Total number: {}".format(len(pos_segs)))
print("Average length: {}".format(np.mean([len(sentence.split()) for sentence in pos_segs])))
print("Max length: {}".format(np.max([len(sentence.split()) for sentence in pos_segs])))
print("Min length: {}".format(np.min([len(sentence.split()) for sentence in pos_segs])))
pos_text_seg = " ".join(pos_segs)
c = Counter(pos_text_seg.split()).most_common(100)
print("Most common words : \n{} \n".format(c))

print("-" * 20 + " NEGATIVE TEXT " + "-" * 20)
print("Total number: {}".format(len(neg_segs)))
print("Average length: {}".format(np.mean([len(sentence.split()) for sentence in neg_segs])))
print("Max length: {}".format(np.max([len(sentence.split()) for sentence in neg_segs])))
print("Min length: {}".format(np.min([len(sentence.split()) for sentence in neg_segs])))
neg_text_seg = " ".join(neg_segs)
c = Counter(neg_text_seg.split()).most_common(100)
print("Most common words : \n{} \n".format(c))


# 正样本数量从25000变成了24967，负样本数量从25000变成了24964.

# 样本的平均长度为17，我们取20作为句子长度。

# In[16]:


SENTENCE_LIMIT_SIZE = 20


# ## 词向量
# 

# ### 构造词典
# 由于语料中还含有很多低频词，在构建词典时需要将这些词过滤掉，这样不仅可以加快模型的执行效率，还能减少特殊词带来的噪声干扰。

# In[17]:


c = Counter(' '.join(pos_segs + neg_segs).split())
sorted(c.most_common(), key=lambda x: x[1])[:20]


# In[18]:


# 初始化两个token：pad和unk
vocab = ["<pad>", "<unk>"]

# 去除出现频次为1次的单词
for w, f in c.most_common():
    if f > 1:
        vocab.append(w)
print("Total size of vocabulary is: {}".format(len(vocab)))


# ### 构造映射
# 构造将单词转换为编码和将编码转为单词的映射

# In[19]:


word_to_token = {word: token for token, word in enumerate(vocab)}
token_to_word = {token: word for word, token in word_to_token.items()}


# In[20]:


# 文本转token
def convert_text_to_token(sentence, word_to_token_map=word_to_token, limit_size=SENTENCE_LIMIT_SIZE):
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]
    
    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower().split()]
    
    # Pad
    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]
    
    return tokens


# In[21]:


pos_tokens = []

for sentence in tqdm.tqdm(pos_segs):
    tokens = convert_text_to_token(sentence)
    pos_tokens.append(tokens)


# In[22]:


pos_tokens[:3]


# In[23]:


neg_tokens = []

for sentence in tqdm.tqdm(neg_segs):
    tokens = convert_text_to_token(sentence)
    neg_tokens.append(tokens)
    
neg_tokens[:3]


# In[24]:


# 合并语料
pos_tokens = np.array(pos_tokens)
neg_tokens = np.array(neg_tokens)
total_tokens = np.concatenate((pos_tokens, neg_tokens), axis=0)
print("Shape of all tokens: ({}, {})".format(*total_tokens.shape))


# In[25]:


# 合并类标
pos_targets = np.ones((pos_tokens.shape[0]))
neg_targets = np.zeros((neg_tokens.shape[0]))
total_targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1)
print("Shape of all labels: ({}, {})".format(*total_tokens.shape))


# In[26]:


# 加载预训练word2vec词向量
with open("/Users/huih/Downloads/sgns.weibo.bigram-char", 'r') as f:
    words = set()
    word_to_vec = {}
    for line in f:
        line = line.strip().split()
        # 当前单词
        curr_word = line[0]
        words.add(curr_word)
        # 当前词向量
        word_to_vec[curr_word] = np.array(line[1:], dtype=np.float32)


# In[27]:


len(words)
print("The number of words which have pretrained-vectors in vocab is: {}".format(len(set(vocab)&set(words))))
print()
print("The number of words which do not have pretrained-vectors in vocab is : {}".format(len(set(vocab))-
                                                                                         len(set(vocab)&set(words))))


# In[28]:


len(vocab)


# In[29]:


# 在预训练词向量中的词
wordSet = set(vocab) & set(words)
print("Number of words in embedding : {}".format(len(wordSet)))
print("Number of words not in embedding : {}".format(len(set(vocab)) - len(wordSet)))


# ### 构造词向量矩阵
# 

# In[30]:


VOCAB_SIZE = len(vocab)
EMBEDDING_SIZE = 300

static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])

for word, token in tqdm.tqdm(word_to_token.items()):
    # 用词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token, :] = word_vector

# 重置PAD为0向量
pad_id = word_to_token["<pad>"]
static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)


# In[31]:


static_embeddings = static_embeddings.astype(np.float32)


# ### 数据集划分

# In[32]:


x_train, x_test, y_train, y_test = train_test_split(total_tokens, total_targets, test_size=0.2, random_state=42, shuffle=True)


# In[33]:


# 将训练集进一步划分为训练和验证集，将测试集留作模型测试使用
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)


# In[34]:


print("train:{}, val:{}, test:{}".format(len(y_train), len(y_val), len(y_test)))


# In[35]:


BATCH_SIZE = 256
def get_batch(x, y, batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")
    # shuffle
    if shuffle:
        shuffled_index = np.random.permutation(range(x.shape[0]))

        x = x[shuffled_index]
        y = y[shuffled_index]
    
    # 统计共几个完整的batch
    n_batches = int(x.shape[0] / batch_size)
    
    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]
    
        yield x_batch, y_batch


# ## CNN

# In[36]:


# 清空图
tf.reset_default_graph()

filters_size = [2, 3, 4, 5]
num_filters = 100
# 超参数
BATCH_SIZE = 256
EPOCHES = 50
LEARNING_RATE = 0.003
L2_LAMBDA = 10
KEEP_PROB = 0.5


# ### 构建模型

# In[37]:


with tf.name_scope("cnn"):
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, 20), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")
    
    # embeddings
    with tf.name_scope("embeddings"):
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
        # 添加channel维度
        embed_expanded = tf.expand_dims(embed, -1, name="embed_expand")
    
    # 用来存储max-pooling的结果
    pooled_outputs = []

    # 迭代多个filter
    for i, filter_size in enumerate(filters_size):
        with tf.name_scope("conv_maxpool_%s" % filter_size):
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, mean=0.0, stddev=0.1), name="W")
            b = tf.Variable(tf.zeros(num_filters), name="b")

            conv = tf.nn.conv2d(input=embed_expanded, 
                                 filter=W, 
                                 strides=[1, 1, 1, 1], 
                                 padding="VALID",
                                 name="conv")

            # 激活
            a = tf.nn.relu(tf.nn.bias_add(conv, b), name="activations")
            # 池化
            max_pooling = tf.nn.max_pool(value=a, 
                                    ksize=[1, SENTENCE_LIMIT_SIZE - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="max_pooling")
            pooled_outputs.append(max_pooling)
    
    # 统计所有的filter
    total_filters = num_filters * len(filters_size)
    total_pool = tf.concat(pooled_outputs, 3)
    flattend_pool = tf.reshape(total_pool, (-1, total_filters))
    
    # dropout
    with tf.name_scope("dropout"):
        dropout = tf.nn.dropout(flattend_pool, KEEP_PROB)
    
    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=(total_filters, 1), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(1), name="b")
        
        logits = tf.add(tf.matmul(dropout, W), b)
        predictions = tf.nn.sigmoid(logits, name="predictions")
    
    # loss
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        loss = loss + L2_LAMBDA * tf.nn.l2_loss(W)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    # evaluation
    with tf.name_scope("evaluation"):
        correct_preds = tf.equal(tf.cast(tf.greater(predictions, 0.5), tf.float32), targets)
        accuracy = tf.reduce_sum(tf.reduce_sum(tf.cast(correct_preds, tf.float32), axis=1))


# ### 训练模型

# In[39]:


cnn_train_accuracy = []
cnn_test_accuracy = []
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter("./graphs/cnn", tf.get_default_graph())
    n_batches = int(x_train.shape[0] / BATCH_SIZE)
    
    for epoch in range(EPOCHES):
        total_loss = 0
        for x_batch, y_batch in get_batch(x_train, y_train):
            _, l = sess.run([optimizer, loss],
                            feed_dict={inputs: x_batch, 
                                       targets: y_batch})
            total_loss += l
        
        train_corrects = sess.run(accuracy, feed_dict={inputs: x_train, targets: y_train})
        train_acc = train_corrects / x_train.shape[0]
        cnn_train_accuracy.append(train_acc)
        
        val_corrects = sess.run(accuracy, feed_dict={inputs: x_val, targets: y_val})
        val_acc = val_corrects / x_val.shape[0]
        cnn_test_accuracy.append(val_acc)
        
        print("Training epoch: {}, Training loss: {:.4f}, Train accuracy: {:.4f}, Val accuracy: {:.4f}".format(epoch + 1, 
                                                                                                                total_loss / n_batches,
                                                                                                                train_acc,
                                                                                                                val_acc))
    
    saver.save(sess, "checkpoints/cnn")
    writer.close()


# In[41]:


plt.plot(cnn_train_accuracy)
plt.plot(cnn_test_accuracy)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of CNN model")
plt.legend(["train", "val"])


# In[42]:


with tf.Session() as sess:
    saver.restore(sess, "checkpoints/cnn")
    
    total_correct = sess.run(accuracy,
                             feed_dict={inputs: x_test, targets: y_test})

    print("The LSTM model accuracy on test set: {:.2f}%".format(100 * total_correct / x_test.shape[0]))


# In[ ]:




