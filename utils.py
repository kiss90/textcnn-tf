import jieba
from collections import Counter
import numpy as np
import tensorflow as tf

STOP_WORDS_FILE = './data/chinese_stopwords.txt'

STOP_WORDS_ADDITION = [",", "'"]
SENTENCE_LIMIT_SIZE = 20
EMBEDDING_DIM = 300

def get_stop_words():
    with open(STOP_WORDS_FILE, 'r') as f:
        stopWords = f.readlines()
    return stopWords + STOP_WORDS_ADDITION

def split_words(sentence):
    ''' 输入句子，分词后，返回词语list
    '''
    return [x.strip() for x in jieba.cut(sentence) if x.strip()]

def data_preprocess(sentence):
    ''' 数据预处理
        1. 分词
        2. 去停用词
        input:  str
        return: str
    '''
    stopWords = get_stop_words()
    wordsList = split_words(sentence)
    return ' '.join([x for x in wordsList if x not in stopWords])

def load_data(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def build_vocabulary(text):
    ''' text 分好词的句子list
    '''
    c = Counter(' '.join(text).split())
    sorted(c.most_common(), key=lambda x: x[1])
    vocab = ["<unk>"]

    # 去除出现频次为1次的单词
    for w, f in c.most_common():
        if f > 1:
            vocab.append(w)
    return vocab

def word_to_token(vocab):
    return {word: token for token, word in enumerate(vocab)}

def convert_text_to_token(sentence, word_to_token_map, limit_size=SENTENCE_LIMIT_SIZE):
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    
    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower().split()]
    
    # Pad
    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]
    
    return np.array(tokens)
    
# inStr = "质量好,做工也不错,尺码标准"
# inStr = data_preprocess(inStr)
# print(inStr)

# pos_text = load_data(POS_TEXT_FILE)
# neg_text = load_data(NEG_TEXT_FILE)
# pos_seg = [data_preprocess(x) for x in pos_text]
# neg_seg = [data_preprocess(x) for x in neg_text]
# vocab = build_vocabulary(pos_seg + neg_seg)
# wordToken = word_to_token(vocab)

# token = convert_text_to_token(inStr, wordToken)
# print(token)

