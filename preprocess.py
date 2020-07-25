import json
import re
import os
import pickle
import random
from keras.preprocessing.text import Tokenizer

'''
we can get the following index file:
1.tokenizer.pkl and word_index.pkl :encode file
2.train_net_addraw.pkl:train数据集添加了raw数据
3.train_net_data.pkl：train数据集没有添加raw数据,用于消歧
4.subject_data.pkl：根据kb_data构建的“kb_id->data”索引字典
5.stockname_data.pkl：根据kb_data构建的"'实体'->data”索引字典
'''

train_net_addraw_is_exist = os.path.exists('./data_deal/train_net_addraw.pkl')  # 训练集，增加了raw数据
train_net_is_exist = os.path.exists('./data_deal/train_net_data.pkl')  # 训练集

count = 0
if train_net_addraw_is_exist:
    with open('./data_deal/train_net_addraw.pkl', 'rb') as f:
        train_net_addraw = pickle.load(f)
if train_net_is_exist:
    with open('./data_deal/train_net_data.pkl', 'rb') as f:  # w
        train_net_data = pickle.load(f)

if not train_net_is_exist:
    f = open('./data/train.json', 'r', encoding='utf-8')
    data = json.load(f)
    # if not train_net_addraw_is_exist:
    #     train_net_addraw = []
    train_net_data = []

    for idx, data_one in enumerate(data):
        text_id = data_one['text_id']
        text = data_one['text'].lower()
        lab_result = data_one['lab_result']
        entity_start = []
        entity_end = []
        entity_list = []
        for lab_result_one in lab_result:
            kb_id = lab_result_one['kb_id']
            entity = lab_result_one['mention'].lower()
            s = int(lab_result_one['offset'])
            e = s + len(entity) - 1
            entity_start.append(s)
            entity_end.append(e)
            entity_list.append((kb_id, entity, s, e))
        if not train_net_is_exist:
            # 存在实体才加入

            if len(entity_list) > 0:
                train_net_data.append({
                    'text_id': text_id,
                    'text': text,
                    'entity_start': entity_start,
                    'entity_end': entity_end,
                    'entity_list': entity_list
                })
            # else:
            #     print(idx)

        # if kb_id == -1:
        #     continue
        #
        # if not train_list_is_exist:
        #     # 存在实体才加入
        #     if len(entity_list) > 0:
        #         train_list_data.append({
        #             'text_id': text_id,
        #             'text': text,
        #             'entity_start': entity_start,
        #             'entity_end': entity_end,
        #             'entity_list': entity_list
        #         })
        # else:
        #     print(idx)
    # add raw
if not train_net_addraw_is_exist:
    f = open('./data/raw_texts.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    train_net_addraw = train_net_data.copy()
    resultlist = random.sample(range(0, len(lines)), 35000)
    with open('./data_deal/addraw_resultlist.pkl', 'wb') as f:
        pickle.dump(resultlist, f)
    for idx, line in enumerate(lines):
        if idx not in resultlist:
            continue
        [entity, text] = line.split('\t')
        text = text.replace(' ', '')
        entity_start = []
        entity_end = []
        entity_list = []
        for i in re.finditer(entity, text):
            entity_start.append(i.span()[0])
            entity_end.append(i.span()[1] - 1)
            entity_list.append((-1, entity, i.span()[0], i.span()[1] - 1))
        train_net_addraw.append({
            'text_id': len(train_net_data) + count,
            'text': text,
            'entity_start': entity_start,
            'entity_end': entity_end,
            'entity_list': entity_list
        })
        count += 1

# 构建文字编码字典
texts = []
for dict in train_net_addraw:
    texts.append(dict['text'])
if os.path.exists('./data_deal/tokenizer.pkl'):
    with open('./data_deal/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
else:
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts)
    with open('./data_deal/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
print('num_words: ', len(tokenizer.word_index))
# num_words = len(tokenizer.word_index)
num_words = 3811

# 保存编码字典
if os.path.exists('./data_deal/word_index.pkl'):
    with open('./data_deal/word_index.pkl', 'rb') as f:
        word_index = pickle.load(f)
else:
    total_num = len(tokenizer.word_index)
    word_index = {}
    for word in tokenizer.word_index:
        word_id = tokenizer.word_index[word]
        if word_id <= num_words:
            word_index.update({word: word_id})
    with open('./data_deal/word_index.pkl', 'wb') as f:
        pickle.dump(word_index, f)

# 训练集文字转编码
text_len = []
for i in train_net_addraw:
    text_len.append(len(i['text']))
    i.update({
        'text_seq': [word_index.get(c, num_words + 1) for c in i['text']]
    })
print('train_net_data最大长度\n', max(text_len))  #
for i in train_net_data:
    text_len.append(len(i['text']))
    i.update({
        'text_seq': [word_index.get(c, num_words + 1) for c in i['text']]
    })
print('train_net_data最大长度', max(text_len))

# 保存处理好的训练集
if os.path.exists('./data_deal/train_net_addraw.pkl'):
    with open('./data_deal/train_net_addraw.pkl', 'rb') as f:
        train_net_addraw = pickle.load(f)
else:
    with open('./data_deal/train_net_addraw.pkl', 'wb') as f:
        pickle.dump(train_net_addraw, f)

if os.path.exists('./data_deal/train_net_data.pkl'):
    with open('./data_deal/train_net_data.pkl', 'rb') as f:
        train_net_data = pickle.load(f)
else:
    with open('./data_deal/train_net_data.pkl', 'wb') as f:
        pickle.dump(train_net_data, f)

# 处理验证集
f = open('./data/dev.json', 'r', encoding='utf-8')
data = json.load(f)

if os.path.exists('./data_deal/dev_data.pkl'):
    with open('./data_deal/dev_data.pkl', 'rb') as f:
        dev_data = pickle.load(f)
else:
    f = open('./data/dev.json', 'r', encoding='utf-8')
    data = json.load(f)
    dev_data = []
    for idx, data_one in enumerate(data):
        text_id = data_one['text_id']
        text = data_one['text'].lower()
        lab_result = data_one['lab_result']
        entity_start = []
        entity_end = []
        entity_list = []
        for lab_result_one in lab_result:
            kb_id = lab_result_one['kb_id']
            if kb_id == -1:
                continue
            entity = lab_result_one['mention'].lower()
            s = int(lab_result_one['offset'])
            e = s + len(entity) - 1
            entity_start.append(s)
            entity_end.append(e)
            entity_list.append((kb_id, entity, s, e))
        texts.append(text)

        # 存在实体才加入
        if len(entity_list) > 0:
            dev_data.append({
                'text_id': text_id,
                'text': text,
                'entity_start': entity_start,
                'entity_end': entity_end,
                'entity_list': entity_list
            })
        else:
            print(idx)

# 验证集文字转编码
text_len = []
for i in dev_data:
    text_len.append(len(i['text']))
    i.update({
        'text_seq': [word_index.get(c, num_words + 1) for c in i['text']]
    })
print('最大长度', max(text_len))  #

# 保存处理好的验证集
if os.path.exists('./data_deal/dev_data.pkl'):
    with open('./data_deal/dev_data.pkl', 'rb') as f:
        dev_data = pickle.load(f)
else:
    with open('./data_deal/dev_data.pkl', 'wb') as f:
        pickle.dump(dev_data, f)

print(train_net_addraw[3])
