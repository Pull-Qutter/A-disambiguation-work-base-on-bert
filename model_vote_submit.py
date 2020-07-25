import pickle
import torch
from net import MyDataset, collate_fn, collate_fn_link, deal_eval, seqs2batch, dataset
from pytorch_pretrained_bert import BertTokenizer, BertModel
import operator
import os
import pandas as pd
from collections import defaultdict
import json

with open('./data_deal/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)


with open('./data_deal/subject_data.pkl', 'rb') as f:
    subject_data = pickle.load(f)


with open('./data_deal/stockname_data.pkl', 'rb') as f:
    stockname_data = pickle.load(f)

device = 'cuda:0'
dataset.device = device
num_words = 3811
max_len = 400
num_layers = 3
hidden_dim = 768
loss_weight = 2

model_name = {'ernie': 12,'wwm':12,'roberta_wwm':9}#embedding_name:link_id
def vote(test_predicts):
    ernie_predict = test_predicts[0]
    wwm_predict = test_predicts[1]
    roberta_wwm_predict = test_predicts[2]

    entity_predict_list = []
    #开始以三个模型投票原则做决策
    for id, _ in enumerate(ernie_predict):
        vote_score = [0,0,0]#以a,b,c注释 用于判空和记录，表示 ernie，wwm，roberta_wwm, 最后输出其中最大值所对应的结果
        ernie_predict_one = [j[1] for j in ernie_predict[id]]
        wwm_predict_one  = [j[1] for j in wwm_predict[id]]
        roberta_wwm_predict_one = [j[1] for j in roberta_wwm_predict[id]]
        data_list = []
        data_list.append(ernie_predict[id])
        data_list.append(wwm_predict[id])
        data_list.append(roberta_wwm_predict[id])

        for i,entity_data in enumerate(data_list):#
            if len(entity_data):  #  not null ，记录a，b,c是否都有结果
                vote_score[i]+=1
            #提前处理一下结果包含两个以上的实体连接




        if sum(vote_score) ==3:#最普遍情况，a,b,c都存在
            if operator.eq(ernie_predict_one,wwm_predict_one): #a,b同否
                vote_score[0] +=1
            elif operator.eq(ernie_predict_one,wwm_predict_one):#a,c同否
                vote_score[0] += 1
            elif operator.eq(wwm_predict_one,roberta_wwm_predict_one):#b,c同否
                vote_score[1] += 1
            else:#a,b,c均不相同，以ernie为准
                vote_score[0] += 1

        elif sum(vote_score) == 2:  # a,b,c中有一个为空
            if operator.eq(ernie_predict_one,wwm_predict_one): #a,b同否
                vote_score[0] +=1
            elif operator.eq(ernie_predict_one,wwm_predict_one):#a,c同否
                vote_score[0] += 1
            elif operator.eq(wwm_predict_one,roberta_wwm_predict_one):#b,c同否
                vote_score[1] += 1
            else:#a,b,c均不相同，判断谁为空
                if not len(ernie_predict_one):#a为空，以b为标准
                    vote_score[1] += 1
                elif not len(wwm_predict_one) or not len(roberta_wwm_predict_one):#b或c为空，以a为标准
                    vote_score[0] += 1
        # elif sum(vote_score) == 1:  # a,b,c中有两个为空,其实不用处理，自然有最大值
        entity_predict_list.append(test_predicts[vote_score.index(max(vote_score))][id])
    return entity_predict_list

def dev_predict(model_name):
    # 读取验证集预处理
    with open('./data_deal/dev_data.pkl', 'rb') as f:
        develop_data = pickle.load(f)
    if not os.path.exists('./model_fuse' ):
        os.makedirs('./model_fuse')
    test_predicts = []

    for embedding_name,link_id in model_name.items():
        if os.path.exists('./model_fuse/%s_devresult.pkl'%embedding_name):
            with open('./model_fuse/%s_devresult.pkl'%embedding_name, 'rb') as f:
                test = pickle.load(f)
                test_predicts.append(test)
        else:
            bert_path = './Bert/' + embedding_name + '/'
            dataset.tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')
            dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
            dataset.BERT.eval()
            dataset.max_len = max_len

            # ==========导入link预训练模型==========
            modelname = 'lstm_%d_%d_%d_len_%d_lf_2_l_2' % (num_layers, hidden_dim, loss_weight, max_len)
            model = torch.load('./results/%s/%s/new_%03d.pth' % (
                embedding_name, modelname, link_id), map_location=device)
            model.device = device
            model.to(device)
            print('%s is running'%embedding_name)
            model.eval()

            entity_predict_list = []
            for idx, data in enumerate(develop_data):
                text_seq = deal_eval([data])
                text_seq = text_seq.to(device)
                text = data['text']
                with torch.no_grad():
                    entity_predict = model(text_seq, text, stockname_data, None)
                entity_predict_list.append(entity_predict)
            with open('./model_fuse/%s_devresult.pkl' % embedding_name, 'wb') as f:
                pickle.dump(entity_predict_list,f)
            test_predicts.append(entity_predict_list)
    #
    entity_predict_list = vote(test_predicts)#投票
    # print(entity_predict_list[1002])
    #计算融合后的实体识别率和实体消歧率
    p_len = 0.001
    l_len = 0.001
    correct_len = 0.001
    p_len1 = 0.001
    l_len1 = 0.001
    correct_len1 = 0.001
    result1 = []
    for idx, data in enumerate(develop_data):
        entity_predict = entity_predict_list[idx]
        p_set = set([j[:-1] for j in entity_predict])
        p_len += len(p_set)
        l_set = set(data['entity_list'])
        l_len += len(l_set)
        correct_len += len(p_set.intersection(l_set))

        p_set1 = set([j[1:-1] for j in entity_predict])
        p_len1 += len(p_set1)
        l_set1 = set([j[1:] for j in data['entity_list']])
        l_len1 += len(l_set1)
        correct_len1 += len(p_set1.intersection(l_set1))
    Precision = correct_len / p_len
    Recall = correct_len / l_len
    F1 = 2 * Precision * Recall / (Precision + Recall)

    Precision1 = correct_len1 / p_len1
    Recall1 = correct_len1 / l_len1
    F1_1 = 2 * Precision1 * Recall1 / (Precision1 + Recall1)

    accu = F1 / F1_1
    result1.append([1,
                    round(Precision1, 4), round(Recall1, 4), round(F1_1, 4), round(accu, 4),
                    round(Precision, 4), round(Recall, 4), round(F1, 4)])
    print('验证集结果：')
    print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (1, Precision, Recall, F1))

    result1_df = pd.DataFrame(result1,
                              columns=['Epoch',
                                       'P_n', 'R_n', 'F_n', 'link_accu',
                                       'P', 'R', 'F1'])

    print(result1_df)
    result1_df.to_csv('./model_fuse/result.csv', index=False)


def showmodeldiff(model_name):
    test_predicts = []
    for embedding_name, link_id in model_name.items():
        if os.path.exists('./model_fuse/%s_devresult.pkl' % embedding_name):
            with open('./model_fuse/%s_devresult.pkl' % embedding_name, 'rb') as f:
                test = pickle.load(f)
                test_predicts.append(test)
        else:
            print("Please run dev_predict firstly")
            break
    ernie_predict = test_predicts[0]
    wwm_predict = test_predicts[1]
    roberta_wwm_predict = test_predicts[2]
    #输出不同之处，以便于验证
    for idx,_ in enumerate(ernie_predict):
        if len(ernie_predict[idx])==0 or len(wwm_predict[idx])==0 or len(roberta_wwm_predict[idx])==0:
            print(idx, ernie_predict[idx], wwm_predict[idx], roberta_wwm_predict[idx])
        else:
            ERNIE_predict_entity = ernie_predict[idx][0][1]
            WWM_predict_entity = wwm_predict[idx][0][1]
            roberta_wwm_predict_entity = roberta_wwm_predict[idx][0][1]
            if((ERNIE_predict_entity != WWM_predict_entity) or (ERNIE_predict_entity != roberta_wwm_predict_entity) or (WWM_predict_entity != roberta_wwm_predict_entity)):
                print(idx, ernie_predict[idx], wwm_predict[idx], roberta_wwm_predict[idx])

def tests_predict_submit(model_name):
    if not os.path.exists('./model_fuse' ):
        os.makedirs('./model_fuse')
    test_predicts = []

    for embedding_name,link_id in model_name.items():
        if os.path.exists('./model_fuse/%s_textresult.pkl'%embedding_name):
            with open('./model_fuse/%s_textresult.pkl'%embedding_name, 'rb') as f:
                test = pickle.load(f)
                test_predicts.append(test)
        else:
            bert_path = './Bert/' + embedding_name + '/'
            dataset.tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')
            dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
            dataset.BERT.eval()
            dataset.max_len = max_len

            # ==========导入link预训练模型==========
            modelname = 'lstm_%d_%d_%d_len_%d_lf_2_l_2' % (num_layers, hidden_dim, loss_weight, max_len)
            model = torch.load('./results/%s/%s/new_%03d.pth' % (
                embedding_name, modelname, link_id), map_location=device)
            model.device = device
            model.to(device)
            print('%s is running'%embedding_name)
            model.eval()

            # 读取测试集
            file = open('./data/text_texts.txt', 'r', encoding='utf-8')
            # file = open('./data_deal/addraw_texts.txt', 'r', encoding='utf-8')
            lines = file.readlines()

            entity_predict_list = []
            for idx, line in enumerate(lines):
                text_id, text = line.split('\t')
                text_seq = [word_index.get(c, num_words + 1) for c in text]
                text_seq = torch.LongTensor([text_seq])
                text_seq = text_seq.to(device)
                with torch.no_grad():
                    entity_predict = model(text_seq,
                                           text,
                                           stockname_data,
                                           None)
                entity_predict_list.append(entity_predict)
            with open('./model_fuse/%s_textresult.pkl' % embedding_name, 'wb') as f:
                pickle.dump(entity_predict_list,f)
            test_predicts.append(entity_predict_list)
    entity_predict_list = vote(test_predicts)  # 投票

    #开始保存结果
    # 读取测试集
    file = open('./data/text_texts.txt', 'r', encoding='utf-8')
    # file = open('./data_deal/addraw_texts.txt', 'r', encoding='utf-8')
    lines = file.readlines()
    predict_results = defaultdict(dict)
    submit_result = []
    flag = 0
    count = 0
    for idx, line in enumerate(lines):
        text_id, text = line.split('\t')
        entitylink_predict = entity_predict_list[idx]
        if flag == 0:
            predict_results.update({"team_name": "驱动先锋"})
            flag = 1

        mention_result = []
        for entity_predict_one in entitylink_predict:
            kb_id, mention, offset, _, confidence = entity_predict_one
            mention_result.append({
                "mention": mention,
                "kb_id": kb_id,
                "offset": offset,
                "confidence": confidence
            })
        submit_result.append({
            "text_id": text_id,
            "text": text,
            "mention_result": mention_result
        })
        count += 1
        if count % 500 == 0:
            print("finsh %d" % count)
    predict_results.update({"submit_result": submit_result})
    with open('./data_deal/result.json', 'w', encoding="utf-8") as f:
        json.dump(predict_results, f, ensure_ascii=False, indent=4, separators=(',', ':'))

if __name__ == '__main__':
    # 对dev预测并将结果进行投票融合，将输出实体识别和实体消歧后的指标
    #dev_predict(model_name)
    # 对主办方下方的texts进行预测并投票融合，结果保存为指定格式data_deal/result.json
    tests_predict_submit(model_name)
    #showmodeldiff(model_name)#观察不同模型结果差异值，以做决策算法



