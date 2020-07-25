import pickle
import os
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from net_ner import MyDataset, collate_fn, deal_eval, seqs2batch, dataset
from net_ner import Net
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertModel
import argparse

cuda_num = '0'  # cuda:0/1
Bert_name = 'bert'  # roberta_wwm,wwm,ernie
loss_weight_ = 2  # 3/2
num_layers = 3  # 3/4
hidden_dim = 768  # 768/1024
BS = 8  # BitchSize
num_words_ = 3811
EMBEDDING_DIM = 300
epochs = 1
k = 0.8500

dataset.device = "cuda:%s" % cuda_num
device = dataset.device
# torch.manual_seed(1)


with open('./data_deal/weight_baidubaike.pkl', 'rb') as f:
    embedding = pickle.load(f)
    embedding = torch.FloatTensor(embedding).to(device)

# 文本编码、词典
with open('./data_deal/word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

# 训练集
with open('./data_deal/train_net_addraw.pkl', 'rb') as f:
    train_data = pickle.load(f)

# 根据"kb_id"检索的实体词典
with open('./data_deal/subject_data.pkl', 'rb') as f:
    subject_data = pickle.load(f)

# 根据"实体"检索的实体词典
with open('./data_deal/stockname_data.pkl', 'rb') as f:
    stockname_data = pickle.load(f)

# 读取验证集预处理
with open('./data_deal/dev_data.pkl', 'rb') as f:
    develop_data = pickle.load(f)

# 拆分训练集
train1_data, train2_data = train_test_split(train_data,
                                            test_size=0.1,
                                            random_state=1)
trainloader1 = torch.utils.data.DataLoader(
    dataset=MyDataset(train1_data, subject_data, stockname_data),
    batch_size=BS, shuffle=True, collate_fn=collate_fn)

train2_data = train2_data
trainloader2 = torch.utils.data.DataLoader(
    dataset=MyDataset(train2_data, subject_data, stockname_data),
    batch_size=BS, shuffle=False, collate_fn=collate_fn)

pwd = '.'
for embedding_name in [Bert_name]:
    bert_path = './Bert/' + embedding_name + '/'
    dataset.tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')
    dataset.BERT = BertModel.from_pretrained(bert_path).to(device)
    dataset.BERT.eval()
    dataset.max_len = 300
    for loss_weight in [loss_weight_]:
        F1_ = 0
        while F1_ < k:
            # vocab_size还有pad和unknow，要+2
            model = Net(vocab_size=len(word_index) + 2,
                        embedding_dim=EMBEDDING_DIM,
                        num_layers=num_layers,
                        hidden_dim=hidden_dim,
                        embedding=embedding,
                        device=device).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            file_name = 'lstm_%d_%d_%d' % (
                num_layers, hidden_dim, loss_weight)

            if not os.path.exists('./results_ner/%s/%s/' % (embedding_name, file_name)):
                os.makedirs('./results_ner/%s/%s/' % (embedding_name, file_name))  # '/result_ner/bert/%s

            score1 = []
            result1 = []
            for epoch in range(epochs):
                print('Start Epoch: %d\n' % (epoch + 1))
                sum_ner_loss = 0.0
                model.train()
                for i, data in enumerate(trainloader1):
                    data_ner = data

                    # 训练ner
                    model.zero_grad()
                    text_features, mask_loss_texts, entity_starts, entity_ends = data_ner

                    # ner损失
                    ner_loss = model.cal_ner_loss(text_features,
                                                  mask_loss_texts,
                                                  entity_starts,
                                                  entity_ends,
                                                  loss_weight)
                    ner_loss.backward()
                    optimizer.step()
                    sum_ner_loss += ner_loss.item()

                    if (i + 1) % 200 == 0:
                        print('\nEpoch: %d ,batch: %d' % (epoch + 1, i + 1))
                        print('ner_loss: %f' % (sum_ner_loss / 200))
                        sum_ner_loss = 0.0

                # train2得分=====================================================================
                model.eval()
                p_len = 0.001
                l_len = 0.001
                correct_len = 0.001
                score_list = []
                for idx, data in enumerate(train2_data):
                    model.zero_grad()
                    text_seqs = deal_eval([data])
                    text_seqs = text_seqs.to(device)
                    text = data['text']
                    with torch.no_grad():
                        entity_predict = model(text_seqs, text, stockname_data)

                    p_set = set(entity_predict)
                    p_len += len(p_set)
                    l_set = set([j[1:] for j in data['entity_list']])
                    l_len += len(l_set)
                    correct_len += len(p_set.intersection(l_set))

                    if (idx + 1) % 2000 == 0:
                        print('finish train_2 %d' % (idx + 1))

                Precision = correct_len / p_len
                Recall = correct_len / l_len
                F1 = 2 * Precision * Recall / (Precision + Recall)

                score1.append([epoch + 1,
                               round(Precision, 4), round(Recall, 4), round(F1, 4)])
                print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

                score1_df = pd.DataFrame(score1,
                                         columns=['Epoch',
                                                  'P', 'R', 'F1'])
                print(score1_df)
                score1_df.to_csv('./results_ner/%s/%s/new_train_2.csv' % (embedding_name, file_name),
                                 index=False)
                F1_ = max(F1_, F1)
                if F1 >= k:
                    # 保存网络参数
                    torch.save(model.state_dict(),
                               pwd + '/results_ner/%s/%s/new_%03d.pth' % (
                                   embedding_name, file_name, epoch + 1))

                    # eval预测结果=====================================================================
                    model.eval()
                    p_len = 0.001
                    l_len = 0.001
                    correct_len = 0.001
                    entity_list_all = []
                    for idx, data in enumerate(develop_data):
                        model.zero_grad()
                        text_seq = deal_eval([data])
                        text_seq = text_seq.to(device)
                        text = data['text']
                        with torch.no_grad():
                            entity_predict = model(text_seq, text, stockname_data)

                        entity_list_all.append(entity_predict)
                        # 1
                        p_set = set(entity_predict)
                        p_len = p_len + len(p_set)
                        l_set = set([j[1:] for j in data['entity_list']])
                        l_len = l_len + len(l_set)
                        correct_len += len(p_set.intersection(l_set))
                        #
                        # if (idx + 1) % 1000 == 0:
                        #     print('finish dev %d' % (idx + 1))
                    # 2
                    Precision = correct_len / p_len
                    Recall = correct_len / l_len
                    F1 = 2 * Precision * Recall / (Precision + Recall)

                    result1.append([epoch + 1, round(Precision, 4), round(Recall, 4), round(F1, 4)])

                    print('验证集识别率：')
                    print('\nEpoch: %d ,Precision:%f, Recall:%f, F1:%f' % (epoch + 1, Precision, Recall, F1))

                    result1_df = pd.DataFrame(result1, columns=['Epoch', 'Precision', 'Recall', 'F1'])
                    print(result1_df)
                    result1_df.to_csv('./results_ner/%s/%s/new_train_2_result.csv' % (embedding_name, file_name),
                                      index=False)
                    F1_ = max(F1_, F1)
