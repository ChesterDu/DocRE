from dgl.batch import batch
import torch.nn as nn
import torch
import fitlog
import tqdm
import torch.nn.functional as F
from model import OUTPUT_NUM
import json
import os

def f1_metric(y_pred,y_true):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1-y_true) * (1-y_pred)).sum().to(torch.float32)
    fp = ((1-y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

#     epsilon = 1e-7
#     precision = tp / (tp + fp + epsilon)
#     recall = tp / (tp + fn + epsilon)
#     f1 = 2* (precision*recall) / (precision + recall + epsilon)

    return tp.item(),tn.item(),fp.item(),fn.item()

def gen_train_facts(data_file_name):
    fact_file_name = data_file_name
    fact_file_name = fact_file_name.replace(".json", ".fact")

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train

class Trainner(nn.Module):
    def __init__(self,config,model,optimizer,criterion):
        super(Trainner,self).__init__()
        
        self.log_pth = config.log_pth
        self.checkpoint_pth = config.checkpoint_pth
        
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.criterion = criterion

        self.time = 0

        self.total_steps = config.total_steps
        self.epoch = config.epoch
        self.step_count = 0
        self.forward_count = 0
        self.num_acumulation = config.num_acumulation
        self.metric_check_freq = config.metric_check_freq
        self.loss_print_freq = config.loss_print_freq

        self.epoch_count = 1

        self.lr = config.lr
        self.device = config.device

        self.fact_in_train = gen_train_facts("../DocRED/data/train_annotated.json")
        truth = json.load(open("../DocRED/data/dev.json",'r'))
        self.std = {}
        self.titleset = set([])

        self.title2vectexSet = {}

        for x in truth:
            title = x['title']
            self.titleset.add(title)

            vertexSet = x['vertexSet']
            self.title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                self.std[(title, r, h_idx, t_idx)] = set(label['evidence'])
        
        if config.debug:
            fitlog.set_log_dir("../debug_logs")
        else:
            fitlog.set_log_dir(self.log_pth)
        fitlog.add_hyper(config)
        
        self.theta = config.theta
    


    def train(self,train_loader,dev_loader):
        print(self.model.parameters)
        self.total_steps = (self.epoch * len(train_loader) - 1) // self.num_acumulation + 1
        bar = tqdm.tqdm(total=self.total_steps)
        bar.update(0)

        # epoch_num = (self.total_steps - 1) // len(train_loader) + 1
        # while(self.step_count < self.total_steps):
        while(self.epoch_count <= self.epoch):
            self.model.train()
            epoch_loss = 0
            for batch_data in train_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                multi_labels = batch_data['multi_label'].to(self.device).reshape(-1,logits.shape[-1]).to(torch.float)
                loss = self.criterion(logits,multi_labels) # reduction = none, shape: [bsz * h_t_pair_num]
                label_mask = batch_data['label_mask'].to(self.device).reshape(-1)
                loss = loss.sum(dim=-1) * label_mask
                loss = loss.sum() / label_mask.sum()

                epoch_loss += loss.item()
                loss.backward()
                self.forward_count += 1

                if (self.forward_count % self.num_acumulation) == 0:
                    self.optimizer.step()
                    self.step_count += 1
                    bar.update(1)
                    fitlog.add_loss(loss.item(),name='Loss',step=self.step_count)
                    self.optimizer.zero_grad()
                    if self.step_count >= self.total_steps:
                        break


                    if (self.step_count % self.loss_print_freq) == 0:
                        print("Epoch:{}/{} || Step:{}/{} || Loss:{}".format(self.epoch_count,self.epoch,self.step_count,self.total_steps,epoch_loss/self.forward_count))
                    
                    # if (self.step_count % self.metric_check_freq) == 0:
            print('Evaluation Start......')
            results = self.official_evaluate(dev_loader)
            print("Eval Results Epoch: {}".format(self.epoch_count))
            for metric in results:
                fitlog.add_metric({"dev":metric},step=self.step_count,epoch=self.epoch_count)
                metric_name = list(metric.keys())[0]
                print("{}: {}".format(metric_name,metric[metric_name]))
            # print("Eval Results Epoch: {} || Step:{}/{} || Precision: {} || Recall: {} || F1: {}".format(self.epoch_count,self.step_count,self.total_steps,p,r,f1))
            # fitlog.add_metric({"dev":{"Precision":p}}, step=self.step_count,epoch=self.epoch_count)
            # fitlog.add_metric({"dev":{"Recall":r}}, step=self.step_count,epoch=self.epoch_count)
            # fitlog.add_metric({"dev":{"F1":f1}}, step=self.step_count,epoch=self.epoch_count)
            # fitlog.add_metric({"dev":{"Loss":test_loss}},step=self.step_count,epoch=self.epoch_count)
            self.epoch_count += 1
    
    def official_evaluate(self,test_loader):
        self.model.eval()

        ## generate submission answer
        tmp = []
        with torch.no_grad():
            for test_batch in test_loader:
                titles = test_batch['title']
                logits = self.model(test_batch)
                batch_label_mask = test_batch['label_mask'].to(self.device)
                batch_orig_pairs = test_batch['orig_pair']
                batch_predictions_re = (torch.sigmoid(logits) > self.theta).to(torch.int64)

                for i in range(len(titles)):
                    title = titles[i]
                    prediction = batch_predictions_re[i]
                    orig_pairs = batch_orig_pairs[i]
                    mask = batch_label_mask[i]
                    prediction = prediction[(mask!=0).nonzero().squeeze(-1)].cpu()
                    assert(prediction.shape[0] == len(orig_pairs))

                    for i,[h_idx,t_idx] in enumerate(orig_pairs):
                        for r in range(prediction.shape[1]):
                            if prediction[i][r] == 1:
                                tmp.append({'title':title, 'h_idx':h_idx, 't_idx':t_idx, 'r':r})


        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i-1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i])

        correct_re = 0

        correct_in_train_annotated = 0
        titleset2 = set([])
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in self.title2vectexSet:
                continue
            vertexSet = self.title2vectexSet[title]

            if (title, r, h_idx, t_idx) in self.std:
                correct_re += 1
                in_train_annotated = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in self.fact_in_train:
                            in_train_annotated = True

                if in_train_annotated:
                    correct_in_train_annotated += 1

        re_p = 1.0 * correct_re / len(submission_answer)
        re_r = 1.0 * correct_re / len(self.std)
        if re_p+re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)


        re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) / (len(submission_answer)-correct_in_train_annotated)
    
        if re_p_ignore_train_annotated+re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

        return {"F1":re_f1},{"Ign F1":re_f1_ignore_train_annotated}
        

    def evaluate_multi(self,test_loader):
        self.model.eval()
        with torch.no_grad():
            total_tp = 0
            total_tn = 0
            total_fp = 0
            total_fn = 0
            for batch_data in test_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['multi_label'].to(self.device).reshape(-1,logits.shape[-1]).to(torch.float)
                label_masks = batch['label_mask'].to(self.device).reshape(-1).to(torch.float)
                predections_re = (torch.sigmoid(logits) > self.theta).to(torch.float)
                
                indices = (label_masks != 0).nonzero().squeeze(-1)
                labels = labels[indices]
                predections_re = predections_re[indices]
                
                tp,tn,fp,fn = f1_metric(predections_re,labels)
                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn

             
            epsilon = 1e-7
            p = total_tp / (total_tp + total_fp + epsilon)
            r = total_tp / (total_tp + total_fn + epsilon)
            f1 = 2* (p*r) / (p + r + epsilon)
        return p,r,f1

        
    def evaluate(self,test_loader):
        self.model.eval()
        with torch.no_grad():
            total_na_pred = None
            total_na_label = None
            total_nonNa_pred = None
            total_nonNa_label = None
            for batch_data in test_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['label'].to(self.device).reshape(-1).to(torch.float)
                # loss = self.criterion(logits,labels)
                # test_loss.append(loss.item())

                prediction = torch.argmax(logits,dim=1)
                indices = (labels != -1).nonzero().squeeze(-1)
                labels = labels[indices]
                prediction = prediction[indices]

                na_indices = (labels == 0).nonzero().squeeze(-1)
                na_pred = F.one_hot((prediction[na_indices] == 0).to(torch.int64),2)
                na_label = F.one_hot((labels[na_indices] == 0).to(torch.int64),2)

                nonNa_indices = (labels != 0).nonzero().squeeze(-1)
                nonNa_pred = F.one_hot(prediction[nonNa_indices],OUTPUT_NUM)
                nonNa_label = F.one_hot(labels[nonNa_indices],OUTPUT_NUM)

                total_na_pred = torch.cat([total_na_pred,na_pred],dim=0) if total_na_pred != None else na_pred
                total_na_label = torch.cat([total_na_label,na_label],dim=0) if total_na_label != None else na_label
                total_nonNa_pred = torch.cat([total_nonNa_pred,nonNa_pred],dim=0) if total_nonNa_pred != None else nonNa_pred
                total_nonNa_label = torch.cat([total_nonNa_label,nonNa_label],dim=0) if total_nonNa_label != None else nonNa_label

            def f1_metric(y_pred,y_true):
                tp = (y_true * y_pred).sum().to(torch.float32)
                tn = ((1-y_true) * (1-y_pred)).sum().to(torch.float32)
                fp = ((1-y_true) * y_pred).sum().to(torch.float32)
                fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

                epsilon = 1e-7
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2* (precision*recall) / (precision + recall + epsilon)

                return precision.item(),recall.item(),f1.item()
            nonNa_p,nonNa_r,nonNa_f1 = f1_metric(total_nonNa_pred,total_nonNa_label)
            na_p,na_r,na_f1 = f1_metric(total_na_pred,total_na_label)

        return {"nonNa_p":nonNa_p,"nonNa_r":nonNa_r,"nonNa_f1":nonNa_f1,"na_p":na_p,"na_r":na_r,"na_f1":na_f1}

