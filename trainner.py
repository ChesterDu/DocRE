import torch.nn as nn
import torch
import fitlog
import tqdm
import torch.nn.functional as F
from model import OUTPUT_NUM
import json
import os
import numpy as np
import sklearn

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
    def __init__(self,config,model,optimizer,criterion,scheduler=None):
        super(Trainner,self).__init__()
        
        self.log_pth = config.log_pth
        self.checkpoint_pth = config.checkpoint_pth
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

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
        self.clip = config.clip
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
        # bar = tqdm.tqdm(total=self.total_steps)
        # bar.update(0)

        # epoch_num = (self.total_steps - 1) // len(train_loader) + 1
        # while(self.step_count < self.total_steps):
        while(self.epoch_count <= self.epoch):
            self.model.train()
            epoch_loss = 0
            total_Na = 0
            correct_Na = 0
            total_not_Na = 0
            correct_not_Na = 0
            for cur_i,batch_data in enumerate(train_loader):
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

                output = torch.argmax(logits, dim=-1)
                output = output.data.cpu().numpy()
                relation_label = batch_data['label'].reshape(-1).data.cpu().numpy()

                for i in range(output.shape[0]):
                    label = relation_label[i]
                    if label < 0:
                        break

                    is_correct = (output[i] == label)
                    if label == 0:
                        total_Na += 1
                        correct_Na += is_correct
                    else:
                        total_not_Na += 1
                        correct_not_Na += is_correct

                if (self.forward_count % self.num_acumulation) == 0:
                    nn.utils.clip_grad_value_(self.model.parameters(),self.clip)
                    self.optimizer.step()
                    if self.scheduler != None:
                        self.scheduler.step(self.epoch_count)
                    self.step_count += 1
                    # bar.update(1)
                    fitlog.add_loss(loss.item(),name='Loss',step=self.step_count)
                    self.optimizer.zero_grad()
                    if self.step_count >= self.total_steps:
                        break


                    if (self.step_count % self.loss_print_freq) == 0:
                        print("Epoch:{}/{} || Step:{}/{} || Loss:{} || NA Acc: {} ||not NA Acc: {}".format( \
                            self.epoch_count,self.epoch,self.step_count,self.total_steps,epoch_loss/(cur_i + 1),\
                            correct_Na/total_Na,correct_not_Na/total_not_Na))
                    
                    # if (self.step_count % self.metric_check_freq) == 0:
            print('Evaluation Start......')
            results = self.evaluate_multi(dev_loader)
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
                        for r in range(1,prediction.shape[1]):
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
        candidate_thetas = np.linspace(0.1,0.9,9)
        with torch.no_grad():
            total_tps = np.zeros(len(candidate_thetas))
            total_tns = np.zeros(len(candidate_thetas))
            total_fps = np.zeros(len(candidate_thetas))
            total_fns = np.zeros(len(candidate_thetas))
            for batch_data in test_loader:
                logits = self.model(batch_data)
                logits = logits.reshape(-1,logits.shape[-1])
                labels = batch_data['multi_label'].to(self.device).reshape(-1,logits.shape[-1]).to(torch.float)
                label_masks = batch_data['label_mask'].to(self.device).reshape(-1).to(torch.float)
                indices = (label_masks != 0).nonzero().squeeze(-1)
                labels = labels[indices]
                prob_score = torch.sigmoid(logits[indices])

                for theta_i,theta in enumerate(candidate_thetas):
                    predections_re = (prob_score > theta).to(torch.float)
                    
                    tp,tn,fp,fn = f1_metric(predections_re[:,1:],labels[:,1:])
                    total_tps[theta_i] += tp
                    total_tns[theta_i] += tn
                    total_fps[theta_i] += fp
                    total_fns[theta_i] += fn

             
            epsilon = 1e-7
            ps = total_tps / (total_tps + total_fps + epsilon)
            rs = total_tps / (total_tps + total_fns + epsilon)
            f1s = 2* (ps*rs) / (ps + rs + epsilon)
            f1 = f1s.max()
            f1_pos = f1.argmax()
            p = ps[f1_pos]
            r = rs[f1_pos]
            theta = candidate_thetas[f1_pos]
        return {'Precision':p},{'Recall':r},{'F1':f1},{'Theta':theta}

        
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

    def evaluate_auc(self, dataloader, input_theta=-1,relation_num=97):

        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        total_steps = len(dataloader)
        bar = tqdm.tqdm(total=total_steps)
        bar.update(0)
        for cur_i, test_batch in enumerate(dataloader):

            with torch.no_grad():
                logits = self.model(test_batch)
                predict_re = torch.sigmoid(logits)

            predict_re = predict_re.data.cpu().numpy()
            labels = test_batch['label']
            L_vertex = test_batch['ent_num']
            titles = test_batch['title']

            for i in range(len(labels)):
                label = labels[i]
                L = L_vertex[i]
                title = titles[i]
                total_recall += len(label)

                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                j = 0

                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            for r in range(1, relation_num):
                                rel_ins = (h_idx, t_idx, r)
                                intrain = label.get(rel_ins, False)

                                test_result.append((rel_ins in label, float(predict_re[i, j, r]), intrain,
                                                        title, h_idx, t_idx, r))

                            j += 1
            
            bar.update(1)

        test_result.sort(key=lambda x: x[1], reverse=True)
        print(test_result[:100])
        print(len(test_result))

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))  # Precision
            pr_x.append(float(correct) / total_recall)  # Recall
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

        print('ma_f1 {:3.4f} | input_theta {:3.4f} test_result P {:3.4f} test_result R {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}' \
            .format(f1, input_theta, pr_y[w], pr_x[w], f1_arr[w], auc))

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0

        # https://github.com/thunlp/DocRED/issues/47
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)

            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        ign_f1 = f1_arr.max()

        ign_auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

        print(
            'Ignore ma_f1 {:3.4f} | inhput_theta {:3.4f} test_result P {:3.4f} test_result R {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}' \
                .format(ign_f1, input_theta, pr_y[w], pr_x[w], f1_arr[w], auc))

        return {'F1':f1}, {'Ign F1':ign_f1}, {'AUC':auc}, {'Ign AUC':ign_auc},{'theta':input_theta}


    def evaluate_fix_theta(self,dataloader,relation_num=97):
        
        candidate_thetas = np.linspace(0.1,0.9,num=9)
        test_result_set_lst = [set() for i in range(len(candidate_thetas))]
        ignore_test_result_set_lst = [set() for i in range(len(candidate_thetas))]
        std_set = set()
        ignore_std_set = set()
        total_recall = 0
        total_steps = len(dataloader)
        bar = tqdm.tqdm(total=total_steps)
        bar.update(0)
        for cur_i, test_batch in enumerate(dataloader):
            labels = test_batch['label']
            L_vertex = test_batch['ent_num']
            titles = test_batch['title']
            with torch.no_grad():
                logits = self.model(test_batch)
                predict_re_prob = torch.sigmoid(logits)

            for batch_i in range(len(labels)):
                label = labels[batch_i]
                L = L_vertex[batch_i]
                title = titles[batch_i]
                total_recall += len(label)

                for h_idx,t_idx,r in label.keys():
                    std_set.add((title,r,h_idx,t_idx))
                    intrain = label[(h_idx,t_idx,r)]
                    if not intrain:
                        ignore_std_set.add((title,r,h_idx,t_idx))

           
                for theta_i,theta in enumerate(candidate_thetas):
                    with torch.no_grad():
                        predict_re = (predict_re_prob[batch_i] > theta).to(torch.int)
                    predict_re = predict_re.data.cpu().numpy()
                    test_result_set = test_result_set_lst[theta_i]
                    ignore_test_result_set = ignore_test_result_set_lst[theta_i]

                    j = 0

                    for h_idx in range(L):
                        for t_idx in range(L):
                            if h_idx != t_idx:
                                for r in range(1, relation_num):
                                    if predict_re[j,r] == 1:
                                        test_result_set.add((title,r,h_idx,t_idx))
                                        intrain = label.get((h_idx,t_idx,r),False)
                                        if not intrain:
                                            ignore_test_result_set.add((title,r,h_idx,t_idx))

                                j += 1
            
            bar.update(1)
        
        def calculate_precision_recall(test_result_set,std_set):
            eplison = 1e-7
            tp = len(test_result_set & std_set)
            p = tp / len(test_result_set)
            r = tp / len(std_set)
            return p,r
        
        p_lst,r_lst,ign_p_lst,ign_r_lst = np.zeros(len(candidate_thetas)),np.zeros(len(candidate_thetas)),np.zeros(len(candidate_thetas)),np.zeros(len(candidate_thetas))
        for theta_i in range(len(candidate_thetas)):
            p,r = calculate_precision_recall(test_result_set_lst[theta_i],std_set)
            ign_p,ign_r = calculate_precision_recall(ignore_test_result_set_lst[theta_i],ignore_std_set)
            p_lst[theta_i],r_lst[theta_i],ign_p_lst[theta_i],ign_r_lst[theta_i] = p,r,ign_p,ign_r
        
        f1_lst = 2 * p_lst * r_lst / (p_lst + r_lst + 1e-7)
        f1 = f1_lst.max()
        f1_theta_pos = f1_lst.argmax()
        f1_theta = candidate_thetas[f1_theta_pos]

        ign_f1_lst = 2 * ign_p_lst * ign_r_lst / (ign_p_lst + ign_r_lst + 1e-7)
        ign_f1 = ign_f1_lst.max()
        ign_f1_pos = ign_f1_lst.argmax()
        ign_f1_theta = candidate_thetas[ign_f1_pos]

        return {'F1':f1},{'F1 Theta':f1_theta},{'Ign F1':ign_f1},{'Ign F1 Theta':ign_f1_theta}


