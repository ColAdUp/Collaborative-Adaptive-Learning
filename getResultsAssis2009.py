from torch.utils.data import TensorDataset, DataLoader
import sklearn.preprocessing as pp
import pandas as pd
import numpy as np
import torch
import scipy
import math

from k_means_constrained import KMeansConstrained
from pyBKT.models import Model
from EduCDM import GDIRT
import itertools
import argparse
import pickle
import time
import sys

from DKT import DKT

columns = ['nb_learners','nb_tests','K','M','method','apt','variant','time','expectedPerf','aptitude','learnGroup','learnGroupAll','precision','recall','correct',\
'improve_p_skill_avg','gain_p_skill_avg','improve_p_skill_avg_2','gain_p_skill_avg_2', 'improve_mean_skills','gain_mean_skills', 'coverage_skills', 'nb_skills_imp', 'at_least']


#DKT
def data_loader_train_dkt(train_loc, seq_len, batch_size, shuffle, pad_val=-1):
    q_seqs = []
    c_seqs = []
    qshft_seqs = []
    cshft_seqs = []
    
    list_questions = train_loc.item_id.tolist()
    list_correct = train_loc.correct.tolist()
    
    i = 0
    while i + seq_len + 1 < len(list_questions):
        q_seq = list_questions[i:i + seq_len + 1]
        c_seq = list_correct[i:i + seq_len + 1]
        
        q_seqs.append(q_seq[:-1])
        c_seqs.append(c_seq[:-1])
        
        qshft_seqs.append(q_seq[1:])
        cshft_seqs.append(c_seq[1:])

        i += seq_len + 1

    q_seq = np.concatenate([list_questions[i:],np.array([pad_val] * (i + seq_len + 1 - len(list_questions)))])
    c_seq = np.concatenate([list_correct[i:],np.array([pad_val] * (i + seq_len + 1 - len(list_correct)))])
    
    q_seqs.append(q_seq[:-1])
    c_seqs.append(c_seq[:-1])

    qshft_seqs.append(q_seq[1:])
    cshft_seqs.append(c_seq[1:])
    
    q_seqs = torch.tensor(q_seqs)
    c_seqs = torch.tensor(c_seqs, dtype=torch.float32)
    qshft_seqs = torch.tensor(qshft_seqs)
    cshft_seqs = torch.tensor(cshft_seqs, dtype=torch.float32)
    
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)
    
    q_seqs, c_seqs, qshft_seqs, cshft_seqs = \
        q_seqs * mask_seqs, c_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        cshft_seqs * mask_seqs
    
    data_set = TensorDataset(
        q_seqs,
        c_seqs,
        qshft_seqs,
        cshft_seqs,
        mask_seqs
    )
    
    return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

def data_loader_test_dkt(train, seq_len, batch_size, pad_val=-1):
    q_seqs = []
    c_seqs = []
    
    list_questions = train.item_id.tolist()
    list_correct = list(np.full(len(list_questions),1))
    
    if seq_len <= len(list_questions):
        q_seq = list_questions[-seq_len:]
        c_seq = list_correct[-seq_len:]
    else:
        q_seq = np.concatenate([list_questions, np.array([pad_val] * (seq_len - len(list_questions)))])
        c_seq = np.concatenate([list_correct, np.array([pad_val] * (seq_len - len(list_correct)))])
        
    q_seqs.append(q_seq)
    c_seqs.append(c_seq)

    q_seqs = torch.tensor(q_seqs)
    c_seqs = torch.tensor(c_seqs, dtype=torch.float32)
    
    mask_seqs = (q_seqs != pad_val)
    
    q_seqs, c_seqs = q_seqs * mask_seqs, c_seqs * mask_seqs
    
    data_set = TensorDataset(
        q_seqs,
        c_seqs,
        mask_seqs
    )
    
    return DataLoader(data_set, batch_size=batch_size)
    
#IRT
def eval_prediction(model, test):
    model.irt_net.eval()
    y_pred = []
    for batch_data in test:
        user_id, item_id, response = batch_data
        user_id: torch.Tensor = user_id
        item_id: torch.Tensor = item_id
        pred: torch.Tensor = model.irt_net(user_id, item_id)
        y_pred.extend(pred.tolist())
    
    return np.array(y_pred)

def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(list(x)),
        torch.tensor(list(y)),
        torch.tensor(list(z), dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)

#Skill Update
def update_skill_2(skill_learner, df_materials):
    max_skill = skill_learner
    
    for row in df_materials.itertuples():
        diff = row[5]
        correct = row[4]
        
        if correct == 1:
            max_skill = diff
        else:
            break
            
    return max_skill

#Metrics
def get_metrics(mean_learners, mean_tests, learner, batch, tests2kc, ground_truth, bkt_model):

    if bkt_model[0] is None:
        return 0,0,0,0,0,0,0,0,0,0,0,0,None

    nb_skills = set(range(mean_learners.shape[1]))

    mean_learners_ = mean_learners.mean(axis=1)

    mean_tests_ = np.true_divide(mean_tests.sum(1),(mean_tests!=0).sum(1))
    mean_tests_ = np.nan_to_num(mean_tests_)

    if learner in ground_truth:
        gt = ground_truth[learner][0]

        prec = len(set(batch).intersection(set(gt)))/len(batch)
        rec = len(set(batch).intersection(set(gt)))/len(gt)
    else:
        prec = 0
        rec = 0
    
    t2k = []
    kc2test = dict()

    df = pd.DataFrame()
    df['item_id'] = batch
    df['user_id'] = learner
    df['id'] = range(len(df))
    df.id += len(bkt_model[1])
    df['correct'] = 1

    for x in list(df.item_id):
        kc = tests2kc[x]
        kc = list(set(kc))

        t2k.append(kc)

        for k in kc:
            if k in kc2test:
                kc2test[k].append(x)
            else:
                kc2test[k] = [x]

    df['kc'] = t2k
    df = df[['id','user_id','item_id','correct','kc']]

    df = pd.concat([bkt_model[1],df], ignore_index = True)

    df = bkt_model[0].predict(data = df.copy()).reset_index()
    df = df.tail(len(batch))
    
    df['correct'] = df.state_predictions.apply(lambda x: 1 if x >= 0.7 else 0)
    df.item_id = df.item_id.apply(lambda x: int(x))

    cor = df.correct.mean()

    gain = 0
    improve = 0

    skills_ = []
    gains_ = []
    improves_ = []

    len_ = len(kc2test)

    for i in nb_skills - kc2test.keys():
        kc2test[i] = []

    #for kc, liste in kc2test.items():
    for kc in nb_skills:
        liste = kc2test[kc]

        if len(liste) == 0:
            sk = mean_learners[learner,kc]

            skills_.append(sk)
            gains_.append(0)
            improves_.append(0)

            continue

        liste = sorted(liste)

        df_tmp = df.sort_values(by='item_id')
        df_tmp = df_tmp[df_tmp.item_id.isin(liste)]

        diffs = list(mean_tests[liste,:][:,kc])

        df_tmp['diff'] = diffs
        #df_tmp['diff'] = df_tmp.item_id.apply(lambda x: mean_tests[int(x)])
        df_tmp = df_tmp.sort_values(by='diff')
        df_tmp = df_tmp[['id','user_id','item_id','correct','diff']]

        sk = mean_learners[learner,kc]
        skill = update_skill_2(sk, df_tmp)

        skills_.append(sk)

        if skill > sk:
            improve += 1
            gain += skill - sk

            gains_.append(skill - sk)
            improves_.append(1)

        else:
            gains_.append(0)
            improves_.append(0)

    at_least = int(improve != 0)

    df['diff'] = df.item_id.apply(lambda x: mean_tests_[int(x)])
    df = df.sort_values(by='diff')
    df = df[['id','user_id','item_id','correct','diff']]

    skill = update_skill_2(mean_learners_[learner], df)

    if skill > mean_learners_[learner]:
        improve_mean = 1
        gain_mean = skill - mean_learners_[learner]
    else:
        improve_mean = 0
        gain_mean = 0

    gain_over = gain/len_
    improve_over = improve/len_

    gain_over_2 = gain/len(kc2test)
    improve_over_2 = improve/len(kc2test)

    dic = {learner: [skills_, improves_, gains_, mean_learners_[learner], gain_mean]}

    return prec, rec, cor, improve_over, improve_over_2, improve_mean, gain_over, gain_over_2, gain_mean, len_/len(kc2test), improve, at_least, dic

#Objectives
def diameter_lg(mean_learners, learners):
    m = mean_learners.reshape(-1,1)
    s = 0
    for grp in learners:
        s += m[grp].max() - m[grp].min()
    return s

def diameter_lg_skills(learners_profile, learners):
    s = 0
    for grp in learners:
        tmp_min = learners_profile[grp,:].min(0)
        tmp_max = learners_profile[grp,:].max(0)
        
        s += (tmp_max-tmp_min).sum()

    return s

def all_lg(mean_learners, learners):
    m = mean_learners.reshape(-1,1)
    s = 0
    for grp in learners:
        combi = list(itertools.combinations(grp, 2))
        for j in combi:
            s += abs(m[j[0]]-m[j[1]])[0]
    return s

def get_pairwise(liste):
    s = 0
    comb = list(itertools.combinations(liste, 2))
    
    for i,j in comb:
        s += abs(i-j)
    
    return s

def all_lg_skills(learners_profile, learners):
    s = 0
    
    for grp in learners:
        tmp = learners_profile[grp,:]
        tmp_s = 0
        for skill in range(tmp.shape[1]):
            tmp_s += get_pairwise(list(tmp[:,skill]))
        
        s += tmp_s
    
    return s

def aptitude_cal(learners, tests, grp, batch, test2ks):    
    apts = []
    
    for l in grp:
        apt = 0
        for t in batch:
            ks = list(test2ks[t])
            apt += (tests[t,ks]-learners[l,ks]).mean()
        apts.append(apt)
    
    return min(apts)

parser = argparse.ArgumentParser()
parser.add_argument("-sk", "--skill", default='irt', type=str, help="Generation of learners skills")
args = parser.parse_args()

type_skill = args.skill

train_rate = 0.7

if type_skill == 'mean':
    file_name = ''
elif type_skill == 'irt':
    file_name = 'irt_'
else:
    sys.exit('ERROR')


defaults = {'order_id': 'id',
            'skill_name': 'item_id',
            'user_id': 'user_id',
            'correct': 'correct',
            'multigs':'kc'}

for simulation in range(5,8):
    if simulation == 0:
        simulation = ''
    else:
        simulation = f'_{simulation}'

    all_res = pd.DataFrame(columns=columns)

    for nb_learners in [100, 500, 1000]:
        ks = [10, 50]
        if nb_learners > 100:
            ks.append(100)

        for nb_tests in [50]:
            ms = [5, 10, 20]

            df_train = pd.read_csv(f'./res_Assis2009/train_{nb_learners}_{nb_tests}.csv')
            df_test = pd.read_csv(f'./res_Assis2009/test_{nb_learners}_{nb_tests}.csv')

            ground_truth = df_test.groupby('user_id')['item_id'].apply(list).reset_index().set_index('user_id').T.to_dict('list')

            with open(f'./res_Assis2009/learners/{file_name}{nb_learners}_{nb_tests}_new{simulation}', 'rb') as fp:
                learners = pickle.load(fp)

            with open(f'./res_Assis2009/tests/{file_name}{nb_learners}_{nb_tests}_new{simulation}', 'rb') as fp:
                tests = pickle.load(fp)

            with open(f'./res_Assis2009/tests2kc/{file_name}{nb_learners}_{nb_tests}_new{simulation}', 'rb') as fp:
                tests2kc = pickle.load(fp)

            with open(f'./res_Assis2009/perfs/{file_name}{nb_learners}_{nb_tests}_new{simulation}', 'rb') as fp:
                perf = pickle.load(fp)

            mean_learners = learners.mean(axis=1)

            mean_tests = np.true_divide(tests.sum(1),(tests!=0).sum(1))
            mean_tests = np.nan_to_num(mean_tests)

            learner_bkts = []
            learner_irt = []
            learner_dkt = []

            eval_df = pd.DataFrame()
            eval_df['item_id'] = range(nb_tests)
            eval_df['kc'] = eval_df.item_id.apply(lambda x: tests2kc[x])

            for i in range(len(mean_learners)):
                temp = df_train[df_train.user_id==i]
                temp['id'] = range(len(temp))
                temp = temp[['id','user_id','item_id','correct','kc']]

                if len(temp) == 0:
                    learner_bkts.append((None,None))
                    learner_irt.append(None)
                    learner_dkt.append(None)
                    continue
                
                #BKT
                bkt = Model(seed=1000)
                bkt.fit(data = temp.copy(), defaults = defaults)
                learner_bkts.append((bkt,temp))
                
                #DKT
                train_batches_dkt = data_loader_train_dkt(temp.copy(), 50, 8, True) 

                dkt = DKT(nb_tests, 50, 50)
                opt = torch.optim.Adam(dkt.parameters(), lr=0.001)
                dkt.train_model(train_batches_dkt, num_epochs=50, opt=opt) #50
                learner_dkt.append(dkt)

                #IRT
                train_batches_irt = transform(np.full(len(temp), 0), temp["item_id"], temp["correct"], 8)
            
                irt = GDIRT(1, nb_tests)
                irt.train(train_batches_irt, epoch=25) #25
                learner_irt.append(irt)

            for k in ks:
                for m in ms:
                    print(f'{nb_learners} - {nb_tests} - {k} - {m}')

                    with open(f'./res_Assis2009/groupings/{file_name}groupings_{nb_learners}_{nb_tests}_{k}_{m}_new{simulation}', 'rb') as fp:
                        grouping_list = pickle.load(fp)

                    with open(f'./res_Assis2009/groupings/{file_name}res_{nb_learners}_{nb_tests}_{k}_{m}_new{simulation}', 'rb') as fp:
                        res = pickle.load(fp)
                    
                    single = True

                    dict_detail = dict()

                    for mth in range(len(grouping_list)):
                        prec, rec, cor, gain_mean, improve_mean, gain_over, improve_over, gain_over_2, improve_over_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0
                        cov, nb_improved_skill, improved_at_least_one, detail = 0, 0, 0, dict()

                        if single:
                            prec_irt, rec_irt, cor_irt, improve_mean_irt, gain_mean_irt, improve_over_irt, gain_over_irt, improve_over_2_irt, gain_over_2_irt = 0, 0, 0, 0, 0, 0, 0, 0, 0 
                            cov_irt, nb_improved_skill_irt, improved_at_least_one_irt, detail_irt = 0, 0, 0, dict()

                            prec_dkt, rec_dkt, cor_dkt, improve_mean_dkt, gain_mean_dkt, improve_over_dkt, gain_over_dkt, improve_over_2_dkt, gain_over_2_dkt = 0, 0, 0, 0, 0, 0, 0, 0, 0
                            cov_dkt, nb_improved_skill_dkt, improved_at_least_one_dkt, detail_dkt = 0, 0, 0, dict()

                        grouping = grouping_list[mth]
                        exp, apt = 0, 0
                        gg = []

                        for i in range(k):
                            us, ts = grouping[i]

                            for u in us:
                                p, r, c, imp_o, imp_o_2, imp_m, gan_o, gan_o_2, gan_m, co, skills_imp, at_least, det_dict = get_metrics(learners, tests, u, ts, tests2kc, ground_truth, learner_bkts[u])
                                prec += p
                                rec += r
                                cor += c

                                cov += co

                                nb_improved_skill += skills_imp

                                improved_at_least_one += at_least

                                improve_over += imp_o
                                gain_over += gan_o

                                improve_over_2 += imp_o_2
                                gain_over_2 += gan_o_2

                                improve_mean += imp_m
                                gain_mean += gan_m

                                if det_dict is not None:
                                    detail[u] = det_dict[u]

                                #IRT
                                if (learner_irt[u] is not None) and (single):
                                    test_irt = transform(np.full(len(eval_df), 0), eval_df["item_id"], np.full(len(eval_df), 0), 8)
                                    irt_list = eval_prediction(learner_irt[u], test_irt)
                                    irt_list = np.array(irt_list)
                                    irt_list = list((np.argsort(irt_list)[::-1]))[:m]

                                    p, r, c, imp_o, imp_o_2, imp_m, gan_o, gan_o_2, gan_m, co, skills_imp, at_least, det_dict = get_metrics(learners, tests, u, irt_list, tests2kc, ground_truth, learner_bkts[u])
                                    prec_irt += p
                                    rec_irt += r
                                    cor_irt += c

                                    cov_irt += co
                                    nb_improved_skill_irt += skills_imp
                                    improved_at_least_one_irt += at_least

                                    improve_over_irt += imp_o
                                    gain_over_irt += gan_o

                                    improve_over_2_irt += imp_o_2
                                    gain_over_2_irt += gan_o_2

                                    improve_mean_irt += imp_m
                                    gain_mean_irt += gan_m    

                                    if det_dict is not None:
                                        detail_irt[u] = det_dict[u]                            

                                #DKT
                                if (learner_dkt[u] is not None) and (single):
                                    dkt_list = data_loader_test_dkt(eval_df, 50, 8)
                                    dkt_list = learner_dkt[u].test_model(dkt_list)
                                    dkt_list = list(dkt_list.detach().numpy()[0].argsort()[::-1])[:m]

                                    p, r, c, imp_o, imp_o_2, imp_m, gan_o, gan_o_2, gan_m, co, skills_imp, at_least, det_dict = get_metrics(learners, tests, u, dkt_list, tests2kc, ground_truth, learner_bkts[u])
                                    prec_dkt += p
                                    rec_dkt += r
                                    cor_dkt += c

                                    cov_dkt += co
                                    nb_improved_skill_dkt += skills_imp
                                    improved_at_least_one_dkt += at_least

                                    improve_over_dkt += imp_o
                                    gain_over_dkt += gan_o

                                    improve_over_2_dkt += imp_o
                                    gain_over_2_dkt += gan_o

                                    improve_mean_dkt += imp_m
                                    gain_mean_dkt += gan_m

                                    if det_dict is not None:
                                        detail_dkt[u] = det_dict[u] 
                            
                            gg.append(list(us))
                            exp += perf[us,:][:,ts].sum()
                            apt += aptitude_cal(learners, tests, us, ts, tests2kc)

                        if res[mth][6] == 'diameter':
                            learn_gr = diameter_lg(mean_learners, gg)
                            learn_gr_skills = diameter_lg_skills(learners, gg)

                        elif res[mth][6] == 'all':
                            learn_gr = all_lg(mean_learners, gg)
                            learn_gr_skills = all_lg_skills(learners, gg)

                        if single:
                            single = False
                        
                        dict_detail[mth] = detail

                        res[mth].extend([exp, apt, learn_gr, learn_gr_skills, prec/nb_learners, rec/nb_learners, cor/nb_learners,\
                        improve_over/nb_learners, gain_over/nb_learners, improve_over_2/nb_learners, gain_over_2/nb_learners, improve_mean/nb_learners, gain_mean/nb_learners,\
                        cov/nb_learners, nb_improved_skill/nb_learners, improved_at_least_one/nb_learners])

                    dict_detail['single_IRT'] = detail_irt
                    dict_detail['single_DKT'] = detail_dkt

                    res_irt = [nb_learners,nb_tests,k,m,'single_IRT', None, None, 0, 0, 0, 0, 0, prec_irt/nb_learners, rec_irt/nb_learners, cor_irt/nb_learners,\
                    improve_over_irt/nb_learners, gain_over_irt/nb_learners, improve_over_2_irt/nb_learners, gain_over_2_irt/nb_learners,\
                    improve_mean_irt/nb_learners, gain_mean_irt/nb_learners, cov_irt/nb_learners, nb_improved_skill_irt/nb_learners, improved_at_least_one_irt/nb_learners]

                    res_dkt = [nb_learners,nb_tests,k,m,'single_DKT', None, None, 0, 0, 0, 0, 0, prec_dkt/nb_learners, rec_dkt/nb_learners, cor_dkt/nb_learners,\
                    improve_over_dkt/nb_learners, gain_over_dkt/nb_learners, improve_over_2_dkt/nb_learners, gain_over_2_dkt/nb_learners,\
                    improve_mean_dkt/nb_learners, gain_mean_dkt/nb_learners, cov_dkt/nb_learners, nb_improved_skill_dkt/nb_learners, improved_at_least_one_dkt/nb_learners]

                    res.append(res_irt)
                    res.append(res_dkt)

                    res = pd.DataFrame(res, columns=columns)

                    all_res = pd.concat([all_res, res], ignore_index=True, sort=False)
                    all_res.to_csv(f'./res_Assis2009/results/{file_name}all_results_neurips_new{simulation}.csv',index=False)

                    with open(f'./res_Assis2009/detailed/{file_name}res_{nb_learners}_{nb_tests}_{k}_{m}_{simulation}', 'wb') as fp:
                        pickle.dump(dict_detail, fp)

    all_res.to_csv(f'./res_Assis2009/results/{file_name}all_results_neurips_new{simulation}.csv',index=False)
