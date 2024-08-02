from torch.utils.data import TensorDataset, DataLoader
import sklearn.preprocessing as pp
import pandas as pd
import numpy as np
import torch
import scipy
import math

from k_means_constrained import KMeansConstrained
from EduCDM import GDIRT
import itertools
import argparse
import time
import sys

from utils import *

import pickle

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import anndata2ri

anndata2ri.activate()
rpy2.robjects.numpy2ri.activate()

anclust = importr('anticlust')

# Expes
parser = argparse.ArgumentParser()
parser.add_argument("-sk", "--skill", default='irt', type=str, help="Generation of learners skills")
args = parser.parse_args()

type_skill = args.skill

train_rate = 0.7

df = pd.read_csv('./Data/anonymized_full_release_competition_dataset.csv')
df = df[['studentId','problemId','skill','correct']]

sk2idx = dict()
for idx, sk in enumerate(df.skill.unique()):
    sk2idx[sk] = idx+1

df_t = df[['problemId','skill']].drop_duplicates().groupby('problemId').skill.apply(set).reset_index()
df_t.skill = df_t.skill.apply(lambda x: [sk2idx[i] for i in x])

df = pd.merge(df.drop(columns=['skill']), df_t,on='problemId')
df_original = df.rename(columns={'studentId':'user_id', 'problemId':'item_id', 'skill':'kc'})

del df_t
del df

for simulation in range(5):
    for nb_learners in [100, 500, 1000]:
        ks = [10, 50]
        if nb_learners > 100:
            ks.append(100)

        for nb_tests in [50]:
            ms = [5, 10, 20]
            df = df_original.copy()

            #Get Most Sized Learners
            df_learner = df.groupby('user_id').size().reset_index(name='nb')
            df_learner = df_learner.sort_values(by='nb', ascending=False).head(nb_learners)

            df = df[df.user_id.isin(df_learner.user_id.unique())]

            #Change Ids of Learners
            df_learner = df_learner.sort_values(by='user_id')
            df_learner['new_l'] = range(len(df_learner))

            #Get Most Sized Tests
            df_tests = df.groupby('item_id').size().reset_index(name='nb')
            df_tests = df_tests.sort_values(by='nb', ascending=False).head(nb_tests)

            df = df[df.item_id.isin(df_tests.item_id.unique())]

            #Change Ids of Tests
            df_tests = df[['item_id']].drop_duplicates().sort_values(by='item_id')
            df_tests['new_id'] = range(len(df_tests))

            #Change Ids of KCs
            kcs = set().union(*df.kc)
            kcs = {i:idx+1 for idx,i in enumerate(kcs)}

            #Apply changings
            df = pd.merge(df,df_tests,on='item_id')
            df = pd.merge(df, df_learner[['user_id','new_l']], on='user_id')
            df = df.drop(columns=['item_id','user_id'])
            df = df.rename(columns={'new_id':'item_id','new_l':'user_id'})
            df.kc = df.kc.apply(lambda x: [kcs[i] for i in x])

            df_train = pd.DataFrame()
            df_test = pd.DataFrame()

            for l in df.user_id.unique():
                temp = df[df.user_id==l]
                len_ = int(len(temp)*train_rate)
                temp_test = temp.tail(len(temp)-len_)

                df_train = pd.concat([df_train, temp.head(len_)])
                df_test = pd.concat([df_test, temp_test])
            
            df_train.to_csv(f'./res_Assis/train_{nb_learners}_{nb_tests}.csv',index=False)
            df_test.to_csv(f'./res_Assis/test_{nb_learners}_{nb_tests}.csv',index=False)

            #Generate profiles of Learners and Tests
            kc2tests, kc2correct, tests2kc, tests = get_tests(df, nb_tests)

            if type_skill == 'mean':
                kc2learners, learners = get_learners_with_mean(df_train, nb_learners, kc2tests)
                file_name = ''
            elif type_skill == 'irt':
                kc2learners, learners = get_learners_with_irt(df_train, nb_learners, nb_tests, kc2tests)
                file_name = 'irt_'
            else:
                sys.exit('ERROR')

            with open(f'./res_Assis/learners/{file_name}{nb_learners}_{nb_tests}_new_{simulation}', 'wb') as fp:
                pickle.dump(learners, fp)

            with open(f'./res_Assis/tests/{file_name}{nb_learners}_{nb_tests}_new_{simulation}', 'wb') as fp:
                pickle.dump(tests, fp)

            with open(f'./res_Assis/tests2kc/{file_name}{nb_learners}_{nb_tests}_new_{simulation}', 'wb') as fp:
                pickle.dump(tests2kc, fp)
            

            sim = similarity_tests(tests)
            perf = expected_perf(df_train, sim, tests, learners)

            with open(f'./res_Assis/perfs/{file_name}{nb_learners}_{nb_tests}_new_{simulation}', 'wb') as fp:
                pickle.dump(perf, fp)

            mean_learners = learners.mean(axis=1)

            for k in ks:
                for m in ms:
                    print(f'{nb_learners} - {nb_tests} - {k} - {m}')
                    res = []
                    grouping_list = []

                    #Our First Solution - Diameter
                    beg = time.time()
                    grouping_diameter_1_with_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='diameter', solution='first')
                    fin = time.time() - beg
                    grouping_list.append(grouping_diameter_1_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'PerfSort', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_diameter_1_without_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='diameter', solution='first', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_diameter_1_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'PerfSort', 'no', 'diameter', fin])

                    #Our Second Solution - Diameter
                    beg = time.time()
                    grouping_diameter_2_with_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='diameter', solution='second')
                    fin = time.time() - beg
                    grouping_list.append(grouping_diameter_2_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Greedy', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_diameter_2_witout_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='diameter', solution='second', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_diameter_2_witout_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Greedy', 'no', 'diameter', fin])

                    #Our First Solution - All
                    beg = time.time()
                    grouping_all_1_with_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='all', solution='first')
                    fin = time.time() - beg
                    grouping_list.append(grouping_all_1_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'PerfSort', 'yes', 'all', fin])
                    # ---
                    beg = time.time()
                    grouping_all_1_without_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='all', solution='first', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_all_1_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'PerfSort', 'no', 'all', fin])

                    #Our Second Solution - All
                    beg = time.time()
                    grouping_all_2_with_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='all', solution='second')
                    fin = time.time() - beg
                    grouping_list.append(grouping_all_2_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Greedy', 'yes', 'all', fin])
                    # ---
                    beg = time.time()
                    grouping_all_2_witout_apt = heuristic(perf, learners, tests, mean_learners, k, m, tests2kc, learning_group='all', solution='second', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_all_2_witout_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Greedy', 'no', 'all', fin])

                    #Kmeans Solution - Diameter
                    beg = time.time()
                    grouping_kmeans_with_apt = kmeans_solution(perf, learners, tests, mean_learners, k, m, tests2kc, n = 1, learning_group='diameter')
                    fin = time.time() - beg
                    grouping_list.append(grouping_kmeans_with_apt)
                    kmeans_yes = [nb_learners, nb_tests, k, m, 'kmeans', 'yes', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'kmeans', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_kmeans_without_apt = kmeans_solution(perf, learners, tests, mean_learners, k, m, tests2kc, n = 1, learning_group='diameter', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_kmeans_without_apt)
                    kmeans_no = [nb_learners, nb_tests, k, m, 'kmeans', 'no', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'kmeans', 'no', 'diameter', fin])

                    #Kmeans Solution - All
                    grouping_list.append(grouping_kmeans_with_apt)
                    res.append(kmeans_yes)

                    # ---
                    grouping_list.append(grouping_kmeans_without_apt)
                    res.append(kmeans_no)

                    #Balanced Solution - Diameter
                    beg = time.time()
                    grouping_dbalanced_1_with_apt = dbalance_1_solution(perf, learners, tests, mean_learners, k, m, tests2kc)
                    fin = time.time() - beg
                    grouping_list.append(grouping_dbalanced_1_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Balanced', 'yes', 'diameter', fin])

                    beg = time.time()
                    grouping_dbalanced_1_without_apt = dbalance_1_solution(perf, learners, tests, mean_learners, k, m, tests2kc, aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_dbalanced_1_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Balanced', 'no', 'diameter', fin])

                    #Balanced Solution - All
                    beg = time.time()
                    grouping_abalanced_with_apt = abalance_solution(perf, learners, tests, mean_learners, k, m, tests2kc)
                    fin = time.time() - beg
                    grouping_list.append(grouping_abalanced_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Balanced', 'yes', 'all', fin])
                    # ---
                    beg = time.time()
                    grouping_abalanced_without_apt = abalance_solution(perf, learners, tests, mean_learners, k, m, tests2kc, aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_abalanced_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'Balanced', 'no', 'all', fin])

                    #RGroup Solution - Diameter
                    beg = time.time()
                    grouping_randomg_with_apt = RGroup_solution(perf, learners, tests, mean_learners, k, m, tests2kc)
                    fin = time.time() - beg
                    grouping_list.append(grouping_randomg_with_apt)

                    random_yes = [nb_learners, nb_tests, k, m, 'RandomGroup', 'yes', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'RandomGroup', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_randomg_without_apt = RGroup_solution(perf, learners, tests, mean_learners, k, m, tests2kc, aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_randomg_without_apt)

                    random_no = [nb_learners, nb_tests, k, m, 'RandomGroup', 'no', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'RandomGroup', 'no', 'diameter', fin])

                    #RGroup Solution - All
                    grouping_list.append(grouping_randomg_with_apt)
                    res.append(random_yes)
                    # ---
                    grouping_list.append(grouping_randomg_without_apt)
                    res.append(random_no)

                    #RPerf Solution
                    beg = time.time()
                    grouping_randomt_with_apt = RPerf_diameter_solution(perf, learners, tests, mean_learners, k, m, tests2kc)
                    fin = time.time() - beg
                    grouping_list.append(grouping_randomt_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'RandomPerf', 'X', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_randomt_a_with_apt = RPerf_all_solution(perf, learners, tests, mean_learners, k, m, tests2kc)
                    fin = time.time() - beg
                    grouping_list.append(grouping_randomt_a_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'RandomPerf', 'X', 'all', fin])

                    #AntiClust Diversity Solution - Diameter
                    beg = time.time()
                    grouping_anti_with_apt = anticlust_solution(perf, learners, tests, 'diversity', k, m, tests2kc)
                    fin = time.time() - beg
                    grouping_list.append(grouping_anti_with_apt)
                    anti_yes = [nb_learners, nb_tests, k, m, 'AntiDiv', 'yes', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'AntiDiv', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_anti_without_apt = anticlust_solution(perf, learners, tests, 'diversity', k, m, tests2kc, aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_anti_without_apt)
                    anti_no = [nb_learners, nb_tests, k, m, 'AntiDiv', 'no', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'AntiDiv', 'no', 'diameter', fin])

                    #AntiClust Diversity Solution - All
                    grouping_list.append(grouping_anti_with_apt)
                    res.append(anti_yes)
                    # ---
                    grouping_list.append(grouping_anti_without_apt)
                    res.append(anti_no)

                    #AntiClust Variance Solution - Diameter
                    beg = time.time()
                    grouping_anti_with_apt = anticlust_solution(perf, learners, tests, 'variance', k, m, tests2kc)
                    fin = time.time() - beg
                    grouping_list.append(grouping_anti_with_apt)
                    anti_yes = [nb_learners, nb_tests, k, m, 'AntiVar', 'yes', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'AntiVar', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_anti_without_apt = anticlust_solution(perf, learners, tests, 'variance', k, m, tests2kc, aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_anti_without_apt)
                    anti_no = [nb_learners, nb_tests, k, m, 'AntiVar', 'no', 'all', fin]
                    res.append([nb_learners, nb_tests, k, m, 'AntiVar', 'no', 'diameter', fin])

                    #AntiClust Variance Solution - All
                    grouping_list.append(grouping_anti_with_apt)
                    res.append(anti_yes)
                    # ---
                    grouping_list.append(grouping_anti_without_apt)
                    res.append(anti_no)

                    
                    #Our AntiClust Solution - Diameter
                    beg = time.time()
                    grouping_ours_with_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='diameter', init='variance')
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursV', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_ours_without_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='diameter', init='variance', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursV', 'no', 'diameter', fin])

                    #Our AntiClust Solution - All
                    beg = time.time()
                    grouping_ours_with_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='all', init='variance')
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursV', 'yes', 'all', fin])
                    # ---
                    beg = time.time()
                    grouping_ours_without_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='all', init='variance', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursV', 'no', 'all', fin])

                    #Our AntiClust 2 Solution - Diameter
                    beg = time.time()
                    grouping_ours_with_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='diameter', init='random')
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursR', 'yes', 'diameter', fin])
                    # ---
                    beg = time.time()
                    grouping_ours_without_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='diameter', init='random', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursR', 'no', 'diameter', fin])

                    #Our AntiClust 2 Solution - All
                    beg = time.time()
                    grouping_ours_with_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='all', init='random')
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_with_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursR', 'yes', 'all', fin])
                    # ---
                    beg = time.time()
                    grouping_ours_without_apt = antiClustering(perf, learners, tests, k, m, tests2kc, learning_group='all', init='random', aptitude=False)
                    fin = time.time() - beg
                    grouping_list.append(grouping_ours_without_apt)
                    res.append([nb_learners, nb_tests, k, m, 'OursR', 'no', 'all', fin])

                    #Collect
                    with open(f'./res_Assis/groupings/{file_name}groupings_{nb_learners}_{nb_tests}_{k}_{m}_new_{simulation}', 'wb') as fp:
                        pickle.dump(grouping_list, fp)

                    with open(f'./res_Assis/groupings/{file_name}res_{nb_learners}_{nb_tests}_{k}_{m}_new_{simulation}', 'wb') as fp:
                        pickle.dump(res, fp)
