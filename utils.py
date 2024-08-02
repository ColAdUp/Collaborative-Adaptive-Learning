


#IRT Model
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

#Data Preparation
def get_tests(df, nb_tests):
    kc2tests = dict()
    tests2kc = dict()
    kc2correct = dict()

    kcs = list(df.groupby('item_id').kc.apply(list).reset_index().kc.apply(lambda x: set().union(*x)))
    tests = list(df.groupby('item_id').size().reset_index().item_id.unique())

    for t, kc in zip(tests, kcs):
        kc = [k-1 for k in kc]

        tests2kc[t] = kc
        for k in kc:
            if k not in kc2tests:
                kc2tests[k] = [t]
            else:
                kc2tests[k].append(t)

    tests_profiles = np.zeros( (nb_tests,len(kc2tests)) )

    for kc, tests in kc2tests.items():
        df_temp = df[df.item_id.isin(tests)]
        df_temp = df_temp.groupby('item_id').correct.mean().reset_index().sort_values(by='correct')
        df_temp.correct = (1 - df_temp.correct)
        
        kc2correct[kc] = df_temp.correct.max()
        
        df_temp.correct = df_temp.correct/kc2correct[kc]
        
        for t,s in df_temp.values.tolist():
            tests_profiles[int(t),kc] = s 

    return kc2tests, kc2correct, tests2kc, tests_profiles

def get_learners_with_mean(df, nb_learners, kc2tests):
    kcs = list(df.groupby('user_id').kc.apply(list).reset_index().kc.apply(lambda x: set().union(*x)))
    learners = list(df.groupby('user_id').size().reset_index().user_id.unique())

    kc2learners = dict()

    for l, kc in zip(learners, kcs):
            for k in kc:
                k = k-1
                if k not in kc2learners:
                    kc2learners[k] = [l]
                else:
                    kc2learners[k].append(l)

    learners_profiles = np.zeros( (nb_learners,len(kc2tests)) )
    
    for kc, learners in kc2learners.items():
        tests = kc2tests[kc]

        df_temp = df[(df.user_id.isin(learners)) & (df.item_id.isin(tests))]
        df_temp = df_temp.groupby('user_id').correct.mean().reset_index().sort_values(by='correct')

        for l,s in df_temp.values.tolist():
            learners_profiles[int(l),kc] = s 
    
    return kc2learners, learners_profiles

def get_learners_with_irt(df, nb_learners, nb_tests, kc2tests):
    kcs = list(df.groupby('user_id').kc.apply(list).reset_index().kc.apply(lambda x: set().union(*x)))
    learners = list(df.groupby('user_id').size().reset_index().user_id.unique())

    kc2learners = dict()

    for l, kc in zip(learners, kcs):
            for k in kc:
                k = k-1
                if k not in kc2learners:
                    kc2learners[k] = [l]
                else:
                    kc2learners[k].append(l)

    learners_profiles = np.zeros( (nb_learners,len(kc2tests)) )
    
    for kc, learners in kc2learners.items():
        tests = kc2tests[kc]

        df_temp = df[(df.user_id.isin(learners)) & (df.item_id.isin(tests))]

        for l in learners:
            df_temp_2 = df_temp[df_temp.user_id == l]
            if len(df_temp_2) == 0:
                continue
            
            df_temp_2.user_id = 0

            train = transform(df_temp_2["user_id"], df_temp_2["item_id"], df_temp_2["correct"], 8)
        
            cdm = GDIRT(1,nb_tests)
            cdm.train(train, epoch=5)

            learners_profiles[int(l),kc] = eval_prediction(cdm, train).mean()
    
    return kc2learners, learners_profiles

def similarity_tests(tests):
    matrix = scipy.sparse.csc_matrix(tests)
    matrix = pp.normalize(matrix, axis=1)
    sim = matrix * matrix.T
    
    return sim

def get_sim(sim_mat, t):
    profile = sim_mat[t,:].toarray()[0]
    return list(np.where(profile > 0.7)[0])

def expected_perf(df, sim_mat, tests, learners):
    nb_learners = learners.shape[0]
    nb_tests = tests.shape[0]
    
    exp = np.zeros( (nb_learners, nb_tests) )
    
    for t in range(nb_tests):
        if t%100==0:
            print(t)
        sim_tests = get_sim(sim_mat, t)
        
        if len(sim_tests) == 0:
            continue
        
        for l in range(nb_learners):
            e = df[(df.user_id==l) & (df.item_id.isin(sim_tests))].correct.mean()
            if math.isnan(e):
                e = 0
            exp[l][t] = e
    
    return exp

#Objectives
def diameter_lg(mean_learners, learners):
    m = mean_learners.reshape(-1,1)
    s = 0
    for grp in learners:
        s += m[grp].max() - m[grp].min()
    return s

def all_lg(mean_learners, learners):
    m = mean_learners.reshape(-1,1)
    s = 0
    for grp in learners:
        combi = list(itertools.combinations(grp, 2))
        for j in combi:
            s += abs(m[j[0]]-m[j[1]])[0]
    return s

def get_best_tests(matrix, learners, max_items):
    matrix_q = matrix[learners,:].sum(0)
    matrix_q = np.argsort(matrix_q)
    tests = matrix_q.tolist()[-max_items:]
    
    return tests

def aptitude_cal(learners, tests, grp, batch, test2ks):    
    apts = []
    
    for l in grp:
        apt = 0
        for t in batch:
            ks = list(test2ks[t])
            apt += (tests[t,ks]-learners[l,ks]).mean()
        apts.append(apt)
    
    return min(apts)

def change_aptitude(matrix, learner_profile, test_profile, learners, tests, test2ks, nb_ite):
    apt = aptitude_cal(learner_profile, test_profile, learners, tests, test2ks)
    
    mean_tests = np.true_divide(test_profile.sum(1),(test_profile!=0).sum(1))
    mean_tests = np.nan_to_num(mean_tests)

    diffs = mean_tests.tolist()
    mean_tests = np.argsort(mean_tests).tolist()[::-1]

    tests_bis = list(tests) 
    tests_diffs = [diffs[i] for i in mean_tests if i in tests]

    ite = 0
    j = 0
    prev_chang = -1

    while apt < 0 and ite < nb_ite:
        while j < len(diffs):
            if mean_tests[j] not in tests_bis:
                j += 1
                break

            j += 1

        idx_changed = np.argmin(tests_diffs)

        temp = list(tests_bis)
        diffs_tem = list(tests_diffs)

        tests_bis[idx_changed] = mean_tests[j]
        tests_diffs[idx_changed] = diffs[ mean_tests[j] ]

        if idx_changed == prev_chang:
            perf_1 = matrix[learners,:][:,temp].sum()
            perf_2 = matrix[learners,:][:,tests_bis].sum()

            if perf_1 >= perf_2:
                tests_bis = temp
                tests_diffs = diffs_tem

        prev_chang = idx_changed

        apt = aptitude_cal(learner_profile, test_profile, learners, tests_bis, test2ks)
        ite += 1
    
    return tests_bis

#Kmeans
def groupsKmean(learners_prof, mean_learners, k, learning_group = 'diameter', n=1):
    learner = list(learners_prof)
    past_lg = 0
    r = len(learners)//k
    
    for i in range(n):
        kmeans = KMeansConstrained(n_clusters=k, size_min=r, size_max=r, random_state=0).fit_predict(learner)
        lg = 0
        groups = [[] for i in range(k)]
        for idx, grp in enumerate(kmeans):
            groups[grp].append(idx)

        if learning_group == 'diameter':
            lg = diameter_lg(mean_learners, groups)
        else:
            lg = all_lg(mean_learners, groups)

        if lg >= past_lg:
            past_lg = lg
            best_kmean = groups
    
    return best_kmean

def kmeans_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks, n=1, 
                    learning_group = 'diameter', aptitude=True):
    
    nb_ite = int(matrix.shape[1]*0.1)
    
    grouping = []
    groups = groupsKmean(learner_profile, mean_learners, k, learning_group, n)
    
    for grp in groups:
        tests = get_best_tests(matrix, grp, m)
        if aptitude:
            tests = change_aptitude(matrix, learner_profile, test_profile, grp, tests, test2ks, nb_ite)
            
        grouping.append( (grp,tests) )
        
    return grouping

#Anticlust
def anticlust_solution(matrix, learner_profile, test_profile, obj, k, m, test2ks, aptitude=True):
    
    nb_ite = int(matrix.shape[1]*0.1)
    
    grp = anclust.anticlustering(matrix, k, objective = obj, method = "exchange")
    groups = [[] for i in range(k)]

    for idx,g in enumerate(grp):
        g = int(g)
        groups[g-1].append(idx)
    
    grouping = []

    for grp in groups:
        tests = get_best_tests(matrix, grp, m)
        if aptitude:
            tests = change_aptitude(matrix, learner_profile, test_profile, grp, tests, test2ks, nb_ite)
            
        grouping.append( (grp,tests) )
        
    return grouping


#Baselines
def dbalance_1_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks, aptitude=True):
    grp, chosen = diameter(mean_learners, k)
    
    r = len(mean_learners)//k
    nb_ite = int(matrix.shape[1]*0.1)
    
    learners = list(np.argsort(mean_learners))[::-1]
    learners = [i for i in learners if i not in chosen]
    
    sum_skills = [mean_learners[list(g)].sum() for g in grp]
    nb_learners = [2 for i in range(k)]
    
    for i in learners:
        idxes = np.argsort(sum_skills)
        
        for j in idxes:
            if nb_learners[j] < r:
                break
        
        grp[j] = grp[j] + (i,)   
        sum_skills[j] += mean_learners[j]
        nb_learners[j] += 1
        
    grouping = []
    
    for g in grp:
        tests = get_best_tests(matrix, g, m)
        
        if aptitude:
            tests = change_aptitude(matrix, learner_profile, test_profile, g, tests, test2ks, nb_ite)
            
        grouping.append( (g,tests) )
    
    return grouping

def dbalance_2_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks, aptitude=True):
    grp, chosen = diameter(mean_learners, k)
    
    r = len(mean_learners)//k
    nb_ite = int(matrix.shape[1]*0.1)
    
    learners = list(np.argsort(mean_learners))[::-1]
    learners = [i for i in learners if i not in chosen]
    
    for idx, i in enumerate(learners):
        idx2 = idx%k
        grp[idx2] = grp[idx2] + (i,)   
        
    grouping = []
    
    for g in grp:
        tests = get_best_tests(matrix, g, m)
        
        if aptitude:
            tests = change_aptitude(matrix, learner_profile, test_profile, g, tests, test2ks, nb_ite)
            
        grouping.append( (g,tests) )
    
    return grouping

def abalance_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks, aptitude=True):
    nb_ite = int(matrix.shape[1]*0.1)
    
    learners = list(np.argsort(mean_learners))[::-1]
    grp = [() for i in range(k)]
    
    for idx, i in enumerate(learners):
        idx2 = idx%k
        grp[idx2] = grp[idx2] + (i,)   
        
    grouping = []
    
    for g in grp:
        tests = get_best_tests(matrix, g, m)
        
        if aptitude:
            tests = change_aptitude(matrix, learner_profile, test_profile, g, tests, test2ks, nb_ite)
            
        grouping.append( (g,tests) )
    
    return grouping

def RGroup_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks, aptitude=True):
    learners = list(np.argsort(mean_learners))
    np.random.shuffle(learners)
    
    r = len(mean_learners)//k
    nb_ite = int(matrix.shape[1]*0.1)
    
    grouping = []
    
    for i in range(0,len(learners),r):
        g = learners[i:i+r]
        tests = get_best_tests(matrix, g, m)
        
        if aptitude:
            tests = change_aptitude(matrix, learner_profile, test_profile, g, tests, test2ks, nb_ite)
            
        grouping.append( (g,tests) )
    
    return grouping

def RPerf_diameter_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks):
    grouping = dbalance_2_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks, False)
    tests = list(range(matrix.shape[1]))
    np.random.shuffle(tests)
    
    for i, g in enumerate(grouping):
        g_ = (g[0],tests[:m])
        np.random.shuffle(tests)
        grouping[i] = g_
    
    return grouping

def RPerf_all_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks):
    grouping = abalance_solution(matrix, learner_profile, test_profile, mean_learners, k, m, test2ks, False)
    tests = list(range(matrix.shape[1]))
    np.random.shuffle(tests)
    
    for i, g in enumerate(grouping):
        g_ = (g[0],tests[:m])
        np.random.shuffle(tests)
        grouping[i] = g_
    
    return grouping

def get_pairwise(liste):
    s = 0
    comb = list(itertools.combinations(liste, 2))
    
    for i,j in comb:
        s += abs(i-j)
    
    return s

def diameter_lg_one(learners_profile, learners):
    tmp = learners_profile[learners,:]
    
    tmp_min = tmp.min(0)
    tmp_max = tmp.max(0)
        
    return (tmp_max-tmp_min).sum()

def all_lg_one(learners_profile, learners):
    
    tmp = learners_profile[learners,:]
    tmp_s = 0
    for skill in range(tmp.shape[1]):
        tmp_s += get_pairwise(list(tmp[:,skill]))

    return tmp_s

# Our Solutions
def diameter(learners, k):
    l = list(np.argsort(learners))[::-1]
    
    groupings = []
    chosen = []
    
    for i,j in zip(l[:k],l[-k:]):
        groupings.append((i,j))
        chosen.append(i)
        chosen.append(j)
    
    return groupings, chosen

def all_(learners, k):
    l = list(np.argsort(learners))[::-1]
    
    categories = dict()
    chosen = []
    
    for i in range(len(learners)):
        categories[l[i]] = i//k
    
    return categories, chosen

def first_solution_diameter(matrix, learner_profile, test_profile, chosen, group, max_users, max_items, test2ks, aptitude = True):
    
    nb_ite = int(matrix.shape[1]*0.1)
    
    chos = list(set(chosen) - set(group))
    free = list( set(range(matrix.shape[0])) - set(chos) )
    
    matrix_u = matrix.sum(1)
    matrix_u = np.argsort(matrix_u).tolist()[::-1]
    
    i = 0
    learners = []
    
    while len(learners) < max_users:
        if matrix_u[i] not in chos:
            learners.append(matrix_u[i])
        
        i += 1
    
    matrix_q = matrix[free,:].sum(0)
    matrix_q = np.argsort(matrix_q)
    tests = matrix_q.tolist()[-max_items:]
    
    missed = list(set(group) - set(learners))
    
    if len(missed) != 0:
        perf = matrix[learners,:][:,tests].sum(axis=1)
        perf = list(np.argsort(perf))
        
        j = 0
        i = 0
        
        while j < len(missed):
            idx_l = perf[i]
            
            if learners[idx_l] not in group:
                learners[idx_l] = missed[j]
                j += 1
            
            i += 1
    
    if aptitude:
        tests = change_aptitude(matrix, learner_profile, test_profile, learners, tests, test2ks, nb_ite)
        
    return learners, tests

def second_solution_diameter(matrix, learner_profile, test_profile, chosen, group, max_users, max_items, test2ks, aptitude = True):
    
    nb_ite = int(matrix.shape[1]*0.1)
    
    chos = list(set(chosen) - set(group))
    
    matrx_tem = matrix.copy()
    matrx_tem[chos,:] = -1

    idx_max = np.argmax(matrx_tem)
    l,t = idx_max//matrix.shape[1], idx_max%matrix.shape[1]
    
    learners = [l]
    tests = [t]
    
    while len(learners) + len(tests) < max_users+max_items:
        if len(tests) < max_items:
            matrix_q = matrix[learners,:].sum(0)
            matrix_q = np.argsort(matrix_q)[::-1]
            
            for i in matrix_q:
                if i not in tests:
                    tests.append(i)
                    break
            
        if len(learners) < max_users:
            matrix_l = matrix[:,tests].sum(1)
            matrix_l = np.argsort(matrix_l)[::-1]
            
            for i in matrix_l:
                if i not in learners and i not in chos:
                    learners.append(i)
                    break
    
    missed = list(set(group) - set(learners))
    
    if len(missed) != 0:
        perf = matrix[learners,:][:,tests].sum(axis=1)
        perf = list(np.argsort(perf))
        
        j = 0
        i = 0
        
        while j < len(missed):
            idx_l = perf[i]
            
            if learners[idx_l] not in group:
                learners[idx_l] = missed[j]
                j += 1
            
            i += 1
    
    if aptitude:
        tests = change_aptitude(matrix, learner_profile, test_profile, learners, tests, test2ks, nb_ite)
        
    return learners, tests

def first_solution_all(matrix, learner_profile, test_profile, chosen, categories, max_users, max_items, test2ks, aptitude = True):
    
    nb_ite = int(matrix.shape[1]*0.1)
    
    matrix_u = matrix.sum(1)
    matrix_u = np.argsort(matrix_u).tolist()[::-1]
    
    i = 0
    learners = []
    cat = []
    
    while len(learners) < max_users:
        idx = matrix_u[i]
        if idx not in chosen and categories[idx] not in cat:
            learners.append(idx)
            cat.append(categories[idx])
        
        i += 1
        
    free = list( set(range(matrix.shape[0])) - set(chosen) )
    
    matrix_q = matrix[free,:].sum(0)
    matrix_q = np.argsort(matrix_q)
    tests = matrix_q.tolist()[-max_items:]
    
    if aptitude:
        tests = change_aptitude(matrix, learner_profile, test_profile, learners, tests, test2ks, nb_ite)
        
    return learners, tests

def second_solution_all(matrix, learner_profile, test_profile, chosen, categories, max_users, max_items, test2ks, aptitude = True):
    
    nb_ite = int(matrix.shape[1]*0.1)
    
    matrx_tem = matrix.copy()
    matrx_tem[chosen,:] = -1

    idx_max = np.argmax(matrx_tem)
    l,t = idx_max//matrix.shape[1], idx_max%matrix.shape[1]
    
    learners = [l]
    tests = [t]
    cat = [categories[l]]
    
    while len(learners) + len(tests) < max_users+max_items:
        if len(tests) < max_items:
            matrix_q = matrix[learners,:].sum(0)
            matrix_q = np.argsort(matrix_q)[::-1]
            
            for i in matrix_q:
                if i not in tests:
                    tests.append(i)
                    break
            
        if len(learners) < max_users:
            matrix_l = matrix[:,tests].sum(1)
            matrix_l = np.argsort(matrix_l)[::-1]
            
            for i in matrix_l:
                if i not in learners and i not in chosen and categories[i] not in cat:
                    learners.append(i)
                    cat.append(categories[i])
                    break
    
    if aptitude:
        tests = change_aptitude(matrix, learner_profile, test_profile, learners, tests, test2ks, nb_ite)
        
    return learners, tests

def heuristic(perf, learner_profile, test_profile, mean_learners, k, m, test2ks, learning_group = 'diameter', 
              solution='first', aptitude=True):
        
    grouping = []

    r = len(mean_learners)//k
    nb_ite = int(perf.shape[1]*0.1)
    
    if learning_group == 'diameter':
        grp, chosen = diameter(mean_learners, k)
        
        if len(chosen) == perf.shape[0]:
            for i in range(k):
                us = grp[i]
                ts = get_best_tests(perf, grp[i], m)
                if aptitude:
                    ts = change_aptitude(perf, learner_profile, test_profile, us, ts, test2ks, nb_ite)

                grouping.append((us,ts))
        else:
            for i in range(k):
                if solution == 'first':
                    us, ts = first_solution_diameter(perf, learner_profile, test_profile, chosen, grp[i], r, m, test2ks, aptitude)
                elif solution == 'second':
                    us, ts = second_solution_diameter(perf, learner_profile, test_profile, chosen, grp[i], r, m, test2ks, aptitude)

                grouping.append((us,ts))
                chosen.extend(us)
        
    else:        
        grp, chosen = all_(mean_learners, k)
        
        for i in range(k):
            if solution == 'first':
                us, ts = first_solution_all(perf, learner_profile, test_profile, chosen, grp, r, m, test2ks, aptitude)
            elif solution == 'second':
                us, ts = second_solution_all(perf, learner_profile, test_profile, chosen, grp, r, m, test2ks, aptitude)

            grouping.append((us,ts))
            chosen.extend(us)
        
    
    return grouping

def get_new_pairwise(learner_profile, prev_learers, new_l, prev_gl):
    
    l_profile = learner_profile[[new_l],:]
    prev_profile = learner_profile[prev_learers,:]
    
    new_gl = prev_profile - l_profile
    new_gl[new_gl < 0] *= -1
    new_gl = new_gl.sum(0)
    
    new_gl = new_gl+prev_gl
    
    return new_gl

def antiClustering(perf, learner_profile, test_profile, k, m, test2ks, learning_group = 'diameter', 
                   init='variance', switch = False, aptitude=True):
    
    r = len(learner_profile)//k
    
    if init == 'variance':
        learners = learner_profile.std(1)
        learners = np.argsort(learners)[::-1]
    else:
        learners = list(range(len(learner_profile)))
        np.random.shuffle(learners)
    
    if learning_group == 'diameter':
        lg_func = diameter_lg_one
    else:
        lg_func = all_lg_one
    
    groups = []
    
    mins = []
    maxs = []
    
    sums = []
    
    nofilled = list(range(k))
    
    learners2groups = dict()
    
    for i in range(k):
        idx = learners[i]
        groups.append([idx])
        
        learners2groups[idx] = [i,0]
        
        if learning_group == 'diameter':
            mins.append(learner_profile[idx,:])
            maxs.append(learner_profile[idx,:])
        else:
            sums.append(learner_profile[idx,:]*0)
    
    learners = learners[k:]
    
    for l in learners:
        profile = learner_profile[l,:]
        
        groupLearn = -1
        bestGroup = -1
        
        bestmin = None
        bestmax = None
        
        bestsum = None
        
        for i in nofilled:
            if learning_group == 'diameter':
                mn = np.min([mins[i], profile], axis=0)
                mx = np.max([mins[i], profile], axis=0)

                gl = (mx-mn).sum()
            else:
                ss = get_new_pairwise(learner_profile, groups[i], l, sums[i])
                gl = ss.sum()
            
            if gl > groupLearn:
                groupLearn = gl
                bestGroup = i
                
                if learning_group == 'diameter':
                    bestmin = mn
                    bestmax = mx
                else:
                    bestsum = ss
        
        learners2groups[l] = [bestGroup, len(groups[bestGroup])]
        
        groups[bestGroup].append(l)
        
        if learning_group == 'diameter':
            mins[bestGroup] = bestmin
            maxs[bestGroup] = bestmax
        else:
            sums[bestGroup] = bestsum
        
        if len(groups[bestGroup]) == r:
            nofilled.remove(bestGroup)
                 
    if switch:
        liste = list(range(learner_profile.shape[0]))
        np.random.shuffle(liste)
        comb = list(itertools.combinations(liste,2))
        np.random.shuffle(comb)
        
        nb_permut = int(len(comb)*0.2)
        
        lgs = [lg_func(learner_profile, grp) for grp in groups]
        best_lg = sum(lgs)
        
        time = 0
        
        for l1, l2 in comb[:nb_permut]:
            i_grp1, idx1 = learners2groups[l1]
            i_grp2, idx2 = learners2groups[l2]

            if i_grp2 == i_grp1:
                continue
            
            grp1 = list(groups[i_grp1])
            grp2 = list(groups[i_grp2])
            
            grp1[idx1] = l2
            grp2[idx2] = l1
            
            tmp_lg = list(lgs)
            tmp_lg[i_grp1] = lg_func(learner_profile, grp1)
            tmp_lg[i_grp2] = lg_func(learner_profile, grp2)
            
            tmp_sum = sum(tmp_lg)
            
            if tmp_sum > best_lg:
                time += 1
                
                best_lg = tmp_sum
                lgs = tmp_lg
                
                groups[i_grp1] = grp1
                groups[i_grp2] = grp2
                
                learners2groups[l1] = [i_grp2, idx2]
                learners2groups[l2] = [i_grp1, idx1]
            
    grouping = []
    nb_ite = int(perf.shape[1]*0.1)

    for i, g in enumerate(groups):
        tests = get_best_tests(perf, g, m)

        if aptitude:
            tests = change_aptitude(perf, learner_profile, test_profile, g, tests, test2ks, nb_ite)

        grouping.append( (g,tests) )

    return grouping

