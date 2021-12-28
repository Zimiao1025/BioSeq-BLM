from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

from ..utils.utils_results import performance, final_results_output, prob_output, print_metric_dict
from ..utils.utils_plot import plot_roc_curve, plot_pr_curve, plot_roc_ind, plot_pr_ind
from ..utils.utils_math import sampling
from ..utils.utils_read import FormatRead

Metric_List = ['Acc', 'MCC', 'AUC', 'BAcc', 'Sn', 'Sp', 'Pr', 'Rc', 'F1']


def ml_cv_process(ml, vectors, labels, folds, metric, sp, multi, res, params_dict):
    results = []

    print_len = 40
    if ml == 'SVM':
        temp_str1 = '  cost = 2 ** ' + str(params_dict['cost']) + ' | ' + 'gamma = 2 ** ' + \
                    str(params_dict['gamma']) + '  '
    else:
        temp_str1 = '  tree = ' + str(params_dict['tree']) + '  '
    print(temp_str1.center(print_len, '+'))

    for train_index, val_index in folds:
        x_train, y_train, x_val, y_val = get_partition(vectors, labels, train_index, val_index)
        if sp != 'none':
            x_train, y_train = sampling(sp, x_train, y_train)
        if ml == 'SVM':
            clf = svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
        else:
            clf = RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
        clf.fit(x_train, y_train)
        y_val_prob = clf.predict_proba(x_val)[:, 1]  # 这里为什么是1呢

        y_val_ = clf.predict(x_val)

        result = performance(y_val, y_val_, y_val_prob, multi, res)
        # acc, mcc, auc, balance_acc, sn, sp, p, r, f1
        results.append(result)

    cv_results = np.array(results).mean(axis=0)

    params_dict['metric'] = cv_results[metric]
    temp_str2 = '  metric value: ' + Metric_List[metric] + ' = ' + '%.3f  ' % cv_results[metric]
    print(temp_str2.center(print_len, '*'))
    print('\n')
    return params_dict


def get_partition(vectors, labels, train_index, val_index):
    x_train = vectors[train_index]
    x_val = vectors[val_index]
    y_train = labels[train_index]
    y_val = labels[val_index]

    return x_train, y_train, x_val, y_val


def ml_cv_results(ml, vectors, labels, folds, sp, multi, res, out_dir, params_dict):
    results = []

    print_len = 60
    print('\n')
    if ml == 'SVM':
        print('  The optimal parameters for SVM are as follows  '.center(print_len, '*'))
        temp_str1 = '    cost = 2 ** ' + str(params_dict['cost']) + ' | ' + 'gamma = 2 ** ' + \
                    str(params_dict['gamma']) + '    '
    else:
        print('The optimal parameters for RF is as follows'.center(print_len, '*'))
        temp_str1 = '    tree = ' + str(params_dict['tree']) + '    '
    print(temp_str1.center(print_len, '*'))
    print('\n')

    cv_labels = []
    cv_prob = []
    predicted_labels = np.zeros(len(labels))
    predicted_prob = np.zeros(len(labels))
    for train_index, test_index in folds:
        x_train, y_train, x_test, y_test = get_partition(vectors, labels, train_index, test_index)
        if sp != 'none':
            x_train, y_train = sampling(sp, x_train, y_train)
        if ml == 'SVM':
            clf = svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
        else:
            clf = RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
        clf.fit(x_train, y_train)
        y_test_prob = clf.predict_proba(x_test)[:, 1]
        y_test_ = clf.predict(x_test)

        result = performance(y_test, y_test_, y_test_prob, multi, res)
        # acc, mcc, auc, balance_acc, sn, sp, p, r, f1
        results.append(result)
        cv_labels.append(y_test)
        cv_prob.append(y_test_prob)
        predicted_labels[test_index] = y_test_
        predicted_prob[test_index] = y_test_prob
    plot_roc_curve(cv_labels, cv_prob, out_dir)  # 绘制ROC曲线
    plot_pr_curve(cv_labels, cv_prob, out_dir)  # 绘制PR曲线

    final_results = np.array(results).mean(axis=0)
    print_metric_dict(final_results, ind=False)
    print('\n')

    final_results_output(final_results, out_dir, ind=False, multi=multi)  # 将指标写入文件
    prob_output(labels, predicted_labels, predicted_prob, out_dir)  # 将标签对应概率写入文件
    # 利用整个数据集训练并保存模型
    if ml == 'SVM':
        model = svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
        model_path = out_dir + 'cost_[' + str(params_dict['cost']) + ']_gamma_[' + str(
            params_dict['gamma']) + ']_svm.model'
    else:
        model = RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
        model_path = out_dir + 'tree_' + str(params_dict['tree']) + '_rf.model'
    if sp != 'none':
        vectors, labels = sampling(sp, vectors, labels)
    model.fit(vectors, labels)

    joblib.dump(model, model_path)  # 使用job lib保存模型


def ml_ind_results(ml, ind_vectors, ind_labels, multi, res, out_dir, params_dict):
    if ml == 'SVM':
        model_path = out_dir + 'cost_[' + str(params_dict['cost']) + ']_gamma_[' + str(
            params_dict['gamma']) + ']_svm.model'
    else:
        model_path = out_dir + 'tree_' + str(params_dict['tree']) + '_rf.model'

    model = joblib.load(model_path)

    ind_prob = model.predict_proba(ind_vectors)[:, 1]
    pre_labels = model.predict(ind_vectors)

    final_result = performance(ind_labels, pre_labels, ind_prob, multi, res)

    print_metric_dict(final_result, ind=True)

    plot_roc_ind(ind_labels, ind_prob, out_dir)  # 绘制ROC曲线
    plot_pr_ind(ind_labels, ind_prob, out_dir)  # 绘制PR曲线

    final_results_output(final_result, out_dir, ind=True, multi=multi)  # 将指标写入文件
    prob_output(ind_labels, pre_labels, ind_prob, out_dir, ind=True)  # 将标签对应概率写入文件


def ml_score_cv_process(ml, vec_files, folds_num, metric, sp, multi, in_format, params_dict):
    dir_name, _ = os.path.splitext(vec_files[0])
    score_dir = dir_name + '/score/'

    print('\n')
    print('Cross Validation Processing...')
    print('\n')

    print_len = 40
    if ml == 'SVM':
        temp_str1 = '  cost = 2 ** ' + str(params_dict['cost']) + ' | ' + 'gamma = 2 ** ' + \
                    str(params_dict['gamma']) + '  '
    else:
        temp_str1 = '  tree = ' + str(params_dict['tree']) + '  '
    print(temp_str1.center(print_len, '+'))

    results = []
    for i in range(folds_num):
        tar_dir = score_dir + 'Fold%d/' % (i+1)
        x_train = FormatRead(tar_dir + 'train_score.txt', in_format).write_to_file()
        x_val = FormatRead(tar_dir + 'test_score.txt', in_format).write_to_file()
        y_train = np.loadtxt(tar_dir + 'train_label.txt')
        y_train = y_train.astype(int)
        y_val = np.loadtxt(tar_dir + 'test_label.txt')
        y_val = y_val.astype(int)
        if sp != 'none':
            x_train, y_train = sampling(sp, x_train, y_train)
        if ml == 'SVM':
            clf = svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
        else:
            clf = RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
        clf.fit(x_train, y_train)
        y_val_prob = clf.predict_proba(x_val)[:, 1]  # 这里为什么是1呢

        y_val_ = clf.predict(x_val)

        result = performance(y_val, y_val_, y_val_prob, multi)
        # acc, mcc, auc, balance_acc, sn, sp, p, r, f1
        results.append(result)

    cv_results = np.array(results).mean(axis=0)
    params_dict['metric'] = cv_results[metric]

    temp_str2 = '  metric value: ' + Metric_List[metric] + ' = ' + '%.3f  ' % cv_results[metric]
    print(temp_str2.center(print_len, '*'))
    print('\n')
    return params_dict


def ml_score_cv_results(ml, vec_files, labels, folds_num, sp, multi, in_format, out_dir, params_dict):
    dir_name, _ = os.path.splitext(vec_files[0])
    score_dir = dir_name + '/score/'

    results = []

    cv_labels = []
    cv_prob = []
    predicted_labels = np.zeros(len(labels))
    predicted_prob = np.zeros(len(labels))
    for i in range(folds_num):
        tar_dir = score_dir + 'Fold%d/' % (i + 1)
        x_train = FormatRead(tar_dir + 'train_score.txt', in_format).write_to_file()
        x_test = FormatRead(tar_dir + 'test_score.txt', in_format).write_to_file()
        # 这里为什么loadtxt时设定dtype在Linux系统上不行呢？
        y_train = np.loadtxt(tar_dir + 'train_label.txt')
        y_train = y_train.astype(int)
        y_test = np.loadtxt(tar_dir + 'test_label.txt')
        y_test = y_test.astype(int)
        test_index = np.loadtxt(tar_dir + 'test_index.txt')
        test_index = list(test_index.astype(int))
        # test_index = np.array(test_index, dtype=int)
        if sp != 'none':
            x_train, y_train = sampling(sp, x_train, y_train)
        if ml == 'SVM':
            clf = svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
        else:
            clf = RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
        clf.fit(x_train, y_train)
        y_test_prob = clf.predict_proba(x_test)[:, 1]
        y_test_ = clf.predict(x_test)

        result = performance(y_test, y_test_, y_test_prob, multi)
        # acc, mcc, auc, balance_acc, sn, sp, p, r, f1
        results.append(result)
        cv_labels.append(y_test)
        cv_prob.append(y_test_prob)
        predicted_labels[test_index] = y_test_
        predicted_prob[test_index] = y_test_prob
    plot_roc_curve(cv_labels, cv_prob, out_dir)  # 绘制ROC曲线
    plot_pr_curve(cv_labels, cv_prob, out_dir)  # 绘制PR曲线

    final_results = np.array(results).mean(axis=0)
    print_metric_dict(final_results, ind=False)

    final_results_output(final_results, out_dir, ind=False, multi=multi)  # 将指标写入文件
    prob_output(labels, predicted_labels, predicted_prob, out_dir)  # 将标签对应概率写入文件


def ml_score_ind_results(ml, ind_vec_file, sp, multi, in_format, out_dir, params_dict):
    dir_name, _ = os.path.split(ind_vec_file)
    tar_dir = dir_name + '/ind_score/'
    x_train = FormatRead(tar_dir + 'train_score.txt', in_format).write_to_file()
    x_test = FormatRead(tar_dir + 'test_score.txt', in_format).write_to_file()
    y_train = np.loadtxt(tar_dir + 'train_label.txt')
    y_train = y_train.astype(int)
    y_test = np.loadtxt(tar_dir + 'test_label.txt')
    y_test = y_test.astype(int)
    if sp != 'none':
        x_train, y_train = sampling(sp, x_train, y_train)

    if ml == 'SVM':
        model_path = out_dir + 'cost_[' + str(params_dict['cost']) + ']_gamma_[' + str(
            params_dict['gamma']) + ']_svm_score.model'
        clf = svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
    else:
        clf = RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
        model_path = out_dir + 'tree_' + str(params_dict['tree']) + '_rf_score.model'

    clf.fit(x_train, y_train)
    ind_prob = clf.predict_proba(x_test)[:, 1]
    pre_labels = clf.predict(x_test)

    final_result = performance(y_test, pre_labels, ind_prob, multi)

    print_metric_dict(final_result, ind=True)

    plot_roc_ind(y_test, ind_prob, out_dir)  # 绘制ROC曲线
    plot_pr_ind(y_test, ind_prob, out_dir)  # 绘制PR曲线

    final_results_output(final_result, out_dir, ind=True, multi=multi)  # 将指标写入文件
    prob_output(y_test, pre_labels, ind_prob, out_dir, ind=True)  # 将标签对应概率写入文件

    joblib.dump(clf, model_path)  # 使用job lib保存模型
