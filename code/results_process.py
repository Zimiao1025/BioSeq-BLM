import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def get_metric(results_file_name):
    results_dict = {}
    with open(results_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[1: 10]:
            # print(line)
            metric = line.split('=')[0].split()[0]
            value = float(line.split('=')[1])
            results_dict[metric] = value
    return results_dict


def plot_roc(true_labels, predict_prob_list, tags, figure_name):
    plt.figure(0)
    count = 0
    color_list = ['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray',
                  'green', 'dodgerblue', 'gold', 'lightcoral']
    # color_list = ['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray',
    #               'green', 'dodgerblue', 'gold', 'lightcoral', 'r', 'k', 'b', 'g', 'y']
    for predict_prob in predict_prob_list:
        fpr, tpr, thresholds = roc_curve(true_labels, predict_prob)
        try:
            auc_val = auc(fpr, tpr)
        except ZeroDivisionError:
            auc_val = 0.0
        annotation = tags[count] + ' (AUC = %0.3f)' % auc_val
        plt.plot(fpr, tpr, lw=2, alpha=0.7, color=color_list[count],
                 label=annotation)
        count += 1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Receiver Operating Characteristic', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc="lower right")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)


def plot_pr(true_labels, predict_prob_list, tags, figure_name):
    plt.figure(0)
    count = 0
    color_list = ['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray',
                  'green', 'dodgerblue', 'gold', 'lightcoral']
    # color_list = ['crimson', 'navy', 'teal', 'darkorange', 'purple', 'gray',
    #               'green', 'dodgerblue', 'gold', 'lightcoral', 'r', 'k', 'b', 'g', 'y']
    for predict_prob in predict_prob_list:
        precision, recall, _ = precision_recall_curve(true_labels, predict_prob)
        try:
            auc_val = auc(recall, precision)
        except ZeroDivisionError:
            auc_val = 0.0
        annotation = tags[count] + ' (AUPR = %0.3f)' % auc_val
        plt.plot(recall, precision, lw=2, alpha=0.7, color=color_list[count],
                 label=annotation)
        count += 1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.legend(loc="lower left")
    ax_width = 1
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(ax_width)
    ax.spines['left'].set_linewidth(ax_width)
    ax.spines['top'].set_linewidth(ax_width)
    ax.spines['right'].set_linewidth(ax_width)

    plt.savefig(figure_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.close(0)


def traverse_folder(path_name, metric="AUC"):
    if os.path.exists(path_name):
        file_list = os.listdir(path_name)
        # print(file_list)
        # exit()
        metric_dict = {}
        for f in file_list:
            # print(f)
            # 将文件名转换为list
            name_list = f.split('_')
            tag = '_'.join(name_list[:2])
            # print(tag)
            if name_list[-1] == "results.txt":
                re_path = os.path.join(path_name, f)
                results_dict = get_metric(re_path)
                # print(results_dict)
                # exit()
                metric_val = results_dict[metric]
                metric_dict[tag] = metric_val
        # 将字典根据val进行从大到小排序，然后输出为元组
        metric_tup = sorted(metric_dict.items(), key=lambda d: d[1], reverse=True)
        return metric_tup
    else:
        return False


def extract_data(path_name, tags):
    true_labels = []
    predict_prob_list = []
    flag = 0
    for tag in tags:
        flag += 1
        predict_prob = []
        prob_file_name = tag + "_" + "prob_out.txt"
        re_path = os.path.join(path_name, prob_file_name)
        with open(re_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                if len(lines[i].strip()) != 0:
                    content = lines[i].split('\t')
                    if flag == 1:
                        true_labels.append(int(content[1]))
                    predict_prob.append(float(content[3]))
        predict_prob_list.append(predict_prob)
    return true_labels, predict_prob_list


def make_tag(tup_list, selected_num=10):
    tags = []
    new_tags = []
    for tup in tup_list[: selected_num]:
        tag = tup[0]
        tags.append(tag)
        tag_list = tag.split('_')
        if tag_list[0] in ['SR', 'WP']:
            new_tag = tag_list[1]
        elif tag_list[0] in ['TR']:
            new_tag = tag_list[1] + '-' + 'TextRank'
        else:
            new_tag = tag_list[1] + '-' + tag_list[0]
        new_tags.append(new_tag)
    return tags, new_tags


def main():
    target_path = "../results/target_rna2/"
    metric_tup = traverse_folder(target_path, "AUC")
    tags, new_tags = make_tag(metric_tup, 10)

    true_labels, predict_prob_list = extract_data(target_path, tags)
    print(len(true_labels))
    print(len(predict_prob_list))
    # exit()

    fig_path = "../results/figure/"
    plot_roc(true_labels, predict_prob_list, new_tags, fig_path + "rna2_roc.png")
    plot_pr(true_labels, predict_prob_list, new_tags, fig_path + "rna2_pr.png")


if __name__ == '__main__':
    main()
