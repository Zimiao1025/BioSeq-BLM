import os
import sys


def get_metric(results_file_name):
    results_dict = {}
    with open(results_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[1: 10]:
            line = line.strip()
            metric = line.split('=')[0].split()[0]
            value = line.split('=')[1].split()[0]
            results_dict[metric] = value
    print("results_dict: ", results_dict)
    return results_dict


def get_params(params_file_name):
    params_dict = {}
    with open(params_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if len(line) != 0:
                param = line.split('=')[0].split()[0]
                value = line.split('=')[1].split()[0]
                params_dict[param] = value
    print("params_dict: ", params_dict)
    return params_dict


def dict2str(pa_dict):
    pa_str = ""
    for k, v in pa_dict.items():
        temp_str = str(k) + "=" + str(v) + "|"
        pa_str += temp_str
    return pa_str


def dict2csv(data_dict, out_path):
    file_name = out_path + "data.csv"
    header = ['BLMs', 'Acc', 'MCC', 'AUC', 'BAcc', 'Sn', 'Sp', 'Precision', 'Recall', 'F1', 'Params']
    with open(file_name, 'w') as f:
        for it in header:
            f.write(it)
            f.write(',')
        f.write('\n')
        for key, val_list in data_dict.items():
            f.write(key)
            f.write(',')
            for val in val_list:
                f.write(str(val))
                f.write(',')
            f.write('\n')


# 遍历文件夹
def walkFile(source_path, target_path):
    data_dict = {}
    for root, dirs, files in os.walk(source_path):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件

        for f in files:
            # print(f)
            if f == "final_results.txt":
                print(os.path.join(root, f))
                re_path = os.path.join(root, f)
                re_dict = get_metric(re_path)
                tag = str(re_path.split('/')[6]) + '_' + str(re_path.split('/')[5])
                re_list = list(re_dict.values())

                if "Opt_params.txt" in files:
                    pa_path = os.path.join(root, "Opt_params.txt")
                    pa_dict = get_params(pa_path)
                    pa_str = dict2str(pa_dict)
                    print(pa_str)
                    re_list.append(pa_str)
                else:
                    re_list.append("NULL")

                data_dict[tag] = re_list
    dict2csv(data_dict, target_path)


def results_dir_check(results_dir):
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
            print('results_dir:', results_dir)
        except OSError:
            pass


def main():
    t = sys.argv[1]
    source_path = '../results/batch/Seq/' + str(t) + '/'
    target_path = '../results/extract_data/' + str(t) + '/'
    results_dir_check(target_path)
    walkFile(source_path, target_path)


if __name__ == '__main__':
    main()
