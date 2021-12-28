import os
import shutil
import sys


# 遍历文件夹
def walkFile(source_path, target_path):
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
                new_file_name = target_path + '_'.join(re_path.split('/')[5:-1]) + '_' + f
                print(new_file_name)
                shutil.copy(re_path, new_file_name)
            
            if f == "prob_out.txt":
                print(os.path.join(root, f))
                re_path = os.path.join(root, f)
                new_file_name = target_path + '_'.join(re_path.split('/')[5:-1]) + '_' + f
                print(new_file_name)
                shutil.copy(re_path, new_file_name)


def main():
    t = sys.argv[1]
    source_path = '../results/batch/Seq/' + str(t) + '/'
    target_path = '../results/target/'
    walkFile(source_path, target_path)


if __name__ == '__main__':
    main()
