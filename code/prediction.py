import joblib
import os

from MachineLearningAlgorithm.utils import utils_read


def read_vectors(in_file):
    vectors = utils_read.FormatRead(in_file, 'csv').write_to_file()
    return vectors


def prob_predict(predicted_labels, prob_list, out_path):
    prob_file = out_path + "predict_out.txt"
    with open(prob_file, 'w') as f:
        head = 'Sample Index' + '\t' + 'predicted labels' + '\t' + 'probability values' + '\n'
        f.write(head)
        for i, (m, n) in enumerate(zip(predicted_labels, prob_list)):
            line = str(i + 1) + '\t' + str(m) + '\t' + str(n) + '\n'
            f.write(line)
    full_path = os.path.abspath(prob_file)
    if os.path.isfile(full_path):
        print('The output file for predict probability can be found:')
        print(full_path)
        print('\n')


def ml_prediction(vectors, out_dir, model_path):
    model = joblib.load(model_path)

    ind_prob = model.predict_proba(vectors)[:, 1]
    pre_labels = model.predict(vectors)
    prob_predict(pre_labels, ind_prob, out_dir)  # 将标签对应概率写入文件


def main(vec_file, model_path):
    current_path = os.path.dirname(os.path.realpath(__file__))
    out_path = current_path + '/results/prediction/'
    vectors = read_vectors(vec_file)

    ml_prediction(vectors, out_path, model_path)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='prediction', description="Prediction for unknown sequences ")
    parse.add_argument('-vec_file', required=True, help="The input feature vector file for prediction.")
    parse.add_argument('-model_path', required=True, help="The model file used for prediction.")

    argv = parse.parse_args()
    print("Prediction Begin!\n")

    main(argv.vec_file, argv.model_path)
