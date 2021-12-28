from .bow_topic_model import bow_tm
from .TF_IDF_topic_model import tf_idf_tm
from .TextRank_topic_model import text_rank_tm
from ..utils.utils_write import vectors2files


def topic_model(from_vec, tm_method, input_file, labels, category, words, fixed_len, sample_num_list, out_format,
                out_file_list, cur_dir, **param_dict):
    if from_vec == 'BOW':
        tm_vectors = bow_tm(tm_method, input_file, labels, category, words, sample_num_list, out_format, out_file_list,
                            cur_dir, **param_dict)
    elif from_vec == 'TF-IDF':
        tm_vectors = tf_idf_tm(tm_method, input_file, labels, category, words, fixed_len, sample_num_list, out_format,
                               out_file_list, cur_dir, **param_dict)
    elif from_vec == 'TextRank':
        tm_vectors = text_rank_tm(tm_method, input_file, labels, category, words, fixed_len, sample_num_list,
                                  out_format, out_file_list, cur_dir, **param_dict)
    else:
        print('The input data type of topic model is wrong, please check!')
        return False
    vectors2files(tm_vectors, sample_num_list, out_format, out_file_list)
