import numpy as np
from hmmlearn import hmm
from data_loader import *
import warnings
from Util import *
import argparse
from MLP import MLP
from hmm_dnn import *
from plot_conf_mat import plot_confusion_matrix
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description='This is a script used to run the tests')
parser.add_argument('-ldmp', '--load-mapper', action='store_true', help='load from saved mapper')

args = parser.parse_args()

percent = 0.01

states_count = 5
mixture_count = 3
iteration_count = 50

words_count = 10

training_dataset, training_length, test_dataset, test_length = dataset_loader()

feature_size = len(training_dataset[0][0])

train_block_size = 660
test_block_size = 220

train_dataset_list = list()
train_len_list = list()
train_dataset_label = list()

test_dataset_list = list()
test_len_list = list()
test_dataset_label = list()

# data set arrangement
#############################################################################
for i in range(words_count):
    train_dataset_list.append(
        training_dataset[i * train_block_size: i * train_block_size + int(train_block_size * percent)])
    train_len_list.append(training_length[i * train_block_size: i * train_block_size + int(train_block_size * percent)])
    train_dataset_label = [label for label in range(10) for _ in range(int(train_block_size * percent))]

    test_dataset_list.append(test_dataset[i * test_block_size: i * test_block_size + int(test_block_size * percent)])
    test_len_list.append(test_length[i * test_block_size: i * test_block_size + int(test_block_size * percent)])
    test_dataset_label = [label for label in range(10) for _ in range(int(test_block_size * percent))]

# normalize data
#############################################################################
temp_train = [np.zeros((0, 13)) for _ in range(words_count)]
temp_test = [np.zeros((0, 13)) for _ in range(words_count)]

for i in range(words_count):

    temp_data = list()
    for j in range(len(train_dataset_list[i])):
        temp_data.append(np.array(train_dataset_list[i][j]))

    train_dataset_list[i] = list()
    for j in range(len(temp_data)):
        train_dataset_list[i].append(
            ((temp_data[j] - np.mean(temp_data[j], axis=0)) / np.std(temp_data[j], axis=0)).tolist())

    temp_data = list()
    for j in range(len(test_dataset_list[i])):
        temp_data.append(np.array(test_dataset_list[i][j]))

    test_dataset_list[i] = list()
    for j in range(len(temp_data)):
        test_dataset_list[i].append((temp_data[j] - np.mean(temp_data[j], axis=0)) / np.std(temp_data[j], axis=0))

# train gmm hmm modules
#############################################################################
print('---- training gmm hmm ')
seq_mapper = list()
gmmhmm_module_list = list()

if not args.load_mapper:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i in range(words_count):
            gmmhmm_module_list.append(hmm.GMMHMM(n_components=states_count, n_mix=mixture_count,
                                                 covariance_type='diag', n_iter=iteration_count))

        for i, module in enumerate(gmmhmm_module_list):
            module.fit(np.concatenate(train_dataset_list[i]), train_len_list[i])

        # data labeling
        #############################################################################
        for i, module in enumerate(gmmhmm_module_list):
            for data in train_dataset_list[i]:
                prob, path = module.decode(np.array(data))
                seq_mapper.append((i, data, path))

        save_list(seq_mapper, 'path.dict')
else:
    seq_mapper = load_list('path.dict')

# language model estimation
#############################################################################
print('---- builing language model')
phonetic_train_data = [np.zeros((0, 13))] * words_count
phonetic_train_label = [np.zeros((0, 1))] * words_count

language_model = [np.zeros(states_count) for _ in range(words_count)]

for label, data, seq in seq_mapper:
    phonetic_train_data[label] = np.vstack([phonetic_train_data[label], np.array(data)])
    phonetic_train_label[label] = np.vstack([phonetic_train_label[label], np.array(seq).reshape(-1, 1)])

    for i, seq_sample in enumerate(seq):
        language_model[label][seq_sample] += 1

observation_count = 0
for i in range(words_count):
    language_model[i] = language_model[i] / np.sum(language_model[i])
    print('lang model: ', language_model[i])
    observation_count += phonetic_train_data[i].shape[0]

# train DNN network
#############################################################################
print('---- training DNN network')
dnn_module_list = list()
for i in range(words_count):
    dnn_module_list.append(MLP(feature_size, states_count))

for i, module in enumerate(dnn_module_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        module.train(phonetic_train_data[i], phonetic_train_label[i])

# if True:
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#
#         acc_mat = np.zeros((words_count, words_count))
#         for i, module in enumerate(dnn_module_list):
#             for j in range(words_count):
#                 acc = 0
#                 tt = 0
#                 pred = module.predict(phonetic_train_data[j])
#                 for k, label in enumerate(pred):
#                     acc += phonetic_train_label[j][k] == label
#                     tt += 1
#                 acc_mat[i, j] = acc / tt
#
#         print('acc_mat : ', acc_mat)

# create hmm dnn modules
#############################################################################
hmm_dnn_module_list = list()
for i in range(words_count):
    hmm_dnn_module_list.append(
        hmm_dnn(dnn_module_list[i], aucoustic_model=language_model[i], observation_count=observation_count,
                n_components=states_count,
                startprob_prior=gmmhmm_module_list[i].startprob_, transmat_prior=gmmhmm_module_list[i].transmat_,
                n_iter=iteration_count))

# train hmm dnn modules
#############################################################################
print('---- training HMM-DNN')
start_train_time = time.time()
for i, module in enumerate(hmm_dnn_module_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        module.fit(np.concatenate(train_dataset_list[i]), train_len_list[i])
print("Train Time: ", time.time() - start_train_time)


def test(dataset_list, dataset_label, module_list):
    predicted_label_list = list()

    print('---- Running Test')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        score_list = [0 for _ in range(words_count)]

        for j, word_data_set in enumerate(dataset_list):
            for data in word_data_set:
                for i, module in enumerate(module_list):
                    score_list[i], _ = module.decode(np.array(data))

                predicted_label_list.append(np.argmax(np.array(score_list)))

        plot_confusion_matrix(dataset_label, predicted_label_list, range(10))


test(train_dataset_list, train_dataset_label, hmm_dnn_module_list)
test(test_dataset_list, test_dataset_label, hmm_dnn_module_list)

plt.show()
