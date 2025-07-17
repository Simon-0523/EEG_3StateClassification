import numpy as np
import datetime
import os
import csv
import h5py
import copy
import os.path as osp
from train_model import *
from utils import Averager
from sklearn.model_selection import KFold

ROOT = os.getcwd()


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        self.log_file = "results.txt"
        file = open(self.log_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)num_epochs:" + str(args.max_epoch) +
                   "\n5)batch_size:" + str(args.batch_size) +
                   "\n6)dropout:" + str(args.dropout) +
                   "\n7)hidden_node:" + str(args.hidden) +
                   "\n8)input_shape:" + str(args.input_shape) +
                   "\n9)T:" + str(args.T) + '\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        :param sub: which subject's data to load
        :return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def prepare_data(self, idx_train, idx_test, data, label):
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        :param idx_train: index of training data
        :param idx_test: index of testing data
        :param data: (trial, segments, 1, channel, data)
        :param label: (trial, segments,)
        :return: data and label
        """
        np.random.shuffle(idx_test)
        data_train = data[idx_train]
        label_train = label[idx_train,0]
        data_test = data[idx_test]
        label_test = label[idx_test,0]

        # data_train = np.concatenate(data_train, axis=0)#按试次进行划分
        # label_train = np.concatenate(label_train, axis=0)

        # the testing data do not need to be concatenated, when doing leave-one-trial-out
        if len(data_test.shape)>4:
            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)

        # data_train, data_test = self.normalize(train=data_train, test=data_test)

        # Prepare the data format for training the model
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data
        :param test: testing data
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data
        mean = 0
        std = 0
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]
        index_2 = np.where(label == 2)[0]
        # index_3 = np.where(label == 3)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        # for class 2
        index_random_2 = copy.deepcopy(index_2)

        # for class 3
        # index_random_3 = copy.deepcopy(index_3)

        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)
            np.random.shuffle(index_random_2)
            # np.random.shuffle(index_random_3)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)],
                                      index_random_2[:int(len(index_random_2) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):],
                                    index_random_2[int(len(index_random_2) * train_rate):]),
                                   axis=0)
        # if random == True:
        #     np.random.shuffle(index_train)
        #     np.random.shuffle(index_val)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    def n_fold_CV(self, subject=[], fold=10, reproduce=False):
        """
        this function achieves n-fold cross-validation
        :param subject: how many subject to load
        :param fold: how many fold
        """
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1

        tta_trial = []   # for trial-wise evaluation
        ttf_trial = []   # for trial-wise evaluation

        for sub in subject:
            data, label = self.load_per_subject(sub)

            data = np.concatenate(data, axis=0)#不按实验试次进行划分，直接按照数据进行划分
            label = np.concatenate(label, axis=0)

            va = []
            va_val = Averager()
            f1_t = []
            preds, acts = [], []
            preds_trial, acts_trial = [], []
            kf = KFold(n_splits=fold, shuffle=True)
            # data: (trial, segment, 1, chan, length) here the KFold is trial-wise
            for idx_fold, (index_train, index_test) in enumerate(kf.split(data)):

                data_train, label_train, data_test, label_test = self.prepare_data(
                    idx_train=index_train, idx_test=index_test, data=data, label=label)

                data_train, label_train, data_val, label_val = self.split_balance_class(
                    data=data_train, label=label_train, train_rate=0.8, random=True
                )
                if reproduce:
                    # to reproduce the reported ACC
                    acc_test, _, _, f1_test = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)
                    acc_val = 0
                else:
                    # to train new models
                    # Check the dimension of the training, validation and test set
                    print(Colors.RESET + 'Training:', data_train.size(), label_train.size())
                    print(Colors.RESET + 'Validation:', data_val.size(), label_val.size())
                    print(Colors.RESET + 'Test:', data_test.size(), label_test.size())

                    acc_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=sub,
                                    fold=idx_fold)
                    # test the model on testing data
                    acc_test, _, _, f1_test = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=self.args.reproduce,
                                               subject=sub, fold=idx_fold)

                va_val.add(acc_val)
                va.append(acc_test)
                f1_t.append(f1_test)

            tva.append(va_val.item())

        # prepare final report
        mACC = np.mean(va)
        mF1 = np.mean(f1_t)
        std_ACC = np.std(va)
        std_F1 = np.std(f1_t)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)


        print(Colors.RESET + 'Final: test mean ACC:{} std:{}'.format(mACC, std_ACC))
        print(Colors.RESET + 'Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        results = 'test mAcc={} std={} mF1={} std={}'.format(mACC, std_ACC, mF1, std_F1)
        self.log2txt(results)

    def log2txt(self, content):
        """
        this function log the content to results.txt
        :param content: string, the content to log
        """
        file = open(self.log_file, 'a')
        file.write(str(content) + '\n')
        file.close()


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
