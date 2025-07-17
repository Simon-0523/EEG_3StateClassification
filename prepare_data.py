import _pickle as cPickle

from train_model import *
import random



class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path

    def run(self, subject_list, split=False, expand=True, feature=False):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        feature: (bool) whether to extract features or not

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)

            if split:
                data_, label_ = self.split(
                    data=data_, label=label_, segment_length=self.args.segment,
                    overlap=self.args.overlap, sampling_rate=self.args.sampling_rate)
                # data_2, label_2 = self.split_random(
                #     data=data_, label=label_, segment_length=self.args.segment,
                #     num_segment=self.args.num_segment, sampling_rate=self.args.sampling_rate)
            # data_ = np.concatenate((data_1, data_2), axis=1)
            # label_ = np.concatenate((label_1, label_2), axis=1)

            if expand:
                # expand one dimension for deep learning(CNNs)
                data_ = np.expand_dims(data_, axis=-3)
            print('Data and label prepared for sub{}!'.format(sub))
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        sub += 1
        if (sub < 10):
            sub_code = str('s0' + str(sub) + '.dat')
        else:
            sub_code = str('s' + str(sub) + '.dat')

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['label']
        data = subject['data'][:, 0:4,:]
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label



    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}'.format(self.args.data_format, self.args.dataset)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def split_random(self, data, label, segment_length=1, num_segment=60, sampling_rate=256):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int(num_segment)
        random_numbers = [random.uniform(0, 57-segment_length) for _ in range(number_segment)]
        for i in range(number_segment):
            data_split.append(data[:, :, int(random_numbers[i]*sampling_rate):int(random_numbers[i]*sampling_rate + data_segment)])
        data_split_array = np.stack(data_split, axis=1)#.astype('float32')
        label = np.stack([np.repeat(label[i:i+1], int(number_segment), axis=0) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label


    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=256):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[2] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i:i+1], int(number_segment + 1), axis = 0) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label
