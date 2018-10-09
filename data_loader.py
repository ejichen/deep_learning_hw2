import numpy as np
import torch.utils.data


class load_data(torch.utils.data.Dataset):
    def __init__(self, features, speakers, nspeakers, seg_length):
        self.features = features
        self.speakers = speakers
        self.nspeakers = nspeakers
        self.seg_length = seg_length
        self.pad_features = np.empty(features.shape)
        self.validIndex = []
        self.speaker_table = []
        self.instances = []
        self.total_instances_count = 0

    def __len__(self):
        return len(self.validIndex)

    def __call__(self, *args, **kwargs):
        # pad the instances in data so that it could be sliced into chunks,
        # pad every instances so that they could be sliced equally
        # maybe pad with symmetric first?
        feature_max_len = max(self.features, key=len).shape[0]
        padded_len = (feature_max_len % self.seg_length) + feature_max_len
        # seg_count id the slice count for each utterance, seg_length is the sample length from one utterance
        self.seg_count = padded_len / self.seg_length
        for i in range(self.features.shape[0]):
            self.pad_features[i] = np.pad(self.features[i], ((0, padded_len - self.features[i].shape[0]),
                                                             (0, 0)), 'symmetric')
        # record the speaker index for each of the sample slice
        for n in range(self.speakers.shape[0]):
            self.speaker_table[n * self.seg_count : (n+1) * self.seg_count] = self.speakers[n] * self.seg_count

        # record the instances for each of the sample slice (should be the same length for every utterance now)
        for i in range(self.features.shape[0]):
            self.instances[i * self.seg_count : (i+1) * self.seg_count] = list(range(self.seg_count))

        # every utterance has fixed segment count
        self.total_instances_count = self.seg_count * self.features.shape[0]

    def __getitem__(self, idx):
        label = self.speakers[idx]
        instance = self.instances[idx]
        feature = self.features[label][instance * self.seg_count : (instance+1) * self.seg_count]
        return label, feature
