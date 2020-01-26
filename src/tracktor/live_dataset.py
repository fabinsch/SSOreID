import torch
from torch import randperm
from torch.utils.data import Subset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id):
        self.id = id
        self.features = torch.tensor([]).to(device)
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.frame_numbers = None
        self.number_of_positive_examples = None

    def append_samples(self, training_set_dict):
        self.features = torch.cat((self.features, training_set_dict['features']))
        self.boxes = torch.cat((self.boxes, training_set_dict['boxes']))
        self.scores = torch.cat((self.scores, training_set_dict['scores']))

    # Filter out all duplicates and add frame number tensor for each data point
    NUMBER_OF_POSITIVE_EXAMPLE_DUPLICATES = 31
    def post_process(self):
        unique_indices = []
        frame_numbers = []
        number_of_duplicates = 0
        frame_number = 0
        current_box = self.boxes[0, :]
        for i, box in enumerate(torch.cat((self.boxes[1:, :], torch.Tensor([[0,0,0,0]]).to(device)))):
            if not torch.equal(box, current_box):
                if number_of_duplicates == self.NUMBER_OF_POSITIVE_EXAMPLE_DUPLICATES:
                    frame_number += 1
                frame_numbers.append(frame_number)
                unique_indices.append(i)
                current_box = box
                number_of_duplicates = 0
            else:
                number_of_duplicates += 1
        self.frame_numbers = torch.Tensor(frame_numbers)
        self.scores = self.scores[torch.LongTensor(unique_indices)]
        self.boxes = self.boxes[torch.LongTensor(unique_indices), :]
        self.features = self.features[torch.LongTensor(unique_indices),:,:,:]
        self.number_of_positive_examples = self.scores[self.scores==1].size()[0]

    def val_test_split(self, percentage_positive_examples_train=None, number_positive_examples_train=None,
                       ordered_by_frame=False, downsample=True):
        if percentage_positive_examples_train is None and number_positive_examples_train is None:
            raise ValueError("either a percentage or absolute number of positive examples should be provided")
        if number_positive_examples_train is not None and number_positive_examples_train >= self.number_of_positive_examples:
            raise ValueError("the number of positive samples you provided exceeds the number of available samples")
        if percentage_positive_examples_train is not None and number_positive_examples_train is not None:
            import warnings
            warnings.warn("you provided a percentage and an absolute number. giving preference to percentage")
        if percentage_positive_examples_train is not None:
            train_positive_examples = int(self.number_of_positive_examples * percentage_positive_examples_train)
        else:
            train_positive_examples = number_positive_examples_train

        positive_examples_indices = (self.scores==1).nonzero().squeeze(1)
        negative_examples_indices = (self.scores==0).nonzero().squeeze(1)
        if not ordered_by_frame:
            positive_examples_indices = positive_examples_indices[randperm(positive_examples_indices.size()[0]).tolist()]
            negative_examples_indices = negative_examples_indices[randperm(negative_examples_indices.size()[0]).tolist()]

        # Do downsampling? or upsampling?
        # ..Downsample
        if downsample:
            train_idx = torch.cat((positive_examples_indices[:train_positive_examples],
                                   negative_examples_indices[:train_positive_examples]))
            val_idx = positive_examples_indices[train_positive_examples:]
            val_idx = torch.cat((val_idx,
                                negative_examples_indices[-val_idx.size()[0]:]))

        return [Subset(self, train_idx), Subset(self, val_idx)]

    def establish_class_balance(self):
        return

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}