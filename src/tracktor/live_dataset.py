import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IndividualDataset(torch.utils.data.Dataset):
    def __init__(self, id):
        self.id = id
        self.features = torch.tensor([]).to(device)
        self.boxes = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)
        self.frame_numbers = None

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
        frame_number = -1
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

    def val_test_split(self, framerange=None):
        return

    def establish_class_balance(self):
        return

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx, :, :, :], 'boxes': self.boxes[idx, :], 'scores': self.scores[idx]}