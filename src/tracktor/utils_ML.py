import time, os
import torch
import torch.nn
import h5py
import learn2learn as l2l
from torch.utils.data import Dataset
import datetime
from tracktor.visualization import VisdomLinePlotter_ML

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(reid, exclude=[], only=[]):
    start_time_load = time.time()
    i_to_dataset = {}
    with h5py.File('./data/ML_dataset/{}.h5'.format(reid['ML']['db']), 'r') as hf:
        datasets = list(hf.keys())
        datasets = [d for d in datasets if d != reid['dataloader']['validation_sequence']]
        datasets = [d for d in datasets if d not in exclude]
        if len(only) > 0:
            datasets = [d for d in datasets if d in only]

        for i,d in enumerate(datasets):
            i_to_dataset[i] = d
        print(f"Train with {datasets} and use {reid['dataloader']['validation_sequence']} as validation set")

        others_FM = []
        others_ID = []
        others_num = []
        others_seq_ID = []
        c = 0

        for i, set in enumerate(datasets):
            seq = hf.get(set)
            d, l = seq.items()
            num = len(l[1])

            others_FM.append(torch.tensor(d[1]))
            others_ID.append(torch.tensor(l[1]))
            others_num.append((c, c+num))
            others_seq_ID.append(torch.ones(num)*i)
            c += num
            print(f"loaded {set} as seq id {i} with {num} samples")


        others_FM = torch.cat(others_FM).to(device)  # contains all feature maps except val sequence
        others_ID = torch.cat(others_ID)
        others_seq_ID = torch.cat(others_seq_ID)
        print(f"Overall {others_FM.shape[0]} samples in others")

        if reid['dataloader']['validation_sequence'] != 'None':
            validation_data = hf.get(reid['dataloader']['validation_sequence'])
            d, l = validation_data.items()
            validation_FM = torch.tensor(d[1]).to(device)
            validation_ID = torch.tensor(l[1])
            print('loaded validation {}'.format(reid['dataloader']['validation_sequence']))
            print('Overall {} samples in val sequence'.format(validation_FM.shape[0]))
        print(f"--- {(time.time() - start_time_load):.4f} seconds --- for loading hdf5 database")

        # create meta learning dataset for validation seq
        if reid['dataloader']['validation_sequence'] != 'None':
            validation_set = ML_dataset(fm=validation_FM, id=validation_ID)
            validation_set = [l2l.data.MetaDataset(validation_set)]
        else:
            validation_set = []

        meta_datasets = []
        for i in others_num:
            meta_datasets.append(ML_dataset(
                fm=others_FM[i[0]:i[1]],
                id=others_ID[i[0]:i[1]],
            ))

        for i, s in enumerate(meta_datasets):
            meta_datasets[i] = l2l.data.MetaDataset(s)

        print(f"--- {(time.time() - start_time_load):.4f} seconds --- for loading db and building meta-datasets ")

        return meta_datasets, validation_set, i_to_dataset, others_FM


def statistics(dataset):
    unique_id, counter = np.unique(dataset[0].numpy(), return_counts=True)
    num_id = len(unique_id)
    num_bb = sum(counter)
    samples_per_id, counter_samples = np.unique(counter, return_counts=True)
    print('in total {} unique IDs, and {} BBs in total, av. {} BB/ID  '.format(num_id, num_bb, (num_bb/num_id)))
    #print('in total {} unique IDs, print until 80 samples per ID'.format(num_id))
    # for i in range(len(counter_samples)):
    #     if samples_per_id[i]<80:
    #         print('{} samples per ID: {} times'.format(samples_per_id[i], counter_samples[i]))
    return num_id, num_bb


def get_ML_settings(reid):
    if reid['ML']['range']:
        nways_list = list(range(2, reid['ML']['nways']+1))
        kshots_list = list(range(1, reid['ML']['kshots']+1))
    else:
        nways_list = reid['ML']['nways']
        if type(nways_list) is int:
            nways_list = [nways_list]
        kshots_list = reid['ML']['kshots']
        if type(kshots_list) is int:
            kshots_list = [kshots_list]

    num_tasks = reid['ML']['num_tasks']
    num_tasks_val = reid['ML']['num_tasks_val']
    adaptation_steps = reid['ML']['adaptation_steps']
    meta_batch_size = reid['ML']['meta_batch_size']

    return nways_list, kshots_list, num_tasks, num_tasks_val, adaptation_steps, meta_batch_size

def get_plotter(reid, info):
    plotter = None
    if reid['solver']["plot_training_curves"]:
        now = datetime.datetime.now()
        run_name = now.strftime("%Y-%m-%d_%H:%M_") + reid['name']
        plotter = VisdomLinePlotter_ML(env=run_name, offline=reid['solver']['plot_offline'],
                                       info=info)
        return plotter

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    #print('[{}] safe checkpoint {}'.format(state['epoch'], filename))

def load_checkpoint(reid, model, opt):
    if reid['solver']['continue_training']:
        if os.path.isfile(reid['solver']['checkpoint']):
            print("=> loading checkpoint '{}'".format(reid['solver']['checkpoint']))
            checkpoint = torch.load(reid['solver']['checkpoint'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(reid['solver']['checkpoint'], checkpoint['epoch']))


class ML_dataset(Dataset):
    def __init__(self, fm, id, flip_p=-1):
        self.fm = fm
        self.id = id
        self.flip_p = flip_p

    def __getitem__(self, item):
        # First way flip was implemented, either flipped or non flipped sample as output
        # if random.random() < self.flip_p:
        #     features = self.data[1][item].flip(-1)
        # else:
        #     features = self.data[1][item]
        # return (features, self.data[0][item].unsqueeze(0))

        # In reID network in tracktor we will always use non flipped and flipped version
        # do the same here in ML setting

        if self.flip_p>0.0:
            features = self.fm[1][item].flip(-1).unsqueeze(0)
            features = torch.cat((self.fm[1][item].unsqueeze(0), features))

            label = self.id[0][item] #.unsqueeze(0)
            #label = torch.cat((self.data[0][item].unsqueeze(0), label))
            return (features.to(device), label)
        else:
            return (self.fm[item], self.id[item].unsqueeze(0))

    def __len__(self):
        return len(self.id)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.best = 1000
        self.best_it = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def update_best(self, val, it):
        save = 0
        if val < self.best:
            self.best = val
            self.best_it = it
            save = 1
        return save


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)