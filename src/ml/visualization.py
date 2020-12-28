import logging
import torch
from visdom import Visdom


class VisdomLinePlotter_ML(object):
    """Plots to Visdom"""

    def __init__(self, env, offline, info):
        logger = logging.getLogger()
        logger.setLevel(40)

        if not offline:
            self.viz = Visdom(port=8097, env=env)
        else:
            self.viz = Visdom(env=env, log_to_filename='experiments/logs/{}'.format(env), offline=True)
            print('\n save plot as experiments/logs/{}'.format(env))
        logger.setLevel(20)
        self.env = env
        self.viz.text(
            env=self.env,
            # text='nWays {}\n kShots {}\n meta batch size {}\n number train tasks {}\n number validation tasks {}\n'
            #      ' validaiton sequence {}'.format(info)
            text='nWays {} kShots {} meta batch size {} number train tasks per sequence {} number validation tasks {} validation sequence {} flip prob {}'.format(info[0], info[1], info[2], info[3], info[4], info[5], info[6])
        )

        self.loss_window = self.viz.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='Iteration',
                                     ylabel='Loss',
                                     env=self.env,
                                     title='Loss val set',
                                     legend=['train_task_val_set']))
        self.accuracy_window = self.viz.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='Iteration',
                                     ylabel='Accuracy',
                                     env=self.env,
                                     title='Accuracy val set',
                                     legend=['train_task_val_set']))
        self.LR_window = self.viz.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='Iteration',
                                     ylabel='LR',
                                     env=self.env,
                                     title='MEAN learned LR',
                                     legend=['LR_fc7_w']))
        self.weight_window = self.viz.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='Iteration',
                                     ylabel='Weight',
                                     env=self.env,
                                     title='MEAN learned weights',
                                     legend=['Mean_fc7_w']))

    def plot(self, epoch, loss, acc, split_name, info=None, LR=-100):
        epoch -= 1
        if split_name == 'inner':
            seq, nways, kshots, it, train_task, taskID = info
            if train_task:
                task='({})train_task'.format(taskID)
            else:
                task='({})val_task'.format(taskID)
            if epoch == 0:
                self.inner_window_loss = self.viz.line(X=torch.zeros((1,)).cpu(),
                                            Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                                            opts=dict(xlabel='Epoch',
                                                      ylabel='Loss',
                                                      env=self.env,
                                                      title='Loss train set {} [{}]'.format(seq, it),
                                                      legend=['{} {}n_{}k'.format(task, nways, kshots)]))
                for i, a in enumerate(acc):
                    if i == 0:
                        self.inner_window_acc = self.viz.line(X=torch.zeros((1,1)).cpu(),
                                                    Y=torch.Tensor([a]).unsqueeze(0).cpu(),
                                                    opts=dict(xlabel='Epoch',
                                                              ylabel='Accuracy',
                                                              env=self.env,
                                                              title='Accuracy train set {} [{}]'.format(seq, it),
                                                              #legend=['TASK {}n_{}k'.format(task, nways, kshots)]))
                                                              legend=['TASK']))
                    if i == 3:
                        self.viz.line(
                            X=torch.ones((1, 1)).cpu() * epoch,
                            Y=torch.Tensor([a]).unsqueeze(0).cpu(),
                            env=self.env,
                            win=self.inner_window_acc,
                            #name='OTHERS {}n_{}k'.format(task, nways, kshots),
                            name='OTHERS',
                            update='append')

            else:
                self.viz.line(
                X=torch.ones((1, 1)).cpu() * epoch,
                Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                env=self.env,
                win=self.inner_window_loss,
                name='{} {}n_{}k'.format(task, nways, kshots),
                update='append')

                for i, a in enumerate(acc):
                    if i == 0:
                        self.viz.line(
                            X=torch.ones((1, 1)).cpu() * epoch,
                            Y=torch.Tensor([a]).unsqueeze(0).cpu(),
                            env=self.env,
                            win=self.inner_window_acc,
                            #name='TASK {}n_{}k'.format(task, nways, kshots),
                            name='TASK',
                            update='append')
                    elif i == 3:
                        self.viz.line(
                            X=torch.ones((1, 1)).cpu() * epoch,
                            Y=torch.Tensor([a]).unsqueeze(0).cpu(),
                            env=self.env,
                            win=self.inner_window_acc,
                            name='OTHERS',
                            update='append')
            return

        elif 'LR' in split_name:
            self.viz.line(X=torch.ones((1, 1)).cpu() * epoch,
                                               Y=torch.Tensor([acc]).unsqueeze(0).cpu(),
                                                env=self.env,
                                                win=self.LR_window,
                                                name=split_name,
                                               update='append')
            return

        elif 'Mean_' in split_name:
            self.viz.line(X=torch.ones((1, 1)).cpu() * epoch,
                                               Y=torch.Tensor([acc]).unsqueeze(0).cpu(),
                                                env=self.env,
                                                win=self.weight_window,
                                                name=split_name,
                                               update='append')
            return

        elif 'idf1' in split_name:
            if not hasattr(self, 'idf1_window'):
                self.idf1_window = self.viz.line(X=torch.zeros((1,)).cpu(),
                                                 Y=torch.Tensor([acc]).unsqueeze(0).cpu(),
                                                 opts=dict(xlabel='Iteration',
                                                           ylabel='Score',
                                                           env=self.env,
                                                           title=f'IDF1 Score Overall',
                                                           legend=['idf1_TrainOn']))

            self.viz.line(X=torch.ones((1, 1)).cpu() * epoch,
                          Y=torch.Tensor([acc]).unsqueeze(0).cpu(),
                          env=self.env,
                          win=self.idf1_window,
                          name=split_name,
                          update='append')
            return

        else:
            name = split_name
        if loss > 0:
            self.viz.line(
                X=torch.ones((1, 1)).cpu() * epoch,
                Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                env=self.env,
                win=self.loss_window,
                name=name,
                update='append')
        self.viz.line(
            X=torch.ones((1, 1)).cpu() * epoch,
            Y=torch.Tensor([acc]).unsqueeze(0).cpu(),
            env=self.env,
            win=self.accuracy_window,
            name=name,
            update='append')

        if LR!=-100:
            self.viz.line(
                X=torch.ones((1, 1)).cpu() * epoch,
                Y=torch.Tensor([LR]).unsqueeze(0).cpu(),
                env=self.env,
                win=self.LR_window,
                name='LR',
                update='append')

    def init_statistics(self, meta_batch_size):
        self.meta_batch_size = meta_batch_size
        ## av for loss and acc ##
        self.loss_meta_train = AverageMeter('Loss', ':.4e')
        self.loss_meta_val = AverageMeter('Loss', ':.4e')
        self.acc_meta_train = AverageMeter('Acc', ':6.2f')
        self.acc_just_zero_meta_train = AverageMeter('Acc', ':6.2f')
        self.acc_all_meta_train = AverageMeter('Acc', ':6.2f')
        self.acc_others_own_meta_train = AverageMeter('Acc', ':6.2f')

        self.acc_meta_val = AverageMeter('Acc', ':6.2f')
        self.acc_just_zero_meta_val = AverageMeter('Acc', ':6.2f')
        self.acc_all_meta_val = AverageMeter('Acc', ':6.2f')
        self.acc_others_own_meta_val = AverageMeter('Acc', ':6.2f')

        self.acc_train_task_train_set_acc_task = AverageMeter('Acc', ':6.2f')
        self.acc_train_task_train_set_acc_others = AverageMeter('Acc', ':6.2f')

        # for LRs
        self.LR_fc7_w = AverageMeter('Acc', ':6.2f')
        self.LR_fc7_b = AverageMeter('Acc', ':6.2f')
        self.LR_fc6_w = AverageMeter('Acc', ':6.2f')
        self.LR_fc6_b = AverageMeter('Acc', ':6.2f')
        self.LR_last_w = AverageMeter('Acc', ':6.2f')
        self.LR_last_b = AverageMeter('Acc', ':6.2f')
        self.LR_template_w = AverageMeter('Acc', ':6.2f')
        self.LR_template_b = AverageMeter('Acc', ':6.2f')

        # for weights
        self.Mean_fc7_w = AverageMeter('Acc', ':6.2f')
        self.Mean_fc7_b = AverageMeter('Acc', ':6.2f')
        self.Mean_fc6_w = AverageMeter('Acc', ':6.2f')
        self.Mean_fc6_b = AverageMeter('Acc', ':6.2f')
        self.Mean_template_w = AverageMeter('Acc', ':6.2f')
        self.Mean_template_b = AverageMeter('Acc', ':6.2f')

    def update_statistics(self, model):
        # using the template neurons
        # LR
        self.LR_fc7_w.update(model.module.lrs[0].mean().item())
        self.LR_fc7_b.update(model.module.lrs[1].mean().item())
        self.LR_fc6_w.update(model.module.lrs[2].mean().item())
        self.LR_fc6_b.update(model.module.lrs[3].mean().item())
        self.LR_template_w.update(model.lrs[0].mean().item())  # template neuron
        self.LR_template_b.update(model.lrs[1].mean().item())  # template neuron

        # weights
        self.Mean_fc7_w.update(model.module.head.fc7.weight.mean())
        self.Mean_fc7_b.update(model.module.head.fc7.bias.mean())
        self.Mean_fc6_w.update(model.module.head.fc6.weight.mean())
        self.Mean_fc6_b.update(model.module.head.fc6.bias.mean())
        self.Mean_template_w.update(model.template_neuron_weight.mean())
        self.Mean_template_b.update(model.template_neuron_bias.mean())

    def reset_batch_stats(self):
        # current batch loss and acc #
        # performance of train task
        self.meta_train_loss = 0.0
        self.meta_train_accuracy = 0.0
        self.meta_train_accuracy_all = 0.0  # with zero
        self.meta_train_accuracy_just_zero = 0.0  # just zero
        self.meta_train_accuracy_others_own = 0.0

        # performance of val task
        self.meta_valid_loss = 0.0
        self.meta_valid_accuracy = 0.0
        self.meta_valid_accuracy_all = 0.0  # with zero
        self.meta_valid_accuracy_just_zero = 0.0  # just zero
        self.meta_valid_accuracy_others_own = 0.0

    def update_batch_val_stats(self, acc, loss):
        evaluation_accuracy, evaluation_accuracy_all, evaluation_accuracy_just_zero, evaluation_accuracy_others_own = acc

        self.meta_valid_loss += loss.item()
        self.meta_valid_accuracy += evaluation_accuracy.item()  # accuracy without zero class
        self.meta_valid_accuracy_all += evaluation_accuracy_all.item()  # accuracy with zeros class
        self.meta_valid_accuracy_just_zero += evaluation_accuracy_just_zero.item()  # track acc just zero class
        self.meta_valid_accuracy_others_own += evaluation_accuracy_others_own.item()  # track acc just zero class

        self.loss_meta_val.update(loss.item())
        self.acc_meta_val.update(evaluation_accuracy.item())
        self.acc_just_zero_meta_val.update(evaluation_accuracy_just_zero.item())  # others from own sequence
        self.acc_all_meta_val.update(evaluation_accuracy_all.item())  # others from others sequences
        self.acc_others_own_meta_val.update(evaluation_accuracy_others_own.item())  # others from OWN sequence

    def update_batch_train_stats(self, acc, loss):
        evaluation_accuracy, evaluation_accuracy_all, evaluation_accuracy_just_zero, evaluation_accuracy_others_own = acc

        self.meta_train_loss += loss.item()
        self.meta_train_accuracy += evaluation_accuracy.item()
        self.meta_train_accuracy_all += evaluation_accuracy_all.item()  # accuracy with zeros class
        self.meta_train_accuracy_just_zero += evaluation_accuracy_just_zero.item()  # track acc just zero class
        self.meta_train_accuracy_others_own += evaluation_accuracy_others_own.item()  # track acc others OWN

        self.loss_meta_train.update(loss.item())
        self.acc_meta_train.update(evaluation_accuracy.item())
        self.acc_just_zero_meta_train.update(evaluation_accuracy_just_zero.item())  # others from own sequence
        self.acc_all_meta_train.update(evaluation_accuracy_all.item())  # others from others sequences
        self.acc_others_own_meta_train.update(evaluation_accuracy_others_own.item())  # others from OWN sequence

    def print_statistics(self, idf1_trainOthers = 0, idf1_NotrainOthers = 0):
        print(f"average over meta batch")
        print(f"IDF1 scores train others in tracktor {idf1_trainOthers}")
        print(f"IDF1 scores NO train others in tracktor {idf1_NotrainOthers}")
        print(f"{'Train Loss':<50} {self.meta_train_loss / self.meta_batch_size:.8f}")
        print(f"{'Train Accuracy task':<50} {self.meta_train_accuracy / self.meta_batch_size:.6f}")
        print(f"{'Train Accuracy task+others':<50} {self.meta_train_accuracy_all / self.meta_batch_size:.6f}")
        print(f"{'Train Accuracy just others':<50} {self.meta_train_accuracy_just_zero / self.meta_batch_size:.6f}")
        print(f"{'Train Accuracy others own sequence':<50} {self.meta_train_accuracy_others_own / self.meta_batch_size:.6f}")

        # print(f"{'Valid Loss':<50} {meta_valid_loss / meta_batch_size}")
        # print(f"{'Valid Accuracy task':<50} {meta_valid_accuracy / meta_batch_size}")
        # print(f"{'Valid Accuracy task+others':<50} {meta_valid_accuracy_all / meta_batch_size}")
        # print(f"{'Valid Accuracy just others:':<50} {meta_valid_accuracy_just_zero / meta_batch_size}")
        # print(f"{'Valid Accuracy others own sequence':<50} {meta_valid_accuracy_others_own / meta_batch_size}")

        print(f"{'Valid Loss':<50} {self.meta_valid_loss:.8f}")
        print(f"{'Valid Accuracy task':<50} {self.meta_valid_accuracy:.6f}")
        print(f"{'Valid Accuracy task+others':<50} {self.meta_valid_accuracy_all:.6f}")
        print(f"{'Valid Accuracy just others:':<50} {self.meta_valid_accuracy_just_zero:.6f}")
        print(f"{'Valid Accuracy others own sequence':<50} {self.meta_valid_accuracy_others_own:.6f}")

        print(f"\n{'Mean Train Loss':<50} {self.loss_meta_train.avg:.8f}")
        print(f"{'Mean Train Accuracy task':<50} {self.acc_meta_train.avg:.6f}")
        print(f"{'Mean Train Accuracy task+others':<50} {self.acc_all_meta_train.avg:.6f}")
        print(f"{'Mean Train Accuracy just others':<50} {self.acc_just_zero_meta_train.avg:.6f}")
        print(f"{'Mean Train Accuracy others own sequence':<50} {self.acc_others_own_meta_train.avg:.6f}")

        print(f"{'Mean Val Loss':<50} {self.loss_meta_val.avg:.8f}")
        print(f"{'Mean Val Accuracy task':<50} {self.acc_meta_val.avg:.6f}")
        print(f"{'Mean Val Accuracy task+others':<50} {self.acc_all_meta_val.avg:.6f}")
        print(f"{'Mean Val Accuracy just others':<50} {self.acc_just_zero_meta_val.avg:.6f}")
        print(f"{'Mean Val Accuracy others own sequence':<50} {self.acc_others_own_meta_val.avg:.6f}")

    def plot_statistics(self, iteration):
        meta_batch_size = self.meta_batch_size
        self.plot(epoch=iteration, loss=self.meta_train_loss / meta_batch_size,
                         acc=self.meta_train_accuracy / meta_batch_size, split_name='train_task_val_set')
        self.plot(epoch=iteration, loss=self.loss_meta_train.avg, acc=self.acc_meta_train.avg,
                     split_name='train_task_val_set MEAN')
        self.plot(epoch=iteration, loss=self.meta_valid_loss, acc=self.meta_valid_accuracy, split_name='val_task_val_set')
        self.plot(epoch=iteration, loss=self.loss_meta_val.avg, acc=self.acc_meta_val.avg, split_name='val_task_val_set MEAN')
        self.plot(epoch=iteration, loss=-1, acc=self.acc_others_own_meta_train.avg,
                     split_name='train_task_val_set_others_own MEAN')
        # self.plot(epoch=iteration, loss=-1, acc=acc_all_meta_train.avg, split_name='train_task_val_set_all MEAN')
        self.plot(epoch=iteration, loss=-1, acc=self.acc_others_own_meta_val.avg,
                     split_name='val_task_val_set_others_own MEAN')
        # self.plot(epoch=iteration, loss=-1, acc=acc_all_meta_val.avg, split_name='val_task_val_set_all MEAN')
        # self.plot(epoch=iteration, loss=-1, acc=acc_just_zero_meta_train.avg, split_name='train_task_val_set_just_zero MEAN')
        # self.plot(epoch=iteration, loss=-1, acc=acc_just_zero_meta_val.avg, split_name='val_task_val_set_just_zero MEAN')
        # self.plot(epoch=iteration, loss=-1, acc=self.train_task_train_set_acc_task_last / meta_batch_size,
        #              split_name='train_task_train_set_task')
        # self.plot(epoch=iteration, loss=-1, acc=self.acc_train_task_train_set_acc_task.avg,
        #              split_name='train_task_train_set_task MEAN')
        # self.plot(epoch=iteration, loss=-1, acc=self.train_task_train_set_acc_others_last / meta_batch_size,
        #              split_name='train_task_train_set_others_own')
        # self.plot(epoch=iteration, loss=-1, acc=self.acc_train_task_train_set_acc_others.avg,
        #              split_name='train_task_train_set_others_own MEAN')

        # plot avg values of LR and weights
        # self.plot(epoch=iteration, loss=-1, acc=self.LR_fc7_w.avg, split_name='LR_fc7_w')
        # self.plot(epoch=iteration, loss=-1, acc=self.LR_fc7_b.avg, split_name='LR_fc7_b')
        # self.plot(epoch=iteration, loss=-1, acc=self.LR_fc6_w.avg, split_name='LR_fc6_w')
        # self.plot(epoch=iteration, loss=-1, acc=self.LR_fc6_b.avg, split_name='LR_fc6_b')
        # self.plot(epoch=iteration, loss=-1, acc=self.LR_last_w.avg, split_name='LR_last6_w')
        # self.plot(epoch=iteration, loss=-1, acc=self.LR_last_b.avg, split_name='LR_last6_b')
        #
        # self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc7_w.avg, split_name='Mean_fc7_w')
        # self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc7_b.avg, split_name='Mean_fc7_b')
        # self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc6_w.avg, split_name='Mean_fc6_w')
        # self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc6_b.avg, split_name='Mean_fc6_b')
        # self.plot(epoch=iteration, loss=-1, acc=self.Mean_last_w.avg, split_name='Mean_last6_w')
        # self.plot(epoch=iteration, loss=-1, acc=self.Mean_last_b.avg, split_name='Mean_last6_b')

        # plot current values of LR and weights
        self.plot(epoch=iteration, loss=-1, acc=self.LR_fc7_w.val, split_name='LR_fc7_w')
        self.plot(epoch=iteration, loss=-1, acc=self.LR_fc7_b.val, split_name='LR_fc7_b')
        self.plot(epoch=iteration, loss=-1, acc=self.LR_fc6_w.val, split_name='LR_fc6_w')
        self.plot(epoch=iteration, loss=-1, acc=self.LR_fc6_b.val, split_name='LR_fc6_b')
        self.plot(epoch=iteration, loss=-1, acc=self.LR_template_w.val, split_name='LR_template_w')
        self.plot(epoch=iteration, loss=-1, acc=self.LR_template_b.val, split_name='LR_template_b')

        self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc7_w.val, split_name='Mean_fc7_w')
        self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc7_b.val, split_name='Mean_fc7_b')
        self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc6_w.val, split_name='Mean_fc6_w')
        self.plot(epoch=iteration, loss=-1, acc=self.Mean_fc6_b.val, split_name='Mean_fc6_b')
        self.plot(epoch=iteration, loss=-1, acc=self.Mean_template_w.val, split_name='Mean_template_w')
        self.plot(epoch=iteration, loss=-1, acc=self.Mean_template_b.val, split_name='Mean_template_b')


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