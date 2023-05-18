import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import time
import os
import numpy as np

from core.noise_augment import GradIntegral


class ClsDcamTrainer:
    def __init__(self, model, criterion, modules, device, optimizer, data_loaders, dataset_sizes, num_epochs,
                 result_path, model_path):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.optimizer = optimizer
        self.data_loaders = data_loaders
        self.dataset_sizes = dataset_sizes
        self.num_epochs = num_epochs
        self.result_path = result_path

        self.gi = GradIntegral(model=model, modules=modules)

        if model_path is not None:
            print('-' * 40)
            print('LOAD CHECKPOINT:', model_path)
            print('-' * 40)
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            self.model.load_state_dict(checkpoint, strict=True)

        self.criterion.to(device)
        self.model = torch.nn.DataParallel(self.model).to(device)
        # self.model.to(device)

        self.history = TrainerHistory(best='loss', save_path=self.result_path)

    def train(self):
        since = time.time()

        # ----------------------------------------
        # Each epochs.
        # ----------------------------------------
        for epoch in range(self.num_epochs):
            print('\nEpoch {}/{}'.format(epoch, self.num_epochs - 1))

            for phase in ['train']:
                if phase == 'train':
                    self.model.train()
                    self.gi.add_noise()
                else:
                    self.model.eval()
                    self.gi.remove_noise()

                running_loss_cls = 1e-5
                running_corrects = 1e-5

                # ----------------------------------------
                # Iterate over data.
                # ----------------------------------------
                for i, samples in enumerate(self.data_loaders[phase]):
                    inputs = samples['img']
                    labels = samples['label']
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(True):
                        outputs = self.model.forward(inputs)
                        sigmoid = torch.sigmoid(outputs)
                        preds = torch.ones(outputs.shape, device=self.device) * (sigmoid >= 0.9)

                        loss_cls = self.criterion(outputs, labels)
                        loss = loss_cls

                        print('\r[{}/{}] loss_cls:{:.4f}'.format(i, len(
                            self.data_loaders[phase]), loss_cls), end='', flush=True)

                        # backward
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss_cls += loss_cls.item() * inputs.size(0)
                    for j in range(len(labels)):
                        running_corrects += torch.equal(labels[j], preds[j])

                # ---------------------------------------------
                # Calculate the loss and acc of each epoch
                # ---------------------------------------------
                epoch_loss_cls = running_loss_cls / self.dataset_sizes[phase]
                epoch_acc = running_corrects / self.dataset_sizes[phase]
                print('\n[{}] Loss CLS: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss_cls, epoch_acc))

                # ----------------------------------------
                # Save epoch data
                # ----------------------------------------
                is_best = self.history.update(phase=phase,
                                              acc=epoch_acc,
                                              loss_cls=epoch_loss_cls,
                                              istrain=True)

                if is_best:
                    print('- Update best weights')
                    if not os.path.exists(self.result_path):
                        os.makedirs(self.result_path)
                    state = self.model.module.state_dict()
                    path = os.path.join(self.result_path, 'checkpoint.pth')
                    torch.save(state, path)

                if phase == 'val':
                    self.history.draw()
                    print('- lr:', self.optimizer.param_groups[0]['lr'])

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def check(self):
        phase = 'val'

        # state = torch.load(self.result_path + '/checkpoint.pth')
        # self.model.module.load_state_dict(state)
        self.model.eval()

        print('-' * 40)
        print('Check data type:', phase)
        print('Load model from:', self.result_path)
        print('Data size:', self.dataset_sizes[phase])
        print('-' * 40)

        running_loss = 0.0
        running_corrects = 0.0

        # ----------------------------------------
        # Iterate over data.
        # ----------------------------------------
        for i, samples in enumerate(self.data_loaders[phase]):
            inputs = samples['img']
            labels = samples['label']
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            with torch.no_grad():
                outputs = self.model.forward(inputs)
                sigmoid = torch.sigmoid(outputs)
                preds = torch.ones(outputs.shape, device=self.device) * (sigmoid >= 0.9)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                for j in range(len(labels)):
                    running_corrects += torch.equal(labels[j], preds[j])

        loss = running_loss / self.dataset_sizes[phase]
        acc = running_corrects / self.dataset_sizes[phase]
        print('\n[{}] Loss CLS: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))


class TrainerHistory(object):
    def __init__(self, best, save_path):
        assert best in ['loss', 'acc']
        self.best = best
        self.best_value = None

        self.save_path = save_path

        self.history = {'train_loss_cls': [],
                        'train_acc': [],
                        'val_loss_cls': [],
                        'val_acc': []}

    def update(self, phase, loss_cls, acc, istrain=False):
        if phase == 'train':
            self.history['train_loss_cls'].append(loss_cls)
            self.history['train_acc'].append(acc)

            # best
            if istrain:
                if self.best == 'loss' and (self.best_value is None or loss_cls <= self.best_value):
                    self.best_value = loss_cls
                    return True
                if self.best == 'acc' and (self.best_value is None or acc >= self.best_value):
                    self.best_value = acc
                    return True

        if phase == 'val':
            self.history['val_loss_cls'].append(loss_cls)
            self.history['val_acc'].append(acc)

            # best
            if not istrain:
                if self.best == 'loss' and (self.best_value is None or loss_cls <= self.best_value):
                    self.best_value = loss_cls
                    return True
                if self.best == 'acc' and (self.best_value is None or acc >= self.best_value):
                    self.best_value = acc
                    return True

        return False

    def draw(self):
        # save history
        np.save(os.path.join(self.save_path, 'model.npy'), self.history)

        # draw history
        num_epochs = len(self.history['train_loss_cls'])

        plt.plot(range(1, num_epochs + 1), self.history['train_loss_cls'], 'r', label='train loss_cls')
        plt.plot(range(1, num_epochs + 1), self.history['train_loss_gc'], 'deeppink', label='train loss_gc')
        plt.plot(range(1, num_epochs + 1), self.history['train_acc'], 'g', label='train acc')
        plt.plot(range(1, num_epochs + 1), self.history['val_loss_cls'], 'b', label='val loss_cls')
        plt.plot(range(1, num_epochs + 1), self.history['val_loss_gc'], 'blueviolet', label='val loss_gc')
        plt.plot(range(1, num_epochs + 1), self.history['val_acc'], 'k', label='val acc')

        plt.title("Acc and Loss of each epoch")
        plt.xlabel("Training Epochs")
        plt.ylabel("Acc Loss")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'model.jpg'))
        plt.clf()
