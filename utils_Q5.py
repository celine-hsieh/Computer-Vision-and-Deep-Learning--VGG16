import re
import sys
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import corrcoef

import torch
from torch import random
from torch.utils import data
from torch.utils.data import sampler
import torchvision
from torchvision.datasets import cifar
import torchvision.transforms as transforms
from torchvision.models import vgg16
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import math
import json

from torchsummary import summary

label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def im_convert(image):
    image = image * (np.array((0.4914, 0.4822, 0.4465)) + np.array((0.2023, 0.1994, 0.2010)))
    image = image.clip(0,1)
    return image

def plot_images(images, true_class, pred_class=None, num_image = 10):
    col = 5
    row = num_image // col
    fig, axes = plt.subplots(row, col)
    for i, ax in enumerate(axes.flat):
        #plot image
        ax.imshow(im_convert(images[i,:,:,:]), interpolation='spline16')
        #show true class
        cls_true_name = label_names[true_class[i]]
        if pred_class is None:
            xlabel = "{0} ({1})".format(cls_true_name, true_class[i])
        else:
            cls_pred_name = label_names[pred_class[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_image(image, true_index, pred_index, pred_data, label_names = label_names):
    true_class = label_names[true_index]
    pred_class = label_names[pred_index]
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image, interpolation='spline16')
    xlabel = "True: {0}\nPred: {1}".format(true_class, pred_class)
    ax[0].set_xlabel(xlabel)

    ax[1].bar(label_names, pred_data)
    plt.show()

def show_acc_plot(history, save=False):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history['train_loss'], color='b', label="Training loss")
    ax[0].plot(history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history['train_acc'], color='b', label="Training accuracy")
    ax[1].plot(history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    if save:
        fig.savefig('history.png')

def bar_plot(pred_x, label_names = label_names):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(label_names, pred_x)
    plt.show()

def update(index, length, epoch_loss, acc, mode):
    if length >= 400:
        update_size = int(length)/400
    else:
        update_size = 10
    if index % update_size == 0 and index != 0:
        print("\r   {} progress: {:.2f}% .... loss: {:.4f}, acc: {:.4f}".format(
            mode, (index/length *100), epoch_loss/index, acc
        ), end = "", flush=True)

def save_history(history:dict, path_save='history.json'):
    json.dump(history, open(path_save, 'w+'))

def load_history(path_load="history.json"):
    return json.load(open(path_load, "r+"))

class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Conv1
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),  # Conv2
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(32, 64, 3, padding=1),  # Conv3
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  # Conv4
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(64, 128, 3, padding=1),  # Conv5
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv6
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # Conv7
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool3
            nn.Conv2d(128, 256, 3, padding=1),  # Conv8
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv9
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv10
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # Pool4
            nn.Conv2d(256, 256, 3, padding=1),  # Conv11
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv12
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # Conv13
            nn.ReLU(True),
            # nn.MaxPool2d(2, 2)  # Pool5 
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 256, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Q5_Cifar10:

    def __init__(self, modeTrain= True, name_classes = label_names):
        self.modeTrain = modeTrain

        self.name_classes = name_classes

        num_classes = len(self.name_classes)

        self.hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "SGD", #SGD or Adam
            "maxepoches": 40,
            "lr_drop": 20,
            "lr_decay": 1e-6,
            "momentum": 0.94
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = VGGNet(num_classes)

        self.model= self.model.to(self.device)

        if self.device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
        if not self.modeTrain:
            self.load_model("model/best.pth")

        self.loss_fn = nn.CrossEntropyLoss()

    def load_train_dataset(self, random_seed, shuffle = True, valid_size=0.2, num_workers = 2, show_sample = False, pin_memory = False):
        print('==> Preparing Data')
        normalize = transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010]
        )
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train= True, download= True,
            transform=transform_train
        )

        valset = torchvision.datasets.CIFAR10(
            root='./data', train= True, download=True,
            transform=transform_val
        )

        num_train = len(trainset)
        indices = list(range(num_train))
        split = (int(np.floor(valid_size * num_train)))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)

        print("Train_length: ", len(train_sampler.indices))
        print("Validation_length: ", len(valid_sampler.indices))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size = self.hyperparameters['batch_size'], 
            shuffle = False, sampler = train_sampler, 
            num_workers = num_workers, 
            pin_memory = pin_memory
        )
        self.val_loader = torch.utils.data.DataLoader(
            valset, batch_size = self.hyperparameters['batch_size'], 
            shuffle=False, sampler = valid_sampler,
            num_workers = num_workers, 
            pin_memory = pin_memory
        )


        if show_sample:
            sample_loader = torch.utils.data.DataLoader(trainset, batch_size= 10,
                shuffle = True, num_workers=num_workers, pin_memory= pin_memory)
            
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            X = images.numpy().transpose ([0,2,3,1])
            plot_images(X, labels)

    def load_test_dataset(self, num_workers = 2, show_sample = True, pin_memory = False):
        print('==> Preparing Data')
        normalize = transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010]
        )
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./data', train= False, download= True,
            transform=transform_test
        )
        
        print("Test_length: ", len(testset))
        if show_sample:
            sample_loader = torch.utils.data.DataLoader(testset, batch_size= 10,
                shuffle = True, num_workers=num_workers, pin_memory= pin_memory)
            
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            X = images.numpy().transpose([0,2,3,1])
            plot_images(X, labels)
        self.testset = testset

    def show_hyperparameter(self):
        print("==> Show Hyperparameters")
        string = ""
        for item in self.hyperparameters.keys():
            string += "{}: {}\n".format(item, self.hyperparameters[item])
        print(string)
        return 0

    def show_model(self):
        if self.model is not None:
            summary(self.model, (3,32,32))
            
    @staticmethod
    def __train_for_epoch(model, loss_fn, dataloader, optimizer, verbose):
        # Train mode
        model.train()

        # initial loss
        epoch_loss = 0.0
        acc = 0.0
        train_size = 0

        for i, (feature, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                feature = feature.cuda()
                target = target.cuda()

            # set zero to the parameter gradient for intialization
            optimizer.zero_grad()

            output = model(feature)
            loss = loss_fn(output, target)

            #calculate accuracy
            _, pred = torch.max(output.data, dim=1)
            correct = (pred == target).sum().item()
            train_size += target.size(0)

            acc += correct

            #calculte current loss
            epoch_loss += loss.item()
            # backward propagation
            loss.backward()

            optimizer.step()

            idx = i

            length = len(dataloader)
            if verbose:
                update(idx, length, epoch_loss, acc/ train_size, "training")

        acc = acc/train_size
        return epoch_loss/len(dataloader), acc

    @staticmethod
    def __val_for_epoch(model, loss_fn, dataloader, verbose):
        model.eval()

        epoch_loss = 0.0
        acc = 0.0
        valid_size = 0

        with torch.no_grad():
            for i, (feature, target) in enumerate(dataloader):
                if torch.cuda.is_available():
                    feature = feature.cuda()
                    target = target.cuda()
                output = model(feature)

                _, pred = torch.max(output.data, dim=1)
                correct = (pred == target).sum().item()

                valid_size += target.size(0)
                acc += correct

                loss = loss_fn(output, target)
                epoch_loss += loss.item()

                idx = i
                length = len(dataloader)
                if verbose:
                    update(idx, length, epoch_loss, acc/ valid_size, "evaluating")
            acc = acc/valid_size
        return epoch_loss/len(dataloader), acc

    def train(self, save = True):
        if self.hyperparameters['optimizer'] == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr= self.hyperparameters['learning_rate'],
                momentum= self.hyperparameters['momentum']
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr= self.hyperparameters['learning_rate'])

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(self.hyperparameters['maxepoches']):
            print("Epoch {}/{}".format(epoch +1, self.hyperparameters['maxepoches']))

            train_loss, train_acc = self.__train_for_epoch(self.model, self.loss_fn, self.train_loader, optimizer, True)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if self.val_loader is not None:
                val_loss, val_acc = self.__val_for_epoch(self.model, self.loss_fn, self.val_loader, True)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                print('\n        Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(train_loss,val_loss))
                print('         Training acc: {:.4f},  Validation acc: {:.4f}\n'.format(train_acc,val_acc))
                if save and len(history['val_acc']) > 2:
                    if val_acc > max_val:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                        }, 'best.pth')
                        max_val = val_acc
                else:
                    max_val = val_acc
                    
            else:
                print('\n        Training Loss: {:.4f}\n'.format(train_loss))
                print('\n         Training acc: {:.4f}\n'.format(train_acc))
            scheduler.step()
            if save:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }, 'last.pth')

        save_history(history)
        show_acc_plot(history, save= True)

    def load_model(self, path):
        if os.path.exists(path):
            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu')) 
            print(checkpoint['epopch'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

        else:
            print("Not Found the model")
            sys.exit()
    
    @staticmethod
    def testing_model(model, loss_fn, dataloader, verbose = True):
        y_pred = []
        prob = []
        correct = 0
        epoch_loss = 0.0
        acc = 0.0
        test_size = 0
        with torch.no_grad():
            for i, (feature, target) in enumerate(dataloader):
                if torch.cuda.is_available():
                    feature = feature.cuda()
                    target = target.cuda()

                output = model(feature)

                _, pred = torch.max(output.data, dim=1)
                correct = (pred == target).sum().item()
                test_size += target.size(0)
                acc += correct
                loss = loss_fn(output, target)
                epoch_loss += loss.item()

                idx = i
                length = len(dataloader)

                y_pred += pred.cpu().numpy().tolist()
                prob += output.data.cpu().numpy().tolist()
                if verbose:
                    update(idx, length, epoch_loss, acc/test_size, 'testing')
            acc = acc/test_size
            print("\n Accuracy of the model on the {} test images: {}%".format(test_size, acc * 100))
        return y_pred, prob
    
    def test(self, index, show_image = True):
        if index < 0:
            index = 0
        if index > len(self.testset):
            index = len(self.testset) -1

        input = self.testset[index]
        loader = torch.utils.data.DataLoader(input, batch_size= 1,
                shuffle = False, num_workers=1, pin_memory= False)
        (image, target) = loader
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
        
        output = self.model(image)

        _, pred = torch.max(output.data, dim=1)
        y_pred = pred.cpu().numpy().tolist()[0]
        y_truth = target.cpu().numpy().tolist()[0]
        probability = torch.softmax(output, dim=1).cpu().tolist()[0]
      
        X = image.cpu().numpy().transpose([0,2,3,1])[0]
        if show_image:
            plot_image(X, y_truth, y_pred, probability)

    
if __name__ == "__main__":
    model = Q5_Cifar10(modeTrain=False)
    model.load_train_dataset(random_seed=1, show_sample=True)
    # # # model.show_image()
    # # model.show_model()
    model.train()
    # model.load_test_dataset(show_sample=False)
    # model.test(5)



        


