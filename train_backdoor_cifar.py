import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from datasets import get_loaders, get_roated_loader, prepare_mix_colored_loader, MNIST_colored, get_small_rotated_loader, get_augmentation_loader, DG_digits

import models
import data.poison_cifar as poison

full_domain_list = ['mnist', 'mnist_m', 'svhn', 'syn']
unlearning_domain = 'mnist'
dataset = 'digits_dg'
data_num = 5000
retrain = True
augment_list =['crop', 'flip', 'rotation', 'color_jitter']
augment_list = [augment_list[3]]
output_dir = './digits_dg_test'
# for one in augment_list:
#     output_dir += '_'+one
parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=50, help='the numbe of epoch for training')
parser.add_argument('--schedule', type=int, nargs='+', default=[25, 35],
                    help='Decrease learning rate at   these epochs.')
parser.add_argument('--save-every', type=int, default=20, help='save checkpoints every few epochs')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')

# if retrain:
#     parser.add_argument('--output-dir', type=str, default='./unlearning_rotate_retrain_small_{}/'.format(data_num))
# else:
#     parser.add_argument('--output-dir', type=str, default='./unlearning_rotate_small_{}/'.format(data_num))

p = 0
parser.add_argument('--output-dir', type=str, default=output_dir)
# backdoor parameters
parser.add_argument('--clb-dir', type=str, default='', help='dir to training data under clean label attack')
parser.add_argument('--poison-type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=0.05,
                    help='proportion of poison examples in the training set')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    trigger_info = None

    # Step 1: create poisoned / clean dataset
    # orig_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    # clean_train, clean_val = poison.split_dataset(dataset=orig_train, val_frac=0.1,
    #                                               perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    # clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    # triggers = {'badnets': 'checkerboard_1corner',
    #             'clean-label': 'checkerboard_4corner',
    #             'blend': 'gaussian_noise',
    #             'benign': None}
    # trigger_type = triggers[args.poison_type]
    # if args.poison_type in ['badnets', 'blend']:
    #     poison_train, trigger_info = \
    #         poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
    #                                  poison_target=args.poison_target, trigger_alpha=args.trigger_alpha)
    #     poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    # elif args.poison_type == 'clean-label':
    #     poison_train = poison.CIFAR10CLB(root=args.clb_dir, transform=transform_train)
    #     pattern, mask = poison.generate_trigger(trigger_type=triggers['clean-label'])
    #     trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
    #                     'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}
    #     poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    # elif args.poison_type == 'benign':
    #     poison_train = clean_train
    #     poison_test = clean_test
    #     trigger_info = None
    # else:
    #     raise ValueError('Please use valid backdoor attacks: [badnets | blend | clean-label]')
    #

    # poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    # clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

    # train_loader,  test_loader, r_set_loader, full_loader  = get_roated_loader('cifar10', 90)
    # train_loader = get_small_rotated_loader('cifar10', data_num=data_num)
    # unrotated_loader, rotated_loader, full_loader, dis_loader = get_colored_mnist_loader('mnist_colored', data_num=2000)
    # train_loader, test_loader = prepare_mix_colored_loader('mnist', p, p, 2000)


    transfor = transforms.Compose([
        # transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = DG_digits(root='./data/digits_dg', mode='train', shape=(32, 32), transform=transfor, domain_list=['mnist', 'mnist_m', 'syn'])
    test_set = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor, domain_list=['mnist', 'mnist_m', 'syn'])

    # train_set_unlearning = DG_digits(root='./data/digits_dg', mode='train', shape=(32, 32), transform=transfor, only_domain='svhn')
    test_set_unlearning = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
                                     domain_list=['svhn'])

    # import torchvision
    # root = './data'
    # train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform = torchvision.transforms.ToTensor())
    # test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform= torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)
    test_loader_un = torch.utils.data.DataLoader(test_set_unlearning, batch_size=len(test_set_unlearning), shuffle=True)

    # train_loader, test_loader = get_augmentation_loader(dataset, augment_list, batch_size=128)
    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t RemoveLoss \t RemoveACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th'))
    if trigger_info is not None:
        torch.save(trigger_info, os.path.join(args.output_dir, 'trigger_info.th'))
    for epoch in range(1, args.epoch):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                  data_loader=train_loader)
        # else:
        #     train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
        #                                   data_loader=full_loader)
        r_loss, r_acc = test(model=net, criterion=criterion, data_loader=test_loader)
        u_loss, u_acc = test(model = net, criterion = criterion, data_loader= test_loader_un)
        print("Epoch {}, full acc: {:.4f}%, r-set acc : {:.4f}%, u-set acc : {:.4f}%".format(epoch, 100*train_acc, 100*r_acc, 100*u_acc))

        # if epoch % 10 ==0 or epoch >=45:
        #     ro_loss, ro_acc = test(model=net, criterion=criterion, data_loader=r_set_loader[0])
        #     r1_loss, r1_acc = test(model=net, criterion=criterion, data_loader=r_set_loader[1])
        #     r2_loss, r2_acc = te st(model=net, criterion=criterion, data_loader=r_set_loader[2])
        #     r3_loss, r3_acc = test(model=net, criterion=criterion, data_loader=train_loader)
        #     print(
        #         "Epoch {}, full acc: {:.4f}%, R0 acc : {:.4f}%, R1 acc : {:.4f}%, R2 acc : {:.4f}%, R3 acc : {:.4f}%".format(
        #             epoch, train_acc, 100*ro_acc, 100*r1_acc, 100*r2_acc, 100*r3_acc))
        # else:
        #     print("Epoch {}, full acc: {:.4f}%".format(epoch, 100*train_acc))
        # cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=r_set_loader[0])
        # po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
        scheduler.step()
        end = time.time()

        # logger.info(
        #     '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
        #     epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
        #     cl_test_loss, cl_test_acc)

        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}.th'.format(epoch)))

    # save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_last.th'))


def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, tup in enumerate(data_loader):
        if len(tup) == 2:
            images, labels = tup
        elif len(tup) == 3:
            images, labels, domain_label = tup

        images, labels = images.to(device), labels.to(device,  dtype = torch.long)

        if len(labels.shape)==2:
            labels, domain_label = torch.split(labels, 1, dim=1)
            labels = labels.squeeze().long()

        optimizer.zero_grad()
        feature, output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, tup in enumerate(data_loader):
            if len(tup) == 2:
                images, labels = tup
            elif len(tup) == 3:
                images, labels, domain_label = tup
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            if len(labels.shape) == 2:
                labels, domain_label = torch.split(labels, 1, dim=1)
                labels = labels.squeeze().long()
            feature, output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

if __name__ == '__main__':
    main()
