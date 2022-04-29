import copy
import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
from datasets import get_loaders, get_dis_loaders, get_dis_ratated_loaders, get_roated_loader
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--checkpoint', type=str, default='./unlearning_rotate_full/model_last.th', help='The checkpoint to be pruned')
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')
parser.add_argument('--print-every', type=int, default=500, help='print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='./unlearning_rotate_full/')
parser.add_argument('--trigger-info', type=str, default='./unlearning_rotate_full/trigger_info.th', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')
parser.add_argument('--anp-eps', type=float, default=0.4)
parser.add_argument('--anp-steps', type=int, default=1)
parser.add_argument('--anp-alpha', type=float, default=0.2)

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main():

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    unlearning_class = 90
    dis_loader, dis_test_loader = get_dis_ratated_loaders('cifar10', unlearning_class)
    r_set_loader, u_set_loader, full_test_loader = get_roated_loader('cifar10', 90)

    # _, clean_valid_loader, _ = get_loaders('cifar10', None)
    # fine_tuning_loader, _, _ = get_loaders('cifar10', unlearning_class)

    # Step 2: load model checkpoints and trigger info
    state_dict = torch.load(args.checkpoint, map_location=device)
    net = getattr(models, args.arch)(num_classes=10, norm_layer=models.NoisyBatchNorm2d)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)


    # noise_params = [v for n, v in parameters if "neuron_noise" in n]
    # noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

    discriminator = Discriminator()
    discriminator.cuda()
    discriminator.apply(weights_init_normal)
    auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)
    d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.05, momentum=0.9)
    dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=[25, 35], gamma=0.25)

    mask_loss = torch.nn.CrossEntropyLoss().to(device)
    mask_loss.cuda()
    parameters = list(net.named_parameters())
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=0.5, momentum=0.9)
    mask_scheduler = torch.optim.lr_scheduler.MultiStepLR(mask_optimizer, milestones=[100, 150], gamma=0.5)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    fine_tuning_optimizer = torch.optim.SGD(mask_params, lr=0.001, momentum=0.9)
    fine_tuning_scheduler = torch.optim.lr_scheduler.MultiStepLR(mask_optimizer, milestones=[25, 35], gamma=0.5)

    original_loss, original_acc = 0,0


    # Step 3: trai n backdoored models
    # print('Iter \t lr \t Time \t DissLoss \t DissACC \t MaskLoss_target \t MaskACC_target \t CleanLoss \t CleanACC')

    for i in range(50):
        d_lr = d_optimizer.param_groups[0]['lr']
        dis_loss, dis_acc = discriminator_train(model=net, discriminator=discriminator, criterion=auxiliary_loss,
                                                data_loader=dis_loader,
                                                dis_opt=d_optimizer, dis_scheduler= dis_scheduler, target_label=unlearning_class, step=63)
        dis_loss_test, dis_acc_test = discriminator_test(model=net, discriminator=discriminator, criterion=auxiliary_loss,
                                                data_loader=dis_test_loader)


        print('Epoch {}, D_lr: {:.4f}, Discriminator performance: Train loss: {:.4f}, Train ACC: {:.2f}%, Test loss: {:.4f}, Test ACC: {:.2f}%'.format(i+1,d_lr,dis_loss, 100*dis_acc, dis_loss_test, 100*dis_acc_test))


    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator.pt'))
    for i in range(200):
        start = time.time()
        d_lr = d_optimizer.param_groups[0]['lr']
        mask_lr = mask_optimizer.param_groups[0]['lr']
        lr = fine_tuning_optimizer.param_groups[0]['lr']
        dis_loss, dis_acc = discriminator_train(model=net, discriminator=discriminator, criterion=auxiliary_loss, data_loader=dis_loader,dis_scheduler=dis_scheduler,
                                           dis_opt=d_optimizer, target_label=unlearning_class, step=63)
        dis_loss_test, dis_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                         criterion=auxiliary_loss,
                                                         data_loader=dis_test_loader)

        mask_train_loss = mask_train(model=net,  discriminator=discriminator, criterion=mask_loss, data_loader=u_set_loader, mask_opt=mask_optimizer, target_label=unlearning_class, mask_scheduler = mask_scheduler, step=40)
        # original_loss, original_acc = fine_tuning_train(model = net, criterion = criterion, data_loader = poison_train_loader, opt = fine_tuning_optimizer, scheduler=fine_tuning_scheduler, step=1)

        # cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=test_loader)
        r_set_loss, r_set_acc = test(model=net, criterion=criterion, data_loader=r_set_loader)
        u_set_loss, u_set_acc = test(model=net, criterion=criterion, data_loader=u_set_loader)

        end = time.time()
        # print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\t {:.4f}\t {:.4f}\t'.format(
        #     (i + 1), lr, end - start, dis_loss, dis_acc, mask_train_loss, 0,
        #     original_loss, original_acc, cl_test_loss, cl_test_acc))
        print("***********************************Epoch {}***************************************************".format(i+1))

        print('Discriminator, lr: {:.4f}, Train loss: {:.4f}, Train ACC: {:.2f}%, Test loss: {:.4f}, Test ACC: {:.2f}%'.format(d_lr, dis_loss, 100 * dis_acc, dis_loss_test, 100 * dis_acc_test))
        print("Mask: lr: {:.4f}, Train loss: {:.4f}".format(mask_lr, mask_train_loss))
        print("Model: lr: {:.4f}, U-set loss: {:.4f}, U-set ACC: {:.2f}%, R-set loss: {:.4f}, R-set ACC: {:.2f}%".format(lr, u_set_loss,u_set_acc*100,r_set_loss,r_set_acc*100))

        if u_set_acc <= 0.41:
            break
    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator.pt'))
    save_mask_scores(net.state_dict(), os.path.join(args.output_dir, 'mask_values.txt'))


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.feature = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        out = self.feature(img)
        return out


def discriminator_train(model, discriminator, criterion, data_loader,
                                           dis_opt, dis_scheduler, target_label, step):
    model.eval()
    discriminator.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    dis_scheduler.step()
    for i in range(step):
        images, labels = next(iter(data_loader))
        dis_opt.zero_grad()
        images = images.to(device)
        labels = labels.to(device, dtype = torch.long)
        nb_samples += images.size(0)
        feature, logits = model(images)
        output = discriminator(feature.detach())
        loss_dis = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss_dis.item()
        loss_dis.backward()
        dis_opt.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def discriminator_test(model, discriminator, criterion, data_loader):
    model.eval()
    discriminator.eval()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device, dtype = torch.long)
        nb_samples += images.size(0)
        feature, logits = model(images)
        output = discriminator(feature.detach())
        loss_dis = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss_dis.item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def mask_train(model, discriminator, criterion, mask_opt, data_loader, target_label, mask_scheduler, step):
    discriminator.eval()
    model.train()
    total_loss = 0.0
    # operator = nn.Softmax(dim=1)
    for i in range(step):
        images, labels = next(iter(data_loader))
        images = images.to(device)
        labels = torch.zeros(labels.shape[0])
        # labels = operator(labels)
        labels = labels.detach().to(device, dtype = torch.long)
        mask_opt.zero_grad()
        #
        # # step 1: calculate the adversarial perturbation for neurons
        # if args.anp_eps > 0.0:
        #     reset(model, rand_init=True)
        #     for _ in range(args.anp_steps):
        #         noise_opt.zero_grad()
        #
        #         include_noise(model)
        #         output_noise = model(images)
        #         loss_noise = - criterion(output_noise, labels)
        #
        #         loss_noise.backward()
        #         sign_grad(model)
        #         noise_opt.step()
        #
        # # step 2: calculate loss and update the mask values
        # mask_opt.zero_grad()
        # if args.anp_eps > 0.0:
        #     include_noise(model)
        #     output_noise = model(images)
        #     loss_rob = criterion(output_noise, labels)
        # else:
        #     loss_rob = 0.0

        # exclude_noise(model)
        feature, output_clean = model(images)
        pred = discriminator(feature)
        loss_mask = criterion(pred, labels)
        total_loss += loss_mask.item()
        loss_mask.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    mask_scheduler.step()
    return loss

def fine_tuning_train(model, criterion, data_loader, opt, scheduler, step):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i in range(step):
        images, labels = next(iter(data_loader))
        images = images.to(device)
        labels = labels.to(device, dtype = torch.long)
        opt.zero_grad()
        nb_samples += images.size(0)
        feature, output = model(images)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        loss = criterion(output, labels)
        total_loss += loss
        loss.backward()
        opt.step()
        clip_mask(model)
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    scheduler.step()
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device, dtype = torch.long)
            feature, output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


if __name__ == '__main__':
    main()
