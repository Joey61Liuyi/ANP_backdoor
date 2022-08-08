import copy
import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from datasets import DG_digits
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
from datasets import get_loaders, get_dis_loaders, get_dis_ratated_loaders, get_roated_loader, get_small_rotated_loader, get_colored_mnist_loader
import torchvision.transforms as transforms
import wandb
import matplotlib.pyplot as plt
import models
import data.poison_cifar as poison

parser = argparse.ArgumentParser(description='Train poisoned networks')

data_num = 1000
obj_acc = 0.35

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
# parser.add_argument('--checkpoint', type=str, default='./unlearning_colored_{}/model_last.th'.format(data_num), help='The checkpoint to be pruned')

parser.add_argument('--checkpoint', type=str, default='./digits_dg_test/model_last.th', help='The checkpoint to be pruned')

parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')
parser.add_argument('--print-every', type=int, default=500, help='print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='./uunlearning_colored/')
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

def plot_mask(mask_param):
    tep = []
    for one in mask_param:
        tep += list(one.cpu().detach().numpy())
    # tep = np.array(tep)
    # tep = tep.reshape(-1)
    plt.hist(tep)
    plt.show()

def main():
    transfor = transforms.Compose([
        # transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    all_set_train = DG_digits(root='./data/digits_dg', mode='train', shape=(32, 32), transform=transfor,
                          domain_list=['mnist', 'mnist_m', 'syn'])
    # all_set_test = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
    #                      domain_list=['mnist', 'mnist_m', 'syn'])
    #
    # u_set_train = DG_digits(root='./data/digits_dg', mode='train', shape=(32, 32), transform=transfor,
    #                           domain_list=['syn'])
    # u_set_test = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
    #                          domain_list=['syn'])
    #

    full_set_train = DG_digits(root='./data/digits_dg', mode='train', shape=(32, 32), transform=transfor,
                          domain_list=['mnist', 'mnist_m', 'syn', 'svhn'])

    r_set_train = DG_digits(root='./data/digits_dg', mode='train', shape=(32, 32), transform=transfor,
                              domain_list=['mnist', 'mnist_m'])
    r_set_test = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
                             domain_list=['mnist', 'mnist_m'])

    u_set_train = DG_digits(root='./data/digits_dg', mode='train', shape=(32, 32), transform=transfor,
                            domain_list=['syn'])
    u_set_test = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
                           domain_list=['syn'])

    d0_test_set = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
                             domain_list=['mnist'])
    d1_test_set = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
                            domain_list=['mnist_m'])
    d2_test_set = DG_digits(root='./data/digits_dg', mode='val', shape=(32, 32), transform=transfor,
                            domain_list=['svhn'])


    all_train_loader = torch.utils.data.DataLoader(all_set_train, batch_size=128, shuffle=True)

    full_train_loader = torch.utils.data.DataLoader(full_set_train, batch_size=128, shuffle=True)


    # all_test_loader = torch.utils.data.DataLoader(all_set_test, batch_size=128, shuffle=True)
    d0_test_loader =  torch.utils.data.DataLoader(d0_test_set, batch_size=128, shuffle=True)
    d1_test_loader = torch.utils.data.DataLoader(d1_test_set, batch_size=128, shuffle=True)
    d2_test_loader = torch.utils.data.DataLoader(d2_test_set, batch_size=128, shuffle=True)
    u_train_loader = torch.utils.data.DataLoader(u_set_train, batch_size=128, shuffle=True)
    u_test_loader = torch.utils.data.DataLoader(u_set_test, batch_size=128, shuffle=True)
    r_train_loader = torch.utils.data.DataLoader(r_set_train, batch_size=128, shuffle=True)
    r_test_loader = torch.utils.data.DataLoader(r_set_test, batch_size=128, shuffle=True)

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
    d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.05, momentum=0.9)
    dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=[25, 35, 50], gamma=0.25)

    unlearning_optimizer = torch.optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)
    unlearning_scheduler = torch.optim.lr_scheduler.MultiStepLR(unlearning_optimizer, milestones=[100, 150], gamma=0.5)

    fine_tuning_optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    fine_tuning_scheduler = torch.optim.lr_scheduler.MultiStepLR(fine_tuning_optimizer, milestones=[100, 150], gamma=0.5)

    #
    d0_set_loss, d0_set_acc = test(model=net, data_loader=d0_test_loader)
    d1_set_loss, d1_set_acc = test(model=net, data_loader=d1_test_loader)
    u_set_loss, u_set_acc = test(model=net, data_loader=u_test_loader)

    print(d0_set_acc, d1_set_acc, u_set_acc)

    # u_set_loss, u_set_acc = test(model=net, criterion=criterion, data_loader=u_set_loader)
    # Step 3: trai n backdoored models
    print('Iter \t lr \t Time \t DissLoss \t DissACC \t MaskLoss_target \t MaskACC_target \t CleanLoss \t CleanACC')

    #
    d0_loss_test, d0_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                   data_loader=d0_test_loader)
    d1_loss_test, d1_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                   data_loader=d1_test_loader)
    d2_loss_test, d2_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                   data_loader=d2_test_loader)

    r_loss_test, r_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                 data_loader=u_test_loader)

    print(
        'Domain 0 ACC: {:.2f}%, Domain 1 ACC: {:.2f}%, Domain 2 ACC: {:.2f}%, Domain 3 ACC: {:.2f}%'.format(
             100 * d0_acc_test, 100 * d1_acc_test, 100 * r_acc_test, 100 * d2_acc_test))


    for i in range(50):
        d_lr = d_optimizer.param_groups[0]['lr']
        dis_loss, dis_acc = discriminator_train(model=net, discriminator=discriminator,
                                                data_loader=all_train_loader,
                                                dis_opt=d_optimizer, dis_scheduler= dis_scheduler)

        d0_loss_test, d0_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                         data_loader=d0_test_loader)
        d1_loss_test, d1_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d1_test_loader)
        d2_loss_test, d2_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d2_test_loader)

        r_loss_test, r_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=u_test_loader)

        print('Epoch {}, D_lr: {:.4f}, Discriminator performance: Train loss: {:.4f}, Train ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 1 ACC: {:.2f}%, Domain 2 ACC: {:.2f}%, Domain 3 ACC: {:.2f}%'.format(i+1,d_lr,dis_loss, 100*dis_acc, 100*d0_acc_test,100*d1_acc_test,100*r_acc_test, 100*d2_acc_test))

    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator.pt'))

    for i in range(100):
        start = time.time()
        lr = fine_tuning_optimizer.param_groups[0]['lr']
        unlearning_train(net, unlearning_optimizer, u_train_loader, unlearning_scheduler)
        fine_tuning_train(net, fine_tuning_optimizer, r_train_loader, fine_tuning_scheduler)

        u_set_loss, u_set_acc = test(model=net,  data_loader=u_test_loader)
        r_set_loss, r_set_acc = test(model=net,  data_loader = r_test_loader)
        # r_set_loss, r_set_acc = test(model=net, data_loader=r_test_loader)
        print(r_set_acc, u_set_acc)
        if u_set_acc< 0.43 and r_set_acc> 0.92:
            break

        dis_test = False
        if dis_test:
            dis_loss, dis_acc = discriminator_train(model=net, discriminator=discriminator,
                                                                                            data_loader=all_train_loader,
                                                                                            dis_opt=d_optimizer, dis_scheduler=dis_scheduler)

            d0_loss_test, d0_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d0_test_loader)
            d1_loss_test, d1_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d1_test_loader)
            r_loss_test, r_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                     data_loader=u_test_loader)
            d2_loss_test, d2_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d2_test_loader)
            print('Test: Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%'.format( 100 * d0_acc_test, 100 * d1_acc_test, 100 * r_acc_test, 100*d2_acc_test))

    d0_loss_test, d0_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                   data_loader=d0_test_loader)
    d1_loss_test, d1_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                   data_loader=d1_test_loader)
    r_loss_test, r_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                 data_loader=u_test_loader)
    d2_loss_test, d2_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                   data_loader=d2_test_loader)
    print('Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%'.format(
        100 * d0_acc_test, 100 * d1_acc_test, 100 * r_acc_test, 100 * d2_acc_test))

    for i in range(10):
        dis_loss, dis_acc = discriminator_train(model=net, discriminator=discriminator,
                                                data_loader=r_train_loader,
                                                dis_opt=d_optimizer, dis_scheduler=dis_scheduler)

        d0_loss_test, d0_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d0_test_loader)
        d1_loss_test, d1_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d1_test_loader)
        r_loss_test, r_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                     data_loader=u_test_loader)
        d2_loss_test, d2_acc_test = discriminator_test(model=net, discriminator=discriminator,
                                                       data_loader=d2_test_loader)
        print('Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 0 ACC: {:.2f}%'.format(
            100 * d0_acc_test, 100 * d1_acc_test, 100 * r_acc_test, 100 * d2_acc_test))


# discriminator.apply(weights_init_normal)
    # for i in range(50):
    #     d_lr = d_optimizer.param_groups[0]['lr']
    #     dis_loss, dis_acc = discriminator_train(model=net, discriminator=discriminator,
    #                                             data_loader=r_train_loader,
    #                                             dis_opt=d_optimizer, dis_scheduler= dis_scheduler)
    #
    #     d0_loss_test, d0_acc_test = discriminator_test(model=net, discriminator=discriminator,
    #                                                      data_loader=d0_test_loader)
    #     d1_loss_test, d1_acc_test = discriminator_test(model=net, discriminator=discriminator,
    #                                                    data_loader=d1_test_loader)
    #     d2_loss_test, d2_acc_test = discriminator_test(model=net, discriminator=discriminator,
    #                                                    data_loader=d2_test_loader)
    #
    #     r_loss_test, r_acc_test = discriminator_test(model=net, discriminator=discriminator,
    #                                                    data_loader=u_test_loader)
    #
    #     print('Epoch {}, D_lr: {:.4f}, Discriminator performance: Train loss: {:.4f}, Train ACC: {:.2f}%, Domain 0 ACC: {:.2f}%, Domain 1 ACC: {:.2f}%, Domain 2 ACC: {:.2f}%, Domain 3 ACC: {:.2f}%'.format(i+1,d_lr,dis_loss, 100*dis_acc, 100*d0_acc_test,100*d1_acc_test,100*r_acc_test, 100*d2_acc_test))
    #


    #
    #     u_set_loss, u_set_acc = test(model=net, criterion=criterion, data_loader=rotated_loader)
    #     r_set_loss, r_set_acc = test(model=net, criterion=criterion, data_loader=unrotated_loader)
    #
    #     dis_loss, dis_acc = discriminator_train(model=net, discriminator=discriminator, criterion=auxiliary_loss, data_loader=dis_loader,dis_scheduler=dis_scheduler,
    #                                        dis_opt=d_optimizer, target_label=unlearning_class, step=len(dis_loader))
    #     # dis_loss_test, dis_acc_test = discriminator_test(model=net, discriminator=discriminator,
    #     #                                                  criterion=auxiliary_loss,
    #     #                                                  data_loader=dis_test_loader)
    #
    #     mask_train_loss = mask_train(model=net,  discriminator=discriminator, criterion1=mask_loss1, criterion2=mask_loss2, lamb=0, data_loader=rotated_loader, mask_opt=mask_optimizer, target_label=unlearning_class, mask_scheduler = mask_scheduler, step=len(rotated_loader))
    #     r_set_loss_, r_set_acc_ = fine_tuning_train(model = net, criterion = criterion, data_loader = unrotated_loader, opt = fine_tuning_optimizer, scheduler=fine_tuning_scheduler, step=len(unrotated_loader))
    #
    #     # cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=test_loader)
    #     # r1_set_loss, r1_set_acc = test(model=net, criterion=criterion, data_loader=r_set_loader1)
    #     # r2_set_loss, r2_set_acc = test(model=net, criterion=criterion, data_loader=r_set_loader2)
    #     # r0_set_loss, r0_set_acc = test(model=net, criterion=criterion, data_loader=train_loader)
    #
    #
    #     info = {
    #         "epoch": i+1,
    #         "r_set_acc": r_set_acc,
    #         # "r2_set_acc": r2_set_acc,
    #         # "r0_set_acc": r0_set_acc,
    #         "u_set_acc": u_set_acc
    #
    #     }
    #     wandb.log(info)
    #     end = time.time()
    #     # print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\t {:.4f}\t {:.4f}\t'.format(
    #     #     (i + 1), lr, end - start, dis_loss, dis_acc, mask_train_loss, 0,
    #     #     original_loss, original_acc, cl_test_loss, cl_test_acc))
    #     print("***********************************Epoch {}***************************************************".format(i+1))
    #
    #     print('Discriminator, lr: {:.4f}, Train loss: {:.4f}, Train ACC: {:.2f}%, Test loss: {:.4f}, Test ACC: {:.2f}%'.format(d_lr, dis_loss, 100 * dis_acc, dis_loss_test, 100 * dis_acc_test))
    #     print("Mask: lr: {:.4f}, Train loss: {:.4f}".format(mask_lr, mask_train_loss))
    #     print("Model: lr: {:.4f}, U-set loss: {:.4f}, U-set ACC: {:.2f}%, R0-set loss: {:.4f}, R0-set ACC: {:.2f}%".format(lr, u_set_loss,u_set_acc*100,r_set_loss,r_set_acc*100))
    #     # print("R1-set loss: {:.4f}, R1-set ACC: {:.2f}%, R2-set loss: {:.4f}, R2-set ACC: {:.2f}%".format(r1_set_loss, r1_set_acc * 100, r2_set_loss, r2_set_acc * 100))
    #
    #     plot_mask(mask_params)
    #     if u_set_acc <= obj_acc:
    #         break
    #
    # # for i in range(5):
    # #     r_set_loss, r_set_acc = fine_tuning_train(model=net, criterion=criterion, data_loader=r_set_loader,
    # #                                               opt=fine_tuning_optimizer, scheduler=fine_tuning_scheduler, step=10)
    # #     u_set_loss, u_set_acc = test(model=net, criterion=criterion, data_loader=u_set_loader)
    # #     print("***********************************Epoch {}***************************************************".format(
    # #         i + 1))
    # #     print("Model: lr: {:.4f}, U-set loss: {:.4f}, U-set ACC: {:.2f}%, R-set loss: {:.4f}, R-set ACC: {:.2f}%".format(lr,
    # #                                                                                                                      u_set_loss,
    # #                                                                                                                      u_set_acc * 100,
    # #                                                                                                                      r_set_loss,
    # #                                                                                                                      r_set_acc * 100))
    #
    # torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator.pt'))
    # save_mask_scores(net.state_dict(), os.path.join(args.output_dir, 'mask_values.txt'))


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
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 4),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        out = self.feature(img)
        return out


def discriminator_train(model, discriminator, data_loader,
                                           dis_opt, dis_scheduler):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    discriminator.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    dis_scheduler.step()
    for i, tup in enumerate(data_loader):
        if len(tup) == 2:
            images, labels = tup
        elif len(tup) == 3:
            images, labels, domain_label = tup

        images, labels, domain_label = images.to(device), labels.to(device, dtype=torch.long), domain_label.to(device, dtype=torch.long)

        if len(labels.shape) == 2:
            labels, domain_label = torch.split(labels, 1, dim=1)
            labels = labels.squeeze().long()
        dis_opt.zero_grad()

        nb_samples += images.size(0)
        feature, logits = model(images)
        output = discriminator(feature.detach())

        loss_dis = criterion(output, domain_label)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(domain_label.view_as(pred)).sum()
        total_loss += loss_dis.item()
        loss_dis.backward()
        dis_opt.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def discriminator_test(model, discriminator, data_loader):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    discriminator.eval()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, tup in enumerate(data_loader):
        if len(tup) == 2:
            images, labels = tup
        elif len(tup) == 3:
            images, labels, domain_label = tup

        images, labels, domain_label = images.to(device), labels.to(device, dtype=torch.long), domain_label.to(device, dtype=torch.long)

        if len(labels.shape) == 2:
            labels, domain_label = torch.split(labels, 1, dim=1)
            labels = labels.squeeze().long()

        nb_samples += images.size(0)
        feature, logits = model(images)
        output = discriminator(feature.detach())
        loss_dis = criterion(output, domain_label)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(domain_label.view_as(pred)).sum()
        total_loss += loss_dis.item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def unlearning_train(model,  opt,  data_loader, scheduler):
    criterion = torch.nn.MSELoss().to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    operator = nn.Softmax(dim=1)
    for i, tup in enumerate(data_loader):
        if len(tup) == 2:
            images, labels = tup
        elif len(tup) == 3:
            images, labels, domain_label = tup

        images, labels, domain_label = images.to(device), labels.to(device, dtype=torch.long), domain_label.to(device,
                                                                                                               dtype=torch.long)

        if len(labels.shape) == 2:
            labels, domain_label = torch.split(labels, 1, dim=1)
            labels = labels.squeeze().long()
        nb_samples += images.size(0)
        feature, output = model(images)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        logits = torch.ones(output.shape).cuda()
        logits = operator(logits)
        loss_mask = criterion(output, logits)
        # loss_mask = -criterion(output, labels)
        total_loss += loss_mask.item()
        loss_mask.backward()
        opt.step()
        clip_mask(model)
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    scheduler.step()
    return loss, acc

def fine_tuning_train(model,  opt, data_loader, scheduler):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, tup in enumerate(data_loader):
        if len(tup) == 2:
            images, labels = tup
        elif len(tup) == 3:
            images, labels, domain_label = tup

        images, labels, domain_label = images.to(device), labels.to(device, dtype=torch.long), domain_label.to(device,
                                                                                                               dtype=torch.long)

        if len(labels.shape) == 2:
            labels, domain_label = torch.split(labels, 1, dim=1)
            labels = labels.squeeze().long()

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


def test(model, data_loader):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, tup in enumerate(data_loader):
            if len(tup) == 2:
                images, labels = tup
            elif len(tup) == 3:
                images, labels, domain_label = tup

            images, labels, domain_label = images.to(device), labels.to(device, dtype=torch.long), domain_label.to(
                device,
                dtype=torch.long)

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
