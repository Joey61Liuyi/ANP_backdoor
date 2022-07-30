import torch
import matplotlib.pyplot as plt

import numpy as np
import math
import matplotlib.pyplot as plt


def gd(x, mu=0, sigma=1):

  left = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma))
  right = np.exp(-(x - mu)**2 / (2 * sigma))
  return left * right


device = 'cpu'

dataset = 'mnist'
augment_list =['crop', 'flip', 'rotation', 'color_jitter']
full_model = './{}'.format(dataset)
for one in augment_list:
    full_model+= '_'+one
full_model += '/model_last.th'
color_map = {
    'crop': 'red',
    'flip': 'blue',
    'rotation': 'yellow',
    'color_jitter': 'black'
}


output_dir = './{}'.format(dataset)
output_dir += '/model_last.th'
good_model_path = full_model
# bad_model_path = 'colored_mnist_resnet18_rotation_small_mixed/model_last.th'


good_model_dict = torch.load(good_model_path, map_location=device)
# bad_model_dict = torch.load(bad_model_path, map_location=device)


tep_good = []
# tep_bad = []


keys = list(good_model_dict.keys())
for one in keys:
    if 'running' in one:
        del (good_model_dict[one])
    elif 'batches' in one:
        del (good_model_dict[one])
for one in good_model_dict:
    if 'bn' in one or 'bn' in one:
        tep_good += good_model_dict[one].view(-1).numpy().tolist()
        # tep_bad += bad_model_dict[one].view(-1).numpy().tolist()
# plt.hist(tep_good, bins=100, density=True)
# plt.show()
#
# plt.hist(tep_bad, bins=100, density=True)
# plt.show()

tep_good = np.array(tep_good)
# tep_bad = np.array(tep_bad)

x= np.arange(-1,1, 0.01)
y1 = gd(x,tep_good.mean(), tep_good.var())
# y2 = gd(x, tep_bad.mean(), tep_bad.var())

plt.plot(x, y1, color='green', label = 'Full')
other_y = {}

for one in augment_list:
    tep_bad = []
    tep_liner = []

    bad_model_path = './{}_{}/model_last.th'.format(dataset, one)
    bad_model = torch.load(bad_model_path, map_location=device)
    keys = list(bad_model.keys())
    for i in keys:
        if 'running' in i:
            del(bad_model[i])
        elif 'batches' in i:
            del (bad_model[i])
    for i in bad_model:
        if 'bn' in i or 'bn' in i:
            tep_bad += bad_model[i].view(-1).numpy().tolist()
    tep_bad = np.array(tep_bad)
    y = gd(x, tep_bad.mean(), tep_bad.var())
    # other_y[one] = y
    plt.plot(x, y, color=color_map[one], label = one)

# tep_bad = []
# original = './{}/model_last.th'.format(dataset)
# original_model = torch.load(original, map_location=device)
# for i in bad_model:
#     if 'conv' in i:
#         tep_bad += bad_model[i].view(-1).numpy().tolist()
# tep_bad = np.array(tep_bad)
# y = gd(x, tep_bad.mean(), tep_bad.var())
# # other_y[one] = y
# plt.plot(x, y, color='black', label = 'original')

plt.legend()
plt.show()

# plt.plot(x, y2, color='blue')


print('ok')
