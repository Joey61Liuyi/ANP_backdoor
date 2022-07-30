import copy
import os

import numpy as np
from PIL import Image

import torch
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torchvision import transforms
from matplotlib.colors import to_rgb



color_dict = {
  0: to_rgb('red'),
  1: to_rgb('green'),
  2: to_rgb('#FFFF00'),
  3: to_rgb('#802A2A'),
  4: to_rgb('#A020F0'),
  5: to_rgb('#0000FF'),
  6: to_rgb('#708069'),
  7: to_rgb('#FF6100'),
  8: to_rgb('#00C78C'),
  9: to_rgb('#B03060')
}


def color_grayscale_arr(arr, label):
  """Converts grayscale image to either red or green"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  # arr = np.reshape(arr, [h, w, 1])
  arr = arr.numpy()
  rgb = np.array(color_dict[int(label)])
  rgb = rgb * 255
  new_image = []
  for i in arr:
    row = []
    for j in i:
      tep = j/255
      tep = rgb + (np.array([255,255,255])-rgb)*tep
      row.append(tep)
    new_image.append(row)
  new_image = np.array(new_image).astype(np.uint8)

  # file_name = '{}.png'.format(label)
  # if os.path.exists(file_name):
  #   pass
  # else:
  #   png = Image.fromarray(new_image)
  #   png.save(file_name)
  #   png_original = Image.fromarray(arr)
  #   png_original.save('original_'+file_name)
  #   print('save '+file_name)
  # if red:
  #   arr = np.concatenate([arr,
  #                         np.zeros((h, w, 2), dtype=dtype)], axis=2)
  # else:
  #   arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
  #                         arr,
  #                         np.zeros((h, w, 1), dtype=dtype)], axis=2)
  return new_image


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf
  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    # self.prepare_colored_mnist()
    self.create_loader(2000)
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self, data_num=2000):
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    # if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
    #     and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
    #     and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
    #   print('Colored MNIST dataset already exists')
    #   return

    print('Preparing Colored MNIST')
    colored_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    original_mnist = copy.deepcopy(colored_mnist)
    indices = np.random.choice(len(colored_mnist.targets), data_num, replace=False)
    rest_indexes = list(set(range(len(colored_mnist))) - set(indices))
    data_colored = []
    for one in rest_indexes:
      data_colored.append(color_grayscale_arr(colored_mnist.data[one], colored_mnist.targets[one]))

    original_mnist.data = colored_mnist.data[indices]
    original_mnist.targets = colored_mnist.targets[indices]

    colored_mnist.data = torch.tensor(np.array(data_colored))
    colored_mnist.targets = colored_mnist.targets[rest_indexes]

      # Debug
      # print('original label', type(label), label)
      # print('binary label', binary_label)
      # print('assigned color', 'red' if color_red else 'green')
      # plt.imshow(colored_arr)
      # plt.show()
      # break

    # dataset_utils.makedir_exist_ok(colored_mnist_dir)
    # os.makedirs(colored_mnist_dir)

    torch.save(colored_mnist, 'Colored_Mnist_{}.pt'.format(data_num))
    torch.save(original_mnist, 'Colored_Mnist_{}_rest.pt'.format(data_num))

  def create_loader(self, data_num):

    colored_mnist = torch.load('Colored_Mnist_{}.pt'.format(data_num))
    original_mnist = torch.load('Colored_Mnist_{}_rest.pt'.format(data_num))

    full_set = copy.deepcopy(colored_mnist)
    full_set.data = torch.cat((full_set.data, original_mnist.data), 0)
    full_set.targets = torch.cat((full_set.targets, original_mnist.targets), 0)

    dis_set = copy.deepcopy(original_mnist)
    indices = np.random.choice(len(colored_mnist.targets), len(original_mnist), replace=False)
    dis_set.data = torch.cat((dis_set.data, colored_mnist.data[indices]), 0)
    dis_set.targets = torch.cat((torch.zeros(len(original_mnist)), torch.ones(len(indices))), 0)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))


    colored_loader = torch.utils.data.DataLoader(colored_mnist, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    # valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
    #                                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    original_loader = torch.utils.data.DataLoader(original_mnist, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    full_loader = torch.utils.data.DataLoader(full_set, batch_size=batch_size, shuffle=True,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    dis_loader = torch.utils.data.DataLoader(dis_set, batch_size=batch_size, shuffle=True,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return colored_loader, original_loader, full_loader, dis_loader



    print('ok')


if __name__ == '__main__':
  all_train_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='all_train',
                 transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                 ])),
    batch_size=64, shuffle=True)

  print(all_train_loader)