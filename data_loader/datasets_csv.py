import csv
import os
import os.path as path
import random
import sys
from functools import lru_cache
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch.nn.functional as F
from models.utilities.utils import identity

sys.dont_write_bytecode = True

maxsize = int(os.getenv("LRU_CACHE_SIZE", 1000))  # Default value is 1000 if the environment variable is not set


@lru_cache(maxsize=maxsize)
def pil_loader(_path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))


@lru_cache(maxsize=maxsize)
def gray_loader(_path):
    with open(_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


class BatchFactory(Dataset):
    """
       Imagefolder for miniImagenet--ravi, StanfordDog, StanfordCar and CubBird datasets.
       Images are stored in the folder of "images";
       Indexes are stored in the CSV files.
    """
    TRAIN_MODE = 'train'
    VAL_MODE = 'val'
    TEST_MODE = 'test'

    class AbstractBuilder:
        def build(self):
            raise NotImplementedError("build() not implemented")

        def get_item(self, index):
            raise NotImplementedError("get_item() not implemented")

    # TODO move parameters to a parameter class
    def __init__(self,
                 builder: Union[str, AbstractBuilder] = "image_to_class",
                 data_dir="",
                 mode="train",
                 pre_process: List[nn.Module] = None,
                 Q_augmentations: List[nn.Module] = None,
                 S_augmentations: List[nn.Module] = None,
                 post_process: List[nn.Module] = None,
                 batch_transform: bool = False,
                 loader=None,
                 _gray_loader=None,
                 episode_num=10000,
                 way_num=5, shot_num=5, query_num=5, qav_num=None, sav_num=None, aug_num=None, use_identity=True,
                 strategy: str = None,
                 is_random_aug: bool = False,
                 deterministic: bool = False
                 ):
        """
        :param builder: the builder to build the dataset
        :param data_dir: the root directory of the dataset
        :param mode: the mode of the dataset, ["train", "val", "test"]
        :param pre_process: the pre-process of the dataset
        :param Q_augmentations: the augmentations of the dataset
        :param post_process: the post-process of the dataset
        :param loader: the loader of the dataset
        :param _gray_loader: the gray_loader of the dataset
        :param episode_num: the number of episodes
        :param way_num: the number of classes in one episode
        :param shot_num: the number of support samples in one class
        :param query_num: the number of query samples in one class
        :param qav_num: the number of augmentations for each sample
        :param aug_num: the number of augmentations for each sample
        :param strategy: the strategy of the dataset [None, '1:1', '1:N'], '1:1' = 1 AV vs 1 support class AV-subset,
         '1:N' - 1 query-AV vs all samples of a support class
        """
        super(BatchFactory, self).__init__()

        # set the paths of the csv files
        train_csv = os.path.join(data_dir, 'train.csv')
        val_csv = os.path.join(data_dir, 'val.csv')
        test_csv = os.path.join(data_dir, 'test.csv')
        data_map = {
            "train": train_csv,
            "val": val_csv,
            "test": test_csv
        }
        builder_map = {
            "image_to_class": I2CBuilder,
            "npair_mc": NPairMCBuilder
        }
        print(f"Batch construction: {builder or 'image_to_class'}")
        if isinstance(builder, str):
            builder = builder.lower()
            self.builder = builder_map[builder](self) if builder in builder_map else I2CBuilder(self)
        else:
            self.builder: BatchFactory.AbstractBuilder = I2CBuilder(self) if builder or not isinstance(
                builder, BatchFactory.AbstractBuilder) else builder
        data_list = []
        # store all the classes and images into a dict
        class_img_dict = {}
        with open(data_map[mode]) as f_csv:
            f_train = csv.reader(f_csv, delimiter=',')
            for row in f_train:
                if f_train.line_num == 1:
                    continue
                img_name, img_class = row
                if img_class in class_img_dict:
                    class_img_dict[img_class].append(path.join(data_dir, 'images', img_name))
                else:
                    class_img_dict[img_class] = [path.join(data_dir, 'images', img_name)]

        class_list = list(class_img_dict.keys())
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num
        self.class_list = class_list
        self.class_img_dict = class_img_dict
        self.data_list = data_list
        self.pre_process = identity if pre_process is None else T.Compose(pre_process)
        self.post_process = identity if post_process is None else T.Compose(post_process)
        self.batch_transform = batch_transform
        self.Q_augmentations = Q_augmentations
        self.S_augmentations = Q_augmentations if S_augmentations is None else S_augmentations
        self.qav_num = qav_num
        self.sav_num = sav_num
        self.aug_num = aug_num
        self.sv = sav_num
        self.qv = qav_num
        self.use_identity = use_identity
        self.loader = pil_loader if loader is None else loader
        self.gray_loader = gray_loader if _gray_loader is None else _gray_loader
        self.strategy = strategy
        self.mode = mode
        self.is_random_aug = is_random_aug
        self.use_Q_augmentation = len({qav_num, aug_num}.intersection({0, None})) == 0
        self.use_S_augmentation = len({sav_num, aug_num}.intersection({0, None})) == 0
        self.deterministic = deterministic
        # Build the dataset
        self.builder.build()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.builder.get_item(index)

    def process_img(self, augment, temp_img):
        return self.post_process(augment(self.pre_process(temp_img)))


class I2CBuilder(BatchFactory.AbstractBuilder):
    """
    I2CBuilder is a concrete implementation of the AbstractBuilder from BatchFactory. It is responsible for
    constructing and managing episodes of images for episodic training.

    Attributes:
    -----------
    factory : BatchFactory
        The BatchFactory object that provides configurations and utilities for data loading and processing.

    Methods:
    --------
    __init__(self, factory: BatchFactory):
        Initializes the I2CBuilder with a reference to the BatchFactory.

    build(self):
        Constructs episodes from the dataset based on the configuration provided by the factory. It performs 
        sampling of classes and images, applies augmentations, and organizes the data into support and query sets.

    get_item(self, index):
        Retrieves a single episode of data, including support and query sets, based on the provided index.
    """

    def __init__(self, factory: BatchFactory):
        """
        Initializes the I2CBuilder with a reference to the BatchFactory.

        Parameters:
          -----------
        factory : BatchFactory
              The BatchFactory object that provides configurations and utilities for data loading and processing.
        """
        self.factory = factory

    def build(self):
        """
        Constructs episodes from the dataset based on the configuration provided by the factory.

        The method samples classes and images to form episodes, applies data augmentations, and organizes the data
        into support and query sets. It also handles the configuration of query and support data augmentations, and
        prints out the configuration settings for verification.

        The constructed episodes are stored in the `data_list` attribute of the factory.
        """
        # assign all values from self
        factory = self.factory
        episode_num = factory.episode_num
        way_num = factory.way_num
        shot_num = factory.shot_num
        query_num = factory.query_num
        class_list = factory.class_list
        class_img_dict = factory.class_img_dict
        data_list = factory.data_list
        # Set seed for deterministic behavior
        if self.factory.deterministic:
            random.seed(42)
        for _ in range(episode_num):

            # construct each episode
            episode = []
            temp_list = random.sample(class_list, way_num)

            for cls, item in enumerate(temp_list):  # for each class
                imgs_set = class_img_dict[item]
                random.shuffle(imgs_set)  # shuffle the images

                # split the images into support and query sets
                support_imgs = imgs_set[:shot_num]
                query_imgs = imgs_set[shot_num:shot_num + query_num]

                cls_subset = {
                    "query_img": query_imgs,  # query_num - query images for `cls`, default(15)
                    "support_set": support_imgs,  # SHOT - support images for `cls`, default(5)
                    "target": cls,
                }
                episode.append(cls_subset)  # (WAY, QUERY (query_num) + SHOT, 3, x, x)
            data_list.append(episode)
        if not factory.use_Q_augmentation:
            factory.qv = 1
        elif factory.use_identity:
            factory.qv = factory.qav_num + 1
        if not factory.use_S_augmentation:
            factory.sv = 1
        elif factory.use_identity:
            factory.sv = factory.sav_num + 1

        print("===========================================")
        print(f"Query   AV ({factory.mode}) : ", factory.qav_num, "(Total: ", factory.qv, ")",
              f"[ AUGMENT : {'✅ ' + '{' + '->'.join([t.__class__.__name__ for t in self.factory.Q_augmentations]) + '}' if factory.use_Q_augmentation else '❌'} ]")
        print(f"Support AV ({factory.mode}) : ", factory.sav_num, "(Total: ", factory.sv, ")",
              f"[ AUGMENT : {'✅ ' + '{' + '->'.join([t.__class__.__name__ for t in self.factory.S_augmentations]) + '}' if factory.use_S_augmentation else '❌'} ]")
        print(
            f"Batch [ AUGMENT : {'✅ ' if factory.batch_transform else '❌'} ]")
        print("USE IDENTITY : ", factory.use_identity)
        print("===========================================")

    def get_item(self, index):
        """
        Retrieves a single episode of data, including support and query sets, based on the provided index.

        Parameters:
        -----------
        index : int
            The index of the episode to retrieve.

        Returns:
        --------
        tuple
            A tuple containing the following:
            - query_images: List of processed query images.
            - query_targets: List of targets corresponding to the query images.
            - support_images: List of processed support images.
            - support_targets: List of targets corresponding to the support images.

        Notes:
        ------
        - The method applies augmentations to the query and support images based on the configuration in the factory.
        - The loader function from the factory is used to load the images.
        - The method handles possible random sampling of augmentations if specified in the factory configuration.
        """
        factory = self.factory
        episode_files = factory.data_list[index]
        loader = factory.loader
        query_images = []
        query_targets = []
        support_images = []
        support_targets = []
        av_num = factory.qav_num
        sav_num = factory.sav_num

        if factory.use_Q_augmentation:
            Q_augment = [
                T.Compose(random.sample(factory.Q_augmentations, min(factory.aug_num, len(factory.Q_augmentations)))
                          if factory.is_random_aug
                          else factory.Q_augmentations[:factory.aug_num]) for _ in range(av_num)]
            if factory.use_identity:
                Q_augment.append(identity)
        else:
            Q_augment = [identity]

        if factory.use_S_augmentation:
            S_augment = [
                T.Compose(random.sample(factory.S_augmentations, min(factory.aug_num, len(factory.Q_augmentations)))
                          if factory.is_random_aug
                          else factory.S_augmentations[:factory.aug_num]) for _ in range(sav_num)]
            if factory.use_identity:
                S_augment.append(identity)
        else:
            S_augment = [identity]

        for cls_subset in episode_files:
            # Randomly select a subset of augmentations to apply per episode
            # load QUERY images, use the cached loader function
            query_dir = cls_subset['query_img']
            temp_imgs = [Image.fromarray(loader(temp_img)) for temp_img in query_dir]
            query_images += [factory.process_img(aug, temp_img) for aug in Q_augment for temp_img in temp_imgs]

            # load SUPPORT images, use the cached loader function
            support_dir = cls_subset['support_set']
            temp_imgs = [Image.fromarray(loader(temp_img)) for temp_img in support_dir]

            temp_support = [factory.process_img(aug, temp_img).unsqueeze(0) for aug in S_augment for
                            temp_img in temp_imgs]  # Use the cached loader function
            support_images.append(torch.cat(temp_support, 0))

            # read the label
            target = cls_subset['target']
            query_targets.extend(np.tile(target, len(query_dir)))
            support_targets.extend(np.tile(target, len(support_dir)))

        return query_images, query_targets, support_images, support_targets


class NPairMCBuilder(BatchFactory.AbstractBuilder):
    """
    NPairsMCBuilder is used to build the dataset for N-pair MC batch construction [Not used in experiments]
    """

    def __init__(self, factory: BatchFactory):
        self.factory = factory
        self.val_builder = I2CBuilder(factory)

    def get_item(self, index):
        """Load an episode each time, including C-way K-shot and Q-query"""
        # write to log file the factory mode

        if self.factory.mode != 'train':
            return self.val_builder.get_item(index)
        factory = self.factory
        episode_files = factory.data_list[index]
        loader = factory.loader
        query_images = []
        targets = []
        positives = []
        for cls_subset in episode_files:
            augment = [identity]
            # Randomly select a subset of augmentations to apply per episode
            if None not in [factory.qav_num, factory.aug_num]:
                augment = [
                    T.Compose(
                        random.sample(factory.Q_augmentations, min(factory.aug_num, len(factory.Q_augmentations))))
                    for _ in range(factory.qav_num)]

            # load query images
            temp_q = Image.fromarray(loader(cls_subset['q']))
            query_images += [factory.process_img(aug, temp_q) for aug in augment]  # Use the cached loader function

            # load support images

            temp_support = [Image.fromarray(loader(pos_img)) for pos_img in cls_subset['+']]
            if factory.mode == 'train' and factory.shot_num > 1:
                augment = [
                    T.Compose(random.sample(factory.Q_augmentations, min(factory.aug_num, len(factory.Q_augmentations)))
                              if factory.is_random_aug else factory.Q_augmentations[:factory.aug_num])]
            positives += [factory.process_img(aug, temp_img) for aug in augment for temp_img in
                          temp_support]  # Use the cached loader function

            # read the label
            targets.append(cls_subset['target'])
        return query_images, positives, targets

    def build(self):
        # assign all values from self
        factory = self.factory
        if self.factory.mode != 'train':
            self.val_builder.build()
            return
        builder = self.factory
        episode_num = builder.episode_num
        way_num = builder.way_num
        # shot_num = builder.shot_num
        # query_num = builder.query_num
        class_list = builder.class_list
        class_img_dict = builder.class_img_dict
        data_list = builder.data_list

        for _ in range(episode_num):
            # construct each episode
            episode = []
            temp_list = random.sample(class_list, self.factory.train_class_num)
            for cls, item in enumerate(temp_list):  # for each class
                imgs_set = class_img_dict[item]
                # split the images into support and query sets
                query, *positives = random.sample(imgs_set, 1 + self.factory.shot_num)
                cls_subset = {
                    "q": query,  # query_num - query images for `cls`, default(15)
                    "+": positives,
                    "target": cls
                }
                episode.append(cls_subset)  # (WAY, QUERY (query_num) + SHOT, 3, x, x)
            data_list.append(episode)
        if not factory.use_Q_augmentation:
            factory.sav_num, factory.qav_num = 1, 1
        print("===========================================")
        print(f"Augmentation : {'ON' if factory.use_Q_augmentation else 'OFF'}")
        print(f"Query   AV ({factory.mode}) : ", factory.qav_num)
        print(f"Support AV ({factory.mode}) : ", factory.sav_num)
        print("===========================================")
