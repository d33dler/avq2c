from __future__ import print_function

import argparse
import traceback
from typing import List

import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
import os
import sys
import time

import numpy as np
import scipy as sp
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import yaml
from PIL import ImageFile
from easydict import EasyDict
from torchvision.transforms import transforms

from data_loader.datasets_csv import BatchFactory
from models import architectures
from models.architectures.DN_X.dnx_arch import DN_X
from models.architectures.dt_model import CNNModel
from models.utilities.utils import AverageMeter, create_confusion_matrix

sys.dont_write_bytecode = True

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cudnn.benchmark = True

"""
This is the main execution script. For training a model simply add the config path, architecture name, 
dataset path & data_name.
"""


class ExperimentManager:
    """
    The ExperimentManager class is responsible for managing and orchestrating the training and testing phases
    of an experiment. It handles dataset preparation, model training, evaluation, and logging results.

    Attributes:
    -----------
    target_bank : numpy.ndarray
        A static attribute to hold targets for creating confusion matrices or other analytics post testing.

    Methods:
    --------
    __init__(self):
        Initializes the ExperimentManager instance, setting up required variables and objects.

    mean_confidence_interval(self, data, confidence=0.95):
        Calculates the mean confidence interval of a given dataset.

    write_losses_to_file(self, losses: List[float]):
        Writes the list of losses to a text file for further analysis or logging.

    test(self, model, F_txt):
        Manages the testing phase of an experiment, evaluating the performance of the model on test data.

    train(self, model: CNNModel, F_txt):
        Manages the training phase of an experiment, including dataset preparation, model training, and evaluation.
    """
    target_bank = np.empty(shape=0)

    def __init__(self):
        self.output_dir = None
        self._args = None

        self.k = None
        self.loss_tracker = AverageMeter()

    def mean_confidence_interval(self, data, confidence=0.95):
        """
         Calculates the mean confidence interval for a given dataset.

         Parameters:
         ----------
         data : list
             A list of numerical values for which the mean confidence interval needs to be calculated.

         confidence : float, optional (default=0.95)
             The confidence level for the interval.

         Returns:
         -------
         m : float
             The mean of the dataset.

         h : float
             The half-width of the confidence interval.
         """
        a = [1.0 * np.array(data[i].cpu()) for i in range(len(data))]
        n = len(a)
        m, se = np.mean(a), sp.stats.sem(a)
        h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
        return m, h

    def write_losses_to_file(self, losses: List[float]):
        """
        Writes the list of losses to a text file for further analysis or logging.

        Parameters:
        ----------
        losses : List[float]
            A list containing loss values to be logged.

        Returns:
        -------
        None
        """
        with open(os.path.join(self.output_dir, "LOSS_LOG.txt"), 'w') as _f:
            for loss in losses:
                _f.write(str(loss) + '\n')

    def test(self, model, F_txt):
        """
         Manages the testing phase of an experiment. It evaluates the model's performance on the test dataset,
         logs the results, and performs any necessary post-processing.

         Parameters:
         ----------
         model : Model
             The model to be evaluated.

         F_txt : file object
             A file object for logging text output.

         Returns:
         -------
         None
         """
        # ============================================ Testing phase ========================================
        print('\n............Start testing............')
        start_time = time.time()
        repeat_num = 5  # repeat running the testing code several times
        total_accuracy = 0.0
        total_h = np.zeros(repeat_num)
        total_accuracy_vector = []
        best_prec1 = 0
        params = model.ds_loader.params
        params.way_num = 5
        model.eval()
        model.data.training(True)
        for r in range(repeat_num):
            print('===================================== Round %d =====================================' % r)
            F_txt.write('===================================== Round %d =====================================\n' % r)

            # ======================================= Folder of Datasets =======================================

            # image transform & normalization
            pre_process = [
                transforms.Resize(92),
            ]
            Q_transform = [
                nn.Identity()
            ]
            post_process = [transforms.CenterCrop(84), transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            testset = BatchFactory(
                data_dir=self._args.DATASET_DIR, mode='test', pre_process=pre_process, post_process=post_process,
                episode_num=params.episode_test_num, way_num=params.way_num, shot_num=params.shot_num,
                query_num=params.query_num  # , qav_num=model.data.qv - 1, aug_num=1, Q_augmentations=Q_transform,

            )
            F_txt.write('Testset: %d-------------%d' % (len(testset), r))

            # ========================================== Load Datasets =========================================
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=params.test_ep_size, shuffle=True,
                num_workers=int(params.workers), drop_last=True, pin_memory=True
            )

            # =========================================== Evaluation ==========================================
            prec1, accuracies = model.validate(test_loader, best_prec1, F_txt)
            best_prec1 = max(prec1, best_prec1)
            test_accuracy, h = self.mean_confidence_interval(accuracies)
            print("Test accuracy=", test_accuracy, "h=", h[0])
            F_txt.write(f"Test accuracy= {test_accuracy} h= {h[0]}\n")
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[r] = h

        aver_accuracy, _ = self.mean_confidence_interval(total_accuracy_vector)
        print("Aver_accuracy:", aver_accuracy, "Aver_h", total_h.mean())
        F_txt.write(f"\nAver_accuracy= {aver_accuracy} Aver_h= {total_h.mean()}\n")
        F_txt.close()
        # create_confusion_matrix(self.target_bank.astype(int), np.argmax(self.out_bank, axis=1))

        # ============================================== Testing end ==========================================

    def train(self, model: CNNModel, F_txt):
        """
        Manages the training phase of an experiment. It handles dataset preparation, model training, evaluation,
        and logging of results.

        Parameters:
        ----------
        model : CNNModel
            The model to be trained.

        F_txt : file object
            A file object for logging text output.

        Returns:
        -------
        None
        """
        best_prec1 = model.best_prec1
        # ======================================== Training phase ===============================================
        print('\n............Start training............\n')
        epoch = model.get_epoch()

        for epoch_index in range(epoch, self._args.EPOCHS):
            print('===================================== Epoch %d =====================================' % epoch_index)
            F_txt.write(
                '===================================== Epoch %d =====================================\n' % epoch_index)
            # ================================= Set the model data to training mode ==============================
            model.data.training()
            # ======================================= Adjust learning rate =======================================

            # ======================================= Folder of Datasets =========================================
            model.load_data(self._args.MODE, F_txt, self._args.DATASET_DIR)
            loaders = model.loaders
            # ============================================ Training ==============================================
            model.train()
            # Freeze the parameters of Batch Normalization after X epochs (root configuration defined)
            model.freeze_auxiliary()
            # Train for 10000 episodes in each epoch
            model.run_epoch(F_txt)

            # torch.cuda.empty_cache()
            # =========================================== Evaluation ==========================================
            print('============ Validation on the val set ============')
            F_txt.write('============ Testing on the test set ============\n')
            try:

                prec1, _ = model.validate(loaders.val_loader, best_prec1, F_txt)
            except Exception as e:
                print("Encountered an exception while running val set validation!")
                print(e)
                prec1 = 0

            # record the best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            model.best_prec1 = best_prec1
            # save the checkpoint
            if epoch_index % 2 == 0 or is_best:
                filename = os.path.join(self._args.OUTF, 'epoch_%d.pth.tar' % model.get_epoch())
                model.save_model(filename)

            # Testing Prase
            print('============ Testing on the test set ============')
            F_txt.write('============ Testing on the test set ============\n')
            try:
                prec1, _ = model.validate(loaders.test_loader, best_prec1, F_txt)
            except Exception as e:
                print("Encountered an exception while running the test set validation!")
                # traceback.print_exc()
                print(traceback.format_exc())
        ###############################################################################
        F_txt.close()
        # save the last checkpoint
        filename = os.path.join(self._args.OUTF, 'epoch_%d.pth.tar' % model.get_epoch())
        model.save_model(filename)
        print('>>> Training and evaluation completed <<<')

    def run(self, _args):
        """
        Execute the job based on the provided arguments and settings.

        Parameters:
        _args: EasyDict
            The configuration and settings for the job.

        Steps:
        1. Assign the provided arguments to an instance variable.
        2. Load the specified model architecture from the ARCHITECTURE_MAP.
        3. Retrieve the parameters and settings from the loaded model.
        4. Create and set up the output directory for saving model checkpoints and log files.
        5. Optionally, resume training from a checkpoint if specified in the settings.
        6. If multiple GPUs are specified, wrap the model with DataParallel for parallel computation.
        7. Print and log the model architecture to a text file.
        8. Execute the training or testing procedure based on the mode specified in the settings.
        9. Close the text file used for logging.

        The paths, model architecture, and other relevant information will be printed to the console and logged to a text file.
        """
        # ======================================== Settings of path ============================================
        self._args = _args
        ARCHITECTURE_MAP = architectures.__all__
        model = ARCHITECTURE_MAP[_args.ARCH](_args.PATH)
        PRMS = model.ds_loader.params
        # create path name for model checkpoints and log files
        _args.OUTF = PRMS.outf + '_'.join(
            [_args.ARCH, _args.BACKBONE.NAME, os.path.basename(_args.DATASET_DIR), str(model.arch), str(PRMS.way_num),
             'Way', str(
                PRMS.shot_num), 'Shot', 'K' + str(model.root_cfg.K_NEIGHBORS),
             'QAV' + str(model.data.qv),
             'SAV' + str(model.data.sv),
             "AUG_" + '_'.join([str(_aug.NAME) for _aug in model.root_cfg.AUGMENTOR.AUGMENTATION])])
        PRMS.outf = _args.OUTF
        self.output_dir = PRMS.outf
        if not os.path.exists(_args.OUTF):
            os.makedirs(_args.OUTF, exist_ok=True)

        # save the opt and results to a txt file
        txt_save_path = os.path.join(_args.OUTF, 'opt_results.txt')
        txt_file = open(txt_save_path, 'a+')
        txt_file.write(str(_args))

        # optionally resume from a checkpoint
        if _args.RESUME:
            model.load_model(_args.RESUME, txt_file)

        if _args.NGPU > 1:
            model: DN_X = nn.DataParallel(model, range(_args.NGPU))

        # Print & log the model architecture
        print(model)
        print(model, file=txt_file)
        self.k = model.k_neighbors

        if _args.MODE == "test":
            self.test(model, F_txt=txt_file)
        else:
            self.train(model, F_txt=txt_file)
        txt_file.close()


# ============================================ Training End ============================================================

def launch_job(args):
    e = ExperimentManager()
    e.run(args)


if __name__ == '__main__':
    """
    This script is designed to launch jobs based on command-line arguments and configurations specified in YAML files. 
    The jobs could be individual model training/testing sessions or other computational tasks.

    Usage:
    ------
    python script_name.py [--jobs JOB [JOB ...]] [--jobfile JOBFILE] [--job_id JOB_ID] [--test]

    Options:
    --------
    --jobs JOB [JOB ...]
        Paths(s) to the model config file(s). Each file should be in YAML format and contain the configuration
        parameters required for a job.

    --jobfile JOBFILE
        Path to a file containing a list of job arrays. The file should be in YAML format and list configurations or
        paths to configurations for multiple jobs.

    --job_id JOB_ID
        An integer representing the index of the job to be launched from the job array specified in the jobfile. 
        Indexing starts at 1.

    --test
        Run the script in test mode. If this flag is set, the jobs will be launched in test mode.

    Functionality:
    --------------
    1. The script starts by setting the start method for multiprocessing to 'spawn'.
    2. It then parses command-line arguments to get the job configurations and operating mode.
    3. If specific job paths are provided with --jobs, it loads the configurations from those files, sets the mode
       (train/test), and launches the jobs.
    4. If a jobfile is provided with --jobfile, it loads the job array from that file, selects the job configuration
       based on the --job_id argument, sets the mode, and launches the job.
    5. If the jobs are to be run in test mode, this can be specified using the --test flag.
    6. The paths to job configuration files, mode of operation, and any other relevant information are printed to the console.
    7. The launch_job function is called with the parsed and processed job arguments to execute the job.

    Note:
    -----
    - The script expects job configurations to be specified in YAML format.
    - The job configurations must be convertible to an EasyDict for easy attribute access.
    - The launch_job function, which is supposed to execute the job, is not defined in this snippet.
    """

    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', default=None, nargs='+', type=str,
                        help='Paths(s) to the model config file(s). Each file should be in YAML format and contain the '
                             'configuration parameters required for a job.')
    parser.add_argument('--jobfile', default=None, type=str,
                        help='Path to a file containing a list of job arrays. The file should be in YAML format and '
                             'list configurations or paths to configurations for multiple jobs.')
    parser.add_argument('--job_id', default=None, type=int,
                        help=' An integer representing the index of the job to be launched from the job array s'
                             'pecified in the jobfile. Indexing starts at 1. None = all jobs')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    arguments = parser.parse_args()
    print(arguments.jobs)
    proc_ls = []
    if arguments.jobs is not None:
        for a in arguments.jobs:
            with open(a, 'r') as f:
                job_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
                job_args.PATH = a
                job_args.MODE = 'train' if not arguments.test else 'test'
                launch_job(job_args)
    elif arguments.jobfile is not None:
        with open(arguments.jobfile, 'r') as f:
            job_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
            job_array = job_args.ARRAY
            try:
                job_cfg = job_array[arguments.job_id - 1]
            except Exception:
                exit(0)
            job_cfg = os.path.join("../models/architectures/configs/", job_cfg)
            with open(job_cfg, 'r') as cfgfile:
                job_args = EasyDict(yaml.load(cfgfile, Loader=yaml.SafeLoader))
                job_args.PATH = job_cfg
                job_args.MODE = 'train' if not arguments.test else 'test'
                launch_job(job_args)
