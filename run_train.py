import os
import argparse
from collections import defaultdict
import time
import torchvision.transforms as transforms
import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from tqdm import tqdm
import sys

from arguments import train_parser
from model import GADBase
from data import MiddleburyDataset, NYUv2Dataset, DIMLDataset
from losses import get_loss
from utils import new_log, to_cuda, seed_all
from datasets import MagicBathyNet
import matplotlib.pyplot as plt
# import nvidia_smi
# nvidia_smi.nvmlInit()

from datasets.geo_tifffile import read_geotiff3D, write_geotiff3D


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = self.args.wandb
        self.dataloaders = self.get_dataloaders(args)

        self.cv_list = []  # Vertical diffusion coefficients
        self.ch_list = []  # Horizontal diffusion coefficients

        seed_all(args.seed)

        self.model = GADBase( 
            args.feature_extractor, 
            Npre=args.Npre,
            Ntrain=args.Ntrain, 
        ).cuda()

        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, args.dataset), args)
        self.args.experiment_folder = self.experiment_folder

        if self.use_wandb:
            wandb.init(project=args.wandb_project, dir=self.experiment_folder)
            wandb.config.update(self.args)
            self.writer = None
        # else:
            # self.writer = SummaryWriter(log_dir=self.experiment_folder)

        if not args.no_opt:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        else:
            self.optimizer = None
            self.scheduler = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

    def __del__(self):
        if not self.use_wandb:
            self.writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    if not args.no_opt:
                        self.scheduler.step()
                        if self.use_wandb:
                            wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                        else:
                            self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1

    def prepare_magic_bathy(self, sample):
            return {
            'guide': sample[3].unsqueeze(0),
            'source': sample[2].unsqueeze(0),
            'mask_lr': (sample[2] != 0).any(dim=0).float(),
            'y_bicubic': torch.nn.functional.interpolate(sample[2].unsqueeze(0), scale_factor=40, mode='bicubic', align_corners=True)
            }



    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()
        
        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # self.train_stats["gpu_used"] = info.used


        with tqdm(self.dataloaders.datasets['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            pref_loss = 0
            img_tensors = []
            for i, sample in enumerate(inner_tnr):
                #sample = self.prepare_magic_bathy(sample)
                sample = to_cuda(sample)

                valid_samples = (sample['y'].sum(dim=(0, 1, 2)) != 0)
                if not valid_samples.any():
                    continue

                if not args.no_opt:
                    self.optimizer.zero_grad()

                output = self.model(sample, train=True)
                self.cv_list.append(output['cv'].detach().cpu().numpy())  # Convert to NumPy for plotting
                self.ch_list.append(output['ch'].detach().cpu().numpy())
                loss, loss_dict = get_loss(output, sample)

                if torch.isnan(loss):
                    print('loss is nan')
                    continue
                    #raise Exception("detected NaN loss..")
                    
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]

                #img = self.dataloaders.denormalize_out(output['y_pred'])
                

                if self.epoch > 0 or not self.args.skip_first:
                    if not args.no_opt:
                        loss.backward()

                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    if not args.no_opt:
                        self.optimizer.step()

                name = sample['img_path'].split('\\')[-1]
                if "359" in name:
                    self.dataloaders.datasets['train'].save_as_tiff(output['y_pred'], sample['img_path'], os.path.join('.', 'save_img_dir', f"epoch_{str(self.epoch)}"))

                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders.datasets['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    if self.use_wandb:
                        wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    else:
                        for key in self.train_stats:
                            self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)
            #self.plot_coefficients(self.cv_list, self.ch_list)

    def plot_coefficients(self, cv_list, ch_list):
        for i, (cv, ch) in enumerate(zip(cv_list, ch_list)):
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title(f'Vertical Coefficients (Step {i})')
            plt.imshow(cv.squeeze(), cmap='viridis')  # Ensure 2D array
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.title(f'Horizontal Coefficients (Step {i})')
            plt.imshow(ch.squeeze(), cmap='viridis')  # Ensure 2D array
            plt.colorbar()

            plt.show()

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders.datasets['val'], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)

                loss, loss_dict = get_loss(output, sample)

                for key in loss_dict:
                    self.val_stats[key] +=  loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

            self.val_stats = {k: v / len(self.dataloaders.datasets['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            else:
                for key in self.val_stats:
                    self.writer.add_scalar('val/' + key, self.val_stats[key], self.epoch)

            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    @staticmethod
    def get_dataloaders(args):
        data_args = {
            'crop_size': (args.crop_size, args.crop_size),
            'in_memory': args.in_memory,
            'max_rotation_angle': args.max_rotation,
            'do_horizontal_flip': not args.no_flip,
            'crop_valid': True,
            'image_transform': Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            'scaling': args.scaling
        }

        phases = ('train', 'val')
        if args.dataset == 'Middlebury':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1122.7]) 
            datasets = {phase: MiddleburyDataset(os.path.join(args.data_dir, 'Middlebury'), **data_args, split=phase,
                        depth_transform=depth_transform, crop_deterministic=phase == 'val') for phase in phases}

        elif args.dataset == 'DIML':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1154.29])
            datasets = {phase: DIMLDataset(os.path.join(args.data_dir, 'DIML'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}

        elif args.dataset == 'NYUv2':
            # Important, do not zero-center the depth, DADA needs positive depths
            depth_transform = Normalize([0.0], [1386.05])
            datasets = {phase: NYUv2Dataset(os.path.join(args.data_dir, 'NYUv2'), **data_args, split=phase,
                        depth_transform=depth_transform) for phase in phases}

        elif args.dataset =='MagicBathy':
                path_to_images = os.path.join('.','datafolder', 'MagicBathyNet_CV4RS_WiSe_2425', 'agia_napa', 'img','s2')
                path_to_images = [os.path.join(path_to_images, x) for x in os.listdir(path_to_images)]
                path_to_labels = os.path.join('.','datafolder', 'MagicBathyNet_CV4RS_WiSe_2425', 'agia_napa', 'img','aerial')
                path_to_labels = [os.path.join(path_to_labels, x) for x in os.listdir(path_to_labels)]
                return MagicBathyNet.MagicBathyNetDataLoader(os.path.join('.','datafolder', 'MagicBathyNet_CV4RS_WiSe_2425'), batch_size=args.batch_size, num_workers=args.num_workers,locations=['agia_napa'], bathymetry=True)
        elif args.dataset == 'resized':
            path_to_images = os.path.join('.', 'datafolder', 'resized', 'agia_napa', 'img', 's2')
            path_to_images = [os.path.join(path_to_images, x) for x in os.listdir(path_to_images)]
            path_to_labels = os.path.join('.', 'datafolder', 'resized', 'agia_napa', 'img',
                                          'aerial')

            return MagicBathyNet.MagicBathyNetDataLoader(
                os.path.join('.', 'datafolder', 'resized'), batch_size=args.batch_size,
                num_workers=args.num_workers, locations=['agia_napa'], bathymetry=True)

        return {phase: DataLoader(datasets[phase], batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=True, drop_last=False) for phase in phases}


    def save_model(self, prefix=''):
        if args.no_opt:
            torch.save({
                'model': self.model.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))
        else:
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])

        if not args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    torch.cuda.empty_cache()  # Clear unused memory in PyTorch's cache
    args = train_parser.parse_args()
    print(train_parser.format_values())

    if args.wandb:
        import wandb

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
