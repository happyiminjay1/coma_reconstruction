import os
import time
import torch
import json
from glob import glob

from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train L1 Loss: {:.4f}, Train Contact Loss: {:.4f}, Train taxonomy Loss: {:.4f}, test_l1: {:.4f}, tax_acc: {:.4f}, f1_score: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_l1'], info['train_contact_loss'], info['train_taxonomy_loss'], info['test_l1'], info['tax_acc'], info['f1_score'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def print_info_tester(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train_test_l1: {:.4f}, Train_tax_acc: {:.4f}, Train_f1_score: {:.4f}, test_l1: {:.4f}, tax_acc: {:.4f}, f1_score: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_l1'], info['train_tax_acc'], info['train_f1_score'], info['test_l1'], info['tax_acc'], info['f1_score'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def s_writer(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/train_l1", info['train_l1'], epoch)
        s_writer.add_scalar("Loss/train_contact_loss", info['train_contact_loss'], epoch)
        s_writer.add_scalar("Loss/train_taxonomy_loss", info['train_taxonomy_loss'], epoch)

        s_writer.add_scalar("Loss/test_l1", info['test_l1'], epoch)
        s_writer.add_scalar("Loss/test_taxonomy_acc", info['tax_acc'], epoch)
        s_writer.add_scalar("Loss/f1_score", info['f1_score'], epoch)

        s_writer.flush()

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))
