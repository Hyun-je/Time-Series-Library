from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from TaPR_pkg import etapr

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_By_Forcast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_By_Forcast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            train_criterion = nn.MSELoss()
        elif self.args.loss == 'L1':
            train_criterion = nn.L1Loss()
        val_criterion = nn.MSELoss()
        return train_criterion, val_criterion
    
    def _init_logger(self, path):

        # Set up logging
        self.logger = logging.getLogger('exp_logger')
        self.logger.setLevel(logging.DEBUG)

        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(path)

        # Set level for handlers
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(vali_loader, ncols=50)):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):

        self.start_time = time.localtime()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.resume is not None:
            print(f"Resume training from {self.args.resume}")
            self.model.load_state_dict(torch.load(self.args.resume))

        self._init_logger(f"{path}/log.txt")
        self.logger.info(f"Args: {self.args}")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        train_criterion, val_criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, ncols=50)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = train_criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            self.logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, val_criterion)
            test_loss = self.vali(test_data, test_loader, val_criterion)

            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(
                epoch,
                {'train_loss': train_loss, 'vali_loss': vali_loss, 'test_loss': test_loss},
                self.model,
                path
            )
            if early_stopping.early_stop:
                self.logger.info(f"Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint_best.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        # self.args.batch_size = 64

        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, ncols=50)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # forcast
                outputs = self.model(batch_x, None, None, None)

                # criterion
                score = torch.mean(self.anomaly_criterion(outputs, batch_y), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        self.logger.info(f"train_energy : {train_energy.shape}")
        self.threshold = np.percentile(train_energy, 100 - self.args.anomaly_ratio)
        self.logger.info(f"Threshold : {self.threshold}")

        return
    
    def prediction(self, setting):

        # self.args.batch_size = 1

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_best.pth')))
        self.model.eval()

        pred_data, pred_loader = self._get_data(flag='pred')

        attens_energy = [np.zeros(self.args.seq_len)]
        for i, (batch_x, batch_y) in enumerate(tqdm(pred_loader, ncols=50)):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            # forcast
            outputs = self.model(batch_x, None, None, None)

            # criterion
            score = torch.mean(self.anomaly_criterion(outputs, batch_y), dim=-1)
            score = score.detach().cpu().numpy().reshape(-1)
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        self.logger.info(f"test_energy : {test_energy.shape}")

        pred = (test_energy > self.threshold).astype(int)
        pred = np.array(pred)
        self.logger.info(f"pred: {pred.shape}")


        # export results
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        sample_submission = pd.read_csv("dataset/DACON/sample_submission.csv")
        sample_submission['anomaly'] = pred
        sample_submission.to_csv(f'{folder_path}/pred.csv', encoding='UTF-8-sig', index=False)

        # Calculate TaPR
        if True:
            gt = pd.read_csv("dataset/DACON/test_label.csv")["anomaly"]
            self.logger.info(f"pred: {gt.shape}")

            TaPR = etapr.evaluate_haicon(gt, pred)
            self.logger.info(f"prediction F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
            self.logger.info(f"탐지된 이상 상황 개수: {len(TaPR['Detected_Anomalies'])}")

            sample_submission['anomaly'] = gt
            sample_submission.to_csv(f'{folder_path}/gt.csv', encoding='UTF-8-sig', index=False)

        # f = open("result_anomaly_detection.txt", 'a')
        # f.write(setting + "  \n")
        # f.write(f"{self.start_time.tm_year}/{self.start_time.tm_mon}/{self.start_time.tm_mday} {self.start_time.tm_hour}:{self.start_time.tm_min}:{self.start_time.tm_sec}\n")
        # f.write(f"Threshold: {best_threshold}\n")
        # f.write(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})\n")
        # f.write(f"탐지된 이상 상황 개수: {len(TaPR['Detected_Anomalies'])}\n")
        # f.write('\n')
        # f.write('\n')
        # f.close()




    def visualize(self, setting):

        def draw_reconstruction_results(self, test_inputs, test_outputs, step, image_path):
            print("Drawing the reconstruction results...")

            for i in range(test_inputs.shape[2]):
                plt.figure(figsize=(100, 15))
                plt.tight_layout()
                plt.grid(True, axis='x')
                
                for w in range(test_inputs.shape[0]):
                    x_offset = w * step
                    data_x = np.arange(x_offset, x_offset + test_inputs.shape[1], 1)
                    data_y1 = test_outputs[w, :, i]
                    data_y2 = test_inputs[w, :, i]

                    plt.plot(data_x, data_y2, linewidth=0.1, color='magenta', alpha=1.0)
                    plt.plot(data_x, data_y1, linewidth=3.0, color='cyan', alpha=0.1)
                    
                    
                plt.savefig(f'{image_path}/test_{i}.png', dpi=300)


        test_data, test_loader = self._get_data(flag='test')

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint_best.pth')))
        self.model.eval()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_inputs = []
        test_outputs = []
        for i, (batch_x, batch_y) in enumerate(tqdm(test_loader, ncols=50)):

            # Downsample loaded data
            batch_x = batch_x[:, ::self.args.downsample, :]
            batch_y = batch_y[:, ::self.args.downsample, :]

            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)

            test_inputs.append(batch_x.detach().cpu().numpy())
            test_outputs.append(outputs.detach().cpu().numpy())

        test_inputs = np.concatenate(test_inputs, axis=0)
        test_outputs = np.concatenate(test_outputs, axis=0)

        draw_reconstruction_results(self, test_inputs, test_outputs, test_data.step, folder_path)

