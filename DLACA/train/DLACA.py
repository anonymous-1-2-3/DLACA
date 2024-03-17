# ========================================== Common Package ===============================================
from itertools import cycle
import os
import argparse
import random
import numpy as np
import tqdm
import platform
# ====================================== Deep Learning Package ===========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# =========================================== Ours Package ================================================
from dataset.fault_dataset import *
from utils.metrics import AverageMeter, accuracy
from utils.op import *
from model.network import *
from loss.CenterLoss import CenterLoss
import copy

if __name__ == "__main__":
    root_path = (os.getcwd()).replace('LOSS_E', '')
    split_sym = get_split_sys()
    data_name = 'PU'
    task = '0_123'
    Fault_Dataset = Task_DataSet(data_name)
    # ========================================== Parameter Config ===============================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--project_name", type=str, default='MTDA_DLACA_' + data_name + task)
    parser.add_argument("--model_name", type=str, default=data_name + task)
    parser.add_argument("--f_num", type=int)
    parser.add_argument("--train_fold", type=list,
                        default=['dataset/data/' + data_name + '/0',
                                 'dataset/data/' + data_name + '/1',
                                 'dataset/data/' + data_name + '/2',
                                 'dataset/data/' + data_name + '/3'])
    parser.add_argument("--test_fold", type=list,
                        default=['dataset/data/' + data_name + '/1',
                                 'dataset/data/' + data_name + '/2',
                                 'dataset/data/' + data_name + '/3'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--first_stage', type=int, default=75)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--record_fold', type=str, default=r'record')
    parser.add_argument('--param_fold', type=str, default=r'model_param')
    parser.add_argument('--txt_fold', type=str, default=r'train_txt')
    parser.add_argument('--snr', type=int, default=10)
    parser.add_argument('--FFT', type=bool, default=False)
    args = parser.parse_args()
    # ========================================== Record Config ===============================================
    Txt_record_path = root_path + args.record_fold + split_sym + args.project_name + split_sym + args.txt_fold
    mkdir(Txt_record_path)
    note_name = Txt_record_path + split_sym + args.model_name + '.txt'
    Note = open(note_name, mode='w')
    Note.writelines('Experiment ' + 'Best_Test_Acc' + '\n')
    Param_record_path = root_path + args.record_fold + split_sym + args.project_name + split_sym + args.param_fold
    mkdir(Param_record_path)
    for exper_idx in range(5):
        # ========================================== Seed Config ============================================
        seed_torch(seed=args.seed)
        # ========================================== Model Config ===========================================
        e1 = Encode().cuda()
        e2 = Encode().cuda()
        w1 = CLS(f_num=args.f_num).cuda()
        w2 = CLS(f_num=4).cuda()
        TS = Transformer(dim=192, num=args.f_num).cuda()
        cent = CenterLoss(num_classes=args.f_num, feat_dim=192, use_gpu=True)
        # ========================================== Data Config =============================================
        source_path = [root_path + ((args.train_fold)[0]).replace('/', split_sym)]
        source_dataset = Fault_Dataset(path=source_path, snr=args.snr, FFT=args.FFT, seed=args.seed)
        Source_Loader = DataLoader(source_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1,
                                   drop_last=True)

        target1_path = [root_path + ((args.train_fold)[1]).replace('/', split_sym)]
        target1_dataset = Fault_Dataset(path=target1_path, snr=args.snr, FFT=args.FFT, seed=args.seed)
        Target1_Loader = DataLoader(target1_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1,
                                    drop_last=True)

        target2_path = [root_path + ((args.train_fold)[2]).replace('/', split_sym)]
        target2_dataset = Fault_Dataset(path=target2_path, snr=args.snr, FFT=args.FFT, seed=args.seed)
        Target2_Loader = DataLoader(target2_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1,
                                    drop_last=True)

        target3_path = [root_path + ((args.train_fold)[3]).replace('/', split_sym)]
        target3_dataset = Fault_Dataset(path=target3_path, snr=args.snr, FFT=args.FFT, seed=args.seed)
        Target3_Loader = DataLoader(target3_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1,
                                    drop_last=True)

        test_path = [root_path + fold.replace('/', split_sym) for fold in args.test_fold]
        test_dataset = Fault_Dataset(path=test_path, snr=args.snr, FFT=args.FFT, seed=args.seed)
        Test_Loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1,
                                 drop_last=True)
        # ========================================== Optimizer Config ===============================================
        e1_optimizer = optim.Adam(e1.parameters(), lr=0.001)
        e2_optimizer = optim.Adam(e2.parameters(), lr=0.001)
        w1_optimizer = optim.Adam(w1.parameters(), lr=0.001)
        w2_optimizer = optim.Adam(w2.parameters(), lr=0.001)
        ts_optimizer = optim.Adam(TS.parameters(), lr=0.001)
        cent_optimizer = optim.Adam(cent.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        Best_ACC = 0.0
        # ==========================================Train Start ===============================================
        for epoch in range(args.epochs):
            print("第%d次实验_第%d轮次" % (exper_idx + 1, epoch + 1))
            TEST_ACC_AVG = AverageMeter()
            for index, data in enumerate(tqdm.tqdm(zip(Source_Loader, Target1_Loader, Target2_Loader, Target3_Loader))):
                # ==================================Get Data and Labels =======================================
                (source_data, source_fault_labels, source_domain), \
                (target1_data, target1_fault_labels, target1_domain), \
                (target2_data, target2_fault_labels, target2_domain), \
                (target3_data, target3_fault_labels, target3_domain) = data
                # ==================================Get Data and Labels =======================================
                source_data = source_data.reshape([-1, 1, 2048]).cuda()
                target1_data = target1_data.reshape([-1, 1, 2048]).cuda()
                target2_data = target2_data.reshape([-1, 1, 2048]).cuda()
                target3_data = target3_data.reshape([-1, 1, 2048]).cuda()

                all_data = torch.cat([source_data, target1_data, target2_data, target3_data], dim=0)
                source_fault_labels = source_fault_labels.cuda()

                all_domain = torch.cat([source_domain, target1_domain, target2_domain, target3_domain], dim=0).cuda()

                # ================================== 训练 =======================================
                e1.train()
                e2.train()
                w1.train()
                w2.train()
                TS.train()

                requires_grad(e1, flag=False)
                requires_grad(e2, flag=False)
                requires_grad(w1, flag=False)
                requires_grad(w2, flag=True)
                requires_grad(TS, flag=False)

                feature_e1 = e1(all_data)
                domain_out_e1 = w2(feature_e1)
                feature_e2 = e2(all_data)
                domain_out_e2 = w2(feature_e2)
                adv_loss = criterion(domain_out_e1, all_domain) * (args.a)  # 对抗
                C1_loss2 = criterion(domain_out_e2, all_domain)

                w2_optimizer.zero_grad()
                (adv_loss + C1_loss2).backward()
                w2_optimizer.step()

                requires_grad(e1, flag=True)
                requires_grad(e2, flag=False)
                requires_grad(w1, flag=False)
                requires_grad(w2, flag=False)
                requires_grad(TS, flag=False)

                feature_e1 = e1(all_data)
                fault_out_e1 = w1(feature_e1)
                domain_out_e1 = w2(feature_e1)
                source_out_e1, _, _, _ = fault_out_e1.chunk(4, dim=0)
                E1_Fault_loss = criterion(source_out_e1, source_fault_labels)
                E1_Domain_loss = criterion(domain_out_e1, all_domain) * (args.a)

                if epoch < args.first_stage:
                    train_loss = E1_Fault_loss - E1_Domain_loss
                else:
                    copy_e1 = copy.deepcopy(e1)
                    copy_w1 = copy.deepcopy(w1)
                    copy_TS = copy.deepcopy(TS)
                    copy_e1.eval()
                    copy_w1.eval()
                    copy_TS.eval()

                    feature = (copy_e1(all_data)).detach()
                    cls_e1 = (copy_w1(feature)).detach()
                    cls_ts = (copy_TS(feature)).detach()  # 得到T的预测标签
                    centroid, c_labels = get_pseudo_centroid(feature_e1, cls_e1, cls_ts, source_fault_labels, size=64,
                                                             n_class=args.f_num, cod=0.8)
                    loss_cent = cent(centroid, c_labels)
                    train_loss = E1_Fault_loss - E1_Domain_loss + loss_cent

                e1_optimizer.zero_grad()
                cent_optimizer.zero_grad()
                train_loss.backward()
                e1_optimizer.step()
                cent_optimizer.step()

                requires_grad(e1, flag=False)
                requires_grad(e2, flag=False)
                requires_grad(w1, flag=True)
                requires_grad(w2, flag=False)
                requires_grad(TS, flag=False)

                feature_e1 = e1(all_data)
                fault_out_e1 = w1(feature_e1)
                source_out_e1, _, _, _ = fault_out_e1.chunk(4, dim=0)

                feature_e2 = e2(all_data)
                fault_out_e2 = w1(feature_e2)
                source_out_e2, _, _, _ = fault_out_e2.chunk(4, dim=0)

                C1_loss1 = criterion(source_out_e1, source_fault_labels)
                C1_loss2 = criterion(source_out_e2, source_fault_labels)
                w1_optimizer.zero_grad()
                (C1_loss1 + C1_loss2).backward()
                w1_optimizer.step()

                requires_grad(e1, flag=False)
                requires_grad(e2, flag=True)
                requires_grad(w1, flag=False)
                requires_grad(w2, flag=False)
                requires_grad(TS, flag=False)

                feature_e2 = e2(all_data)
                fault_out_e2 = w1(feature_e2)
                domain_out_e2 = w2(feature_e2)
                source_out_e2, _, _, _ = fault_out_e2.chunk(4, dim=0)

                E2_Fault_loss = criterion(source_out_e2, source_fault_labels)
                E2_Domain_loss = criterion(domain_out_e2, all_domain)
                e2_optimizer.zero_grad()
                (E2_Domain_loss - E2_Fault_loss).backward()
                e2_optimizer.step()

                requires_grad(e1, flag=False)
                requires_grad(e2, flag=False)
                requires_grad(w1, flag=False)
                requires_grad(w2, flag=False)
                requires_grad(TS, flag=True)

                ts_optimizer.zero_grad()
                feature_e1 = (e1(all_data)).detach()
                fault_out_ts = TS(feature_e1)
                source_out_ts, _, _, _ = fault_out_ts.chunk(4, dim=0)
                ts_loss = criterion(source_out_ts, source_fault_labels)
                ts_optimizer.zero_grad()
                ts_loss.backward()
                ts_optimizer.step()

            for index, data in enumerate(tqdm.tqdm(Test_Loader)):
                # ==================================Get Data and Labels =======================================
                target_data, target_fault_labels, _ = data
                target_data = target_data.reshape([-1, 1, 2048]).cuda()
                target_fault_labels = target_fault_labels.cuda()
                # ================================== 测试 =======================================
                e1.eval()
                w1.eval()
                # ================================== 测试 =======================================
                target_out = w1(e1(target_data))
                test_acc = accuracy(target_out, target_fault_labels, topk=(1,))
                TEST_ACC_AVG.update(test_acc[0].item(), args.test_batch_size)
            Note.writelines(str(epoch + 1) + ' ' + str(round(TEST_ACC_AVG.avg, 4)) + '\n')
            if TEST_ACC_AVG.avg >= Best_ACC:
                Best_ACC = TEST_ACC_AVG.avg
                encode_model_path = Param_record_path + split_sym + args.model_name + '_' + 'encode' + '_' + str(
                    exper_idx) + '.pkl'
                cls_model_path = Param_record_path + split_sym + args.model_name + '_' + 'cls' + '_' + str(
                    exper_idx) + '.pkl'
                torch.save(e1, encode_model_path)
                torch.save(w1, cls_model_path)
            else:
                pass
        Note.writelines(str(exper_idx + 1) + ' ' + 'Best_ACC' + ' ' + str(round(Best_ACC, 4)) + '\n')
        Note.writelines('===============================' + '\n')
    Note.close()
