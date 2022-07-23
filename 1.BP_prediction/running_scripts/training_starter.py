import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from running_scripts.MyDataset import MainDataset
import os
from running_scripts.eval_util import result_evaluate, evaluate_with_normed_bp

record_columns = ['MAPE', 'MAE', 'RMSE']
BATCH_SIZE = 32

# 配置文件路径
RESULT_ROOT = r'G:\PythonPro\BP_prediction\result'


def model_training(MODEL_NAME, user_id, model, optimizer, scheduler, epoch_num, loss_function, train_dataset,
                   vali_dataset, targetBP=['SBP', 'DBP']):
    best_model_param = os.path.join(RESULT_ROOT, 'params', user_id, MODEL_NAME + '_param.pkl')
    # 数据集设置
    train_dataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, drop_last=True)
    vali_dataLoader = DataLoader(vali_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    train_records = [pd.DataFrame(dtype=np.float64, columns=record_columns) for bp in targetBP]
    vali_records = [pd.DataFrame(dtype=np.float64, columns=record_columns) for bp in targetBP]

    best_epoch = 0
    for epoch in range(epoch_num):
        print('epoch ' + str(epoch + 1) + ' started;')
        model.train()
        for i, data in enumerate(train_dataLoader):
            seq, ft, bp = data[0].cuda(), data[1].cuda(), data[2].cuda()
            prediction = model(seq, ft)
            if i == 0:
                train_pred = prediction.detach().cpu()
                train_truth = bp.cpu()
            else:
                train_pred = torch.cat([train_pred, prediction.detach().cpu()])
                train_truth = torch.cat([train_truth, bp.cpu()])
            loss = loss_function(prediction, bp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        for i, data in enumerate(vali_dataLoader):
            seq, ft, bp = data[0].cuda(), data[1].cuda(), data[2].cuda()
            prediction = model(seq, ft)
            if i == 0:
                vali_pred = prediction.detach().cpu()
                vali_truth = bp.cpu()
            else:
                vali_pred = torch.cat([vali_pred, prediction.detach().cpu()], dim=0)
                vali_truth = torch.cat([vali_truth, bp.cpu()], dim=0)

        # 计算本次epoch中train和eval中 SBP和 DBP的不同指标
        for i, bp in enumerate(targetBP):
            train_result = pd.Series(result_evaluate(torch.flatten(train_truth[:, i]), torch.flatten(train_pred[:, i])),
                                     index=record_columns)
            train_records[i].loc[epoch] = train_result
            vali_result = pd.Series(result_evaluate(torch.flatten(vali_truth[:, i]), torch.flatten(vali_pred[:, i])),
                                    index=record_columns)
            vali_records[i].loc[epoch] = vali_result

        # 保存模型参数
        train_effect = True
        for i in range(len(targetBP)):
            train_effect = train_effect and vali_records[i].at[epoch, 'MAE'] < vali_records[i].at[best_epoch, 'MAE']
        if train_effect:
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_param)

    # 保存train和eval中 SBP和 DBP的不同指标
    for index, bp in enumerate(targetBP):
        train_log = os.path.join(RESULT_ROOT, 'log_file', user_id, MODEL_NAME, bp + '_train.csv')
        train_records[index].to_csv(train_log, header=True, index=False)
        valid_log = os.path.join(RESULT_ROOT, 'log_file', user_id, MODEL_NAME, bp + '_valid.csv')
        vali_records[index].to_csv(valid_log, header=True, index=False)


def model_testing(MODEL_NAME, user_id, model, test_dataset, targetBP=['SBP', 'DBP']):
    best_model_param = os.path.join(RESULT_ROOT, 'params', user_id, MODEL_NAME + '_param.pkl')
    model.load_state_dict(torch.load(best_model_param))
    model.eval()
    test_dataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    for i, data in enumerate(test_dataLoader):
        seq, ft, bp = data[0].cuda(), data[1].cuda(), data[2].cuda()
        prediction = model(seq, ft)
        if i == 0:
            test_pred = prediction.detach().cpu()
            test_truth = bp.cpu()
        else:
            test_pred = torch.cat([test_pred, prediction.detach().cpu()])
            test_truth = torch.cat([test_truth, bp.cpu()])

    for index, bp in enumerate(targetBP):
        test_result = pd.DataFrame(test_pred[:, index].numpy().T, columns=['test_result'])
        test_result_file = os.path.join(RESULT_ROOT, 'test_output', user_id, MODEL_NAME, bp + '.csv')
        test_result.to_csv(test_result_file, index=False, header=False)

        test_log = os.path.join(RESULT_ROOT, 'log_file', user_id, MODEL_NAME, bp + '_test.csv')
        test_eval = pd.Series(result_evaluate(torch.flatten(test_truth[:, index]), torch.flatten(test_pred[:, index])),
                              index=record_columns).to_frame().T
        test_eval.to_csv(test_log, header=True, index=False)
    print('testing finished')
