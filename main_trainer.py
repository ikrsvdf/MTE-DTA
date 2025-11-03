# -*-coding:utf-8-*-
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess.dataloder import create_DTA_dataset
from model.model_initial import HierarchyT
from preprocess.utils import *
import logging
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
FORCE_REPROCESS = True
project_root = 'MTE-DTA'  # 根据你的实际路径调整
os.chdir(project_root)

def train(model, device, train_loader, optimizer, loss_fn):
    train_losses_in_epoch = []  # 存储一个 epoch 内的所有样本损失
    model.train()
    for data in tqdm(train_loader):  # 遍历 batch
        compounds = data[0].to(device)
        protein = data[1].to(device)
        interaction = compounds.y.to(device)
        optimizer.zero_grad()  # 梯度清零
        # 对每个样本进行独立的前向传播，避免批次堆叠问题
        output = model(protein, compounds)  # 前向传播
        train_loss = loss_fn(output, interaction.view(-1, 1).float().to(device))  # 计算损失
        train_losses_in_epoch.append(train_loss.item())  # 记录每个 batch 的平均损失
        train_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 计算并记录一个 epoch 的平均训练损失
    train_loss_a_epoch = np.average(train_losses_in_epoch)

    return train_loss_a_epoch

def test(dataloader, model, loss_fn):
    valid_losses_in_epoch = []
    model.eval()
    total_preds = []  # 存储所有预测值
    total_labels = []  # 存储所有真实标签
    with torch.no_grad():
        for data in tqdm(dataloader):  # 遍历 batch
            compounds = data[0].to(device)
            protein = data[1].to(device)
            interaction = compounds.y.to(device)

            output = model.forward(protein, compounds)  # 前向传播
            val_loss = loss_fn(output, interaction.view(-1, 1).float().to(device))  # 计算损失
            valid_losses_in_epoch.append(val_loss.item())
            # 将预测值和真实标签添加到列表中
            total_preds.extend(output.cpu().numpy().flatten())
            total_labels.extend(interaction.cpu().numpy().flatten())

    valid_loss_a_epoch = np.average(valid_losses_in_epoch)

    return valid_loss_a_epoch, np.array(total_labels), np.array(total_preds)

dataset_name = ['Davis', 'Kiba', 'Parasite']
type = ['random', 'filter_radom', 'cold']
model_st = 'Davis'
datasets = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
logging.basicConfig(filename=f'{model_st}.log', level=logging.DEBUG)
BATCH_SIZE = 16
device = torch.device("cuda:0")
LR = 1e-5

all_mse, all_rmse, all_mae, all_rm2, all_ci, all_spearman, all_pearson = [], [], [], [], [], [], []

for n, dataset in enumerate(datasets, start=1):
    print('\nrunning on ', model_st + '_' + dataset)
    ## 1) 数据准备
    train_data, val_data, test_data = create_DTA_dataset(dataset_name[0], type[0], dataset,
                                                         force_reprocess=FORCE_REPROCESS)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=0, pin_memory=True)

    ##  2） 构建结构保存路径
    fold_name = "{}_Fold/".format(n)
    save_path_i = os.path.join(dataset_name[0], fold_name)
    if not os.path.exists(save_path_i):
        os.makedirs(save_path_i)

    ##  3） 重新初始化模型、损失函数和优化器
    model = HierarchyT(protein_dim=1280, drug_dim=78, hid_dim=128, atom_mid_dim=32, num_features_mol=384,
                       num_features_pro=33, n_heads=8, n_enlayers=2, n_delayers=2, device=device)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    epochs = 800
    best_mse = 100
    patience = 0  # 耐心计数器，用于早停
    Patience = 50

    ##  4） 开始训练
    print('第---%d---折训练开始' % (n))
    logging.info(f'第---{n}---折训练开始')
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, loss_fn)
        val_loss, G, P = test(val_loader, model, loss_fn)
        val_mse = mse1(G, P)
        val_mae = mean_absolute_error(G, P)
        val_person = pearson(G, P)
        val_spearman = spearman(G,P)

        print('epoch-%d, train_loss-%.3f, val_loss-%.3f, val_mse-%.3f, val_mae-%.3f, val_person-%.3f, val_spearman-%.3f' % (
            epoch + 1, train_loss, val_loss, val_mse, val_mae, val_person, val_spearman))
        # 在训练循环中添加
        with open('training_log.txt', 'a') as f:
            f.write('epoch-%d, train_loss-%.3f, val_loss-%.3f, val_mse-%.3f, val_mae-%.3f, val_person-%.3f, val_spearman-%.3f\n' % (
                epoch + 1, train_loss, val_loss, val_mse, val_mae, val_person, val_spearman))

        if  val_mse < best_mse:
            best_mse = val_mse
            patience = 0
            torch.save(model.state_dict(), save_path_i + model_st + '-valid_best_model.pth')
        else:
            patience += 1
            
        # 每10个epoch保存一次测试集预测结果
        if (epoch + 1) % 1 == 0:
            # 加载当前模型进行测试
            model.load_state_dict(torch.load(save_path_i + model_st + "-valid_best_model.pth"))
            test_loss, G_test, P_test = test(test_loader, model, loss_fn)
            
            test_csv_path = os.path.join('data', dataset_name[0], type[0], dataset, 'data_test.csv')
            df_test_full = pd.read_csv(test_csv_path, encoding='gbk')
            
            # 创建包含epoch信息的文件名
            epoch_output_csv_path = f'{model_st}_test_predictions.csv'
            
            fold_results = pd.DataFrame({
                'Smiles': df_test_full['Smiles'],
                'Sequence': df_test_full['Sequence'],
                'UniProt Accessions': df_test_full['UniProt Accessions'],
                'BindingDB MonomerID': df_test_full['BindingDB MonomerID'],
                'Average_pIC50': df_test_full['Average_pIC50'],
                'Predicted_pIC50': P_test
            })
            
            fold_results.to_csv(epoch_output_csv_path, index=False)
            print(f'第 {epoch+1} 轮测试集预测结果已保存到: {epoch_output_csv_path}')
            logging.info(f'第 {epoch+1} 轮测试集预测结果已保存到: {epoch_output_csv_path}')
            
        if patience == Patience:  # 如果耐心计数器达到设定值，停止训练
            break
    
    all_test_results = []  # 存储所有测试集结果
    ##  5）保存最终模型并测试
    torch.save(model.state_dict(), save_path_i + model_st + '-stable_model.pth')
    """load trained model"""  # 加载最佳模型
    model.load_state_dict(torch.load(save_path_i + model_st + "-valid_best_model.pth"))

    test_loss, G, P = test(test_loader, model, loss_fn)

    test_csv_path = os.path.join('data', dataset_name[0], type[0], dataset, 'data_test.csv')
    df_test_full = pd.read_csv(test_csv_path, encoding='gbk')
    fold_results = pd.DataFrame({
        'Smiles': df_test_full['Smiles'],
        'Sequence': df_test_full['Sequence'],
        'UniProt Accessions': df_test_full['UniProt Accessions'],
        'BindingDB MonomerID': df_test_full['BindingDB MonomerID'],
        'Average_pIC50': df_test_full['Average_pIC50'],
        'Predicted_pIC50': P
    })


    all_test_results.append(fold_results)
    test_mse, test_rmse, test_mae, test_person, test_sperman, test_ci, test_rm2 = mse1(G, P), rmse(G,P), mean_absolute_error(G, P), pearson(G, P), spearman(G, P), CI2(G, P), get_rm2(G, P)
    print('test_mse-%.4f, test_rmse-%.4f,test_mae-%.4f, test_ci-%.4f, test_rm2-%.4f' % (test_mse, test_rmse, test_mae, test_ci, test_rm2))
    logging.info(
        f'Test mse {test_mse}, rmse {test_rmse}, mae {test_mae}, person {test_person}, sperman {test_sperman}, ci{test_ci}, rm2 {test_rm2}')
    logging.info(f'第---{n}---折训练完成')
    print('第---%d---折训练完成' % (n))
    all_mse.append(test_mse)
    all_rmse.append(test_rmse)
    all_mae.append(test_mae)
    all_pearson.append(test_person)
    all_spearman.append(test_sperman)
    all_ci.append(test_ci)
    all_rm2.append(test_rm2)
    final_test_results = pd.concat(all_test_results, ignore_index=True)
    output_csv_path = f'{model_st}_test_predictions_final.csv'
    final_test_results.to_csv(output_csv_path, index=False)
    print(f'最终测试集预测结果已保存到: {output_csv_path}')
    logging.info(f'最终测试集预测结果已保存到: {output_csv_path}')

# 计算平均值
average_mse = np.mean(all_mse)
average_rmse = np.mean(all_rmse)
average_mae = np.mean(all_mae) 
average_pearson = np.mean(all_pearson)
average_spearman = np.mean(all_spearman)
average_ci = np.mean(all_ci)
average_rm2 = np.mean(all_rm2)

print(f'五折平均 MSE: {average_mse:.4f}')
print(f'五折平均 RMSE: {average_rmse:.4f}')
print(f'五折平均 MAE: {average_mae:.4f}')
print(f'五折平均 Pearson: {average_pearson:.4f}')
print(f'五折平均 Spearson: {average_spearman:.4f}')
print(f'五折平均 CI: {average_ci:.4f}')
print(f'五折平均 RM2: {average_rm2:.4f}')

logging.info(f'The results on five_data:')
logging.info(f'average_mse :{average_mse}')
logging.info(f'average_rmse :{average_rmse}')
logging.info(f'average_mae :{average_mae}')
logging.info(f'average_pearson :{average_pearson}')
logging.info(f'average_spearman :{average_spearman}')
logging.info(f'average_ci :{average_ci}')
logging.info(f'average_rm2 :{average_rm2}')
