import argparse
import json

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataset.DataLoaderVec import CodeDataset
from configs import config
from Model.SiameseNetwork import SiameseNetwork
from os import path, mkdir

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(siamese_model, db_file, model_path, batch_size, num_epochs, learning_rate=0.001, weight_decay=0.01):
    # 将模型移动到指定设备
    siamese_model.to(device)

    # 创建训练，验证和测试数据集
    print('Info: Start initializing training data···')
    train_dataset = CodeDataset(
        db_file=db_file,
        table_name="train_vec",
        data_size=config.TRAIN_DATASET_SIZE,
        node_size=config.MAX_NODE_SIZE
    )
    print('Info: Start initializing validation data···')
    valid_dataset = CodeDataset(
        db_file=db_file,
        table_name="valid_vec",
        data_size=config.VALID_DATASET_SIZE,
        node_size=config.MAX_NODE_SIZE
    )
    # print('Info: Start initializing test data···')
    # test_dataset = CodeDataset(
    #     db_file=db_file,
    #     table_name="test_vec",
    #     data_size=config.TEST_DATASET_SIZE,
    #     node_size=config.MAX_NODE_SIZE
    # )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义损失函数和优化器
    criterion = nn.BCELoss(reduction='mean')

    optimizer = torch.optim.Adam(siamese_model.parameters(), lr=learning_rate)

    # 初始化记录变量
    record_loss = {'train': [], 'valid': []}
    record_correct = {'train': [], 'valid': []}
    min_loss = 1000.
    early_stop_cnt = 0

    for epoch in range(num_epochs):
        # 训练模型
        train_loss = 0.
        train_correct = 0.
        siamese_model.train()
        for input_l, input_r, labels in tqdm(train_loader, desc=f'epoch:{epoch+1}, train'):
            ast_l, features_l = input_l
            ast_r, features_r = input_r
            ast_l, ast_r, = ast_l.to(device), ast_r.to(device)
            for key in features_l.keys():
                features_l[key] = features_l[key].to(device)
            for key in features_r.keys():
                features_r[key] = features_r[key].to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = siamese_model(ast_l, ast_r, features_l, features_r)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算loss和准确度
            current_batch_size = len(labels.detach().cpu())
            loss_cpu = loss.detach().cpu().item()
            correct_cpu = (outputs.detach() > config.THRESHOLD).eq(labels.detach().byte()).cpu().sum().item()
            train_loss += loss_cpu * current_batch_size
            train_correct += correct_cpu
            record_loss['train'].append(loss_cpu)
            record_correct['train'].append(correct_cpu / current_batch_size)

        # 在验证集上评估模型
        valid_loss = 0.
        valid_correct = 0.
        siamese_model.eval()
        with torch.no_grad():
            for input_l, input_r, labels in tqdm(valid_loader, desc=f'epoch:{epoch+1}, valid'):
                ast_l, features_l = input_l
                ast_r, features_r = input_r
                ast_l, ast_r, = ast_l.to(device), ast_r.to(device)
                for key in features_l.keys():
                    features_l[key] = features_l[key].to(device)
                for key in features_r.keys():
                    features_r[key] = features_r[key].to(device)
                labels = labels.to(device)
                outputs = siamese_model(ast_l, ast_r, features_l, features_r)
                loss = criterion(outputs, labels)

                # 计算loss和准确度
                current_batch_size = len(labels.detach().cpu())
                loss_cpu = loss.detach().cpu().item()
                correct_cpu = (outputs.detach() > config.THRESHOLD).eq(labels.detach().byte()).cpu().sum().item()
                valid_loss += loss_cpu * current_batch_size
                valid_correct += correct_cpu
                record_loss['valid'].append(loss_cpu)
                record_correct['valid'].append(correct_cpu / current_batch_size)

        # 计算每个epoch的准确度和loss
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        valid_accuracy = valid_correct / len(valid_loader.dataset)

        # 打印训练和验证损失及准确率
        print(f'Info: Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')

        if valid_loss < min_loss:
            min_loss = valid_loss
            early_stop_cnt = 0
            print('Info: Saving model (epoch={}, loss={:.4f})'.format(epoch+1, min_loss))
            # 保存模型
            save_dir = path.dirname(model_path)
            if not path.exists(save_dir):
                 mkdir(save_dir)
            torch.save(siamese_model.state_dict(), model_path)
        else:
            early_stop_cnt += 1

        if early_stop_cnt > config.EARLY_STOP:
            print('Info: Loss did not decrease for 10 consecutive epochs and completed training ahead of schedule!')
            print('Info: Finish training after {} epochs, the loss={:.4f}'.format(epoch+1, min_loss))
            break

    return record_loss, record_correct


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing program.')
    parser.add_argument('--db', type=str, required=True, help='SQLite database file for storing datasets.')
    parser.add_argument('--model', type=str, required=True, help='Model file to save path.')
    parser.add_argument('--record', type=str, required=True, help='Record file to save path.')
    args = parser.parse_args()

    # 初始化模型
    model = SiameseNetwork(
        config.SIAMESE_EMBEDDING_DIM,
        config.EMBEDDING_DIM,
        config.HIDDEN_DIM,
        config.OUTPUT_DIM,
        config.NUM_LAYERS,
        config.NUM_HEADS,
        config.DROPOUT
    )
    record_loss, record_correct = train_model(model, args.db, args.model, config.BATCH_SIZE, config.NUM_EPOCHS, config.LEARNING_RATE, config.WEIGHT_DECAY)

    # 保存loss和correct
    record = {
        'loss': record_loss,
        'correct': record_correct
    }
    with open(args.record, 'w') as f:
        json.dump(record, f)
        print(f'Info: The accuracy and loss have been saved in {args.record}!')
