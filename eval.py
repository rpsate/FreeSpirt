import argparse
import json

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Dataset.DataLoaderVec import CodeDataset
from configs import config
from Model.SiameseNetwork import SiameseNetwork

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessing program.')
    parser.add_argument('--db', type=str, required=True, help='SQLite database file for storing datasets.')
    parser.add_argument('--model', type=str, required=True, help='Model file to save path.')
    parser.add_argument('--out', type=str, required=True, help='Output file for saving results.')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ckpt = torch.load(args.model, map_location=device)  # Load your best model
    model.load_state_dict(ckpt)

    # 加载数据
    print('Info: Start initializing test data···')
    test_dataset = CodeDataset(
        db_file=args.db,
        table_name="test_vec",
        data_size=config.TEST_DATASET_SIZE,
        node_size=config.MAX_NODE_SIZE
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model.eval()

    result = {
        'label': [],
        'pred': [],
    }
    with torch.no_grad():
        for input_l, input_r, labels in tqdm(test_loader):
            ast_l, features_l = input_l
            ast_r, features_r = input_r
            ast_l, ast_r, = ast_l.to(device), ast_r.to(device)
            for key in features_l.keys():
                features_l[key] = features_l[key].to(device)
            for key in features_r.keys():
                features_r[key] = features_r[key].to(device)
            labels = labels.to(device)
            outputs = model(ast_l, ast_r, features_l, features_r)
            result['label'].extend(labels.detach().cpu().tolist())
            result['pred'].extend(outputs.detach().cpu().tolist())

    with open(args.out, 'w') as f:
        json.dump(result, f)