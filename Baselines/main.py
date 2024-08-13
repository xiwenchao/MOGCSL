import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from datasets.kaggle import KaggleDataset
from datasets.rc15 import RC15Dataset
from models.sharedbottom import SharedBottomModel
from models.mmoe import MMoEModel
from models.SASRec import SASRecModel
from models.utility import calculate_hit, pad_history, pareto_step
import pandas as pd
import math

import logging
import time as Time
logging.getLogger().setLevel(logging.INFO)


def get_dataset(name, path):
    if 'Kaggle' in name:
        return KaggleDataset(path)
    elif 'RC15' in name:
        return RC15Dataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, outupt_dim, numerical_num, task_num, expert_num, embed_dim):  
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    
    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(numerical_num, embed_dim=embed_dim, bottom_mlp_dims=[64], tower_mlp_dims=[outupt_dim], task_num=task_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(numerical_num, embed_dim=embed_dim, bottom_mlp_dims=[64], tower_mlp_dims=[outupt_dim], task_num=task_num, expert_num=expert_num, dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(base_model, botom_model, optimizer, data_loader, criterion, device, task_num, loss_weights, trained_vars, log_interval=200):
    global total_step
    base_model.train()
    botom_model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    w_a, w_b = 0.5, 0.5
    c_a, c_b = 0.4, 0.2
    loss_dwa = [[1, 1], [1, 1]]
    T_dwa = 20
    for i, (numerical_fields, labels) in enumerate(loader):
        loss_list = list()
        train_numerical_fields = numerical_fields.to(device)
        state = train_numerical_fields[:, :-2]
        len_state = train_numerical_fields[:, -2]
        state = state.int().to(device)
        len_state = len_state.int().to(device)
        latent = base_model(state, len_state, True)
        y = botom_model(latent)
        for j in range(task_num):
            if j == 1:
                valid = (np.array(labels)!=-1).all(axis=1)
                if True in valid:
                    train_labels = labels[valid, j].to(device)
                    y_valid = torch.stack([y[j][v] for v in np.where(valid==True)[0]], dim=0)
                    loss_list.append(criterion(y_valid, train_labels.long()))
            else:
                train_labels = labels[:, j].to(device)
                loss_list.append(criterion(y[j], train_labels.long()))
        if args.weights_opt == "pareto" and len(loss_list) == 2:
            start_t = Time.time()
            a_gradients, b_gradients = [], []
            for w in trained_vars: 
                # calculate the gradient of loss_list[0] with respect to w
                gradient_1 = torch.autograd.grad(loss_list[0], w, retain_graph=True, allow_unused=True)[0]
                gradient_2 = torch.autograd.grad(loss_list[1], w, retain_graph=True, allow_unused=True)[0]
                if gradient_1 == None or gradient_2 == None:
                    continue
                a_gradients.append(gradient_1.cpu().clone().reshape(-1, 1))
                b_gradients.append(gradient_2.cpu().clone().reshape(-1, 1))
            a_gradients = np.concatenate(a_gradients, axis=0)
            b_gradients = np.concatenate(b_gradients, axis=0)
            paras = np.hstack((a_gradients, a_gradients))
            paras = np.transpose(paras)
            loss_weights = pareto_step(w_a, w_b, c_a, c_b, paras)
            # logging.info("pareto cost: {}".format(Time.time() - start_t))
            w_a, w_b = loss_weights[0], loss_weights[1]
            loss_weights = torch.tensor(loss_weights).to(device)
        if args.weights_opt == "dwa" and len(loss_list) == 2:
            w = [l_1 / l_2 for l_1, l_2 in zip(loss_dwa[0], loss_dwa[1])]
            task_n = len(w)
            lamb = [math.exp(v / T_dwa) for v in w]
            lamb_sum = sum(lamb)
            loss_weights = torch.tensor([task_n * l / lamb_sum for l in lamb]).to(device)
            loss_dwa[1] = loss_dwa[0]
            loss_dwa[0] = [loss_list[0].cpu().detach().numpy(), loss_list[1].cpu().detach().numpy()]

        loss = torch.sum(loss_weights * torch.stack(loss_list))

        base_model.zero_grad()
        botom_model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_step += 1
        if total_step % 200 == 0:
            logging.info("the loss in %dth batch is: %f" % (total_step, loss.item()))
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(base_model, botom_model, data_loader, task_num, outupt_dim, state_size, device):
    base_model.eval()
    botom_model.eval()
    labels_dict, predicts_dict = {}, {}
    total_clicks=0.0
    total_purchase = 0.0
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]

    test_sessions = pd.read_pickle(os.path.join("./data/Kaggle", 'sampled_test.df'))
    test_sessions['valid_session'] = test_sessions.session_id.map(
        test_sessions.groupby('session_id')['item_id'].size() < 50 + 1)
    test_sessions = test_sessions.loc[test_sessions.valid_session].drop('valid_session', axis=1)
    test_ids = test_sessions.session_id.unique()
    groups = test_sessions.groupby('session_id')
    batch = 100
    tested = 0

    with torch.no_grad():
        while tested < len(test_ids):
            states, labels = [], []
            for j in range(batch):
                # if evaluated in eval_ids:
                id = test_ids[tested]
                group = groups.get_group(id)
                history = []
                step = 0
                for index, row in group.iterrows():
                    state = list(history)
                    len_state = state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state)
                    state = pad_history(state, state_size, outupt_dim)
                    states.append(state+[len_state]+[step])
                    item = row['item_id']
                    is_buy = row['is_buy']
                    if is_buy == 1:
                        total_purchase += 1.0
                        labels.append([item, item])
                    else:
                        total_clicks += 1.0
                        labels.append([item, -1])
                    history.append(row['item_id'])
                    step += 1
                tested += 1
                if tested == len(test_ids):
                    break
            test_numerical_fields = torch.tensor(np.array(states).astype(np.float32)).to(device)
            labels = np.array(labels).astype(np.int32)
            state = test_numerical_fields[:, :-2]
            len_state = test_numerical_fields[:, -2]
            state = state.int().to(device)
            len_state = len_state.int().to(device)
            latent = base_model(state, len_state, False)
            y = botom_model(latent)

            for i in range(task_num):
                predicts_dict[i] = y[i].tolist()
                labels_dict[i] = labels[:, i].tolist()

            calculate_hit(predicts_dict,topk,labels_dict,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase,task_num)

    logging.info('#############################################################')
    logging.info('Start test!')
    logging.info('total clicks: %d, total purchase:%d' % (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click=hit_clicks[i]/total_clicks
        hr_purchase=hit_purchase[i]/total_purchase
        ng_click=ndcg_clicks[i]/total_clicks
        ng_purchase=ndcg_purchase[i]/total_purchase
        logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # logging.info('cumulative reward @ %d: %f' % (topk[i],total_reward[i]))
        logging.info('clicks hr ndcg @ %d : %f, %f' % (topk[i],hr_click,ng_click))
        logging.info('purchase hr and ndcg @%d : %f, %f' % (topk[i], hr_purchase, ng_purchase))
    logging.info('#############################################################')

def main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir,
         click_w):
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/train.pickle')
    test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.pickle')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    loss_weights = torch.tensor([click_w, 1-click_w]).to(device)

    data_statis = pd.read_pickle('./data/Kaggle/data_statis.df')  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]

    # field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    outupt_dim = train_dataset.outupt_dim
    state_size = train_dataset.state_size
    base_model = SASRecModel(hidden_size=64, item_num=item_num,state_size=state_size, device=device).to(device) 
    botom_model = get_model(model_name, outupt_dim, 64, task_num, expert_num, embed_dim).to(device) 
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    trained_vars = list(base_model.parameters()) + list(botom_model.parameters())
    optimizer = torch.optim.Adam(params=trained_vars, lr=learning_rate, weight_decay=weight_decay)
    save_path=f'{save_dir}/{dataset_name}_{model_name}.pt'
    # early_stopper = EarlyStopper(num_trials=2, save_path=save_path)
    for epoch_i in range(epoch):
        current_time = Time.time()
        train(base_model, botom_model, optimizer, train_data_loader, criterion, device, task_num, loss_weights, trained_vars)
        end_time = Time.time()
        logging.info("the time in %dth epoch is: %f" % (epoch_i, end_time - current_time))
        total = sum([param.nelement() for param in base_model.parameters()] + [param.nelement() for param in botom_model.parameters()])
        logging.info("the total number of parameters is: %d" % total)
        
        if epoch_i >= 13 and epoch_i % 2 == 0:
            test(base_model, botom_model, test_data_loader, task_num, outupt_dim, state_size, device)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='Kaggle', choices=['RC15'])
    parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--model_name', default='mmoe', choices=['sharedbottom','mmoe'])
    parser.add_argument('--epoch', type=int, default=70)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)  # change from 0.001 to 0.005
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--click_w', type=float, default=0.167)  # 0.167
    parser.add_argument('--weights_opt', default="pareto", choices=["None", "pareto", "dwa"])
    args = parser.parse_args()
    topk=[5,10,15,20]
    total_step = 0
    logging.basicConfig(
        filename="./log/Kaggle/model:SASRec_{}_bs:{}_click_w:{}_weights_opt:{}_{}".format(args.model_name, args.batch_size, args.click_w, 
                                                                                          args.weights_opt, Time.strftime("%m-%d %H:%M:%S", Time.localtime())))


    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.click_w)