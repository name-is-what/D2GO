def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os

from model import HCL
from data_loader import *
from mix_data_loader import get_ood_dataset_new, get_ood_dataset_new_diverse
import argparse
import numpy as np
import torch
import random
import sklearn.metrics as skm
import torch_geometric
import datetime
import pytz
from queue import PriorityQueue
import logging
import time
from function_sp import *

from torch_geometric.loader import DataLoader

import os
import psutil

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_type', type=str, default='oodd', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default="BZR+COX2")
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=32)
    parser.add_argument('-num_trial', type=int, default=1)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-weight_in', type=float, default=1.0)
    parser.add_argument('-weight_out', type=float, default=1.0)

    parser.add_argument('-lam_range', type=str, default="[0.7,1.0]")
    parser.add_argument('-top_k', type=int, default=5)
    return parser.parse_args()

def get_time():
    utc_now = datetime.datetime.utcnow()
    utc_now = utc_now.replace(tzinfo=pytz.utc)
    return utc_now

def batch_knn_cosine_score(query_feats: torch.Tensor,
                           dict_feats: torch.Tensor,
                           k: int = 5) -> torch.Tensor:
    """
    query_feats: (B1, D)
    dict_feats:  (B2, D)
    return: scores of shape (B1,), where each is kth largest cosine similarity
    """
    # Normalize
    query_feats = torch.nn.functional.normalize(query_feats, dim=1)
    dict_feats = torch.nn.functional.normalize(dict_feats, dim=1)

    # Compute cosine similarity: (B1, B2)
    sim_matrix = torch.matmul(query_feats, dict_feats.T)

    # Get top-k values per row (largest=True), then get kth largest (index k-1)
    topk_vals, _ = torch.topk(sim_matrix, k=k, dim=1, largest=True, sorted=True)
    kth_vals = topk_vals[:, -1]  # shape: (B1,)

    return kth_vals  # higher means more similar to dict


def setup_logger(args):
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    lam_range = eval(args.lam_range)

    filepath = f'D2GO_{args.DS_pair}'
    log_file = f"{filepath}_{get_time().strftime('%m%d-%H%M')}.log"
    log_file = os.path.join(log_dir, log_file)

    def normal(sec, what):
        normal_time = datetime.datetime.now() + datetime.timedelta(hours=12)
        return normal_time.timetuple()

    logging.Formatter.converter = normal
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger()

if __name__ == '__main__':
    setup_seed(0)
    args = arg_parse()
    log = setup_logger(args)
    num_epoch=args.num_epoch
    log.info(args)
    root_path = '.'

    aucs=[]
    for trial in range(args.num_trial):
        log.info(f'Trial {trial}')
        setup_seed(trial + 1)

        dataloader, dataloader_test, meta, dataset_triple = get_ood_dataset_new(args)
        # DS='PTC_MR', DS_ood='COX2'
        # DS='PTC_MR', DS_ood='NCI1'
        # DS='IMDB-BINARY', DS_ood='REDDIT-BINARY'
        # DS='BZR', DS_ood='MUTAG'
        # DS='ENZYMES', DS_ood='DD'
        # DS='AIDS', DS_ood='PROTEINS'
        # dataloader, dataloader_test, meta, dataset_triple = get_ood_dataset_new_diverse(args, DS='BZR', DS_ood='MUTAG')
        log.info(f'meta: {meta}')

        dataset_num_features = meta['num_feat']
        n_train = meta['num_train']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HCL(args.hidden_dim, args.num_layer, dataset_num_features, args.dg_dim+args.rw_dim).to(device)


        save_path = os.path.join(os.getcwd(), "OODCL")
        file_name = f"OOD_{args.DS_pair}.pth"
        file_path = os.path.join(save_path, file_name)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if trial == 0:
            log.info('================')
            log.info('Exp_type: {}'.format(args.exp_type))
            log.info('DS: {}'.format(args.DS_pair if args.DS_pair is not None else args.DS))
            log.info('num_train: {}'.format(n_train))
            log.info('num_features: {}'.format(dataset_num_features))
            log.info('num_structural_encodings: {}'.format(args.dg_dim + args.rw_dim))
            log.info('hidden_dim: {}'.format(args.hidden_dim))
            log.info('num_gc_layers: {}'.format(args.num_layer))
            log.info('dataset:{}'.format(args.DS_pair))
            log.info('================')

        model.eval()
        y_true_all = []
        y_score_all = []


        print("===================== TEST TIME =====================")
        weight_in,weight_out=1, 1
        y_true_all = []
        y_score_all = []
        torch.cuda.reset_peak_memory_stats()
        start_gpu_alloc = torch.cuda.memory_allocated()
        start_time = time.time()
        for data in dataloader_test:
            data = data.to(device)
            b, g_f, g_s, n_f, n_s = model(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
            y_score_g = model.calc_loss_g(g_f, g_s)
            y_score_n = model.calc_loss_n(n_f, n_s, data.batch)
            y_score=y_score_g + y_score_n

            print("Pre Judging with model...")
            y_pred = g_f.softmax(dim=1).cpu()
            y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
            y_pred = torch.tensor(y_pred).to(device)

            dataset_for_search_id=dataset_triple[0]
            dataset_for_search_ood=dataset_triple[1]
            ID_list=[]
            OOD_list=[]
            list_id,list_ood=split_dict(data,y_pred,25,75)
            ID_list=ID_list+list_id
            OOD_list=OOD_list+list_ood

            print("Estimate and Mixup Graphon...")
            ID_graphon_queue=mixup_dataloader(ID_list,dataset_triple[0],args,lam_range=(0.5,0.5),aug_num=5,batch_size=9999,shuffle=True,keep_graphon_size=False)
            OOD_graphon_queue=mixup_dataloader(OOD_list,dataset_triple[1],args,lam_range=(0.5,0.5),aug_num=5,batch_size=9999,shuffle=True,keep_graphon_size=False)

            # social: batch_size=32
            # ID_graphon_queue=mixup_dataloader(ID_list,dataset_triple[0],args,lam_range=(0.5,0.5),aug_num=5,batch_size=128,shuffle=True,keep_graphon_size=False)
            # OOD_graphon_queue=mixup_dataloader(OOD_list,dataset_triple[1],args,lam_range=(0.5,0.5),aug_num=5,batch_size=128,shuffle=True,keep_graphon_size=False)

            for ID_data,OOD_data in zip(ID_graphon_queue,OOD_graphon_queue):
                break
            ID_data=ID_data.to(device)
            OOD_data=OOD_data.to(device)

            b_oe, g_f_in, _, _, _ = model(ID_data.x, ID_data.x_s, ID_data.edge_index, ID_data.batch, ID_data.num_graphs)
            b_oe, g_f_out, _, _, _ = model(OOD_data.x, OOD_data.x_s, OOD_data.edge_index, OOD_data.batch, OOD_data.num_graphs)


            torch.cuda.reset_peak_memory_stats()
            start_time_queue = time.time()
            start_mem_queue = torch.cuda.memory_allocated()

            if len(ID_graphon_queue) > args.top_k or len(OOD_graphon_queue) > args.top_k:
                score_in = -batch_knn_cosine_score(g_f, g_f_in, k=5)
                score_out = batch_knn_cosine_score(g_f, g_f_out, k=5)
                y_score += score_in * weight_in + score_out * weight_out

            end_time_queue = time.time()
            end_mem_queue = torch.cuda.memory_allocated()
            peak_mem_queue = torch.cuda.max_memory_allocated()
            print(f"Queue Time: {(end_time_queue - start_time_queue)*1000:.2f} ms, GPU Memory: {(end_mem_queue - start_mem_queue) / 1024 / 1024:.2f} MB, Peak GPU Memory: {peak_mem_queue / 1024 / 1024:.2f} MB")

            max_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

            y_true = data.y
            y_score_all = y_score_all + y_score.detach().cpu().tolist()
            y_true_all = y_true_all + y_true.detach().cpu().tolist()
            # torch.cuda.reset_peak_memory_stats()
        auc = skm.roc_auc_score(y_true_all, y_score_all)
        end_time = time.time()
        runtime = end_time - start_time  # 以秒为单位
        end_gpu_alloc = torch.cuda.memory_allocated()
        log.info('[TEST TIME] Trial: {:01d} | AUC: {:.4f} | Runtime: {:.2f}s | Max GPU Memory: {:.4f}MiB'.format(
            trial, auc, runtime, max_gpu_memory))
        print('all gpu memory:', (end_gpu_alloc - start_gpu_alloc) / 1024 / 1024, 'MiB')
        # log.info('[TEST TIME] trial: {:01d} | AUC:{:.4f}'.format(trial, auc))
    aucs.append(auc)
    log.info('[TEST TIME] AUC_mean:{:.4f}'.format(np.mean(aucs)))
    log.info('[TEST TIME] AUC_std:{:.4f}'.format(np.std(aucs)))



# python d2go.py -exp_type oodd -DS_pair AIDS+DHFR -batch_size_test 128 -num_epoch 100 -hidden_dim 32
# python d2go.py -exp_type oodd -DS_pair PTC_MR+MUTAG -batch_size_test 128 -num_epoch 100 -hidden_dim 32
# python d2go.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -batch_size_test 128 -num_epoch 100 -hidden_dim 32
# python d2go.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -batch_size_test 128 -num_epoch 100 -hidden_dim 32
