import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import random
import os
import sys
from typing import List, Dict, Tuple
import torch_geometric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.graphon_utils import split_graphs, align_tensor_graphs, universal_tensor_svd, two_graphon_mixup, two_graphon_mixup_random_align, adjust_graphon_size, stat_graph, prepare_dataset_x_xs_mean
from torch_geometric.utils import dense_to_sparse
import math

def split_by_model(model, test_data_list, device, threshold=0.5):
    """
    使用模型计算预测标签并将测试数据分为ID和OOD两组

    参数:
    - model: 已训练好的模型
    - test_data_list: 测试数据的DataLoader或单个DataBatch对象
    - device: 计算设备 (CPU或GPU)
    - threshold: 分类阈值，默认为0.5

    返回:
    - ID_list: 内分布数据列表
    - OOD_list: 分布外数据列表
    """
    model.eval()
    all_pred_labels = []

    with torch.no_grad():
        for batch in test_data_list:
            batch = batch.to(device)
            # 获取模型预测
            b, g_f, g_s, n_f, n_s = model(batch.x, batch.x_s, batch.edge_index, batch.batch, batch.num_graphs)
            # 计算预测标签
            y_pred = g_f.softmax(dim=1).cpu()
            y_pred_labels = np.argmax(y_pred.numpy(), axis=1)
            all_pred_labels.extend(y_pred_labels)

    # 调用现有的分类函数
    return split_by_pred(test_data_list, np.array(all_pred_labels), threshold)

def split_by_pred(test_data_list, y_pred_labels, threshold=0.5, method=1):
    """
    根据外部传入的预测标签将测试数据分为ID和OOD两组

    参数:
    - test_data_list: 测试数据的DataLoader或单个DataBatch对象
    - y_pred_labels: 外部传入的预测标签
    - threshold: 分类阈值，默认为0.5
    - method: 二元化方法选择
        1: 二聚类
        2: 平均数
        3: 中位数

    返回:
    - ID_list: 内分布数据列表
    - OOD_list: 分布外数据列表
    """
    ID_list = []
    OOD_list = []

    # 如果传入的是单个DataBatch，将其包装成列表
    if isinstance(test_data_list, torch_geometric.data.Batch):
        test_data_list = [test_data_list]

    # 确保y_pred_labels是numpy数组
    if isinstance(y_pred_labels, torch.Tensor):
        y_pred_labels = y_pred_labels.cpu().numpy()

    # 根据选择的方法进行二元化
    if method == 1:  # 二聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=0)
        binary_labels = kmeans.fit_predict(y_pred_labels.reshape(-1, 1))
    elif method == 2:  # 平均数
        mean_value = np.mean(y_pred_labels)
        binary_labels = (y_pred_labels > mean_value).astype(int)
    elif method == 3:  # 中位数
        median_value = np.median(y_pred_labels)
        binary_labels = (y_pred_labels > median_value).astype(int)
    else:
        raise ValueError("不支持的二元化方法，请选择1(二聚类)、2(平均数)或3(中位数)")

    # 初始化标签索引
    label_idx = 0

    for batch in test_data_list:
        num_graphs = batch.num_graphs

        # 遍历每个图
        for graph_idx in range(num_graphs):
            # 获取当前图
            graph_data = extract_graph(batch, graph_idx)

            # 获取当前图的二元化标签
            if label_idx < len(binary_labels):
                binary_label = binary_labels[label_idx]
                label_idx += 1

                # 根据二元化结果分类
                if binary_label == 0:  # 假设第一个类别为ID
                    ID_list.append(graph_data)
                else:  # 第二个类别为OOD
                    OOD_list.append(graph_data)

    return ID_list, OOD_list

def split_by_label(test_data_list, device):
    """
    根据真实标签将测试数据分为ID和OOD两组
    在这个函数中，假设y=0表示ID数据，y=1表示OOD数据

    参数:
    - test_data_list: 测试数据的DataLoader
    - device: 计算设备 (CPU或GPU)

    返回:
    - ID_list: 内分布数据列表
    - OOD_list: 分布外数据列表
    """
    ID_list = []
    OOD_list = []

    for batch in test_data_list:
        batch = batch.to(device)
        num_graphs = batch.num_graphs

        # 遍历每个图
        for graph_idx in range(num_graphs):
            # 获取当前图
            graph_data = extract_graph(batch, graph_idx)

            # 获取图的标签
            if graph_data.y.item() == 0:  # 假设0表示ID
                ID_list.append(graph_data)
            else:  # 假设非0表示OOD
                OOD_list.append(graph_data)

    return ID_list, OOD_list

def split_by_score(test_data_list, scores, threshold=0.5):
    """
    根据外部传入的异常得分将测试数据分为ID和OOD两组

    参数:
    - test_data_list: 测试数据的DataLoader
    - scores: 外部传入的异常得分，形状为(num_graphs,)
    - threshold: 分数阈值，高于此值被视为OOD，默认为0.5

    返回:
    - ID_list: 内分布数据列表
    - OOD_list: 分布外数据列表
    """
    ID_list = []
    OOD_list = []

    for batch in test_data_list:
        num_graphs = batch.num_graphs

        # 遍历每个图
        for graph_idx in range(num_graphs):
            # 获取当前图
            graph_data = extract_graph(batch, graph_idx)

            # 获取当前图的得分
            if graph_idx < len(scores):
                current_score = scores[graph_idx]

                # 根据得分对图进行分类
                if current_score < threshold:  # 低分为ID，高分为OOD
                    ID_list.append(graph_data)
                else:
                    OOD_list.append(graph_data)

    return ID_list, OOD_list

def split_by_sim(test_data_list, similarities, threshold=0.5):
    """
    根据外部传入的余弦相似度将测试数据分为ID和OOD两组

    参数:
    - test_data_list: 测试数据的DataLoader
    - similarities: 外部传入的余弦相似度得分，形状为(num_graphs,)
    - threshold: 相似度阈值，高于此值被视为ID，默认为0.5

    返回:
    - ID_list: 内分布数据列表
    - OOD_list: 分布外数据列表
    """
    ID_list = []
    OOD_list = []

    for batch in test_data_list:
        num_graphs = batch.num_graphs

        # 遍历每个图
        for graph_idx in range(num_graphs):
            # 获取当前图
            graph_data = extract_graph(batch, graph_idx)

            # 获取当前图的相似度得分
            if graph_idx < len(similarities):
                current_similarity = similarities[graph_idx]

                # 根据相似度对图进行分类
                if current_similarity > threshold:  # 高相似度为ID，低相似度为OOD
                    ID_list.append(graph_data)
                else:
                    OOD_list.append(graph_data)

    return ID_list, OOD_list

def split_by_threshold(test_data_list, scores, threshold1, threshold2):
    """
    根据分数和阈值将测试数据分为ID和OOD两组

    参数:
    - test_data_list: 测试数据的DataLoader或单个DataBatch对象
    - scores: 分数张量，形状为(样本数量,)
    - threshold1: ID阈值，低于此阈值的样本被视为ID
    - threshold2: OOD阈值，高于此阈值的样本被视为OOD

    返回:
    - ID_list: 内分布数据列表
    - OOD_list: 分布外数据列表
    """
    ID_list = []
    OOD_list = []

    # 如果传入的是单个DataBatch，将其包装成列表
    if isinstance(test_data_list, torch_geometric.data.Batch):
        test_data_list = [test_data_list]

    # 确保scores是numpy数组
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # 初始化标签索引
    label_idx = 0

    for batch in test_data_list:
        num_graphs = batch.num_graphs

        # 遍历每个图
        for graph_idx in range(num_graphs):
            # 获取当前图
            graph_data = extract_graph(batch, graph_idx)

            # 获取当前图的分数
            if label_idx < len(scores):
                score = scores[label_idx]
                label_idx += 1

                # 根据阈值分类
                if score < threshold1:  # 低于threshold1的为ID
                    ID_list.append(graph_data)
                elif score > threshold2:  # 高于threshold2的为OOD
                    OOD_list.append(graph_data)
                # 介于threshold1和threshold2之间的样本不分类

    return  ID_list,OOD_list

def extract_graph(batch, graph_idx):
    """
    从批次中提取单个图数据

    参数:
    - batch: 图数据批次
    - graph_idx: 要提取的图索引

    返回:
    - 单个图的Data对象
    """
    mask = batch.batch == graph_idx

    # 获取节点特征和结构特征
    x = batch.x[mask]
    x_s = batch.x_s[mask] if hasattr(batch, 'x_s') else None

    # 获取边索引 (需要调整为新的节点索引)
    edge_mask = torch.zeros(batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device)
    for i in range(batch.edge_index.size(1)):
        if mask[batch.edge_index[0, i]] and mask[batch.edge_index[1, i]]:
            edge_mask[i] = True

    edge_index = batch.edge_index[:, edge_mask]

    # 创建节点映射 (从批次索引到图内索引)
    node_map = torch.zeros(mask.size(0), dtype=torch.long, device=edge_index.device)
    node_map[mask] = torch.arange(mask.sum(), device=edge_index.device)
    edge_index = node_map[edge_index]

    # 获取图的标签
    y = batch.y[graph_idx] if batch.y.dim() == 1 else batch.y[graph_idx:graph_idx+1]

    # 创建新的图数据对象
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    if x_s is not None:
        graph_data.x_s = x_s

    return graph_data

def create_loader(graph_list, batch_size=32, shuffle=False):
    """
    从图列表创建DataLoader

    参数:
    - graph_list: 图数据列表
    - batch_size: 批次大小
    - shuffle: 是否打乱数据

    返回:
    - 图数据的DataLoader
    """
    return DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)

def create_graphon(dataset_list, resolution=None):
    """
    从数据集创建graphon

    参数:
    - dataset_list: 图数据列表
    - resolution: graphon的分辨率(大小)，如果为None则使用图中节点数的中位数

    返回:
    - graphon: 生成的graphon矩阵
    """
    # 统计图结构信息
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, min_num_nodes, max_num_nodes = stat_graph(dataset_list)

    # 如果未指定分辨率，使用节点数中位数
    if resolution is None:
        resolution = int(median_num_nodes)

    # 将图拆分为列表形式
    graphs = split_graphs(dataset_list)

    # 对图进行对齐
    align_graphs_list, normalized_node_degrees, max_num, min_num = align_tensor_graphs(graphs, padding=True, N=resolution)

    # 使用SVD生成graphon
    graphon = universal_tensor_svd(align_graphs_list, threshold=0.01)

    return graphon

def create_graphon_dict(dataset_list, cross_mix_ratio=0.2, mix_samples_per_pair=5, lam_range=(0.3, 0.7), resolution=None, save_dir=None,type=0):
    """
    创建graphon字典，包含直接生成的graphon和交叉混合的graphon

    参数:
    - dataset_list: 图数据列表
    - cross_mix_ratio: 交叉混合的比例，值在[0,1]之间
    - mix_samples_per_pair: 每对图交叉混合生成的样本数
    - lam_range: 混合系数范围，元组形式(min, max)
    - resolution: graphon的分辨率(大小)，如果为None则使用中位数
    - save_dir: 保存graphon的目录，如果为None则不保存

    返回:
    - graphon_dict: 包含graphon的字典，键为graphon的名称，值为graphon矩阵
    """
    graphon_dict = {}

    # 1. 创建直接生成的graphon
    print("创建主graphon...")
    main_graphon = create_graphon(dataset_list, resolution)
    graphon_dict["main"] = main_graphon

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "main.npy"), main_graphon)

    # 如果数据集太小，不进行交叉混合
    if len(dataset_list) <= 1:
        return graphon_dict

    # 2. 创建交叉混合的graphon
    # 确定要选择的数据对数量
    n = len(dataset_list)
    max_pairs = n * (n-1) // 2  # 总的可能对数
    num_pairs = max(1, int(max_pairs * cross_mix_ratio))  # 实际选择的对数

    # 随机选择数据对进行混合
    pairs = []
    while len(pairs) < num_pairs:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i != j and (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))

    print(f"创建{len(pairs)}对交叉混合graphon...")
    for idx, (i, j) in enumerate(pairs):
        # 为这两个图创建graphon
        subset_i = [dataset_list[i]]
        subset_j = [dataset_list[j]]

        graphon_i = create_graphon(subset_i, resolution)
        graphon_j = create_graphon(subset_j, resolution)

        # 统一大小
        if graphon_i.shape[0] != graphon_j.shape[0]:
            max_size = max(graphon_i.shape[0], graphon_j.shape[0])
            graphon_i = adjust_graphon_size(graphon_i, max_size)
            graphon_j = adjust_graphon_size(graphon_j, max_size)

        # 对每对生成多个混合graphon，使用不同的混合系数
        for sample_idx in range(mix_samples_per_pair):
            # 随机生成混合系数
            lam = random.uniform(lam_range[0], lam_range[1])

            # 生成混合graphon
            mixed_graphon = two_graphon_mixup(graphon_i, graphon_j, la=lam, num_sample=1)[0]

            # 保存到字典
            key = f"mix_{idx}_{sample_idx}_lam_{lam:.2f}"
            graphon_dict[key] = mixed_graphon

            if save_dir is not None:
                np.save(os.path.join(save_dir, f"{key}.npy"), mixed_graphon)

    return graphon_dict

def create_ID_OOD_dicts(ID_list, OOD_list,
                       id_cross_mix_ratio=0.2,
                       ood_cross_mix_ratio=0.2,
                       mix_samples_per_pair=3,
                       lam_range=(0.3, 0.7),
                       resolution=None,
                       save_dir="graphon_lib"):
    """
    为ID和OOD数据创建graphon字典

    参数:
    - ID_list: ID数据列表
    - OOD_list: OOD数据列表
    - id_cross_mix_ratio: ID数据交叉混合的比例
    - ood_cross_mix_ratio: OOD数据交叉混合的比例
    - mix_samples_per_pair: 每对图交叉混合生成的样本数
    - lam_range: 混合系数范围，元组形式(min, max)
    - resolution: graphon的分辨率(大小)，如果为None则使用中位数
    - save_dir: 保存graphon的基础目录

    返回:
    - ID_dict: ID graphon字典
    - OOD_dict: OOD graphon字典
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 创建ID和OOD子目录
    id_save_dir = os.path.join(save_dir, "ID")
    ood_save_dir = os.path.join(save_dir, "OOD")
    os.makedirs(id_save_dir, exist_ok=True)
    os.makedirs(ood_save_dir, exist_ok=True)

    print(f"为{len(ID_list)}个ID数据创建graphon字典...")
    ID_dict = create_graphon_dict(
        ID_list,
        cross_mix_ratio=id_cross_mix_ratio,
        mix_samples_per_pair=mix_samples_per_pair,
        lam_range=lam_range,
        resolution=resolution,
        save_dir=id_save_dir
    )

    print(f"为{len(OOD_list)}个OOD数据创建graphon字典...")
    OOD_dict = create_graphon_dict(
        OOD_list,
        cross_mix_ratio=ood_cross_mix_ratio,
        mix_samples_per_pair=mix_samples_per_pair,
        lam_range=lam_range,
        resolution=resolution,
        save_dir=ood_save_dir
    )

    return ID_dict, OOD_dict

def create_graphon_list(dataset_list, cross_mix_ratio=0.2, mix_samples_per_pair=5, lam_range=(0.3, 0.7), resolution=None, save_dir=None,type=0):
    """
    创建graphon列表，包含直接生成的graphon和交叉混合的graphon

    参数:
    - dataset_list: 图数据列表
    - cross_mix_ratio: 交叉混合的比例，值在[0,1]之间
    - mix_samples_per_pair: 每对图交叉混合生成的样本数
    - lam_range: 混合系数范围，元组形式(min, max)
    - resolution: graphon的分辨率(大小)，如果为None则使用中位数
    - save_dir: 保存graphon的目录，如果为None则不保存

    返回:
    - graphon_list: 包含graphon的列表
    """
    graphon_list = []

    # 1. 创建直接生成的graphon
    print("创建主graphon...")
    #main_graphon = create_graphon(dataset_list, resolution)
    for i in dataset_list:
        graphon=create_graphon([i], resolution)
        graphon=two_graphon_mixup(graphon, graphon, la=0.5, num_sample=1)[0]
        graphon.y=type
        graphon_list.append(graphon)
    # graphon_list.append(main_graphon)

    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)
    #     np.save(os.path.join(save_dir, "main.npy"), )

    # 如果数据集太小，不进行交叉混合
    if len(dataset_list) <= 1:
        return graphon_list

    # 2. 创建交叉混合的graphon
    # 确定要选择的数据对数量
    n = len(dataset_list)
    max_pairs = n * (n-1) // 2  # 总的可能对数
    num_pairs = max(1, int(max_pairs * cross_mix_ratio))  # 实际选择的对数
    num_pairs = min(num_pairs,100 )

    # 随机选择数据对进行混合
    pairs = []
    while len(pairs) < num_pairs:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i != j and (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))

    print(f"创建{len(pairs)}对交叉混合graphon...")
    for idx, (i, j) in enumerate(pairs):
        # 为这两个图创建graphon
        subset_i = [dataset_list[i]]
        subset_j = [dataset_list[j]]

        graphon_i = create_graphon(subset_i, resolution)
        graphon_j = create_graphon(subset_j, resolution)

        # 统一大小
        if graphon_i.shape[0] != graphon_j.shape[0]:
            max_size = max(graphon_i.shape[0], graphon_j.shape[0])
            graphon_i = adjust_graphon_size(graphon_i, max_size)
            graphon_j = adjust_graphon_size(graphon_j, max_size)

        # 对每对生成多个混合graphon，使用不同的混合系数
        for sample_idx in range(mix_samples_per_pair):
            # 随机生成混合系数
            lam = random.uniform(lam_range[0], lam_range[1])

            # 生成混合graphon
            mixed_graphon = two_graphon_mixup(graphon_i, graphon_j, la=lam, num_sample=1)[0]
            mixed_graphon.y=type
            # 添加到列表
            graphon_list.append(mixed_graphon)

            if save_dir is not None:
                np.save(os.path.join(save_dir, f"mix_{idx}_{sample_idx}_lam_{lam:.2f}.npy"), mixed_graphon)

    return graphon_list

def create_ID_OOD_lists(ID_list, OOD_list,
                       id_cross_mix_ratio=0.2,
                       ood_cross_mix_ratio=0.2,
                       mix_samples_per_pair=3,
                       lam_range=(0.3, 0.7),
                       resolution=None,
                       save_dir="graphon_lib"):
    """
    为ID和OOD数据创建graphon列表

    参数:
    - ID_list: ID数据列表
    - OOD_list: OOD数据列表
    - id_cross_mix_ratio: ID数据交叉混合的比例
    - ood_cross_mix_ratio: OOD数据交叉混合的比例
    - mix_samples_per_pair: 每对图交叉混合生成的样本数
    - lam_range: 混合系数范围，元组形式(min, max)
    - resolution: graphon的分辨率(大小)，如果为None则使用中位数
    - save_dir: 保存graphon的基础目录

    返回:
    - ID_list: ID graphon列表
    - OOD_list: OOD graphon列表
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 创建ID和OOD子目录
    id_save_dir = os.path.join(save_dir, "ID")
    ood_save_dir = os.path.join(save_dir, "OOD")
    os.makedirs(id_save_dir, exist_ok=True)
    os.makedirs(ood_save_dir, exist_ok=True)

    print(f"为{len(ID_list)}个ID数据创建graphon列表...")
    ID_graphon_list = create_graphon_list(
        ID_list,
        cross_mix_ratio=id_cross_mix_ratio,
        mix_samples_per_pair=mix_samples_per_pair,
        lam_range=lam_range,
        resolution=resolution,
        save_dir=id_save_dir,
        type=0
    )

    print(f"为{len(OOD_list)}个OOD数据创建graphon列表...")
    OOD_graphon_list = create_graphon_list(
        OOD_list,
        cross_mix_ratio=ood_cross_mix_ratio,
        mix_samples_per_pair=mix_samples_per_pair,
        lam_range=lam_range,
        resolution=resolution,
        save_dir=ood_save_dir,
        type=1
    )

    return ID_graphon_list, OOD_graphon_list

def create_ID_OOD_datasets(ID_list, OOD_list, dataset_for_search, close_k=1, args=None):
    """
    将ID和OOD的graphon列表转换为dataset形式

    参数:
    - ID_list: ID graphon列表或Data对象列表
    - OOD_list: OOD graphon列表或Data对象列表
    - dataset_for_search: 用于搜索的参考数据集
    - close_k: 邻近搜索的k值
    - args: 其他参数

    返回:
    - ID_dataset: ID数据的dataset
    - OOD_dataset: OOD数据的dataset
    """
    # 导入必要的模块
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from util.graphon_utils import prepare_dataset_x_xs_mean
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse

    # 处理dataset_for_search参数
    if isinstance(dataset_for_search, (list, tuple)) and len(dataset_for_search) >= 1:
        # 如果是tuple或list，使用第一个元素
        dataset_for_search = dataset_for_search[0]

    # 检查输入数据类型并相应处理
    ID_graph_list = []
    for i, item in enumerate(ID_list):
        if isinstance(item, Data):
            # 如果已经是Data对象，直接使用
            ID_graph_list.append(item)
        else:
            # 如果是numpy数组，转换为Data对象
            adj_matrix = torch.tensor(item, dtype=torch.float)
            edge_index, edge_attr = dense_to_sparse(adj_matrix)
            graph_data = Data(edge_index=edge_index, num_nodes=adj_matrix.size(0), y=torch.tensor([0], dtype=torch.long))
            ID_graph_list.append(graph_data)

    OOD_graph_list = []
    for i, item in enumerate(OOD_list):
        if isinstance(item, Data):
            # 如果已经是Data对象，直接使用
            OOD_graph_list.append(item)
        else:
            # 如果是numpy数组，转换为Data对象
            adj_matrix = torch.tensor(item, dtype=torch.float)
            edge_index, edge_attr = dense_to_sparse(adj_matrix)
            graph_data = Data(edge_index=edge_index, num_nodes=adj_matrix.size(0), y=torch.tensor([1], dtype=torch.long))
            OOD_graph_list.append(graph_data)

    # 处理ID列表
    print(f"为{len(ID_list)}个ID graphon创建dataset...")
    ID_dataset = prepare_dataset_x_xs_mean(ID_graph_list, dataset_for_search=dataset_for_search, close_k=close_k, args=args)

    # 处理OOD列表
    print(f"为{len(OOD_list)}个OOD graphon创建dataset...")
    OOD_dataset = prepare_dataset_x_xs_mean(OOD_graph_list, dataset_for_search=dataset_for_search, close_k=close_k, args=args)

    return ID_dataset, OOD_dataset

def create_ID_OOD_dataloaders(ID_list, OOD_list, dataset_for_search, close_k=1, batch_size=128, shuffle=True, args=None):
    """
    将ID和OOD的graphon列表转换为dataloader形式

    参数:
    - ID_list: ID graphon列表
    - OOD_list: OOD graphon列表
    - dataset_for_search: 用于搜索的参考数据集
    - close_k: 邻近搜索的k值
    - batch_size: 批次大小
    - shuffle: 是否打乱数据
    - args: 其他参数

    返回:
    - ID_dataloader: ID数据的dataloader
    - OOD_dataloader: OOD数据的dataloader
    """
    # 获取dataset
    ID_dataset, OOD_dataset = create_ID_OOD_datasets(ID_list, OOD_list, dataset_for_search, close_k, args)

    # 创建dataloader
    ID_dataloader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=shuffle)
    OOD_dataloader = DataLoader(OOD_dataset, batch_size=batch_size, shuffle=shuffle)

    return ID_dataloader, OOD_dataloader

def create_mixup_dataloader(ID_list, dataset_for_search, args, lam_range=(0.3, 0.7), aug_num=10, batch_size=128, shuffle=True, keep_graphon_size=False,lan=None):
    """
    执行ID-ID混合并返回一个dataloader

    参数:
    - ID_list: 内分布数据列表，包含Data类型的图数据
    - dataset_for_search: 用于搜索的参考数据集
    - args: 其他参数
    - lam_range: 混合系数范围，元组形式(min, max)
    - aug_num: 每对graphon混合生成的样本数量
    - batch_size: 批次大小
    - shuffle: 是否打乱数据
    - keep_graphon_size: 是否保持graphon大小

    返回:
    - dataloader_mixup: 混合后的数据加载器
    """

    # 1. 统计ID图的节点数
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, min_num_nodes, max_num_nodes = stat_graph(ID_list)
    resolution = int(median_num_nodes)

    # 2. 创建ID graphon列表
    id_graphon_list = []
    for item in ID_list:
        # 创建单个图的graphon
        graphon = create_graphon([item], resolution)
        id_graphon_list.append(graphon)

    # 3. 生成随机混合系数
    lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

    # 4. 执行ID-ID混合
    new_graph = []
    id_inter_sample_times = (len(id_graphon_list)^2) // 2  # 修正了原代码中的^运算符错误

    # 计算每对需要生成的样本数，确保总样本数与ID_list数量相当
    num_sample = math.ceil(len(ID_list) / (aug_num * max(1, id_inter_sample_times)))
    if lan is None:
        for lam in lam_list:
            # 随机选择不同的graphon对进行混合
            pairs = []
            n = len(id_graphon_list)
            max_pairs = min(n * (n-1) // 2, id_inter_sample_times)  # 限制最大对数

            while len(pairs) < max_pairs:
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j and (i, j) not in pairs and (j, i) not in pairs:
                    pairs.append((i, j))

            for i, j in pairs:
                id_graphon1 = id_graphon_list[i]
                id_graphon2 = id_graphon_list[j]

                if keep_graphon_size:
                    # 保持大小统一
                    max_size = max(id_graphon1.shape[0], id_graphon2.shape[0])
                    adjust_graphon1 = adjust_graphon_size(id_graphon1, max_size)
                    adjust_graphon2 = adjust_graphon_size(id_graphon2, max_size)
                    mixup_graph_new = two_graphon_mixup(adjust_graphon1, adjust_graphon2, la=lam, num_sample=num_sample)
                else:
                    # 随机大小
                    min_size = max(2, min(id_graphon1.shape[0], id_graphon2.shape[0]) // 2)
                    max_size = max(id_graphon1.shape[0], id_graphon2.shape[0])
                    mixup_graph_new = two_graphon_mixup_random_align(id_graphon1, id_graphon2,
                                                                min_size=min_size,
                                                                max_size=max_size,
                                                                la=lam,
                                                                num_sample=num_sample)

                new_graph += mixup_graph_new
    else:
        pairs = []
        n = len(id_graphon_list)
        max_pairs = min(n * (n-1) // 2, id_inter_sample_times)  # 限制最大对数

        while len(pairs) < max_pairs:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i != j and (i, j) not in pairs and (j, i) not in pairs:
                pairs.append((i, j))

        for i, j in pairs:
            id_graphon1 = id_graphon_list[i]
            id_graphon2 = id_graphon_list[j]

            if keep_graphon_size:
                # 保持大小统一
                max_size = max(id_graphon1.shape[0], id_graphon2.shape[0])
                adjust_graphon1 = adjust_graphon_size(id_graphon1, max_size)
                adjust_graphon2 = adjust_graphon_size(id_graphon2, max_size)
                mixup_graph_new = two_graphon_mixup(adjust_graphon1, adjust_graphon2, la=lan, num_sample=num_sample)
            else:
                # 随机大小
                min_size = max(2, min(id_graphon1.shape[0], id_graphon2.shape[0]) // 2)
                max_size = max(id_graphon1.shape[0], id_graphon2.shape[0])
                mixup_graph_new = two_graphon_mixup_random_align(id_graphon1, id_graphon2,
                                                            min_size=min_size,
                                                            max_size=max_size,
                                                            la=lan,
                                                            num_sample=num_sample)

            new_graph += mixup_graph_new

    # 5. 处理混合数据
    mixup_dataset = prepare_dataset_x_xs_mean(
        new_graph,
        dataset_for_search=dataset_for_search,
        close_k=args.close_k if hasattr(args, 'close_k') else 1,
        args=args
    )

    # 6. 创建DataLoader
    dataloader_mixup = DataLoader(mixup_dataset, batch_size=batch_size, shuffle=shuffle)

    #print(f"创建了包含{len(mixup_dataset)}个混合样本的DataLoader")
    return dataloader_mixup

def split_by_percentile(test_data_list, scores, p_percentile, q_percentile):
    """
    根据分数的百分位数将测试数据分为ID和OOD两组

    参数:
    - test_data_list: 测试数据的DataLoader或单个DataBatch对象
    - scores: 分数张量，形状为(样本数量,)
    - p_percentile: ID百分位数，分数低于此百分位的样本被视为ID
    - q_percentile: OOD百分位数，分数高于此百分位的样本被视为OOD

    返回:
    - ID_list: 内分布数据列表
    - OOD_list: 分布外数据列表
    """
    ID_list = []
    OOD_list = []

    # 如果传入的是单个DataBatch，将其包装成列表
    if isinstance(test_data_list, torch_geometric.data.Batch):
        test_data_list = [test_data_list]

    # 确保scores是numpy数组
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # 计算百分位数阈值
    p_threshold = np.percentile(scores, p_percentile)
    q_threshold = np.percentile(scores, q_percentile)
    # print(p_threshold,q_threshold)

    # 初始化标签索引
    label_idx = 0

    for batch in test_data_list:
        num_graphs = batch.num_graphs

        # 遍历每个图
        for graph_idx in range(num_graphs):
            # 获取当前图
            graph_data = extract_graph(batch, graph_idx)

            # 获取当前图的分数
            if label_idx < len(scores):
                score = scores[label_idx]
                label_idx += 1

                # 根据百分位数分类
                if score <= p_threshold:  # 低于p百分位的为ID
                    ID_list.append(graph_data)
                elif score >= q_threshold:  # 高于q百分位的为OOD
                    OOD_list.append(graph_data)
                # 介于p和q百分位之间的样本不分类

    return ID_list,OOD_list

def split_dict(batch, scores, p_percentile, q_percentile):
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    p_thresh = np.percentile(scores, p_percentile)
    q_thresh = np.percentile(scores, q_percentile)
    graph_list = batch.to_data_list()
    ID_list = [graph for graph, s in zip(graph_list, scores) if s <= p_thresh]
    OOD_list = [graph for graph, s in zip(graph_list, scores) if s >= q_thresh]
    return ID_list, OOD_list

'''
# 使用模型直接分类
ID_list, OOD_list = split_by_model(model, test_data_list, device)

# 或者使用外部计算的预测标签
y_pred_labels = model(data).argmax(dim=1).cpu().numpy()
ID_list, OOD_list = split_by_pred(test_data_list, y_pred_labels)

# 或者使用外部计算的异常得分
scores = model.calc_anomaly_score(data).cpu().numpy()
ID_list, OOD_list = split_by_score(test_data_list, scores)

# 或者使用外部计算的余弦相似度
similarities = model.calc_cosine_similarity(data).cpu().numpy()
ID_list, OOD_list = split_by_sim(test_data_list, similarities)

# 创建ID和OOD的DataLoader
ID_dataloader = create_loader(ID_list, batch_size=args.batch_size_test)
OOD_dataloader = create_loader(OOD_list, batch_size=args.batch_size_test)

# 创建ID和OOD的graphon字典
ID_dict, OOD_dict = create_ID_OOD_dicts(
    ID_list,
    OOD_list,
    id_cross_mix_ratio=0.2,
    ood_cross_mix_ratio=0.2,
    mix_samples_per_pair=3,
    lam_range=(0.3, 0.7)
)

# 或者创建ID和OOD的graphon列表
ID_graphon_list, OOD_graphon_list = create_ID_OOD_lists(
    ID_list,
    OOD_list,
    id_cross_mix_ratio=0.2,
    ood_cross_mix_ratio=0.2,
    mix_samples_per_pair=3,
    lam_range=(0.3, 0.7)
)

# 将graphon列表转换为dataset
ID_dataset, OOD_dataset = create_ID_OOD_datasets(
    ID_graphon_list,
    OOD_graphon_list,
    dataset_for_search=dataset_triple[0],
    close_k=args.close_k,
    args=args
)

# 或者直接创建dataloader
ID_dataloader, OOD_dataloader = create_ID_OOD_dataloaders(
    ID_graphon_list,
    OOD_graphon_list,
    dataset_for_search=dataset_triple[0],
    close_k=args.close_k,
    batch_size=args.batch_size,
    shuffle=True,
    args=args
)
'''




from itertools import combinations
def mixup_dataloader(ID_list, dataset_for_search, args, lam_range=(0.3, 0.7), aug_num=10, batch_size=128, shuffle=True, keep_graphon_size=False):
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, min_num_nodes, max_num_nodes = stat_graph(ID_list)
    resolution = int(median_num_nodes)

    import time
    torch.cuda.reset_peak_memory_stats()
    start_time_estimate = time.time()
    start_mem_estimate = torch.cuda.memory_allocated()
    id_graphon_list = []
    for item in ID_list:
        graphon = create_graphon([item], resolution)
        id_graphon_list.append(graphon)

    print("len(graphon_list)=",len(id_graphon_list))
    end_time_estimate = time.time()
    end_mem_estimate = torch.cuda.memory_allocated()
    peak_mem_estimate = torch.cuda.max_memory_allocated()
    print(f"Graphon Estiamte Time: {(end_time_estimate - start_time_estimate)*1000:.2f} ms, GPU Memory: {(end_mem_estimate - start_mem_estimate) / 1024 / 1024:.2f} MB, Peak GPU Memory: {peak_mem_estimate / 1024 / 1024:.2f} MB")

    lam_list = [0.5]
    new_graph = []
    id_inter_sample_times = (len(id_graphon_list)^2) // 2
    num_sample = math.ceil(len(ID_list) / (aug_num * max(1, id_inter_sample_times)))


    torch.cuda.reset_peak_memory_stats()
    start_time_mixup = time.time()
    start_mem_mixup = torch.cuda.memory_allocated()
    for lam in lam_list:
        for id_graphon1, id_graphon2 in combinations(id_graphon_list, 2):
            if keep_graphon_size:
                    # 保持大小统一
                max_size = max(id_graphon1.shape[0], id_graphon2.shape[0])
                adjust_graphon1 = adjust_graphon_size(id_graphon1, max_size)
                adjust_graphon2 = adjust_graphon_size(id_graphon2, max_size)
                mixup_graph_new = two_graphon_mixup(adjust_graphon1, adjust_graphon2, la=lam, num_sample=num_sample)

            else:
                # 随机大小
                min_size = max(2, min(id_graphon1.shape[0], id_graphon2.shape[0]) // 2)
                max_size = max(id_graphon1.shape[0], id_graphon2.shape[0])
                mixup_graph_new = two_graphon_mixup_random_align(id_graphon1, id_graphon2,
                                                                min_size=min_size,
                                                                max_size=max_size,
                                                                la=lam,
                                                                num_sample=num_sample)

            new_graph += mixup_graph_new

    end_time_mixup = time.time()
    end_mem_mixup = torch.cuda.memory_allocated()
    peak_mem_mixup = torch.cuda.max_memory_allocated()
    print(f"Mixup Time: {(end_time_mixup - start_time_mixup)*1000:.2f} ms, GPU Memory: {(end_mem_mixup - start_mem_mixup) / 1024 / 1024:.2f} MB, Peak GPU Memory: {peak_mem_mixup / 1024 / 1024:.2f} MB")

    # 5. 处理混合数据
    mixup_dataset = prepare_dataset_x_xs_mean(
        new_graph,
        dataset_for_search=dataset_for_search,close_k=args.close_k if hasattr(args, 'close_k') else 1,args=args)

    dataloader_mixup = DataLoader(mixup_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader_mixup