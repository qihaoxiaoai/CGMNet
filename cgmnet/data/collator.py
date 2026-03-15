# cgmnet/data/collator.py
import torch
import dgl
import random
from typing import List, Dict
from cgmnet.utils.register import register_collator


def _batch_knodes(knodes_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not knodes_list or not knodes_list[0]:
        return {}
    batched = {}
    for name in knodes_list[0].keys():
        tensors = [k[name] for k in knodes_list]
        batched[name] = torch.stack(tensors) if tensors[0].ndim == 1 else torch.cat(tensors)
    return batched


def _node_id_shift(graph: dgl.DGLGraph, id_list: List[torch.Tensor]) -> torch.Tensor:
    # Calculate the cumulative sum of nodes in each graph of the batch
    # This provides the offset for each graph's node indices
    offsets = torch.cat(
        [torch.tensor([0], device=graph.device), graph.batch_num_nodes().cumsum(dim=0)[:-1]]
    )
    # Add the offset to each list of indices
    return torch.cat([ids + offset for ids, offset in zip(id_list, offsets)])


@register_collator('cgmnet_finetune')
class CGMNetFinetuneCollator:
    def __call__(self, batch: List):
        valid_samples = [item for item in batch if item[0] is not None]
        if not valid_samples:
            return None, None

        # Unpack samples into separate lists
        graphs, labels, knodes, smiles = zip(*valid_samples)

        # Batch the graphs and knowledge nodes
        data = {
            'atom_graph': dgl.batch([g['atom_graph'] for g in graphs]),
            'fragment_graph': dgl.batch([g['fragment_graph'] for g in graphs]),
            'knodes': _batch_knodes(knodes),
            'smiles': smiles
        }

        # Rename keys to match what the model's forward pass expects.
        data['node_ids'] = _node_id_shift(
            data['atom_graph'], [g['node_ids'] for g in graphs]
        )
        data['macro_node_ids'] = _node_id_shift(
            data['fragment_graph'], [g['macro_node_ids'] for g in graphs]
        )

        return data, torch.stack(labels)


@register_collator('cgmnet_pretrain')
class CGMNetPretrainCollator:
    """
    预训练 collator：
      - 复用 finetune collator 做图的 batch（保持行为一致）
      - 随机 mask 一部分 fragment，构造：
          * data['masked_indices']         : 被 mask 的 fragment 的全局节点索引
          * labels                         : 每个被 mask fragment 的“主”id（兼容旧的 MCP 单标签）
          * data['hier_row_idx'] / ['hier_col_idx'] :
                用 final_details 生成的多标签（DOVE 嵌套 fragment）
                供 PretrainTrainer 构建 multi-hot hierarchical label
    """
    def __init__(self, mask_rate: float = 0.3):
        self.mask_rate = mask_rate
        # Instantiate the parent collator to use its logic
        self.base_collator = CGMNetFinetuneCollator()

    def __call__(self, batch: List):
        valid_samples = [item for item in batch if item[0] is not None]
        if not valid_samples:
            return None, None

        # 预训练 dataset: (graph_dict, dummy_label, knodes)
        graphs, labels_dummy, knodes = zip(*valid_samples)

        # 加 dummy smiles 复用 finetune collator
        temp_batch_for_parent = [(g, l, k, "") for (g, l, k) in valid_samples]
        data, _ = self.base_collator(temp_batch_for_parent)

        if data is None:
            return None, None

        frag_graph = data['fragment_graph']
        num_nodes = frag_graph.num_nodes()

        # 没有 fragment，直接返回空
        if num_nodes == 0:
            data['masked_indices'] = torch.tensor([], dtype=torch.long)
            # 保持 label 类型为 long tensor
            labels = torch.tensor([], dtype=torch.long)
            # hierarchical index 也给空的，Trainer 那边会自动跳过
            data['hier_row_idx'] = torch.tensor([], dtype=torch.long)
            data['hier_col_idx'] = torch.tensor([], dtype=torch.long)
            return data, labels

        # 随机选择要 mask 的 fragment（全局节点索引）
        num_mask = int(self.mask_rate * num_nodes)
        if num_mask == 0:
            num_mask = 1
        masked_indices = torch.randperm(num_nodes)[:num_mask]

        # 1) 原来 MCP 的单标签：主 id（graph.ndata['id'] 里保存的那个）
        labels = frag_graph.ndata['id'][masked_indices]

        # 2) 用 final_details 构造 hierarchical 多标签的稀疏 index
        #
        # 思路：
        #   - DGL batch 后 fragment 节点按图顺序拼在一起：
        #         graph0: 0 .. n0-1
        #         graph1: n0 .. n0+n1-1
        #         ...
        #   - 用 batch_num_nodes() 做前缀和得到 offsets
        #   - 对每个 masked_indices[i] 反推出属于哪个 graph、在该 graph 里的 local fragment idx
        #   - 再从对应 graph_dict['final_details'][local_idx] 取出这一 fragment 的所有 nested ids
        #   - 记录到 (hier_row_idx, hier_col_idx) 中：
        #         row = 该 masked fragment 在 masked_indices 中的行号（0..num_mask-1）
        #         col = vocab id
        num_frags_per_graph = frag_graph.batch_num_nodes().tolist()
        offsets = [0]
        for n in num_frags_per_graph[:-1]:
            offsets.append(offsets[-1] + n)
        offsets_t = torch.tensor(offsets, dtype=torch.long)  # 长度 = num_graphs
        if len(offsets_t) > 1:
            boundaries = offsets_t[1:]  # 用作 bucket 边界
            graph_ids = torch.bucketize(masked_indices, boundaries)
        else:
            # 只有一个图的简单情况
            graph_ids = torch.zeros_like(masked_indices, dtype=torch.long)
        local_ids = masked_indices - offsets_t[graph_ids]

        hier_row_idx: List[int] = []
        hier_col_idx: List[int] = []

        # 注意：row_idx = 在 masked_indices 里的位置，对应 logits 的第几行
        for row_idx, (g_idx, local_idx) in enumerate(
            zip(graph_ids.tolist(), local_ids.tolist())
        ):
            # 每个 graph 的 final_details: List[List[int]]，长度 = num_fragments
            details_for_graph = graphs[g_idx].get('final_details', None)
            if details_for_graph is None:
                continue
            if local_idx < 0 or local_idx >= len(details_for_graph):
                continue

            nested_ids = details_for_graph[local_idx]
            # nested_ids 是一组 vocab id（int），可能会有重复，简单去重一下
            # 以免对同一个 (row, col) 重复计数（虽然 BCE 理论上也能收敛）
            seen = set()
            for frag_id in nested_ids:
                if not isinstance(frag_id, int):
                    continue
                if frag_id in seen:
                    continue
                seen.add(frag_id)
                hier_row_idx.append(row_idx)
                hier_col_idx.append(frag_id)

        if len(hier_row_idx) == 0:
            data['hier_row_idx'] = torch.tensor([], dtype=torch.long)
            data['hier_col_idx'] = torch.tensor([], dtype=torch.long)
        else:
            data['hier_row_idx'] = torch.tensor(hier_row_idx, dtype=torch.long)
            data['hier_col_idx'] = torch.tensor(hier_col_idx, dtype=torch.long)

        # 3) 用一个新的 vocab 尾部 id 作为 mask token
        mask_token_id = frag_graph.ndata['id'].max().item() + 1
        frag_graph.ndata['id'][masked_indices] = mask_token_id

        data['masked_indices'] = masked_indices
        return data, labels

