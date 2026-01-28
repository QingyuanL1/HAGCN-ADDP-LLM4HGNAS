import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import move_to
from torch.utils.data import DataLoader
import time
from datetime import timedelta
import random
import json



from utils.functions import parse_softmax_temperature, torch_load_cpu, _load_model_file
from pdp.problem_pdp import PDP
from nets.attention_model import AttentionModel
from torch.nn import DataParallel

mp = torch.multiprocessing.get_context('spawn')


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def load_architectures(model_path, arch_source='best', top_k=1):
    """
    从 nas_history.json 加载架构

    Args:
        model_path: 模型目录路径
        arch_source: 'best' 使用最佳架构, 'history' 从历史中选择, 'none' 使用默认
        top_k: 返回 Top-K 个架构（按 val_cost 排序）

    Returns:
        list of dict: 架构列表，每个包含 arch, val_cost, source 等信息
    """
    nas_history_path = os.path.join(model_path, 'nas_history.json')

    # 向后兼容：文件不存在返回 None
    if not os.path.exists(nas_history_path):
        print(f"警告: 未找到 nas_history.json 文件: {nas_history_path}")
        print("使用默认架构（全注意力）")
        return [{'arch': None, 'val_cost': None, 'source': 'default'}]

    try:
        with open(nas_history_path, 'r', encoding='utf-8') as f:
            nas_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: nas_history.json 格式无效: {e}")
        return [{'arch': None, 'val_cost': None, 'source': 'default'}]

    architectures = []

    if arch_source == 'best' and 'best' in nas_data:
        # 使用最佳架构
        best = nas_data['best']
        architectures.append({
            'arch': best['arch'],
            'val_cost': best.get('val_cost'),
            'epoch': best.get('epoch'),
            'source': 'best'
        })
    elif 'history' in nas_data:
        # 从历史中选择 Top-K
        history = nas_data['history']
        # 按 val_cost 排序（升序，成本越低越好）
        sorted_history = sorted(
            [h for h in history if 'val_cost' in h and h['val_cost'] is not None],
            key=lambda x: x['val_cost']
        )
        for i, entry in enumerate(sorted_history[:top_k]):
            architectures.append({
                'arch': entry['arch'],
                'val_cost': entry.get('val_cost'),
                'epoch': entry.get('epoch'),
                'source': f'history_rank_{i+1}'
            })

    # 如果没有找到有效架构，使用默认
    if not architectures:
        print("警告: 未找到有效架构，使用默认架构")
        return [{'arch': None, 'val_cost': None, 'source': 'default'}]

    return architectures


def validate_architecture(arch, n_layers=3, n_relations=7):
    """
    验证架构格式是否正确

    Args:
        arch: 架构（可以是旧格式列表或新格式字典）
        n_layers: 期望的层数
        n_relations: 每层的关系数

    Returns:
        bool: 架构是否有效
    """
    if arch is None:
        return True  # None 表示使用默认架构

    # 新格式：{'layers': [...], 'aggregation': [...]}
    if isinstance(arch, dict) and 'layers' in arch:
        layers = arch['layers']
        if not isinstance(layers, (list, tuple)) or len(layers) != n_layers:
            print(f"错误: 期望 {n_layers} 层，但得到 {len(layers) if isinstance(layers, (list, tuple)) else 'invalid'}")
            return False

        for i, layer in enumerate(layers):
            if not isinstance(layer, (list, tuple)) or len(layer) != n_relations:
                print(f"错误: 第 {i} 层有 {len(layer) if isinstance(layer, (list, tuple)) else 'invalid'} 个操作，期望 {n_relations}")
                return False
            for op in layer:
                if not (0 <= int(op) <= 4):
                    print(f"错误: 第 {i} 层中的操作码 {op} 无效")
                    return False

        if 'aggregation' in arch:
            agg = arch['aggregation']
            if len(agg) != n_layers:
                print(f"错误: 期望 {n_layers} 个聚合方式，但得到 {len(agg)}")
                return False
            valid_aggs = {'sum', 'mean', 'max'}
            for a in agg:
                if a not in valid_aggs:
                    print(f"错误: 无效的聚合方式 '{a}'")
                    return False

        return True

    # 旧格式：[[...], [...], ...]
    elif isinstance(arch, (list, tuple)):
        if len(arch) != n_layers:
            print(f"错误: 期望 {n_layers} 层，但得到 {len(arch)}")
            return False

        for i, layer in enumerate(arch):
            if not isinstance(layer, (list, tuple)) or len(layer) != n_relations:
                print(f"错误: 第 {i} 层有 {len(layer) if isinstance(layer, (list, tuple)) else 'invalid'} 个操作，期望 {n_relations}")
                return False
            for op in layer:
                if not (0 <= int(op) <= 4):
                    print(f"错误: 第 {i} 层中的操作码 {op} 无效")
                    return False

        return True

    print(f"错误: 架构必须是字典或列表，但得到 {type(arch)}")
    return False


def load_model(model, device, arch=None):
    model_name = model
    problem = PDP()
    model = AttentionModel(
        device,
        embedding_dim=128,
        hidden_dim=128,
        problem=problem,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization='batch',
        tanh_clipping=8,
        checkpoint_encoder=False,
        shrink_size=None,
        arch=arch  # 传入架构参数
    )

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_name)

    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_name, model)

    # 确保架构被设置
    if arch is not None:
        model.set_arch(arch)

    model.eval()  # Put in eval mode

    return model, None


def get_best(sequences, cost, ids=None, batch_size=None):

    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes, arch) = args

    model, _ = load_model(opts.model, opts.device, arch=arch)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(num_samples=val_size,filename=dataset_path, offset=opts.offset + val_size * i)

    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts, arch=None):
    # load the model
    model, _ = load_model(opts.model, opts.device, arch=arch)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda

    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0


        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes, arch) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    return costs, tours, durations


def eval_dataset_multi_arch(dataset_path, width, softmax_temp, opts, architectures=None):
    """
    评估数据集，支持多架构批量评估

    Args:
        dataset_path: 数据集路径
        width: beam width
        softmax_temp: softmax temperature
        opts: 命令行参数
        architectures: 架构列表

    Returns:
        dict: 每个架构的评估结果
    """
    if architectures is None:
        architectures = [{'arch': None, 'source': 'default'}]

    results = {}

    for arch_info in architectures:
        arch = arch_info['arch']
        source = arch_info.get('source', 'unknown')

        print(f"\n{'='*60}")
        print(f"评估架构: {source}")
        if arch:
            if isinstance(arch, dict) and 'layers' in arch:
                print(f"  Layers: {arch['layers']}")
                print(f"  Aggregation: {arch.get('aggregation', 'default')}")
            else:
                print(f"  Layers: {arch}")
        else:
            print("  使用默认架构（全注意力）")
        print(f"{'='*60}")

        # 验证架构
        if not validate_architecture(arch):
            print(f"跳过无效架构: {source}")
            results[source] = {'error': 'Invalid architecture'}
            continue

        # 执行评估
        start_time = time.time()
        costs, tours, durations = eval_dataset(dataset_path, width, softmax_temp, opts, arch=arch)
        eval_time = time.time() - start_time

        avg_cost = np.mean(costs)
        std_cost = np.std(costs)

        results[source] = {
            'arch': arch,
            'avg_cost': float(avg_cost),
            'std_cost': float(std_cost),
            'min_cost': float(np.min(costs)),
            'max_cost': float(np.max(costs)),
            'eval_time': eval_time,
            'train_val_cost': arch_info.get('val_cost')
        }

        print(f"\n{source} 的结果:")
        print(f"  平均成本: {avg_cost:.4f}")
        print(f"  标准差: {std_cost:.4f}")
        print(f"  评估时间: {eval_time:.2f}s")

    return results


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # shape: (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            elif model.problem.NAME == "pdp":
                seq = np.trim_zeros(seq).tolist() + [0]
            else:
                seq = None
                # assert False, "Unkown problem: {}".format(model.problem.NAME)

            results.append((cost, seq, duration))

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=None, nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=500,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), ' 
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true', default=False,
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--seed', type=int, default=8888, help='Random seed to use')

    # NAS 架构评估相关参数
    parser.add_argument('--arch_source', type=str, default='best',
                        choices=['best', 'history', 'none'],
                        help="架构来源: 'best' 使用最佳架构, 'history' 从历史中选择 top-k, 'none' 使用默认架构")
    parser.add_argument('--top_k', type=int, default=1,
                        help="评估前 K 个架构 (配合 --arch_source history 使用)")
    parser.add_argument('--save_eval_results', action='store_true',
                        help="将评估结果保存到 JSON 文件")

    opts = parser.parse_args()


    torch.manual_seed(opts.seed)
    random.seed(opts.seed)

    widths = opts.width if opts.width is not None else [0]

    for width in widths:
        # for dataset_path in opts.datasets:
        opts.use_cuda = torch.cuda.is_available()

        opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

        dataset_path = opts.datasets

        # 加载架构
        model_dir = os.path.dirname(opts.model)
        if opts.arch_source == 'none':
            architectures = [{'arch': None, 'source': 'default'}]
        else:
            architectures = load_architectures(
                model_dir,
                arch_source=opts.arch_source,
                top_k=opts.top_k
            )

        print(f"\n加载了 {len(architectures)} 个架构进行评估")

        # 执行多架构评估
        results = eval_dataset_multi_arch(
            dataset_path=dataset_path,
            width=width,
            softmax_temp=opts.softmax_temperature,
            opts=opts,
            architectures=architectures
        )

        # 输出汇总结果
        print("\n" + "="*60)
        print("评估结果汇总")
        print("="*60)

        sorted_results = sorted(
            [(k, v) for k, v in results.items() if 'error' not in v],
            key=lambda x: x[1]['avg_cost']
        )

        for rank, (source, result) in enumerate(sorted_results, 1):
            print(f"\n排名 {rank}: {source}")
            print(f"  平均成本: {result['avg_cost']:.4f}")
            print(f"  标准差: {result['std_cost']:.4f}")
            if result.get('train_val_cost'):
                print(f"  训练验证成本: {result['train_val_cost']:.4f}")

        # 保存结果
        if opts.save_eval_results:
            output_path = os.path.join(model_dir, 'eval_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model': opts.model,
                    'dataset': opts.datasets,
                    'width': width,
                    'results': results,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到 {output_path}")




