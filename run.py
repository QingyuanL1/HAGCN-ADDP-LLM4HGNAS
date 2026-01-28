#!/usr/bin/env python

import os
import json
import pprint as pp
import random
import re
import time
import urllib.error
import urllib.request

import torch
import torch.optim as optim

import sys
sys.path.append("..")

from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline,  RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu
from pdp.problem_pdp import PDP



def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        try:
            from tensorboard_logger import Logger as TbLogger
        except Exception as e:
            print("  [!] Failed to import tensorboard_logger, disabling tensorboard (error: {})".format(e))
            tb_logger = None
        else:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))


    # Save arguments so exact configuration can always be found
    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # APDP problem
    problem = PDP()


    def validate_arch(arch, n_layers):
        if not isinstance(arch, dict):
             # Plan B: Backward compatibility or strict check?
             # Let's enforce new dict format
             return False
        
        layers = arch.get('layers')
        aggr = arch.get('aggregation')
        
        if not isinstance(layers, (list, tuple)) or len(layers) != n_layers:
            return False
        
        if not isinstance(aggr, (list, tuple)) or len(aggr) != n_layers:
            return False

        for layer in layers:
            if not isinstance(layer, (list, tuple)) or len(layer) != 7:
                return False
            for v in layer:
                if int(v) not in (0, 1, 2, 3, 4):
                    return False
        
        for a in aggr:
            if a not in ('sum', 'mean', 'max'):
                return False

        return True


    def normalize_arch(arch):
        # arch is expected to be a dict now
        if not isinstance(arch, dict):
             return arch # Should fail validation
        
        layers = [[int(v) for v in layer] for layer in arch['layers']]
        aggr = [str(a) for a in arch['aggregation']]
        return {'layers': layers, 'aggregation': aggr}


    def arch_signature(arch):
        return json.dumps(arch, separators=(',', ':'), sort_keys=False)


    def random_arch(n_layers):
        # Sample 0-4.
        # distribution: 0 (30%), 1-4 (random)
        layers = []
        aggr = []
        for _ in range(n_layers):
            layer = []
            for _ in range(7):
                if random.random() < 0.3:
                    layer.append(0)
                else:
                    layer.append(random.randint(1, 4))
            layers.append(layer)
            
            # Sample aggregation
            aggr.append(random.choice(['sum', 'mean', 'max']))
            
        return ensure_arch_safe({'layers': layers, 'aggregation': aggr})


    def ensure_arch_safe(arch):
        arch = normalize_arch(arch)
        safe_layers = []
        for layer in arch['layers']:
            layer = [int(v) for v in layer]
            # Always ensure global relation is active (or at least some relation)
            if layer[0] == 0:
                 layer[0] = 1 # Default to Attention
            if sum(layer) == 0:
                 layer[0] = 1
            safe_layers.append(layer)
        return {'layers': safe_layers, 'aggregation': arch['aggregation']}


    def extract_first_json_array(text):
        if not isinstance(text, str):
            return None
        # Try finding JSON object first
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m is None:
             # Fallback to finding array if object not found (though prompt asks for object now, legacy might return array)
             # But we strictly want object.
             return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


    def llm_arch(n_layers, tried_signatures, tried_history_records):
        api_key = opts.llm_api_key or os.environ.get('OPENAI_API_KEY')
        if api_key is None:
            raise RuntimeError('OPENAI_API_KEY is not set. Please set it via env var or pass --llm_api_key')

        base_url = getattr(opts, 'llm_base_url', 'https://api.openai.com/v1')
        base_url = str(base_url).rstrip('/')

        # Prepare performance feedback
        # valid records with cost
        valid_records = [r for r in tried_history_records if r.get('cost') is not None]
        # Sort by cost ascending (lower is better for APDP)
        valid_records.sort(key=lambda x: x['cost'])
        
        top_k = 5
        best_performers = valid_records[:top_k]
        
        perf_text = ""
        if len(best_performers) > 0:
            perf_text += "\n\nTop {} Performing Architectures (Lower Cost is Better):\n".format(len(best_performers))
            for i, r in enumerate(best_performers):
                 perf_text += "{}. Cost={:.4f}, Arch={}\n".format(i+1, r['cost'], r['sig'])
        
        # Also avoid repeating recent ones (already in tried_signatures set logic, but can list them for context)
        # We can just list the prompt requirement.

        prompt = (
            "Task Description:\n"
            "Your task is to design the optimal Encoder architecture for the Asymmetric Pickup and Delivery Problem (APDP).\n"
            "The goal is to minimize the total travel distance of the vehicle.\n\n"
            
            "Dataset Structure:\n"
            "Nodes: Pickup Nodes (P), Delivery Nodes (D), Depot.\n"
            "Relations (Message Passing Paths):\n"
            "1. Global: All-to-all attention among all nodes (Depot, P, D).\n"
            "2. P-D_pair: Interaction from a pickup node to its specific paired delivery node.\n"
            "3. P-P: Interactions among all pickup nodes.\n"
            "4. P-D_all: Interaction from a pickup node to all delivery nodes.\n"
            "5. D-P_pair: Interaction from a delivery node to its specific paired pickup node.\n"
            "6. D-D: Interactions among all delivery nodes.\n"
            "7. D-P_all: Interaction from a delivery node to all pickup nodes.\n\n"

            "Search Space:\n"
            "For each relation type in each layer, select one operation from:\n"
            "[0: Zero (Mask out), 1: Attention (Standard), 2: GCN (Isotropic), 3: GAT (Additive), 4: MLP (Linear)].\n"
            "Also select an aggregation function for each layer from: ['sum', 'mean', 'max'].\n\n"

            "Constraint:\n"
            "You must output ONLY one valid JSON object with detailed structure:\n"
            "{{\n"
            "  \"layers\": [ [row_1_7_ints], [row_2], ... ],\n"
            "  \"aggregation\": [ \"agg_layer_1\", \"agg_layer_2\", ... ]\n"
            "}}\n"
            "n_layers = {}\n"
            "The 7 positions per layer correspond to relations [1..7] above.\n"
            "The aggregation array must have length n_layers.\n\n"
            
            "Performance History (Reference):\n"
            "{}"
            "\n\nInstructions:\n"
            "1. Analyze the history. If previous architectures converged poorly, try DIFFERENT operators OR DIFFERENT AGGREGATION methods (e.g. max or mean).\n"
            "2. AGGRESSIVELY EXPLORE combinations with GCN(2), GAT(3), and MLP(4).\n"
            "3. Do not simply repeat the best history; iterate on it to find a better structure.\n"
            "4. Return ONLY the JSON object."
        ).format(n_layers, perf_text)

        payload = {
            'model': opts.llm_model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You output strictly valid JSON only, no extra text.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ], # Message end
            'temperature': opts.llm_temperature
        }

        req = urllib.request.Request(
            '{}/chat/completions'.format(base_url),
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Authorization': 'Bearer {}'.format(api_key),
                'Content-Type': 'application/json'
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=opts.llm_timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        content = data['choices'][0]['message']['content']
        arch = extract_first_json_array(content)
        if arch is None:
            raise RuntimeError('Failed to parse JSON arch from LLM response')

        if not validate_arch(arch, n_layers):
            raise RuntimeError('LLM arch has invalid shape or values')

        arch = ensure_arch_safe(arch)
        sig = arch_signature(arch)
        if sig in tried_signatures:
            raise RuntimeError('LLM produced a repeated architecture')
        return arch


    tried_arch_signatures = set()
    # History of (signature, val_cost) tuples
    # If val_cost is None, it means it was tried but maybe failed or just initialized
    tried_arch_history_records = [] 

    def sample_arch(n_layers):
        def sample_random_unique():
            for _ in range(200):
                arch = random_arch(n_layers)
                sig = arch_signature(arch)
                if sig not in tried_arch_signatures:
                    tried_arch_signatures.add(sig)
                    tried_arch_history_records.append({'sig': sig, 'cost': None, 'arch': arch})
                    return arch

            arch = random_arch(n_layers)
            sig = arch_signature(arch)
            tried_arch_signatures.add(sig)
            tried_arch_history_records.append({'sig': sig, 'cost': None, 'arch': arch})
            return arch

        if getattr(opts, 'nas_arch_generator', 'llm') == 'random':
            arch = sample_random_unique()
            return arch, 'random'

        for _ in range(3):
            try:
                # Pass full history records
                arch = llm_arch(n_layers, tried_arch_signatures, tried_arch_history_records)
                sig = arch_signature(arch)
                tried_arch_signatures.add(sig)
                tried_arch_history_records.append({'sig': sig, 'cost': None, 'arch': arch})
                return arch, 'llm'
            except Exception as e:
                print("  [!] LLM arch generation failed, retrying (error: {})".format(e))
                time.sleep(1)

        arch = sample_random_unique()
        return arch, 'random_fallback' 

    def load_trained_archs_from_outputs():
        base = os.path.join(opts.output_dir, "{}_{}".format(opts.problem, opts.graph_size))
        if not os.path.isdir(base):
            return

        loaded = 0
        for root, dirs, files in os.walk(base):
            if 'nas_history.json' not in files:
                continue
            path = os.path.join(root, 'nas_history.json')
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            history = data.get('history', []) if isinstance(data, dict) else []
            for rec in history:
                if not isinstance(rec, dict) or 'arch' not in rec:
                    continue
                try:
                    arch = ensure_arch_safe(rec['arch'])
                except Exception:
                    continue
                sig = arch_signature(arch)
                # We want to keep all valid records
                cost = rec.get('val_cost', None)
                
                # Check duplication in set for uniqueness, but we might want multiple records if different costs?
                # For Plan B, we treat same arch as same signature. 
                # We will just load them.
                if sig not in tried_arch_signatures:
                    tried_arch_signatures.add(sig)
                    tried_arch_history_records.append({'sig': sig, 'cost': cost, 'arch': arch})
                    loaded += 1

        if loaded > 0:
            print("  [*] Loaded {} previously-trained architectures from {}".format(loaded, base))


    load_trained_archs_from_outputs()


    def build_model(arch=None):
        model_class = {
            'attention': AttentionModel,
        }.get(opts.model, None)

        assert model_class is not None, "Unknown model: {}".format(model_class)

        model = model_class(
            opts.device,
            opts.embedding_dim,
            opts.hidden_dim,
            problem,
            n_encode_layers=opts.n_encode_layers,
            arch=arch,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size,
        ).to(opts.device)

        if opts.use_cuda and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        return model

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume   # None
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    ######################## Initialize model ###########################################
    model = build_model(arch=None)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)

    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    def build_baseline(model_):
        if opts.baseline == 'exponential':
            baseline_ = ExponentialBaseline(opts.exp_beta)
        elif opts.baseline == 'rollout':
            baseline_ = RolloutBaseline(model_, problem, opts)
        else:
            assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
            baseline_ = NoBaseline()

        if opts.bl_warmup_epochs > 0:
            baseline_ = WarmupBaseline(baseline_, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
        return baseline_

    baseline = build_baseline(model)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    def build_optimizer(model_, baseline_):
        return optim.Adam(
            [{'params': model_.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline_.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline_.get_learnable_parameters()) > 0
                else []
            )
        )

    optimizer = build_optimizer(model, baseline)

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    def build_lr_scheduler(optimizer_):
        return optim.lr_scheduler.LambdaLR(optimizer_, lambda epoch: opts.lr_decay ** epoch)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch
    lr_scheduler = build_lr_scheduler(optimizer)


    ################################ Start training ######################################
    # Generate the data required for training
    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        if opts.nas_enabled:
            nas_history = []
            best = {
                'arch': None,
                'val_cost': None,
                'epoch': None
            }

        current_arch = None
        current_arch_source = None
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):

            if opts.nas_enabled and ((epoch - opts.epoch_start) % opts.arch_switch_interval == 0):
                current_arch, current_arch_source = sample_arch(opts.n_encode_layers)

                model = build_model(arch=current_arch)
                model_ = get_inner_model(model)
                model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

                baseline = build_baseline(model)
                optimizer = build_optimizer(model, baseline)
                lr_scheduler = build_lr_scheduler(optimizer)

            avg_cost = train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )

            if opts.nas_enabled:
                avg_cost_val = float(avg_cost.item()) if torch.is_tensor(avg_cost) else float(avg_cost)
                
                # Update in-memory history for LLM feedback
                if len(tried_arch_history_records) > 0:
                     tried_arch_history_records[-1]['cost'] = avg_cost_val
                
                record = {
                    'epoch': int(epoch),
                    'arch': current_arch,
                    'arch_source': current_arch_source,
                    'val_cost': avg_cost_val
                }
                nas_history.append(record)
                if best['val_cost'] is None or avg_cost_val < best['val_cost']:
                    best = {
                        'arch': current_arch,
                        'val_cost': avg_cost_val,
                        'epoch': int(epoch)
                    }

                with open(os.path.join(opts.save_dir, 'nas_history.json'), 'w') as f:
                    json.dump({'history': nas_history, 'best': best}, f, indent=2)


if __name__ == "__main__":
    run(get_options())
