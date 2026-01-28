import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        out = input + self.module(input)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        self.arch_mask = None
        self.agg_type = 'sum'

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)


        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # pickup
        self.W1_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W2_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W3_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))


        # delivery
        self.W4_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W5_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.W6_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        # GAT parameters
        self.a_gat = nn.Parameter(torch.Tensor(1, n_heads, 1, 2 * val_dim)) # Concatenated size
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.init_parameters()


    def set_arch_mask(self, arch_mask, agg_type='sum'):
        self.arch_mask = arch_mask
        self.agg_type = agg_type


    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    """
    q: queries (batch_size, n_query, input_dim)
    h: data (batch_size, graph_size, input_dim)  
    mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
    Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency) 
    """
    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)  # 21
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)  # [batch_size * graph_size, embed_dim]
        qflat = q.contiguous().view(-1, input_dim)  # [batch_size * n_query, embed_dim]

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # pickup -> its delivery attention
        n_pick = (graph_size - 1) // 2
        shp_delivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_pick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all pickups attention
        shp_allpick = (self.n_heads, batch_size, n_pick, -1)
        shp_q_allpick = (self.n_heads, batch_size, n_pick, -1)

        # pickup -> all delivery attention
        shp_alldelivery = (self.n_heads, batch_size, n_pick, -1)
        shp_q_alldelivery = (self.n_heads, batch_size, n_pick, -1)


        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)


        # pickup -> its delivery
        pick_flat = h[:, 1:n_pick + 1, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]
        delivery_flat = h[:, n_pick + 1:, :].contiguous().view(-1, input_dim)  # [batch_size * n_pick, embed_dim]


        # pickup -> its delivery attention W1
        Q_pick = torch.matmul(pick_flat, self.W1_query).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, key_size)
        K_delivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_delivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_delivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> all pickups attention   W2
        Q_pick_allpick = torch.matmul(pick_flat, self.W2_query).view(
            shp_q_allpick)  # (self.n_heads, batch_size, n_pick, -1)
        K_allpick = torch.matmul(pick_flat, self.W_key).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]
        V_allpick = torch.matmul(pick_flat, self.W_val).view(
            shp_allpick)  # [self.n_heads, batch_size, n_pick, key_size]

        # pickup -> all delivery attention    W3
        Q_pick_alldelivery = torch.matmul(pick_flat, self.W3_query).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_alldelivery = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_alldelivery = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # pickup -> its delivery
        V_additional_delivery = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            V_delivery,  # [n_heads, batch_size, n_pick, key/val_size]
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype,
                        device=V.device)
        ], 2)

        # delivery -> its pickup attention  W4
        Q_delivery = torch.matmul(delivery_flat, self.W4_query).view(
            shp_delivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_pick = torch.matmul(pick_flat, self.W_key).view(shp_q_pick)  # (self.n_heads, batch_size, n_pick, -1)
        V_pick = torch.matmul(pick_flat, self.W_val).view(shp_q_pick)  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> all delivery attention   W5
        Q_delivery_alldelivery = torch.matmul(delivery_flat, self.W5_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        K_alldelivery2 = torch.matmul(delivery_flat, self.W_key).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]
        V_alldelivery2 = torch.matmul(delivery_flat, self.W_val).view(
            shp_alldelivery)  # [self.n_heads, batch_size, n_pick, key_size]

        # delivery -> all pickup   W6
        Q_delivery_allpickup = torch.matmul(delivery_flat, self.W6_query).view(
            shp_alldelivery)  # (self.n_heads, batch_size, n_pick, key_size)
        K_allpickup2 = torch.matmul(pick_flat, self.W_key).view(
            shp_q_alldelivery)  # (self.n_heads, batch_size, n_pick, -1)
        V_allpickup2 = torch.matmul(pick_flat, self.W_val).view(
            shp_q_alldelivery)  # (n_heads, batch_size, n_pick, key/val_size)

        # delivery -> its pick up
        V_additional_pick = torch.cat([  # [n_heads, batch_size, graph_size, key_size]
            torch.zeros(self.n_heads, batch_size, 1, self.input_dim // self.n_heads, dtype=V.dtype, device=V.device),
            torch.zeros(self.n_heads, batch_size, n_pick, self.input_dim // self.n_heads, dtype=V.dtype,
                        device=V.device),
            V_pick  # [n_heads, batch_size, n_pick, key/val_size]
        ], 2)


        # Helper to get op index
        ops = [1] * 7 # Default to Attention
        if self.arch_mask is not None:
            arch_mask = self.arch_mask
            if torch.is_tensor(arch_mask):
                arch_mask = arch_mask.detach().cpu().tolist()
            if isinstance(arch_mask, (list, tuple)) and len(arch_mask) == 7:
                 ops = [int(x) for x in arch_mask]

        # Precompute GAT components if needed
        # V shape: (n_heads, batch, graph_size, val_dim)
        # a_gat shape: (1, n_heads, 1, 2*val_dim)
        use_gat = 3 in ops
        if use_gat:
             # Split a_gat into a_l and a_r
             a_l = self.a_gat[..., :self.val_dim]   # (1, n_heads, 1, val_dim)
             a_r = self.a_gat[..., self.val_dim:]   # (1, n_heads, 1, val_dim)
             # E_l = (V * a_l).sum(-1) -> (n_heads, batch, graph_size)
             # Use matmul for efficiency?
             # a_l: (n_heads, val_dim). V: (n_heads, batch, graph_size, val_dim)
             # V * a_l.transpose?
             # Let's reshape a_l to (n_heads, 1, val_dim, 1) to broadcast over batch size
             a_l_t = a_l.view(self.n_heads, 1, self.val_dim, 1)
             a_r_t = a_r.view(self.n_heads, 1, self.val_dim, 1)
             
             # Calculate scores for all nodes
             # V: (n_heads, batch, graph, val)
             E_l = torch.matmul(V, a_l_t).squeeze(-1) # (n_heads, batch, graph)
             E_r = torch.matmul(V, a_r_t).squeeze(-1) # (n_heads, batch, graph)

        def compute_block(op, q_sub, k_sub, v_sub_q=None, v_sub_k=None, gat_el=None, gat_er=None):
            # q_sub: query vectors (for Attn)
            # k_sub: key vectors (for Attn)
            # v_sub_q, v_sub_k: V vectors (for GAT)
            # gat_el, gat_er: precomputed GAT scores (for GAT)
            
            # Shape check
            # q_sub: (n_heads, batch, n_q, dim)
            # k_sub: (n_heads, batch, n_k, dim)
            
            if op == 0: # Zero
                sh = list(q_sub.size())
                sh[-1] = k_sub.size(2) # n_k
                return -np.inf * torch.ones(sh, dtype=q_sub.dtype, device=q_sub.device)
            
            elif op == 1: # Attention
                return self.norm_factor * torch.matmul(q_sub, k_sub.transpose(2, 3))
            
            elif op == 2 or op == 4: # GCN / MLP
                # Isotropic: return 0.
                sh = list(q_sub.size())
                sh[-1] = k_sub.size(2)
                return torch.zeros(sh, dtype=q_sub.dtype, device=q_sub.device)
            
            elif op == 3: # GAT
                # Score = LeakyReLU(E_l + E_r)
                # gat_el: (n_heads, batch, n_q)
                # gat_er: (n_heads, batch, n_k)
                score = gat_el.unsqueeze(-1) + gat_er.unsqueeze(-2) # (n_heads, batch, n_q, n_k)
                return self.leaky_relu(score)
            
            return -np.inf # Default fallback

        # 0. Global Compat
        # Q: (n_heads, batch, n_query, dim)
        # K: (n_heads, batch, graph, dim)
        # E_l (if GAT): (n_heads, batch, graph) -> need slicing for n_query? n_query is usually equal to graph
        # Wait, q is usually h, so n_query = graph_size.
        comp_0 = compute_block(ops[0], Q, K, V, V, E_l if use_gat else None, E_r if use_gat else None)
        
        ## Pick up
        # 1. P->D_pair
        # Q_pick: (heads, batch, n_pick, dim)
        # K_delivery: (heads, batch, n_pick, dim) (Wait, K is reshaped?)
        # Original: sum(Q * K, -1). This implies element-wise corresponding P_i and D_i.
        # This is strictly diagonal! 
        # If GAT, E_l_pick + E_r_delivery.
        # If GCN, 0.
        if ops[1] == 0:
             comp_1 = -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=comp_0.dtype, device=comp_0.device)
        elif ops[1] == 1:
             comp_1 = self.norm_factor * torch.sum(Q_pick * K_delivery, -1)
        elif ops[1] == 2 or ops[1] == 4:
             comp_1 = torch.zeros(self.n_heads, batch_size, n_pick, dtype=comp_0.dtype, device=comp_0.device)
        elif ops[1] == 3:
             # E_l_pick (n_heads, batch, n_pick)
             # E_r_delivery (n_heads, batch, n_pick)
             # Element-wise add
             el_p = E_l[:, :, 1:n_pick+1]
             er_d = E_r[:, :, n_pick+1:]
             comp_1 = self.leaky_relu(el_p + er_d)

        # 2. P->P (All Pick)
        # Q_pick_allpick: (heads, batch, n_pick, dim)
        # K_allpick: (heads, batch, n_pick, dim)
        # Original: matmul.
        el_p = E_l[:, :, 1:n_pick+1] if use_gat else None
        er_p = E_r[:, :, 1:n_pick+1] if use_gat else None
        comp_2 = compute_block(ops[2], Q_pick_allpick, K_allpick, None, None, el_p, er_p)

        # 3. P->D_all
        el_p = E_l[:, :, 1:n_pick+1] if use_gat else None
        er_d = E_r[:, :, n_pick+1:] if use_gat else None
        comp_3 = compute_block(ops[3], Q_pick_alldelivery, K_alldelivery, None, None, el_p, er_d)

        ## Delivery
        # 4. D->P_pair
        if ops[4] == 0:
             comp_4 = -np.inf * torch.ones(self.n_heads, batch_size, n_pick, dtype=comp_0.dtype, device=comp_0.device)
        elif ops[4] == 1:
             comp_4 = self.norm_factor * torch.sum(Q_delivery * K_pick, -1)
        elif ops[4] == 2 or ops[4] == 4:
             comp_4 = torch.zeros(self.n_heads, batch_size, n_pick, dtype=comp_0.dtype, device=comp_0.device)
        elif ops[4] == 3:
             el_d = E_l[:, :, n_pick+1:]
             er_p = E_r[:, :, 1:n_pick+1]
             comp_4 = self.leaky_relu(el_d + er_p)
        
        # 5. D->D
        # K_alldelivery2
        # Q_delivery_alldelivery
        el_d = E_l[:, :, n_pick+1:] if use_gat else None
        er_d = E_r[:, :, n_pick+1:] if use_gat else None
        comp_5 = compute_block(ops[5], Q_delivery_alldelivery, K_alldelivery2, None, None, el_d, er_d)
        
        # 6. D->P_all
        el_d = E_l[:, :, n_pick+1:] if use_gat else None
        er_p = E_r[:, :, 1:n_pick+1] if use_gat else None
        comp_6 = compute_block(ops[6], Q_delivery_allpickup, K_allpickup2, None, None, el_d, er_p)


        # Construct final compatibility matrix
        # Need to reconstruct the structure:
        # [Global, P->D_pair, P->P, P->D_all, D->P_pair, D->D, D->P_all]
        # Wait, the structure is:
        # compatibility (Global)
        # compatibility_additional_delivery (P row, D col scalar?) -> Wait
        # The structure of tensor concatenation was:
        # P row:
        #   Col Global (from comp_0)
        #   Col Pair D (from comp_1) -> Size (Batch, 1) per row? No, Size 1 (scalar)
        #   Col All P (from comp_2)
        #   Col All D (from comp_3)
        
        # Let's recreate the blocks:
        
        # P->D_pair block (column)
        compatibility_additional_delivery = torch.cat([ 
            -np.inf * torch.ones(self.n_heads, batch_size, 1, 1, dtype=comp_0.dtype, device=comp_0.device),
            comp_1.unsqueeze(-1),  # [n_heads, batch, n_pick, 1]
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, 1, dtype=comp_0.dtype, device=comp_0.device)
        ], 2)
        
        # P->all P block
        compatibility_additional_allpick = torch.cat([ 
             -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=comp_0.dtype, device=comp_0.device),
             comp_2, 
             -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=comp_0.dtype, device=comp_0.device)
        ], 2)

        # P->all D block
        compatibility_additional_alldelivery = torch.cat([ 
             -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=comp_0.dtype, device=comp_0.device),
             comp_3,
             -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=comp_0.dtype, device=comp_0.device)
        ], 2)

        # D->P_pair block
        compatibility_additional_pick = torch.cat([ 
            -np.inf * torch.ones(self.n_heads, batch_size, 1, 1, dtype=comp_0.dtype, device=comp_0.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, 1, dtype=comp_0.dtype, device=comp_0.device),
            comp_4.unsqueeze(-1)
        ], 2)

        # D->all D block
        compatibility_additional_alldelivery2 = torch.cat([
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=comp_0.dtype, device=comp_0.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=comp_0.dtype, device=comp_0.device),
            comp_5
        ], 2)
        
        # D->all P block
        compatibility_additional_allpick2 = torch.cat([
            -np.inf * torch.ones(self.n_heads, batch_size, 1, n_pick, dtype=comp_0.dtype, device=comp_0.device),
            -np.inf * torch.ones(self.n_heads, batch_size, n_pick, n_pick, dtype=comp_0.dtype, device=comp_0.device),
            comp_6
        ], 2)

        compatibility = torch.cat([comp_0, compatibility_additional_delivery, compatibility_additional_allpick,
                                   compatibility_additional_alldelivery,
                                   compatibility_additional_pick, compatibility_additional_alldelivery2,
                                   compatibility_additional_allpick2], dim=-1)

        # Remove old arch_mask logic loops since handled above

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility,
                             dim=-1)  # [n_heads, batch_size, n_query, graph_size+1+n_pick*2]

        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        if self.agg_type == 'mean':
             # heads is currently a sum of 7 components (plus original V)
             # This is tricky because the "sum" above is actually 
             # heads = heads_0 + heads_1 + ...
             # We want mean(heads_0, heads_1, ...)
             # But we computed it iteratively.
             # Wait, the current implementation iteratively adds to `heads`.
             # To do mean/max properly, we should stack them?
             # BUT: The architecture allows opting out (zero).
             # Efficient way: Stack results and aggregate.
             # Refactoring the iterative add to stacking:
             components = []
             
             # 0. Global
             components.append(torch.matmul(attn[:, :, :, :graph_size], V))
             
             # 1. P->D_pair
             components.append(attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size, 1) * V_additional_delivery)
             
             # 2. P->P
             components.append(torch.matmul(
                attn[:, :, :, graph_size + 1:graph_size + 1 + n_pick].view(self.n_heads, batch_size, graph_size, n_pick),
                V_allpick
             ))
             
             # 3. P->D_all
             components.append(torch.matmul(
                attn[:, :, :, graph_size + 1 + n_pick:graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size,
                                                                                        graph_size, n_pick), V_alldelivery))
                                                                                        
             # 4. D->P_pair
             components.append(attn[:, :, :, graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size,
                                                                            1) * V_additional_pick)
            
             # 5. D->D
             components.append(torch.matmul(
                attn[:, :, :, graph_size + 1 + 2 * n_pick + 1:graph_size + 1 + 3 * n_pick + 1].view(self.n_heads,
                                                                                                    batch_size, graph_size,
                                                                                                    n_pick), V_alldelivery2))
             
             # 6. D->P_all
             components.append(torch.matmul(
                attn[:, :, :, graph_size + 1 + 3 * n_pick + 1:].view(self.n_heads, batch_size, graph_size, n_pick),
                V_allpickup2))
             
             stacked = torch.stack(components, dim=0) # [7, n_heads, batch, graph, val]
             heads = stacked.mean(dim=0)
             
        elif self.agg_type == 'max':
             components = []
             components.append(torch.matmul(attn[:, :, :, :graph_size], V))
             components.append(attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size, 1) * V_additional_delivery)
             components.append(torch.matmul(attn[:, :, :, graph_size + 1:graph_size + 1 + n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_allpick))
             components.append(torch.matmul(attn[:, :, :, graph_size + 1 + n_pick:graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_alldelivery))
             components.append(attn[:, :, :, graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, 1) * V_additional_pick)
             components.append(torch.matmul(attn[:, :, :, graph_size + 1 + 2 * n_pick + 1:graph_size + 1 + 3 * n_pick + 1].view(self.n_heads, batch_size, graph_size, n_pick), V_alldelivery2))
             components.append(torch.matmul(attn[:, :, :, graph_size + 1 + 3 * n_pick + 1:].view(self.n_heads, batch_size, graph_size, n_pick), V_allpickup2))
             
             stacked = torch.stack(components, dim=0)
             heads, _ = stacked.max(dim=0)
             
        # Default 'sum' falls through to original behavior (but we need to rewrite it to use components if we want to be clean, or just keep original iterative add for sum as it might be marginally faster/less memory? Stacking is cleaner for maintentance though).
        # Let's keep original iterative logic for sum for now, or just use stack sum.
        # However, the iterative logic was modifying `heads` in place.
        # If I want to support all 3 cleanly, I should replace lines 406-434 with the component logic for ALL cases.
        else: # 'sum'
             components = []
             components.append(torch.matmul(attn[:, :, :, :graph_size], V))
             components.append(attn[:, :, :, graph_size].view(self.n_heads, batch_size, graph_size, 1) * V_additional_delivery)
             components.append(torch.matmul(attn[:, :, :, graph_size + 1:graph_size + 1 + n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_allpick))
             components.append(torch.matmul(attn[:, :, :, graph_size + 1 + n_pick:graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, n_pick), V_alldelivery))
             components.append(attn[:, :, :, graph_size + 1 + 2 * n_pick].view(self.n_heads, batch_size, graph_size, 1) * V_additional_pick)
             components.append(torch.matmul(attn[:, :, :, graph_size + 1 + 2 * n_pick + 1:graph_size + 1 + 3 * n_pick + 1].view(self.n_heads, batch_size, graph_size, n_pick), V_alldelivery2))
             components.append(torch.matmul(attn[:, :, :, graph_size + 1 + 3 * n_pick + 1:].view(self.n_heads, batch_size, graph_size, n_pick), V_allpickup2))
             
             stacked = torch.stack(components, dim=0)
             heads = stacked.sum(dim=0)


        #Zi*Wo
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None   # None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))




    def forward(self, x, mask=None):   # x: [embed_depot, embed_pick, embed_delivery]
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


    def set_arch(self, arch):
        if arch is None:
            for layer in self.layers:
                layer[0].module.set_arch_mask(None)
            return
        
        # New format: {'layers': [...], 'aggregation': [...]}
        if isinstance(arch, dict) and 'layers' in arch and 'aggregation' in arch:
            arch_per_layer = arch['layers']
            agg_per_layer = arch['aggregation']
        elif isinstance(arch, (list, tuple)):
            # Legacy format fallback (assumes sum)
            arch_per_layer = arch
            agg_per_layer = ['sum'] * len(self.layers)
        else:
            raise ValueError("Invalid arch format")

        if len(arch_per_layer) != len(self.layers) or len(agg_per_layer) != len(self.layers):
            raise ValueError("arch must have per-layer masks matching n_layers")

        for layer, layer_mask, agg_type in zip(self.layers, arch_per_layer, agg_per_layer):
            layer[0].module.set_arch_mask(layer_mask, agg_type)

