"""A demo of ReFactor GNN induced by DistMult optimised with SGD over a cross-entropy loss + N3 regularizer . 
Implemented using Pytorch Geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing


def parse_infer_cmd(infer_with):
    return int(infer_with.split('-')[0])


class ReFactorConv(MessagePassing):
    def __init__(self, n_ent=None, n_rel=None, n_hid=None, 
                 fm_alpha=0.1, fm_lmbda=5e-3, fm_optim='SGD', fm_score_func='DistMult',
                 train_mp_init='input', infer_with='message-passing'):
        """ ReFactorConv implements the ReFactor Layer in our paper (see sec 4 in https://arxiv.org/pdf/2207.09980.pdf)
        Args:
            train_mp_init: str, specify how to two options, default to 'input'
                           option1: 'input', each series of message-passings starts with raw input
                           option2: 'state_caching', each series of message-passings starts with cached states from previous calculation
            infer_with: str, two options
                        option1: 'input', simply use input as node states;
                        option2: 'state_caching', use cashed node states; 
                        option3: 'k-message_passing', use node states produced by a message-passing; 
        """
        super(ReFactorConv, self).__init__()
        self.fm_alpha, self.fm_lmbda, self.fm_optim, self.fm_score_func = fm_alpha, fm_lmbda, fm_optim, fm_score_func
        self.train_mp_init, self.infer_with = train_mp_init, infer_with

        self.rel_emb = nn.Embedding(n_rel, n_hid)
        self.rel_emb.weight.data *= 1e-3 

        if self.train_mp_init == 'state_caching' or self.infer_with == 'state_caching': # node state cache/memory
            self.ent_state_cache = nn.Embedding(n_ent, n_hid)
            self.ent_state_cache.weight.data *= 1e-3
            self.ent_state_cache.weight.requires_grad = False # Caution!!! The node state memory is not updated by auto-differentiation
            self.init_node_states = None

    def forward(self, x, edge_index, edge_type, g_node_idx=None, clr_ent_state=False):
        if self.training == False: # inference-time 
            if self.infer_with.endswith('-message-passing'): 
                l_round = parse_infer_cmd(self.infer_with) # on-the-fly l-round message-passing
                adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_type) # use to incur message_and_aggregate
                for _ in range(l_round):
                    x = self.propagate(adj, x=x, g_node_idx=g_node_idx) # l-round MESSAGE-PASSING
                x_new = x   
            elif self.infer_with == 'state_caching':
                x_new = self.pull(g_node_idx)
            elif self.infer_with == 'input':
                x_new = x            
        else: # training time
            if clr_ent_state and (self.train_mp_init == 'state_caching' or self.infer_with == 'state_caching'):
                self.clear() # restart the message-passing by clearing node states/optim states
            if self.train_mp_init == 'state_caching':
                x = self.pull(g_node_idx) 
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_type) # use to incur message_and_aggregate
            x_new = self.propagate(adj, x=x, g_node_idx=g_node_idx)
            if self.train_mp_init == 'state_caching' or self.infer_with == 'state_caching':
                self.push(g_node_idx, x_new)
        return x_new
    
    def pull(self, node_idx):
        """Read from the node state cache"""
        x = self.ent_state_cache(node_idx)
        return x

    def push(self, node_idx, node_state):
        """Write to the node state cache"""
        with torch.no_grad():
            self.ent_state_cache.weight.data[node_idx] = node_state
    
    def clear(self):
        """Reset the node state cache to init_node_states"""
        if self.init_node_states is None:
            self.ent_state_cache.weight.data.normal_()
            self.ent_state_cache.weight.data *= 1e-3 
            self.ent_state_cache.weight.requires_grad = False
        else:
            print('Reset ent_emb to init_node_states!')            
            self.ent_state_cache.weight.data = self.init_node_states.clone().detach().type_as(self.ent_state_cache.weight.data)
            self.ent_state_cache.weight.requires_grad = False

    def set_init_node_states(self, node_features):
        self.init_node_states = node_features.clone().detach()

    def embed_rel(self, p):
        """Get relation embedding"""
        rp = self.rel_emb(p)
        return rp
            
    def fm_score(self, hv, rp, ws):
        """Score function, corresponding to Gamma in the paper
        Might lead to huge memory consumption if |B| (num queries) or N (num candidate nodes) is big

        Args:
            hv: subject representation, |B| X K
            rp: predicate representation, |B| X K
            ws: representations for all the objects, N X K 
        
        Return: probabilities for P(w|v, r), |B| x N (num_queries x num_nodes) 
        """
        if self.fm_score_func == 'DistMult':
            Z = F.softmax(hv * rp @ ws.t(), dim=1)
        return Z
    
    def message_and_aggregate(self, adj, x=None):
        """ Compute the aggregated message, 
        avoiding explicitly materialising each message vector

        args: 
            adj: sparse tensor indicating index, transposed
        """ 
        n_nodes = x.shape[0]
        v, w, p = adj.coo()
        hv, rp, hw = x[v], self.embed_rel(p), x[w]

        Z = self.fm_score(hv, rp, ws=x) # |B| x N
        Zw = torch.gather(Z, dim=1, index=w.unsqueeze(1)) # |B| x 1
        Zv = torch.gather(Z, dim=1, index=v.unsqueeze(1)) # |B| x 1

        if self.fm_score_func == 'DistMult':
            aggr_out = [] 
            '''Direction 1: w2v, z_{v<-*} = - grad_v'''
            msg_fit = rp * hw - rp * (Z @ x)
            msg_reg = - self.fm_lmbda * 3 * (hv ** 2) * torch.sign(hv)
            aggr_other2v = scatter(src=msg_fit + msg_reg, 
                                       index=v, dim=0, dim_size=n_nodes)
            aggr_out.append(aggr_other2v)

            '''Direction 2: v2w, z_{w<-v} = - grad_w'''
            msg_fit = (1 - Zw) * rp * hv
            msg_reg = - self.fm_lmbda * 3 * (hw ** 2) * torch.sign(hw)
            aggr_v2neighbor = scatter(src=msg_fit + msg_reg, 
                                      index=w, dim=0, dim_size=n_nodes)
            aggr_out.append(aggr_v2neighbor)
    
            '''Global nomaliser: z_{u<-v}= - grad_u'''
            msg_fit = - Z.T @ (rp * hv) # N x K
            msg_fit[w, :] += Zw * (rp * hv)
            msg_fit[v, :] += Zv * (rp * hv)
            aggr_v2negative = scatter(src=msg_fit, index=torch.arange(n_nodes).type_as(v), dim=0, dim_size=n_nodes)
            aggr_out.append(aggr_v2negative)
            aggr_out = torch.stack(aggr_out, dim=0).sum(dim=0)
        return aggr_out

    def update(self, aggr_out, x, g_node_idx):
        return x + self.fm_alpha * aggr_out