# coding=utf-8
"""
@author: Yantong Lai
@description: Code of [24 ICASSP] Adaptive Spatial-Temporal Hypergraph Fusion Learning for Next POI Recommendation
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiSemanticHyperConvLayer(nn.Module):
    """
    Multi-Semantic Hypergraph Convolutional Layer
    """
    def __init__(self, emb_dim, device, dropout):
        super(MultiSemanticHyperConvLayer, self).__init__()

        # self.fusion = nn.Sequential(nn.Linear(7 * emb_dim, emb_dim), nn.Linear(emb_dim, emb_dim))
        self.fusion = nn.Linear(7 * emb_dim, emb_dim)
        self.emb_dim = emb_dim
        self.device = device
        self.dropout = dropout

    def forward(self, col_pois_embs, geo_pois_embs, seq_pois_embs, users_embs, HG_up, HG_pu):
        # geo_pois_embs = [L, d]
        # seq_pois_embs = [L, d]
        # users_embs = [U, d]
        # H_pu = [L, U] = H
        # H_up = [U, L] = HT

        # node -> hyperedge message
        # 1) poi node aggregation to get msg_g, msg_t, msg_c
        msg_geo_agg = torch.sparse.mm(HG_up, geo_pois_embs)    # [U, d]
        msg_seq_agg = torch.sparse.mm(HG_up, seq_pois_embs)    # [U, d]
        msg_poi_agg = torch.sparse.mm(HG_up, col_pois_embs)   # [U, d]

        # generate finer-grained message by multiplication: msg_gt, msg_gc, msg_tc, msg_gtc
        msg_geo_seq = msg_geo_agg * msg_seq_agg
        msg_geo_poi = msg_geo_agg * msg_poi_agg
        msg_seq_poi = msg_seq_agg * msg_poi_agg
        msg_geo_seq_poi = msg_geo_agg * msg_seq_agg * msg_poi_agg

        # concat above 7 message
        msg = torch.cat([msg_geo_agg, msg_seq_agg, msg_poi_agg, msg_geo_seq, msg_geo_poi, msg_seq_poi, msg_geo_seq_poi], dim=1)
        msg_emb = self.fusion(msg)  # [U, d]
        # msg_emb = F.dropout(msg_emb, self.dropout)    # [U, d]

        # adaptive fusion method to generate user representation
        hg_users_emb = (msg_emb + users_embs) + (msg_emb * users_embs)    # [U, d]

        # propagation: hyperedge -> node
        propag_pois_embs = torch.sparse.mm(HG_pu, hg_users_emb)  # [L, d]

        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """Directed hypergraph convolutional layer"""
    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src


class MultiSemanticHyperConvNetwork(nn.Module):
    """
    Multi-Semantic Hypergraph Convolutional Network
    """
    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiSemanticHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.ms_hconv_layer = MultiSemanticHyperConvLayer(emb_dim, device, dropout)

    def forward(self, init_pois_embs, geo_pois_embs, seq_pois_embs, users_embs, HG_up, HG_pu):
        final_pois_embs = [init_pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.ms_hconv_layer(init_pois_embs, geo_pois_embs, seq_pois_embs, users_embs, HG_up, HG_pu)

            # add residual connection to alleviate over-smoothing issue
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)    # [L, d]

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device):
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # add residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)    # [L, d]

        return final_pois_embs


class GeoConvNetwork(nn.Module):
    def __init__(self, num_layers):
        super(GeoConvNetwork, self).__init__()

        self.num_layers = num_layers

    def forward(self, pois_embs, geo_graph):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            # pois_embs = geo_graph @ pois_embs
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return output_pois_embs


class ASTHL(nn.Module):
    """Adaptive Spatial-Temporal Hypergraph Fusion Learning for Next POI Recommendation (ASTHL)"""
    def __init__(self, num_users, num_pois, args, device):
        super(ASTHL, self).__init__()

        # definition
        self.num_users = num_users
        self.num_pois = num_pois
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature

        # embedding
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=num_pois)

        # embedding init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)

        # self gating
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)

        # network
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers)
        self.di_hconv_network = DirectedHyperConvNetwork(args.num_di_layers, device)
        self.multi_semantic_network = MultiSemanticHyperConvNetwork(args.num_mv_layers, args.emb_dim, args.dropout, device)

    def cal_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]

        return loss

    def cal_cross_view_loss(self, view1_embs, view2_embs):
        loss_cross_view = 0.0
        loss_cross_view += self.cal_loss_infonce(view1_embs, view2_embs)
        loss_cross_view += self.cal_loss_infonce(view2_embs, view1_embs)

        return loss_cross_view / 2

    def forward(self, dataset, batch):
        # self-gating input
        geo_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_geo) + self.b_gate_geo))
        seq_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_seq) + self.b_gate_seq))
        col_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))

        # geographical view: poi-poi geographical graph convolutional network
        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)    # [L, d]

        # sequential view: directed hypergraph convolutional network
        seq_pois_embs = self.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)  # [L, d]

        # normalize pois embeddings from geographical and sequential views
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_seq_pois_embs = F.normalize(seq_pois_embs, p=2, dim=1)

        # cross-view contrastive learning
        loss_cl_pois = self.cal_cross_view_loss(norm_geo_pois_embs, norm_seq_pois_embs)

        # user-poi interaction view
        hg_pois_embs = self.multi_semantic_network(col_gate_pois_embs, norm_geo_pois_embs, norm_seq_pois_embs, self.user_embedding.weight, dataset.HG_up, dataset.HG_pu)

        # normalize
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)    # [L, d]

        # fuse for final pois embs
        fusion_pois_embs = norm_hg_pois_embs + norm_geo_pois_embs + norm_seq_pois_embs

        hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, fusion_pois_embs)  # [U, d]
        batch_users_embs = hg_structural_users_embs[batch["user_idx"]]  # [BS, d]

        final_batch_users_embs = F.normalize(batch_users_embs, p=2, dim=1)

        return final_batch_users_embs, fusion_pois_embs, loss_cl_pois


