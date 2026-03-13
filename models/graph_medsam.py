import torch
import torch.nn as nn
from models.medsam_encoder import MedSAMEncoder
from models.fusion_layer import GraphResidualFusion
from models.decoder_wrapper import MedSAMDecoderWrapper
from graph.supervoxel_cluster import SupervoxelClusterer
from graph.edge_weights import EdgeWeightProcessor
from graph.gnn_model import VesselGNN
from graph.broadcast import FeatureBroadcaster

class GraphModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clusterer = SupervoxelClusterer(config)
        self.edge_processor = EdgeWeightProcessor(config)
        self.gnn = VesselGNN(config)
        self.broadcaster = FeatureBroadcaster(config)

class GraphMedSAM(nn.Module):
    def __init__(self, config, sam_model):
        super().__init__()
        self.encoder = MedSAMEncoder(sam_model, config['model']['encoder_frozen_layers'])
        self.graph_module = GraphModule(config) # Grouped for Trainer compatibility
        self.fusion = GraphResidualFusion(config)
        self.decoder_wrapper = MedSAMDecoderWrapper(config, sam_model.mask_decoder)

    def forward(self, x, coords):
        original_size = x.shape[-3:]
        
        # Encoder tokens: (B, 2048, 768)
        z_visual = self.encoder(x)
        
        # Graph Pipeline via graph_module
        cluster_ids, node_feats, node_coords = self.graph_module.clusterer(z_visual, coords)
        edge_index, weights = self.graph_module.edge_processor(node_feats, node_coords)
        
        B, K, C = node_feats.shape
        node_feats_flat = node_feats.view(-1, C)
        
        # GNN Refinement
        refined_nodes = self.graph_module.gnn(node_feats_flat, edge_index, weights)
        
        # Broadcast & Fusion
        z_graph = self.graph_module.broadcaster(refined_nodes, cluster_ids)
        z_fused = self.fusion(z_visual, z_graph)
        
        # Decoder Output
        logits = self.decoder_wrapper(z_fused, original_size)
        return logits