import torch
import torch.nn as nn



class Spatial_MHA(nn.Module):
    def __init__(self, input_size):
        super(Spatial_MHA, self).__init__()
        self.MHA = nn.MultiheadAttention(input_size, num_heads=1, batch_first=True)
        

    def forward(self, x, residual=True):
        _, adj = self.MHA(x, x, x)

        out = torch.bmm(adj, x)

        # non linear?
        out = torch.tanh(out)
        
        if residual:
            out = out + x
        
        return out

class Spatial_Linear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Spatial_Linear, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(0.5)
    def forward(self, x, residual=True):
        # x_new = self.linear(x)
        # adj = torch.bmm(x_new, x_new.transpose(1, 2))
        adj = torch.bmm(x, x.transpose(1, 2))
        adj = nn.functional.relu(adj)
        row_sum = adj.sum(dim=1)
        adj = adj / row_sum.unsqueeze(1)
        out = torch.bmm(adj, x)

        out = self.linear(out)
        # non-linear?
        out = torch.tanh(out)
        out = self.drop(out)
        # print(out.shape)
        # exit(0)
        # out = nn.functional.normalize(out, p=2, dim=2)
        if residual:
            out = out + x
        
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        return out
    
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.linear1(x)
        out = nn.functional.relu(out)
        out = self.drop(out)
        out = self.linear2(out)
        return out
    
class IDSA(nn.Module):
    def __init__(self, input_feat_dim, hidden_size, num_layers, num_classes, channel_num, mode='mha'):
        super(IDSA, self).__init__()
        if mode == 'mha':
            self.spatial_layer = Spatial_MHA(input_feat_dim)
        else:
            self.spatial_layer = Spatial_Linear(input_feat_dim, input_feat_dim)
        self.temporal_layer = LSTM(channel_num, hidden_size, num_layers)
        self.classifier = Classifier(hidden_size, hidden_size, num_classes)

        self.source_embed = nn.Embedding(channel_num, input_feat_dim)
        self.target_embed = nn.Embedding(channel_num, input_feat_dim)
        self.mapping_matrix = nn.Parameter(torch.randn(channel_num, channel_num))
        self.transform_layer = nn.Linear(input_feat_dim, input_feat_dim, bias=True)
        nn.init.eye_(self.transform_layer.weight)


    def EMA(self, alpha=0.9):
        self.target_embed.weight.data = alpha * self.target_embed.weight.data + (1-alpha) * self.source_embed.weight.data   
    
    def Position_transform(self, alpha = 0.9):
        transformed_embed =  self.transform_layer(self.source_embed.weight)
        self.target_embed.weight.data = transformed_embed.data
        
    def EMA_map(self, alpha=0.9):
        mapped_source = torch.matmul(self.mapping_matrix, self.source_embed.weight.data)
        self.target_embed.weight.data = alpha * self.target_embed.weight.data + (1-alpha) * mapped_source

        
    def inter_domain_graph_generation(self):
        source_embed_w = self.source_embed.weight
        target_embed_w = self.target_embed.weight
        adj = torch.mm(source_embed_w, target_embed_w.t())
        adj = nn.functional.relu(adj)
        row_sum = adj.sum(dim=1)
        adj = adj / row_sum.unsqueeze(1)
        return adj, source_embed_w, target_embed_w
    
    def forward(self, x, target=False):
        B, C, T = x.shape
        
        # Spatial layer
        adj, source_embed_w, target_embed_w = self.inter_domain_graph_generation()
        if target:
            
            x = x + self.target_embed.weight.unsqueeze(0).repeat(B, 1, 1)
            x = torch.bmm(adj.unsqueeze(0).repeat(B, 1, 1), x)
        else:
            x = x + self.source_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        out = self.spatial_layer(x, residual=True)
        
        # Reshape for temporal processing
        out = out.permute(0, 2, 1)
        
        # Temporal layer
        out = self.temporal_layer(out)
        
        # Pooling
        st_out = out[:, -1, :]

        # Classification
        out = self.classifier(st_out)
        return out, st_out, adj, source_embed_w, target_embed_w
    

class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        return loss
    
class Inter_domain_loss(nn.Module):
    def __init__(self):
        super(Inter_domain_loss, self).__init__()

    def forward(self, source, target, graph, source_embed_w, target_embed_w):
        mean_src = torch.mean(source, dim=0, keepdim=True)
        mean_trg = torch.mean(target, dim=0, keepdim=True)
        ### compute distance matrix
        distance_matrix = torch.zeros((graph.shape[1], graph.shape[1]))
        for i in range(graph.shape[1]):
            for j in range(graph.shape[1]):
                dist_element = torch.sqrt(torch.sum((mean_trg[0,i] - mean_src[0, j])**2))
                distance_matrix[i,j] = dist_element
        distance_matrix = distance_matrix.to(mean_src.device)
        graph = self.doubly_stochastic(graph)
        #graph = self.marginalization(graph, source_embed_w, target_embed_w)
        ### compute the loss
        loss = torch.sum(torch.mul(distance_matrix, graph))
        return loss
    
    def doubly_stochastic(self, graph):
        # graph = torch.mean(graph, dim=0, keepdim=True) # Not need batch-wise calculation now 
        graph += 1e-5
        rsum = torch.zeros(graph.shape[0])
        csum = torch.zeros(graph.shape[1])

        tolerance = 1e-5
        max_iterations = 10000
        iteration = 0

        while (torch.any(torch.abs(rsum - 1) > tolerance) or torch.any(torch.abs(csum - 1) > tolerance)) and iteration < max_iterations:
            graph /= graph.sum(dim=0, keepdim=True)
            graph /= graph.sum(dim=1, keepdim=True)
            
            rsum = graph.sum(dim=1)  
            csum = graph.sum(dim=0)  
 
            iteration += 1

        if iteration == max_iterations:
            print("WARNING: Doubly stochastic algorithm did not converge!!")

        return graph
    
    def marginalization(self, graph, source_embed_w, target_embed_w):
        src_prob = torch.mean(source_embed_w, dim=1, keepdim=False)
        src_prob = src_prob/src_prob.sum()
        trg_prob = torch.mean(target_embed_w, dim=1, keepdim=False)
        trg_prob = trg_prob/trg_prob.sum()

        graph = graph*src_prob.view(-1, 1)
        graph = graph*trg_prob.view(1, -1) 
        return graph

