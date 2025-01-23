import torch
import torch.nn as nn



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, dilation_factor, num_channels, kernel_size=7, dropout=0.2):
        super(TCN, self).__init__()
        # self.tcn = TemporalBlock(input_size, num_channels, kernel_size, dropout)
        layers = []
        num_levels = len(num_channels)
        dilation_factor = dilation_factor
        for i in range(num_levels):
            dilation_size = dilation_factor ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        y1 = self.tcn(x)  # (B, D, T)
        y1 = y1.permute(0, 2, 1)  # (B, T, D)
        return y1


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
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.batch_norm(out)
        out = nn.functional.relu(out)
        out = self.drop(out)
        out = self.linear2(out)
        return out
    
class IDSA(nn.Module):
    def __init__(self, input_feat_dim, hidden_size, num_layers, num_classes, channel_num, spa_mode='mha', temp_mode='lstm', dilation_factor=3):
        super(IDSA, self).__init__()
        if spa_mode == 'mha':
            self.spatial_layer = Spatial_MHA(input_feat_dim)
        else:
            self.spatial_layer = Spatial_Linear(input_feat_dim, input_feat_dim)
        if temp_mode == 'lstm':
            self.temporal_layer = LSTM(channel_num, hidden_size, num_layers)
        else:
            self.temporal_layer = TCN(channel_num, dilation_factor, [hidden_size]*num_layers)
        self.classifier = Classifier(hidden_size, hidden_size, num_classes)

        self.source_embed = nn.Embedding(channel_num, input_feat_dim)
        self.target_embed = nn.Embedding(channel_num, input_feat_dim)
        self.mapping_matrix = nn.Parameter(torch.randn(channel_num, channel_num))


    def EMA(self, alpha=0.9):
        self.target_embed.weight.data = alpha * self.target_embed.weight.data + (1-alpha) * self.source_embed.weight.data   

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
        
        # checking only temporal layer (no spatial layer)
        # out = x

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
        # graph = self.marginalization(graph, source_embed_w, target_embed_w)
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

