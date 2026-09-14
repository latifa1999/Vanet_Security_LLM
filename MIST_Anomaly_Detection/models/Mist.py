import torch
import torch.nn as nn
# # from layers.Embed import PositionalEmbedding

# # class Model(nn.Module):
# #     def __init__(self, configs):
# #         super(Model, self).__init__()

# #         # get parameters
# #         self.seq_len = configs.seq_len
# #         self.pred_len = configs.pred_len
# #         self.enc_in = configs.enc_in
# #         self.period_len = configs.period_len
# #         self.d_model = configs.d_model
# #         self.model_type = configs.model_type
# #         assert self.model_type in ['linear', 'mlp']

# #         self.seg_num_x = self.seq_len // self.period_len
# #         self.seg_num_y = self.pred_len // self.period_len

# #         self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
# #                                 stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

# #         if self.model_type == 'linear':
# #             self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
# #         elif self.model_type == 'mlp':
# #             self.mlp = nn.Sequential(
# #                 nn.Linear(self.seg_num_x, self.d_model),
# #                 nn.ReLU(),
# #                 nn.Linear(self.d_model, self.seg_num_y)
# #             )


# #     def forward(self, x):
# #         batch_size = x.shape[0]
# #         # normalization and permute     b,s,c -> b,c,s
# #         seq_mean = torch.mean(x, dim=1).unsqueeze(1)
# #         x = (x - seq_mean).permute(0, 2, 1)

# #         # 1D convolution aggregation
# #         x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

# #         # downsampling: b,c,s -> bc,n,w -> bc,w,n
# #         x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

# #         # sparse forecasting
# #         if self.model_type == 'linear':
# #             y = self.linear(x)  # bc,w,m
# #         elif self.model_type == 'mlp':
# #             y = self.mlp(x)

# #         # upsampling: bc,w,m -> bc,m,w -> b,c,s
# #         y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

# #         # permute and denorm
# #         y = y.permute(0, 2, 1) + seq_mean

# #         return y

# class Model(nn.Module):
#     """
#     SparseTSF for Classification
#     Modified from original forecasting version
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = 0  # Not used in classification
#         self.period_len = configs.period_len
#         self.enc_in = configs.enc_in
#         self.num_classes = configs.num_classes  # NEW: number of classes
        
#         # Calculate downsampled length
#         self.d_model = self.seq_len // self.period_len
        
#         # Sparse cross-period extraction
#         self.conv = nn.Conv1d(
#             in_channels=self.period_len,
#             out_channels=self.period_len,
#             kernel_size=self.d_model,
#             bias=False,
#             groups=self.period_len
#         )
        
#         # Channel-wise projection
#         self.projection = nn.Linear(self.enc_in, self.enc_in)
        
#         # Classification head
#         self.flatten_dim = self.period_len * self.enc_in
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.flatten_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, self.num_classes)
#         )
        
#     def forward(self, x):
#         # x: [Batch, seq_len, enc_in]
#         B, L, C = x.shape
        
#         # Reshape for period-based processing
#         x = x.reshape(B, self.period_len, self.d_model, C)
#         x = x.permute(0, 3, 1, 2)  # [B, C, period_len, d_model]
        
#         # Process each channel
#         outputs = []
#         for i in range(C):
#             channel_data = x[:, i, :, :]  # [B, period_len, d_model]
            
#             # Apply sparse cross-period convolution
#             out = self.conv(channel_data)  # [B, period_len, 1]
#             outputs.append(out)
        
#         # Concatenate channels
#         x = torch.stack(outputs, dim=1)  # [B, C, period_len, 1]
#         x = x.squeeze(-1)  # [B, C, period_len]
#         x = x.permute(0, 2, 1)  # [B, period_len, C]
        
#         # Channel projection
#         x = self.projection(x)  # [B, period_len, C]
        
#         # Classification
#         logits = self.classifier(x)  # [B, num_classes]
        
#         return logits




# # class Model(nn.Module):
# #     """
# #     SparseTSF for Classification
# #     Modified from original forecasting version
# #     """
# #     def __init__(self, configs):
# #         super(Model, self).__init__()
# #         self.seq_len = configs.seq_len
# #         self.pred_len = 0  # Not used in classification
# #         self.period_len = configs.period_len
# #         self.enc_in = configs.enc_in
# #         self.channels = configs.enc_in
# #         self.num_classes = configs.num_classes  # Number of classes
        
# #         # Calculate downsampled length
# #         self.d_model = self.seq_len // self.period_len
        
# #         # Sparse cross-period extraction
# #         self.conv = nn.Conv1d(
# #             in_channels=self.period_len,
# #             out_channels=self.period_len,
# #             kernel_size=self.d_model,
# #             bias=False,
# #             groups=self.period_len
# #         )
        
# #         # Additional convolution layers
# #         self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
# #         self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        
# #         # Batch normalization layers
# #         self.bn1 = nn.BatchNorm1d(self.channels)
# #         self.bn2 = nn.BatchNorm1d(self.channels)
        
# #         # Activation
# #         self.act = nn.GELU()
        
# #         # Channel-wise projection
# #         self.projection = nn.Linear(self.enc_in, self.enc_in)
        
# #         # Output convolution for feature aggregation
# #         self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
# #         self.bn3 = nn.BatchNorm1d(1)
        
# #         # Classification head
# #         self.classifier = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(self.period_len, 128),
# #             nn.ReLU(),
# #             nn.Dropout(0.5),
# #             nn.Linear(128, self.num_classes)
# #         )
    
# #     def forward(self, x):
# #         # x: [Batch, seq_len, enc_in]
# #         B, L, C = x.shape
        
# #         # Reshape for period-based processing
# #         x = x.reshape(B, self.period_len, self.d_model, C)
# #         x = x.permute(0, 3, 1, 2)  # [B, C, period_len, d_model]
        
# #         # Process each channel
# #         outputs = []
# #         for i in range(C):
# #             channel_data = x[:, i, :, :]  # [B, period_len, d_model]
# #             # Apply sparse cross-period convolution
# #             out = self.conv(channel_data)  # [B, period_len, 1]
# #             outputs.append(out)
        
# #         # Concatenate channels
# #         x = torch.stack(outputs, dim=1)  # [B, C, period_len, 1]
# #         x = x.squeeze(-1)  # [B, C, period_len]
        
# #         # Apply initial convolution with batch norm
# #         x = self.init_conv(x)  # [B, C, period_len]
# #         x = self.bn1(x)
# #         x = self.act(x)
        
# #         # Apply inner convolution with batch norm
# #         x = self.inner_conv(x)  # [B, C, period_len]
# #         x = self.bn2(x)
# #         x = self.act(x)
        
# #         # Apply output convolution to aggregate channels
# #         x = self.out_conv(x)  # [B, 1, period_len]
# #         x = self.bn3(x)
# #         x = self.act(x)
        
# #         x = x.squeeze(1)  # [B, period_len]
        
# #         # Classification
# #         logits = self.classifier(x)  # [B, num_classes]
        
# #         return logits





class Model(nn.Module):
    """
    Enhanced SparseTSF for Classification
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.period_len = configs.period_len
        self.enc_in = configs.enc_in
        self.channels = configs.enc_in
        self.num_classes = configs.num_classes
        
        # Calculate downsampled length
        self.d_model = self.seq_len // self.period_len
        
        # 1. IMPROVEMENT: Add learnable embeddings for temporal position
        self.temporal_embedding = nn.Parameter(torch.randn(1, self.period_len, self.channels))
        
        # Sparse cross-period extraction with MULTIPLE scales
        self.conv = nn.Conv1d(
            in_channels=self.period_len,
            out_channels=self.period_len,
            kernel_size=self.d_model,
            bias=False,
            groups=self.period_len
        )
        
        # 2. IMPROVEMENT: Multi-scale feature extraction
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(self.channels, self.channels, kernel_size=k, padding=k//2, bias=False)
            for k in [3, 5, 7]
        ])
        
        # 3. IMPROVEMENT: Residual connections with better architecture
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        
        # 4. IMPROVEMENT: Use LayerNorm instead of BatchNorm for better generalization
        self.ln1 = nn.LayerNorm(self.channels)
        self.ln2 = nn.LayerNorm(self.channels)
        
        # 5. IMPROVEMENT: Attention mechanism for important features
        self.attention = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.ln_attn = nn.LayerNorm(self.channels)
        
        # 6. IMPROVEMENT: Better channel aggregation
        hidden_dim = 256
        self.channel_aggregation = nn.Sequential(
            nn.Conv1d(self.channels, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim, 64, 1, bias=False),
            nn.GELU()
        )
        
        self.act = nn.GELU()
        
        # 7. IMPROVEMENT: Enhanced classification head with residual
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        classifier_input_dim = 64 * 2  # avg + max pooling
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        # 8. IMPROVEMENT: Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [Batch, seq_len, enc_in]
        B, L, C = x.shape
        
        # Reshape for period-based processing
        x = x.reshape(B, self.period_len, self.d_model, C)
        x = x.permute(0, 3, 1, 2)  # [B, C, period_len, d_model]
        
        # Process each channel with sparse convolution
        outputs = []
        for i in range(C):
            channel_data = x[:, i, :, :]  # [B, period_len, d_model]
            out = self.conv(channel_data)  # [B, period_len, 1]
            outputs.append(out)
        
        x = torch.stack(outputs, dim=1)  # [B, C, period_len, 1]
        x = x.squeeze(-1)  # [B, C, period_len]
        
        # 9. IMPROVEMENT: Add temporal embeddings
        x_perm = x.permute(0, 2, 1)  # [B, period_len, C]
        x_perm = x_perm + self.temporal_embedding
        x = x_perm.permute(0, 2, 1)  # [B, C, period_len]
        
        # 10. IMPROVEMENT: Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        x = torch.stack(multi_scale_features, dim=0).mean(dim=0)  # Average multi-scale
        
        # Residual block 1
        residual = x
        x = self.init_conv(x)
        x = x.permute(0, 2, 1)  # [B, period_len, C]
        x = self.ln1(x)
        x = x.permute(0, 2, 1)  # [B, C, period_len]
        x = self.act(x)
        x = x + residual  # Residual connection
        
        # Residual block 2
        residual = x
        x = self.inner_conv(x)
        x = x.permute(0, 2, 1)  # [B, period_len, C]
        x = self.ln2(x)
        x = x.permute(0, 2, 1)  # [B, C, period_len]
        x = self.act(x)
        x = x + residual  # Residual connection
        
        # 11. IMPROVEMENT: Self-attention for temporal dependencies
        x_attn = x.permute(0, 2, 1)  # [B, period_len, C]
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        x_attn = self.ln_attn(x_attn + attn_out)  # Residual
        x = x_attn.permute(0, 2, 1)  # [B, C, period_len]
        
        # Channel aggregation
        x = self.channel_aggregation(x)  # [B, 64, period_len]
        
        # 12. IMPROVEMENT: Dual pooling (avg + max)
        x_avg = self.global_avg_pool(x).squeeze(-1)  # [B, 64]
        x_max = self.global_max_pool(x).squeeze(-1)  # [B, 64]
        x = torch.cat([x_avg, x_max], dim=1)  # [B, 128]
        
        # Classification
        logits = self.classifier(x)  # [B, num_classes]
        
        return logits