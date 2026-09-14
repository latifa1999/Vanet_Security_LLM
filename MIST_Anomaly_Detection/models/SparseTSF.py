import torch
import torch.nn as nn

class Model(nn.Module):
    """
    SparseTSF adapted for classification.
    Encoder preserved from original forecasting model.
    Adds classification head after latent representation.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # ----- Original SparseTSF parameters -----
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = 'mlp' # 'linear' or 'mlp'
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len

        # 1D convolution for sparse period extraction
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False
        )

        # Linear or MLP segment-level aggregation
        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_x, bias=False)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_x)
            )

        # ----- Classification head -----
        self.num_classes = configs.num_classes
        self.feature_dim = self.enc_in * self.period_len * self.seg_num_x

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        """
        Input:  [B, seq_len, enc_in]
        Output: logits [B, num_classes]
        """
        B = x.shape[0]

        # Normalize + permute: [B, seq_len, enc_in] -> [B, enc_in, seq_len]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D Convolution: reshape to [B*enc_in, 1, seq_len] for conv1d
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(B, self.enc_in, self.seq_len) + x

        # Downsampling into segments: [B, enc_in, seq_len] -> [B*enc_in, seg_num_x, period_len]
        x = x.reshape(B * self.enc_in, self.seg_num_x, self.period_len)
        
        # Permute for segment-wise processing: [B*enc_in, period_len, seg_num_x]
        x = x.permute(0, 2, 1)

        # Sparse forecasting/transformation
        if self.model_type == 'linear':
            y = self.linear(x)  # [B*enc_in, period_len, seg_num_x]
        else:
            y = self.mlp(x)

        # Reshape back: [B*enc_in, period_len, seg_num_x] -> [B, enc_in, period_len, seg_num_x]
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.seg_num_x, self.period_len)
        
        # Flatten spatial dimensions for classification: [B, enc_in * seg_num_x * period_len]
        logits = self.classifier(y)

        return logits