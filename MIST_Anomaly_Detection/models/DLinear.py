# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
#
#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x
#
#
# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)
#
#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
#
# class Model(nn.Module):
#     """
#     Decomposition-Linear
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#
#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = configs.individual
#         self.channels = configs.enc_in
#
#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
#
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
#                 self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
#
#                 # Use this two lines if you want to visualize the weights
#                 # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#                 # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         else:
#             self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
#             self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
#
#             # Use this two lines if you want to visualize the weights
#             # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#             # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seasonal_init, trend_init = self.decompsition(x)
#         seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
#         if self.individual:
#             seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
#             trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
#             for i in range(self.channels):
#                 seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
#                 trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
#         else:
#             seasonal_output = self.Linear_Seasonal(seasonal_init)
#             trend_output = self.Linear_Trend(trend_init)
#
#         x = seasonal_output + trend_output
#         return x.permute(0,2,1) # to [Batch, Output length, Channel]


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """

#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x


# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """

#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean


# class Model(nn.Module):
#     """
#     Decomposition-Linear
#     """

#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len

#         # Decompsition Kernel Size
#         kernel_size = 25
#         self.decompsition = series_decomp(kernel_size)
#         self.individual = configs.individual
#         self.enc_in = configs.enc_in
#         self.period_len = 24

#         self.seg_num_x = self.seq_len // self.period_len
#         self.seg_num_y = self.pred_len // self.period_len

#         # self.Linear_Seasonal = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
#         # self.Linear_Trend = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
#         self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
#         self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seasonal_init, trend_init = self.decompsition(x)
#         seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

#         seasonal_output = self.Linear_Seasonal(seasonal_init)
#         trend_output = self.Linear_Trend(trend_init)

#         # seasonal_init = seasonal_init.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
#         # seasonal_output = self.Linear_Seasonal(seasonal_init)  # bc,w,m
#         # seasonal_output = seasonal_output.permute(0, 2, 1).reshape(x.size(0), self.enc_in, self.pred_len)
#         #
#         # trend_init = trend_init.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
#         # trend_output = self.Linear_Trend(trend_init)  # bc,w,m
#         # trend_output = trend_output.permute(0, 2, 1).reshape(x.size(0), self.enc_in, self.pred_len)

#         x = seasonal_output + trend_output
#         return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    DLinear for Classification Task
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """
    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = 'classification'
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len  # For classification, we use the same length
        
        # Series decomposition block from Autoformer
        kernel_size = configs.moving_avg if hasattr(configs, 'moving_avg') else 25
        self.decompsition = series_decomp(kernel_size)
        
        self.individual = individual if individual is not None else configs.individual if hasattr(configs, 'individual') else False
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Initialize weights
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Initialize weights
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # Classification head
        self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        """
        Encode the input time series using decomposition and linear layers
        """
        # Decompose into seasonal and trend components
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        
        # Combine seasonal and trend
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [B, seq_len, channels]

    def classification(self, x_enc):
        """
        Classification forward pass
        """
        # Encoder
        enc_out = self.encoder(x_enc)  # [B, seq_len, channels]
        
        # Output
        # Flatten: (batch_size, seq_length * channels)
        output = enc_out.reshape(enc_out.shape[0], -1)
        
        # Project to num_classes: (batch_size, num_classes)
        output = self.projection(output)
        
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass for classification
        x_enc: [B, seq_len, channels]
        Returns: [B, num_classes]
        """
        dec_out = self.classification(x_enc)
        return dec_out  # [B, num_classes]