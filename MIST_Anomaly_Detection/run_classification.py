import argparse
import torch
from exp.exp_classification import Exp_Classification
import random
import numpy as np

parser = argparse.ArgumentParser(description='Mist for Classification')

# Basic config
parser.add_argument('--task_name', type=str, default='classification')
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--model_id', type=str, default='Mist_classification')
parser.add_argument('--model', type=str, default='Mist')

# Data loader
parser.add_argument('--data', type=str, default='Custom')
parser.add_argument('--root_path', type=str, default='/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Data/Generated_data_2/')
parser.add_argument('--data_path', type=str, default='real_time_cleaned.csv')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--scaler_path', type=str, 
                    default='/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/AD_models/scalers/veremi_scaler.pkl',
                    help='Path to save/load scaler')

# Model parameters
parser.add_argument('--seq_len', type=int, default=96) #96
parser.add_argument('--pred_len', type=int, default=0)  # Not used for classification
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--enc_in', type=int, default=8)  # features
# parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=8, help='output size')
parser.add_argument('--num_classes', type=int, default=2)  # Binary classification
parser.add_argument('--period_len', type=int, default=4) #24 #4

parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from latest checkpoint')

parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=False)

# Autoformer architecture
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers (not used)')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')

# FEDformer specific parameters
parser.add_argument('--version', type=str, default='Fourier', 
                    help='Fourier or Wavelets')
parser.add_argument('--mode_select', type=str, default='random',
                    help='mode selection method for Fourier')
parser.add_argument('--modes', type=int, default=64,
                    help='modes to be selected')

# For Wavelets version
parser.add_argument('--L', type=int, default=1,
                    help='wavelet level')
parser.add_argument('--base', type=str, default='legendre',
                    help='wavelet base')
parser.add_argument('--cross_activation', type=str, default='tanh',
                    help='cross activation for wavelets')


# Training parameters
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256) #default=32, help='batch size (smaller for Autoformer)'
parser.add_argument('--learning_rate', type=float, default=0.001) #0.001
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')

# Misc
parser.add_argument('--seed', type=int, default=2021)

args = parser.parse_args()

# Set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Set device
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

# Create experiment
Exp = Exp_Classification

if args.is_training:
    setting = f'{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.period_len}'
    
    exp = Exp(args)
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)
    
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)

    # If benchmark flag is set, also test on benchmark
    # if args.test_benchmark:
    #     print(f'\n>>>>>>>testing on BENCHMARK set: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #     # Change data_path to benchmark
    #     original_data_path = args.data_path
    #     args.data_path = args.benchmark_data_path
    #     exp.test(setting, test=1, flag='benchmark')
    #     args.data_path = original_data_path  # Restore
else:
    setting = f'{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.period_len}'
    
    exp = Exp(args)

    # Test on regular test set
    print(f'>>>>>>>testing on TEST set: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1, flag='test')

print("\n✓ All testing completed!")
