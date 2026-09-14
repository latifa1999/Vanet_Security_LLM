import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from sklearn.preprocessing import LabelEncoder
import joblib  # More robust than pickle for sklearn objects


warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            #self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)


        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        if self.features == 'S':
            df_raw = df_raw.iloc[:, -1:]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class CustomDataset(Dataset):
    """
    Custom Dataset for Time Series Classification
    Features: posx, posy, spdx, spdy, aclx, acly, hedx, hedy
    Target: label
    """
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', scale=True, timeenc=0, freq='h', scaler_path=None):
        self.seq_len = size[0] if size else 96
        self.features = features
        self.scale = scale
        self.flag = flag
        self.scaler_path = scaler_path
        
        self.root_path = root_path
        self.data_path = data_path

        print(f"\n{'='*80}")
        print(f"Loading {flag.upper()} Dataset")
        print(f"{'='*80}")
        print(f"Data path: {root_path}/{data_path}")
        print(f"Sequence length: {self.seq_len}")
        print(f"Features: {features}")
        print(f"Scaling: {scale}")
        
        self.__read_data__()
        
    def __read_data__(self):
        print(f"\nReading data...")
        df_raw = pd.read_csv(f'{self.root_path}/{self.data_path}')

        # take 1000 samples
        # df_raw = df_raw.sample(n=100000, random_state=42)

        # shuffle the data to ensure randomness
        df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Raw data shape: {df_raw.shape}")
        print(f"Columns: {list(df_raw.columns)}")
        print(f"Total rows: {len(df_raw)}")

        # Check for missing values
        missing = df_raw.isnull().sum()
        if missing.sum() > 0:
            print(f"\nWarning: Found missing values:")
            print(missing[missing > 0])
        else:
            print(f"✓ No missing values found")
        
        # Feature columns
        feature_cols = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']
        
        # ===== STEP 1: Extract RAW features and labels =====
        data = df_raw[feature_cols].values  # RAW data
        labels = df_raw['label'].values

        print(f"Feature matrix shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # ===== STEP 2: Calculate split boundaries on RAW data =====
        # We need to account for the fact that sequences will be created
        # So total_rows - seq_len + 1 = number of possible sequences
        total_rows = len(data)
        
        # Split ratios
        train_ratio = 0.7
        val_ratio = 0.1
        test_ratio = 0.2
        
        # Calculate row indices for splits (not sequence indices)
        train_rows = int(total_rows * train_ratio)
        val_rows = int(total_rows * val_ratio)
        # test_rows = remaining
        
        print(f"\n{'='*80}")
        print(f"SPLIT BOUNDARIES (on RAW data)")
        print(f"{'='*80}")
        print(f"Total rows: {total_rows}")
        print(f"Train rows: 0 to {train_rows}")
        print(f"Val rows: {train_rows} to {train_rows + val_rows}")
        print(f"Test rows: {train_rows + val_rows} to {total_rows}")
        
        # ===== STEP 3: Split the RAW data =====
        if self.flag == 'train':
            data_split = data[:train_rows]
            labels_split = labels[:train_rows]
        elif self.flag == 'val':
            data_split = data[train_rows:train_rows + val_rows]
            labels_split = labels[train_rows:train_rows + val_rows]
        else:  # test or benchmark or any external data
            # For test: use the test split
            # For external: use ALL data from external file
            if 'benchmark' in self.data_path or 'external' in self.data_path:
                # External data - use all of it
                data_split = data
                labels_split = labels
                print(f"Using EXTERNAL data (all rows)")
            else:
                # Internal test split
                data_split = data[train_rows + val_rows:]
                labels_split = labels[train_rows + val_rows:]
        
        print(f"\nSplit '{self.flag}' size: {len(data_split)} rows")
        
        # ===== STEP 4: Handle Scaling =====
        if self.scale:
            print(f"\n{'='*80}")
            print(f"SCALING STRATEGY")
            print(f"{'='*80}")
            
            scaler_loaded = False
            
            # Try to load existing scaler
            if self.scaler_path and os.path.exists(self.scaler_path):
                print(f"Attempting to load scaler from: {self.scaler_path}")
                try:
                    self.scaler = joblib.load(self.scaler_path)
                    
                    # Validate scaler
                    if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                        print(f"✓ Scaler loaded successfully")
                        print(f"  Scaler mean: {self.scaler.mean_[:3]}... (first 3 features)")
                        print(f"  Scaler std: {self.scaler.scale_[:3]}... (first 3 features)")
                        scaler_loaded = True
                    else:
                        print(f"⚠ WARNING: Loaded scaler is not properly fitted")
                        scaler_loaded = False
                        
                except Exception as e:
                    print(f"⚠ WARNING: Failed to load scaler: {e}")
                    print(f"  Error type: {type(e).__name__}")
                    scaler_loaded = False
            
            if not scaler_loaded and self.flag == 'train':
                # TRAIN: Fit scaler ONLY on training split
                print(f"Training mode: Fitting NEW scaler on training data ONLY")
                self.scaler = StandardScaler()
                
                # ===== KEY: fit_transform on training data =====
                data_split = self.scaler.fit_transform(data_split)
                
                print(f"  Fitted on {len(data_split)} training rows")
                print(f"  Scaler mean: {self.scaler.mean_[:3]}... (first 3 features)")
                print(f"  Scaler std: {self.scaler.scale_[:3]}... (first 3 features)")
                
                # Save scaler for later use
                if self.scaler_path:
                    os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
                    
                    try:
                        joblib.dump(self.scaler, self.scaler_path, compress=3)
                        
                        # Verify the save worked
                        test_load = joblib.load(self.scaler_path)
                        
                        if np.allclose(test_load.mean_, self.scaler.mean_):
                            print(f"✓ Scaler saved and verified at: {self.scaler_path}")
                        else:
                            print(f"⚠ WARNING: Scaler verification failed!")
                    except Exception as e:
                        print(f"⚠ WARNING: Failed to save scaler: {e}")
            
            elif scaler_loaded:
                # VAL/TEST/BENCHMARK: Use existing scaler (transform only)
                print(f"Using existing scaler for {self.flag} data")
                
                # ===== KEY: transform only (no fit) =====
                data_split = self.scaler.transform(data_split)
                print(f"✓ Data transformed using training scaler")
            
            else:
                # ERROR: Need scaler but don't have one
                raise ValueError(
                    f"ERROR: Testing on {self.flag} data but no valid scaler found!\n"
                    f"Scaler path: {self.scaler_path}\n"
                    f"File exists: {os.path.exists(self.scaler_path) if self.scaler_path else False}\n"
                    f"\nYou must:\n"
                    f"  1. Delete corrupted/invalid scaler: rm {self.scaler_path}\n"
                    f"  2. Retrain model to create new scaler\n"
                    f"This is CRITICAL to prevent data leakage!"
                )
            
            print(f"{'='*80}")
        
        # ===== STEP 5: Create sequences from the (now scaled) split data =====
        print(f"\nCreating sequences...")
        self.data_x = []
        self.data_y = []
        
        # Calculate how many sequences we can create from this split
        num_sequences = len(data_split) - self.seq_len + 1
        print(f"Can create {num_sequences} sequences from {len(data_split)} rows")
        
        for i in range(num_sequences):
            seq = data_split[i:i+self.seq_len]
            label = labels_split[i+self.seq_len-1]  # Use last label in sequence
            self.data_x.append(seq)
            self.data_y.append(label)
        
        self.data_x = np.array(self.data_x)
        self.data_y = np.array(self.data_y)

        print(f"\n{self.flag} data shape: {self.data_x.shape}, labels shape: {self.data_y.shape}")
        
        # Show sample statistics
        if self.scale:
            print(f"\nSample sequence statistics (after scaling):")
            print(f"  Mean: {self.data_x[0].mean(axis=0)[:3]}... (first 3 features)")
            print(f"  Std: {self.data_x[0].std(axis=0)[:3]}... (first 3 features)")
        
        print(f"{'='*80}")
        print(f"✓ {self.flag.upper()} dataset loaded successfully!")
        print(f"  Total sequences: {len(self)}")
        print(f"  Input shape per sample: {self.data_x[0].shape}")
        print(f"  Number of classes: {len(np.unique(self.data_y))}")
        print(f"  Class distribution: {np.bincount(self.data_y)}")
        print(f"{'='*80}\n")
    
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def __len__(self):
        return len(self.data_x)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) if self.scale else data