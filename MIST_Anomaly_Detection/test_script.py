"""
Simple test script for Informer model with seq_len=1
Uses your saved scaler.pkl for normalization.
"""
import os
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib

# Import your experiment class
import sys
sys.path.insert(0, '.')
from exp.exp_classification import Exp_Classification


def initialize_model(checkpoint_path, args):
    """Initialize the model from checkpoint"""
    exp = Exp_Classification(args)
    exp.model = exp._build_model()
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_dict = exp.model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    exp.model.load_state_dict(model_dict)
    exp.model.eval()
    return exp


def main():
    parser = argparse.ArgumentParser(description='Test Informer model row-by-row')
    
    parser.add_argument('--checkpoint_path', type=str, default='/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/AD_models/checkpoints/Generated_SparseTSF_Generated/checkpoint.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--test_csv', type=str, default='/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Data/Scanrios_2/Scenario3_Madrid/new_madrid/new_madrid200_50%.csv',
                        help='Path to test CSV file')
    parser.add_argument('--scaler_path', type=str, default='/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/AD_models/scalers/veremi_scaler.pkl',
                        help='Path to scaler.pkl')
    parser.add_argument('--output_path', type=str, default='/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Data/Generated_data_2/Attacks',
                        help='Path to save results')
    parser.add_argument('--label_col', type=str, default='label',
                        help='Label column name')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    feature_cols = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']
    if not all(col in pd.read_csv(args.test_csv).columns for col in feature_cols):
        feature_cols = ['PosX', 'PosY', 'SpdX', 'SpdY', 'AclX', 'AclY', 'HedX', 'HedY']
    print(f"Using feature columns: {feature_cols}")

    
    # Set device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Model args
    model_args = Namespace(
        device=device,
        use_multi_gpu=False,
        use_gpu=args.use_gpu,
        model="SparseTSF",
        task_name="classification",
        data="veremi",
        model_id="verClssSingle",
        root_path=os.path.dirname(args.test_csv),
        data_path=os.path.basename(args.test_csv),
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=1,
        period_len=1,
        label_len=0,
        pred_len=0,
        enc_in=8,
        dec_in=8,
        c_out=2,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        decomp_method="moving_avg",
        num_workers=0,
        itr=1,
        train_epochs=7,
        batch_size=256,
        patience=5,
        learning_rate=0.0001,
        des="test",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        top_k=5,
        num_classes=2,
    )
    
    # Load scaler
    print(f"\nLoading scaler from: {args.scaler_path}")
    scaler = joblib.load(args.scaler_path)
    print(f"  Mean: {scaler.mean_}")
    print(f"  Std:  {scaler.scale_}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint_path}")
    exp = initialize_model(args.checkpoint_path, model_args)
    exp.args = model_args
    exp.model = exp.model.to(device)
    print("✓ Model loaded")
    
    # Load test data
    print(f"\nLoading test data from: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    print(f"✓ Loaded {len(df)} rows")
    
    # Extract features and labels
    X = df[feature_cols].values.astype(np.float32)
    y = df[args.label_col].values.astype(np.int64) if args.label_col in df.columns else None
    
    # Scale features using your scaler
    X_scaled = scaler.transform(X)
    
    # Run inference row by row
    print(f"\nRunning inference...")
    predictions = []
    probabilities = []
    
    exp.model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(X_scaled)), desc="Predicting"):
            # Shape: (1, 1, 8) - batch=1, seq_len=1, features=8
            batch_x = torch.FloatTensor(X_scaled[i:i+1]).unsqueeze(0).to(device)
            padding_mask = torch.ones(1, 1).float().to(device)
            
            # Model forward
            outputs = exp.model(batch_x)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
            
            predictions.append(pred)
            probabilities.append(probs.cpu().numpy()[0])
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    print(f"\n✓ Done! {len(predictions)} predictions")
    
    # Print results
    if y is not None:
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(y, predictions)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Normal  Malicious")
        print(f"Actual Normal    {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"Actual Malicious {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    
    df['Prediction'] = predictions
    df['Prob_Normal'] = probabilities[:, 0]
    df['Prob_Malicious'] = probabilities[:, 1]
    
    output_file = os.path.join(args.output_path, 'temp2.csv')
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")


if __name__ == "__main__":
    main()