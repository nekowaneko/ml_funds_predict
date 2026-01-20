import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 導入自定義模組
import config
import utils

# 定義 TCN 模型
# Input shape: (Batch, Window_Size, Features)
class TCN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TCN, self).__init__()
        # input_size: 視窗大小 (config.WINDOW_SIZE)
        # 這裡假設 Features 數量為標的股票數
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 動態計算 Flatten 後的維度
        # 輸入形狀: (Batch, Window_Size, Num_Features)
        # Conv1d 作用在 Num_Features 維度上
        self.fc1 = None 
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # x shape: (Batch, Window, Features)
        x = self.conv1(x) 
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
            
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_and_align_data():
    """
    載入基金與所有觀察標的股票資料並進行時間對齊
    """
    # 載入基金資料
    fund_path = os.path.join(config.FUND_DATA_DIR, f"{config.TARGET_FUND['id']}.csv")
    if not os.path.exists(fund_path):
        print(f"⚠️ 找不到基金資料: {fund_path}")
        return None
    
    fund_df = pd.read_csv(fund_path)
    fund_df = fund_df.rename(columns={'net_value': 'target_val', 'date': 'date'})
    fund_df['date'] = pd.to_datetime(fund_df['date'])
    
    merged_df = fund_df[['date', 'target_val']]
    
    # 載入並合併觀察標的股票
    for stock_code, name in config.WATCH_STOCKS.items():
        file_name = f"{stock_code}{name}.csv"
        stock_path = os.path.join(config.STOCK_DATA_DIR, file_name)
        
        if os.path.exists(stock_path):
            stock_df = pd.read_csv(stock_path)
            # 證交所資料欄位: ['日期', '成交股數', ..., '收盤價', ...]
            # 確保欄位正確
            col_map = {'日期': 'date', '收盤價': f'close_{stock_code}'}
            stock_df = stock_df.rename(columns=col_map)
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            
            # 只取日期與收盤價
            stock_df = stock_df[['date', f'close_{stock_code}']]
            
            # 清理收盤價中的逗號
            stock_df[f'close_{stock_code}'] = stock_df[f'close_{stock_code}'].astype(str).str.replace(',', '').astype(float)
            
            merged_df = pd.merge(merged_df, stock_df, on='date', how='inner')
        else:
            print(f"⚠️ 缺少觀察標的資料: {name} ({stock_code})")
            
    return merged_df.sort_values('date').reset_index(drop=True)

def preprocess_data(df):
    """
    資料前處理：排除日期，標準化所有特徵
    """
    features = df.drop(columns=['date', 'target_val'])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return pd.DataFrame(scaled_features, columns=features.columns), scaler

def create_labels(df, window_size):
    """
    根據基金淨值 (target_val) 建立標籤
    """
    vals = df['target_val'].values
    labels = []
    # 預測 N 天後的淨值是否高於當前
    for i in range(len(vals) - window_size):
        if vals[i + window_size] > vals[i]:
            labels.append(1)
        else:
            labels.append(0)
    return labels

def train_fund_model():
    """
    訓練基金預測模型
    """
    print(f"🚀 開始訓練基金預測模型: {config.TARGET_FUND['name']}")
    
    df = load_and_align_data()
    if df is None or len(df) < 50:
        print(f"⚠️ 資料不足，無法訓練。 (目前筆數: {len(df) if df is not None else 0})")
        return

    # 前處理
    clean_features, scaler = preprocess_data(df)
    window_size = config.WINDOW_SIZE
    
    x_data = []
    labels = create_labels(df, window_size)
    
    for i in range(len(labels)):
        window = clean_features.iloc[i : i + window_size]
        x_data.append(window.values)
        
    x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    
    # 分割資料
    x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(x_train, F.one_hot(y_train, 2).float())
    test_dataset = TensorDataset(x_test, F.one_hot(y_test, 2).float())
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = TCN(window_size, 2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 訓練
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 評估模式 (Evaluation Mode)
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # 計算準確度
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(targets.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if (epoch + 1) % 10 == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            accuracy = correct / total if total > 0 else 0
            print(f'Epoch [{epoch+1}/{config.EPOCHS}]')
            print(f'  Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
            print(f'  Test Accuracy: {accuracy:.2%}')

    # 儲存
    utils.ensure_dir_exists(config.MODEL_SAVE_DIR)
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'features': list(clean_features.columns)
    }, os.path.join(config.MODEL_SAVE_DIR, 'fund_model.pth'))
    print("✅ 基金預測模型訓練完成並已儲存。")

def main():
    train_fund_model()

if __name__ == "__main__":
    main()