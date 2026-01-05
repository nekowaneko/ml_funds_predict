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
from . import config
from . import utils

# 定義 TCN 模型
# Input shape: (Batch, Window_Size, Features)
class TCN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TCN, self).__init__()
        # input_size 這裡對應 config.WINDOW_SIZE
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 經過 Conv1d 和 MaxPool1d 後的維度計算:
        # Input: (N, 10, 7)
        # Conv1d: (N, 64, 7) (padding=1, kernel=3 維持長度)
        # MaxPool1d: (N, 64, 3) (7 // 2 = 3)
        # Flatten: 64 * 3 = 192 (注意：這取決於 Feature 數量，這裡假設為 7)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # 展平 (Flatten)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def preprocess_data(df):
    """
    資料前處理：清洗與標準化
    
    參數:
        df (pd.DataFrame): 原始股票資料
        
    回傳:
        pd.DataFrame: 處理後的資料
    """
    # 排除不需要的欄位
    columns_to_exclude = ['index', 'date', 'change', '日期', '漲跌', '漲跌幅(%)']
    # 只保留存在於 df 的欄位
    cols_to_drop = [c for c in columns_to_exclude if c in df.columns]
    data_to_normalize = df.drop(columns=cols_to_drop)
    
    # 確保數值型別 (移除逗號並轉為浮點數)
    for column in data_to_normalize.columns:
        if data_to_normalize[column].dtype == 'object':
             data_to_normalize[column] = data_to_normalize[column].str.replace(',', '').astype(float)
        else:
             data_to_normalize[column] = data_to_normalize[column].astype(float)
             
    # 標準化處理 (Features)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_to_normalize)
    clean_data = pd.DataFrame(normalized_data, columns=data_to_normalize.columns)
    
    return clean_data

def create_labels(df, window_size):
    """
    建立標籤：比較當前淨值與未來淨值
    
    參數:
        df (pd.DataFrame): 原始股票資料
        window_size (int): 預測的時間跨度
        
    回傳:
        list: 標籤列表 (1: 上漲, 0: 下跌或持平)
    """
    # 鎖定收盤價欄位
    target_col = 'close' if 'close' in df.columns else '收盤價'
    
    prices = None
    if target_col in df.columns:
         if df[target_col].dtype == 'object':
            prices = df[target_col].str.replace(',', '').astype(float).values
         else:
            prices = df[target_col].values
    else:
        # 如果找不到明確的收盤價欄位，嘗試使用最後一欄
        prices = df.iloc[:, -1].values

    labels = []
    length = len(prices)
    
    # 產生標籤：如果 N 天後的價格 > 當前價格，則標記為 1
    for i in range(length - window_size):
        if prices[i + window_size] > prices[i]:
            labels.append(1)
        else:
            labels.append(0)
            
    return labels

def train_model(stock_code):
    """
    訓練指定股票的預測模型
    """
    print(f"🚀 開始訓練模型: {stock_code}")
    
    file_name = f"{stock_code}{config.SYMBOL_DICT.get(stock_code, '')}.csv"
    file_path = os.path.join(config.DATA_DIR, file_name)
    
    # 1. 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到資料檔：{file_path}，跳過訓練。")
        return

    try:
        # 讀取 CSV
        df = pd.read_csv(file_path)
        
        # 簡單映射欄位名稱 (如果 CSV 沒有 Header)
        if len(df.columns) == 10:
             df.columns = ['index', 'date', 'volume', 'amount', 'open', 'high', 'low', 'close', 'change', 'transactions']
        
        # 2. 檢查資料量是否足夠
        if len(df) < 50:
            print(f"⚠️ 資料不足 ({len(df)} 筆)，跳過訓練。")
            return
            
        # 資料前處理
        clean_data = preprocess_data(df)
        
        # 準備訓練資料 (Sliding Window)
        window_size = config.WINDOW_SIZE
        x_data = []
        labels = create_labels(df, window_size) 
        
        # 確保資料長度一致
        valid_length = len(labels)
        
        if valid_length < window_size:
             print("⚠️ 有效資料長度不足以建立 Window，跳過。")
             return

        for i in range(valid_length):
             window = clean_data.iloc[i : i + window_size]
             # 檢查特徵數量是否符合模型預期 (例如 7 個特徵)
             if window.shape[1] != 7:
                 # 這裡可以加入動態調整模型或報錯的邏輯
                 pass
             x_data.append(window.values)
             
        # 轉為 Tensor
        x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        
        # 分割資料集
        x_train, x_temp, y_train, y_temp = train_test_split(x_tensor, y_tensor, test_size=0.2, random_state=42)
        
        # 準備 DataLoader
        batch_size = config.BATCH_SIZE
        train_dataset = TensorDataset(x_train, F.one_hot(y_train, num_classes=2).float())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        input_size = window_size
        output_size = 2
        model = TCN(input_size, output_size)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # 訓練迴圈
        num_epochs = config.EPOCHS
        print(f"開始訓練 {num_epochs} Epochs...")
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # 定期輸出訓練狀態
            if (epoch + 1) % 10 == 0:
                print(f'Stock {stock_code} | Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
                
        print(f"✅ 模型 {stock_code} 訓練完成！")
        
        # 儲存模型
        utils.ensure_dir_exists(config.MODEL_SAVE_DIR)
        model_path = os.path.join(config.MODEL_SAVE_DIR, f'model_{stock_code}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"💾 模型已儲存至: {model_path}")

    except Exception as e:
        print(f"❌ 訓練過程發生未預期錯誤 ({stock_code}): {e}")

def main():
    target_stocks = list(config.SYMBOL_DICT.keys())
    for stock in target_stocks:
        train_model(stock)

if __name__ == "__main__":
    main()