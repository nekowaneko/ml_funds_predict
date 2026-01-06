import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 導入自定義模組
import config
import utils
from train_model import TCN, preprocess_data 

def load_model(stock_code):
    """
    載入指定股票的訓練模型
    """
    model_path = os.path.join(config.MODEL_SAVE_DIR, f'model_{stock_code}.pth')
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到模型檔案: {model_path}")
        return None
    
    # 初始化模型架構 (參數需與訓練時一致)
    input_size = config.WINDOW_SIZE
    output_size = 2
    model = TCN(input_size, output_size)
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval() # 設定為評估模式
        return model
    except Exception as e:
        print(f"❌ 載入模型失敗 ({stock_code}): {e}")
        return None

def get_latest_data(stock_code, window_size):
    """
    讀取並處理最新的股票數據以供推論使用
    """
    file_name = f"{stock_code}{config.SYMBOL_DICT.get(stock_code, '')}.csv"
    file_path = os.path.join(config.DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到資料檔: {file_path}")
        return None, None
    
    try:
        df = pd.read_csv(file_path)
        
        # 簡單映射欄位名稱
        if len(df.columns) == 10:
             df.columns = ['index', 'date', 'volume', 'amount', 'open', 'high', 'low', 'close', 'change', 'transactions']
             
        # 資料量檢查
        if len(df) < window_size:
            print(f"⚠️ 資料量不足 ({len(df)} < {window_size})，無法進行預測")
            return None, None
            
        # 取得最後 window_size 筆資料的日期 (用於報告)
        last_date = df['date'].iloc[-1] if 'date' in df.columns else "未知日期"
            
        # 資料前處理 (需與訓練時一致)
        clean_data = preprocess_data(df)
        
        # 取最後一段視窗的資料
        input_data = clean_data.iloc[-window_size:].values
        
        # 轉為 Tensor (Batch Size = 1)
        # Shape: (1, Window_Size, Features)
        input_tensor = torch.tensor(np.array([input_data]), dtype=torch.float32)
        
        return input_tensor, last_date
        
    except Exception as e:
        print(f"❌ 讀取資料失敗 ({stock_code}): {e}")
        return None, None

def predict_signal(stock_code):
    """
    對指定股票執行推論，回傳預測結果
    
    回傳:
        tuple: (是否建議進場 bool, 信心分數 float, 最後資料日期 str)
    """
    model = load_model(stock_code)
    if model is None:
        return False, 0.0, None
        
    input_tensor, last_date = get_latest_data(stock_code, config.WINDOW_SIZE)
    if input_tensor is None:
        return False, 0.0, None
        
    with torch.no_grad():
        output = model(input_tensor)
        # 使用 Softmax 取得機率
        probabilities = torch.softmax(output, dim=1)
        # Class 1 代表 "上漲/進場"
        confidence = probabilities[0][1].item()
        prediction = torch.argmax(probabilities, dim=1).item()
        
    is_buy_signal = (prediction == 1)
    return is_buy_signal, confidence, last_date

def main():
    target_stocks = list(config.SYMBOL_DICT.keys())
    
    print("🔎 開始執行預測...")
    for stock in target_stocks:
        buy, conf, date = predict_signal(stock)
        stock_name = config.SYMBOL_DICT.get(stock, stock)
        
        if date:
            signal_str = "🔴 進場 (看漲)" if buy else "🟢 觀望 (看跌/盤整)"
            print(f"股票: {stock} {stock_name}")
            print(f"資料日期: {date}")
            print(f"預測結果: {signal_str} (信心度: {conf:.2%})")
            print("-" * 30)

if __name__ == "__main__":
    main()
