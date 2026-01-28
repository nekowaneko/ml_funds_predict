import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 導入自定義模組
import config
import utils
from train_model import TCN, preprocess_data 

def load_fund_model():
    """
    載入基金訓練模型與標量
    """
    model_path = os.path.join(config.MODEL_SAVE_DIR, 'fund_model.pth')
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到模型檔案: {model_path}")
        return None, None
    
    checkpoint = torch.load(model_path, weight_only=False)
    
    # 初始化模型架構
    input_size = config.WINDOW_SIZE
    output_size = 2
    model = TCN(input_size, output_size)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, checkpoint['scaler']

def get_inference_data(scaler):
    """
    讀取最新資料並準備推論 Tensor
    """
    from train_model import load_and_align_data
    df = load_and_align_data()
    
    if df is None or len(df) < config.WINDOW_SIZE:
        print("⚠️ 累積資料量不足，無法執行預測")
        return None, None
        
    last_date = df['date'].iloc[-1]
    
    # 僅提取特徵欄位並標準化
    features = df.drop(columns=['date', 'target_val'])
    scaled_features = scaler.transform(features)
    
    # 取最後一段視窗
    input_data = scaled_features[-config.WINDOW_SIZE:]
    input_tensor = torch.tensor(np.array([input_data]), dtype=torch.float32)
    
    return input_tensor, last_date

def predict_fund_signal():
    """
    執行基金推論，回傳預測結果
    """
    model, scaler = load_fund_model()
    if model is None:
        return False, 0.0, None
        
    input_tensor, last_date = get_inference_data(scaler)
    if input_tensor is None:
        return False, 0.0, None
        
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][1].item()
        prediction = torch.argmax(probabilities, dim=1).item()
        
    is_buy_signal = (prediction == 1)
    return is_buy_signal, confidence, last_date

def main():
    print(f"🔎 開始執行基金預測: {config.TARGET_FUND['name']}...")
    buy, conf, date = predict_fund_signal()
    
    if date:
        signal_str = "🔴 進場 (看漲)" if buy else "🟢 觀望 (看跌/盤整)"
        print(f"基金: {config.TARGET_FUND['name']}")
        print(f"最後資料對齊日期: {date}")
        print(f"預測結果: {signal_str} (信心度: {conf:.2%})")
        print("-" * 30)

if __name__ == "__main__":
    main()
