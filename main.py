import argparse
import sys
from datetime import datetime

# 導入自定義模組
import config
import scraper
import train_model
import predict
import notifier

def run_pipeline(do_train=False):
    """
    執行完整工作流
    
    參數:
        do_train (bool): 是否在預測前執行模型訓練
    """
    print("==========================================")
    print(f"🚀 啟動基金/股票預測自動化工作流 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========================================\n")

    # 1. 爬取最新資料
    print("--- 步驟 1: 更新股票資料 ---")
    try:
        scraper.main()
    except Exception as e:
        print(f"❌ 爬蟲執行失敗: {e}")
        # 如果爬蟲失敗，視情況決定是否繼續 (若舊資料可用)
    
    # 2. 模型訓練 (可選)
    if do_train:
        print("\n--- 步驟 2: 重新訓練模型 ---")
        try:
            train_model.main()
        except Exception as e:
            print(f"❌ 訓練執行失敗: {e}")
    else:
        print("\n--- 步驟 2: 跳過訓練 (使用現有模型) ---")

    # 3. 執行預測與發送通知
    print("\n--- 步驟 3: 執行預測與通知 ---")
    results = []
    
    try:
        is_buy, conf, last_date = predict.predict_fund_signal()
        fund_name = config.TARGET_FUND['name']
        
        if last_date:
            signal_emoji = "🔴" if is_buy else "🟢"
            signal_text = "建議進場 (看漲)" if is_buy else "建議觀望 (看跌/盤整)"
            
            result_str = (
                f"{signal_emoji} {fund_name}\n"
                f"資料對齊日期: {last_date.strftime('%Y/%m/%d')}\n"
                f"訊號: {signal_text}\n"
                f"信心: {conf:.1%}"
            )
            results.append(result_str)
            print(result_str)
            print("-" * 20)
            
    except Exception as e:
        print(f"❌ 預測 {config.TARGET_FUND['name']} 時發生錯誤: {e}")

    # 4. 發送匯總通知
    if results:
        summary_msg = "\n\n".join(results)
        header = f"\n📊 【每日股票預測報告】 {datetime.now().strftime('%Y/%m/%d')}\n"
        full_msg = header + summary_msg
        
        print("\n📤 正在發送 Discord 通知...")
        status, resp = notifier.send_message(full_msg)
        if status == 200:
            print("✅ 通知發送成功")
        else:
            print(f"❌ 通知發送失敗 (Code: {status})")
    else:
        print("\n⚠️ 沒有產生任何預測結果，未發送通知。")

    print("\n==========================================")
    print("🏁 工作流執行結束")
    print("==========================================")

if __name__ == "__main__":
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='執行股票預測自動化工作流')
    parser.add_argument('--train', action='store_true', help='是否執行模型訓練')
    
    args = parser.parse_args()
    
    run_pipeline(do_train=args.train)
