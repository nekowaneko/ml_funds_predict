import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 導入自定義模組
import config
import utils

# 重試裝飾器：針對網路錯誤進行指數退避重試
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.RequestException)
)
def fetch_url(url):
    """
    發送 GET 請求並包含錯誤重試機制
    """
    response = requests.get(url, headers=config.HEADERS, timeout=10)
    response.raise_for_status()
    return response

def get_data(date_str, stock_code):
    """
    抓取證交所個股日成交資訊
    
    參數:
        date_str (str): 日期字串，格式為 YYYYMM01
        stock_code (str): 股票代碼
        
    回傳:
        pd.DataFrame: 包含該月份成交資訊的 DataFrame，若無資料則回傳 None
    """
    print(f'正在抓取 {stock_code} 於 {date_str} 的資料...')
    
    # 格式化目標網址
    url = config.BASE_URL_PATTERN.format(date_str, stock_code)
    
    try:
        res = fetch_url(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # 尋找表格標題與內容
        thead = soup.find('thead')
        if thead is None:
            print(f"⚠️ 警告: 無法找到表格標題 (可能該月無資料或休市) - {stock_code} {date_str}")
            return None
        
        title_row = thead.find('tr')
        if title_row is None:
            return None
            
        # 修正：標題列通常使用 <th> 標籤，而非 <td>
        columns = [th.text.strip() for th in title_row.find_all(['th', 'td'])]
        
        datalist = []
        tbody = soup.find('tbody')
        if tbody:
             for row in tbody.find_all('tr'):
                datalist.append([col.text.strip() for col in row.find_all('td')])
        
        if not datalist:
            return None

        # 建立 DataFrame
        df = pd.DataFrame(datalist, columns=columns)
        
        # 轉換日期格式 (使用 utils 模組)
        if '日期' in df.columns:
            df['日期'] = df['日期'].apply(utils.transform_date)
            
        print(f'✅ {stock_code} {config.SYMBOL_DICT.get(stock_code, "")} {date_str} 資料搜集成功')
        return df
        
    except Exception as e:
        print(f"❌ 抓取失敗 {stock_code} {date_str}: {e}")
        return None

def save_to_csv(df, stock_code):
    """
    將 DataFrame 儲存為 CSV 檔案
    
    參數:
        df (pd.DataFrame): 要儲存的資料
        stock_code (str): 股票代碼，用於生成檔名
    """
    if df is None or df.empty:
        return

    # 確保資料夾存在 (使用 utils 模組)
    utils.ensure_dir_exists(config.DATA_DIR)
    
    file_name = f"{stock_code}{config.SYMBOL_DICT.get(stock_code, '')}.csv"
    file_path = os.path.join(config.DATA_DIR, file_name)
    
    mode = 'a' if os.path.exists(file_path) else 'w'
    header = not os.path.exists(file_path)
    
    try:
        # 如果檔案存在，檢查重複日期以避免寫入重複資料
        if mode == 'a':
            try:
                existing_data = pd.read_csv(file_path)
                # 檢查新資料的第一筆日期是否已存在於舊資料中
                if not df.empty and df['日期'].iloc[0] in existing_data['日期'].values:
                    print('ℹ️ 資料檢查結果：有重複日期，不寫入')
                    return
            except Exception as read_err:
                 print(f"⚠️ 讀取現有檔案時發生錯誤 (可能檔案損毀)，將嘗試直接寫入: {read_err}")

        df.to_csv(file_path, mode=mode, header=header, index=False)
        print('💾 寫入完成！')
        
    except Exception as e:
        print(f"❌ 存檔錯誤: {e}")

def main():
    # 設定爬取範圍：從 2023 年 1 月到當前月份
    today = datetime.today()
    current_year = today.year
    current_month = today.month
    
    target_dates = utils.generate_date_list(2023, 1, current_year, current_month)
    # 從設定檔讀取目標股票清單
    target_stocks = list(config.SYMBOL_DICT.keys())
    
    print(f"開始爬取任務: {len(target_dates)} 個月份 x {len(target_stocks)} 支股票")
    
    for stock in target_stocks:
        for date_str in target_dates:
            df = get_data(date_str, stock)
            save_to_csv(df, stock)
            time.sleep(2) # 禮貌性延遲，避免對伺服器造成過大負擔

if __name__ == "__main__":
    main()