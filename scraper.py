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

def get_fund_data():
    """
    抓取目標基金的最新淨值與日期
    """
    print(f'正在抓取基金 {config.TARGET_FUND["name"]} 的最新淨值...')
    try:
        res = fetch_url(config.TARGET_FUND['url'])
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # 根據 climb-warm.ipynb 的邏輯定位日期與淨值
        date_tag = soup.select('.ywm_fi_sec')
        if len(date_tag) < 3:
            print("⚠️ 無法定位基金日期標籤")
            return None
            
        date_text = date_tag[2].text.strip() # 格式通常為 YYYY/MM/DD
        
        data_tag = soup.select('.ywm_fi_cell')
        price_element = data_tag[1].find('h4', {'class': 'red'})
        if not price_element:
            print("⚠️ 無法定位基金淨值標籤")
            return None
            
        price_text = price_element.text.strip().replace('TWD', '').replace(',', '').strip()
        price = float(price_text)
        
        print(f'Fund Data Success: {date_text}, Value: {price}')
        return {'date': date_text, 'net_value': price}
    except Exception as e:
        print(f"Fund Fetch Failed: {e}")
        return None

def get_fund_history():
    """
    抓取目標基金的歷史淨值
    使用台北富邦的歷史資料介面
    """
    print(f'Fetching History for {config.TARGET_FUND["name"]}...')
    # 台北富邦歷史淨值 URL
    history_url = "https://fund.taipeifubon.com.tw/w/wr/wr02_ACPS02-0603.djhtm"
    
    try:
        res = fetch_url(history_url)
        res.encoding = 'big5'
        soup = BeautifulSoup(res.text, 'html.parser')
        
        history_data = []
        # 尋找所有表格並篩選包含關鍵字的
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            if not rows:
                continue
                
            # 檢查標題列是否包含日期與淨值
            header_text = rows[0].get_text()
            if '日期' in header_text and '淨值' in header_text:
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        date_str = cols[0].get_text().strip()
                        price_str = cols[1].get_text().strip().replace(',', '')
                        
                        # 驗證日期格式 YYYY/MM/DD
                        if '/' in date_str and len(date_str) >= 8:
                            try:
                                # 嘗試轉換日期確保格式正確
                                valid_date = datetime.strptime(date_str, '%Y/%m/%d').strftime('%Y/%m/%d')
                                price_val = float(price_str)
                                history_data.append({'date': valid_date, 'net_value': price_val})
                            except ValueError:
                                continue
        
        # 去重
        seen_dates = set()
        unique_history = []
        for item in history_data:
            if item['date'] not in seen_dates:
                unique_history.append(item)
                seen_dates.add(item['date'])
        
        print(f'Successfully fetched {len(unique_history)} fund history records')
        return unique_history
    except Exception as e:
        print(f"Fund History Fetch Failed: {e}")
        return []

def save_fund_data(data):
    """
    儲存基金資料 (支援單筆 dict 或多筆 list)
    """
    if not data:
        return
    
    if isinstance(data, dict):
        data = [data]
    
    utils.ensure_dir_exists(config.FUND_DATA_DIR)
    file_path = os.path.join(config.FUND_DATA_DIR, f"{config.TARGET_FUND['id']}.csv")
    
    new_df = pd.DataFrame(data)
    
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        # 合併並去重
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['date'], keep='last')
        combined_df = combined_df.sort_values('date', ascending=True)
        combined_df.to_csv(file_path, index=False)
    else:
        new_df = new_df.sort_values('date', ascending=True)
        new_df.to_csv(file_path, index=False)
        
    print(f'SAVE FUND: {file_path} (Total: {len(pd.read_csv(file_path))})')

def get_stock_data(date_str, stock_code):
    """
    抓取證交所個股日成交資訊
    """
    print(f'Fetching {stock_code} at {date_str}...')
    
    url = config.BASE_URL_PATTERN.format(date_str, stock_code)
    
    try:
        res = fetch_url(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        thead = soup.find('thead')
        if thead is None:
            print(f"Warning: No table header - {stock_code} {date_str}")
            return None
        
        title_rows = thead.find_all('tr')
        if not title_rows:
            return None
            
        columns = [th.text.strip() for th in title_rows[-1].find_all(['th', 'td'])]
        
        datalist = []
        tbody = soup.find('tbody')
        if tbody:
             for row in tbody.find_all('tr'):
                cols = [col.text.strip() for col in row.find_all('td')]
                if len(cols) == len(columns):
                    datalist.append(cols)
        
        if not datalist:
            return None

        df = pd.DataFrame(datalist, columns=columns)
        
        if '日期' in df.columns:
            df['日期'] = df['日期'].apply(utils.transform_date)
            
        print(f'Stock {stock_code} {date_str} Success')
        return df
        
    except Exception as e:
        print(f"Fetch Failed {stock_code} {date_str}: {e}")
        return None

def save_stock_data(df, stock_code):
    """
    將股票 DataFrame 儲存為 CSV 檔案
    """
    if df is None or df.empty:
        return

    utils.ensure_dir_exists(config.STOCK_DATA_DIR)
    
    file_name = f"{stock_code}{config.WATCH_STOCKS.get(stock_code, '')}.csv"
    file_path = os.path.join(config.STOCK_DATA_DIR, file_name)
    
    mode = 'a' if os.path.exists(file_path) else 'w'
    header = not os.path.exists(file_path)
    
    try:
        if mode == 'a':
            existing_data = pd.read_csv(file_path)
            if not df.empty and df['日期'].iloc[0] in existing_data['日期'].values:
                print(f'INFO: {stock_code} Duplicate date, skipping')
                return

        df.to_csv(file_path, mode=mode, header=header, index=False)
        print(f'SAVE STOCK: {stock_code} DONE')
        
    except Exception as e:
        print(f"Error saving stock: {e}")

def main():
    # 1. 抓取觀察標的股票資料
    today = datetime.today()
    # 縮短時間範圍以進行測試：從 2025 年 11 月開始抓取
    target_dates = utils.generate_date_list(2025, 11, today.year, today.month)
    watch_stocks = list(config.WATCH_STOCKS.keys())
    
    print(f"Starting Scraper...")
    for stock in watch_stocks:
        for date_str in target_dates:
            df = get_stock_data(date_str, stock)
            save_stock_data(df, stock)
            time.sleep(3) 

    # 2. 抓取目標基金資料
    # 先嘗試抓取歷史資料以確保資料量足夠訓練
    fund_history = get_fund_history()
    if fund_history:
        save_fund_data(fund_history)
    
    # 再抓取最新一筆 (確保當日最新)
    fund_latest = get_fund_data()
    if fund_latest:
        save_fund_data(fund_latest)


if __name__ == "__main__":
    main()