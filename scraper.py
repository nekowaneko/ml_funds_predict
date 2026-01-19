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
        
        print(f'基金抓取成功: {date_text}, 淨值: {price}')
        return {'date': date_text, 'net_value': price}
    except Exception as e:
        print(f"基金抓取失敗: {e}")
        return None

def save_fund_data(data):
    """
    儲存基金資料
    """
    if not data:
        return
    
    utils.ensure_dir_exists(config.FUND_DATA_DIR)
    file_path = os.path.join(config.FUND_DATA_DIR, f"{config.TARGET_FUND['id']}.csv")
    
    df = pd.DataFrame([data])
    mode = 'a' if os.path.exists(file_path) else 'w'
    header = not os.path.exists(file_path)
    
    if mode == 'a':
        existing = pd.read_csv(file_path)
        if data['date'] in existing['date'].values:
            print('ℹ️ 基金資料已存在，不重複寫入')
            return
            
    df.to_csv(file_path, mode=mode, header=header, index=False)
    print('💾 基金資料寫入完成！')

def get_stock_data(date_str, stock_code):
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
            print(f"警告: 無法找到表格標題 (可能該月無資料或休市) - {stock_code} {date_str}")
            return None
        
        title_rows = thead.find_all('tr')
        if not title_rows:
            return None
            
        # 證交所的表格可能有兩層 tr，我們取最後一層包含實際欄位名稱的
        columns = [th.text.strip() for th in title_rows[-1].find_all(['th', 'td'])]
        
        # 檢查欄位數量是否與資料對齊
        datalist = []
        tbody = soup.find('tbody')
        if tbody:
             for row in tbody.find_all('tr'):
                cols = [col.text.strip() for col in row.find_all('td')]
                if len(cols) == len(columns):
                    datalist.append(cols)
        
        if not datalist:
            return None

        # 建立 DataFrame
        df = pd.DataFrame(datalist, columns=columns)
        
        # 轉換日期格式 (使用 utils 模組)
        if '日期' in df.columns:
            df['日期'] = df['日期'].apply(utils.transform_date)
            
        print(f'股票 {stock_code} {config.WATCH_STOCKS.get(stock_code, "")} {date_str} 資料搜集成功')
        return df
        
    except Exception as e:
        print(f"抓取失敗 {stock_code} {date_str}: {e}")
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
            
        print(f'✅ {stock_code} {config.WATCH_STOCKS.get(stock_code, "")} {date_str} 資料搜集成功')
        return df
        
    except Exception as e:
        print(f"❌ 抓取失敗 {stock_code} {date_str}: {e}")
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
                print(f'ℹ️ {stock_code} 資料已重複，跳過寫入')
                return

        df.to_csv(file_path, mode=mode, header=header, index=False)
        print(f'💾 {stock_code} 寫入完成！')
        
    except Exception as e:
        print(f"❌ 股票存檔錯誤: {e}")

def main():
    # 1. 抓取觀察標的股票資料
    today = datetime.today()
    target_dates = utils.generate_date_list(2023, 1, today.year, today.month)
    watch_stocks = list(config.WATCH_STOCKS.keys())
    
    print(f"開始爬取股票資料...")
    for stock in watch_stocks:
        for date_str in target_dates:
            df = get_stock_data(date_str, stock)
            save_stock_data(df, stock)
            time.sleep(3) 

    # 2. 抓取目標基金資料
    fund_data = get_fund_data()
    save_fund_data(fund_data)


if __name__ == "__main__":
    main()