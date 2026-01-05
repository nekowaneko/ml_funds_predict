import requests
from typing import Optional, Tuple, Dict, Any
from . import config

def send_message(msg: str, image_path: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    """
    發送訊息 (可選圖片) 到 Discord 頻道。

    Args:
        msg (str): 要發送的文字訊息。
        image_path (str, optional): 圖片檔案路徑。 Defaults to None.

    Returns:
        Tuple[int, Dict]: HTTP 狀態碼與 API 回傳的 JSON。
    """
    headers = {
        'Authorization': f'Bot {config.DISCORD_TOKEN}'
    }

    # 準備 Payload
    # 注意：若有檔案上傳，Content-Type 不能設為 application/json，requests 會自動處理 multipart/form-data
    data = {
        'content': msg
    }

    files = {}
    if image_path:
        try:
            # Discord API 接收檔案的欄位名稱通常為 'file' 或 'files[0]'
            # 我們在此開啟檔案，並讓 requests 處理關閉 (或手動關閉)
            f = open(image_path, 'rb')
            files = {'file': (image_path.split('/')[-1], f)}
        except FileNotFoundError:
            print(f"⚠️ 警告: 找不到圖片檔案 {image_path}，將僅發送文字。")

    try:
        if files:
            # 發送文字 + 圖片
            response = requests.post(config.DISCORD_API_URL, headers=headers, data=data, files=files)
            files['file'][1].close() # 關閉檔案
        else:
            # 僅發送文字
            response = requests.post(config.DISCORD_API_URL, headers=headers, data=data)
            
        if response.status_code in [200, 201]:
            # print("✅ Discord 通知發送成功")
            return response.status_code, response.json()
        else:
            print(f"❌ Discord 通知發送失敗 (Code: {response.status_code}): {response.text}")
            return response.status_code, response.json()
    
    except requests.RequestException as e:
        print(f"❌ Discord 連線錯誤: {e}")
        if files:
            files['file'][1].close()
        return -1, {'error': str(e)}
