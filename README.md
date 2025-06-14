# 投資商品AI預測專案

## 主要目標客群
想投資的上班族、想增加退休金的退休人士

## 想解決的問題
投資者在下單前，容易因市場資訊過載、自身缺乏專業判斷與驗證機制，以及受情緒干擾影響，最後做出錯誤的策略判斷，從而產生焦慮。

因此，本專案透過以下服務，讓投資者遠離焦慮：

 - 利用 AI 預測資產淨值走勢。
 - 即時驗證投資策略報酬。
 - 將策略結果推播至行動裝置。

## 最小可行性產品(MVP)
### 功能預覽

step 1 使用者選定指定基金中的成分股與其預測賣出時間。
 
step 2 系統自動下載成分股歷史報價資料。

step 3 AI 模型預測未來賣出時間的淨值。

step 4 系統判斷是否可進場。

step 5 透過dc機器人推播給使用者，內容包含是否可進場和其策略現實執行的報酬率。

### 📸 實作畫面

![系統實作畫面](./operation.png)


## 📊 功能開發進度

### 投資預測與策略系統
- 投資預測 模組
  - &#10004; 資料擷取與預處理
  - &#10004; 模型訓練
  - &#10004; git action 部屬
  - - [ ] 資料來源加廣(新聞、財報、分析師等)
  - - [ ] LLM模型預測(關鍵字分析、情緒分析等)加權

### 使用者互動與介面系統
- 使用者互動介面 模組
  - - [ ] 策略選擇(勾選)
  - - [ ] 使用 gradio 介面化
  - - [ ] 廣告位（底部橫幅）
  - - [ ] 策略結果折線圖

### 資料庫維護與管理系統
- 資料庫 模組
  - &#10004; 使用json檔紀錄虛擬買入的投報率
  - - [ ] 儲存使用者資料(帳號、email、登入資訊、權限、點數餘額)
  - - [ ] 個人化策略紀錄
  - - [ ] 點數使用紀錄
  - - [ ] 簽到記錄
  - - [ ] 任務完成紀錄
  - - [ ] 廣告管理
  - - [ ] 系統公告
  - - [ ] 異常紀錄
  - - [ ] 報酬率排行榜（計算、排名、發獎）
- 系統資源與推播 模組
  - &#10004; discord機器人推播

### 點數與會員成長系統
- 點數獎勵 模組
  - - [ ] 每日簽到功能
  - - [ ] 社群分享回報 API（分享至 X/Facebook）
  - - [ ] 點數消費與解鎖

