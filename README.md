# DSAI HW1

**重要**: 有新增 argument，請使用'python app.py --training 2021.csv --training2 2020to2021.csv --output submission.csv'執行。

### Data preprocessing

對於[台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)資料做的處理如下
- 將**台電燃煤**('林口#1', '林口#2', '林口#3', '台中#1', '台中#2', '台中#3', '台中#4', '台中#5','台中#6', '台中#7', '台中#8', '台中#9', '台中#10', '興達#1', '興達#2', '興達#3', '興達#4','大林#1', '大林#2')、**核能發電**('核一#1(萬瓩)', '核一#2(萬瓩)', '核二#1(萬瓩)', '核二#2(萬瓩)', '核三#1','核三#2')、**燃氣發電**('大潭 (#1-#6)', '通霄 (#1-#6)', '興達 (#1-#5)', '南部 (#1-#4)', '大林(#5-#6)')資料整合
- 將資料標準化至(0 ,1)區間
---
- 將[台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)以及[台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850)根據時間連接起來(2020/01/01~2021/03/22)
- 只取**備轉容量**做為訓練使用

### Data analysis

- 觀察**備轉容量**隨著時間的變化
- 觀察**民生用電**、**工業用電**隨著時間的變化
- 觀察**備轉容量**與**民生用電**、**工業用電**的關係
- 觀察**備轉容量**與**台電燃煤**、**核能**、**燃氣**的關係

![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/2019.png)

可以發現備轉容量隨著時間的變化起伏很大

![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/1.png)
![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/2.png)

上圖為工業用電，下圖為民生用電。可以發現兩者相比的話， 工業用電相對穩定。

![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/3.png)
從此圖可以看出民生用電和備轉容量呈正相關。若是將其他圖畫出來，則會發現沒有那麼明顯的趨勢。所以模型不考慮這些columns，決定單純的使用過去的備轉容量來預測未來的備轉容量。  

於是我將 **2020/01/01 - 2021/01/31** 還有 **2021/02/01 - 2021/03/22**的資料合併，並且只取日期和備轉容量，做為我的訓練資料。 
![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/2020.png)

### Model

- 使用基礎的 AR 和 ARIMA 模型
- 使用 Grid Search 找到最佳的模型

![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/AR.png)
使用 AR 預測的話，Test RMSE為93.920。
![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/ARIMA.png)
使用 ARIMA 預測的話，Test RMSE為79.250。

ARIMA 的效果較好，所以使用 ARIMA 做為預測備轉容量的模型。
![image](https://github.com/P76094046/DSAI_HW1/blob/main/image/forecast.png)
預測出的結果如上。


```python

```
