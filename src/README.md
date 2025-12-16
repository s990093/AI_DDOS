3. 程序將計算特徵的熵和斜率，並生成相應的可視化圖表。

## 函數說明

### `calculate_feature_entropy(data, feature, window_size)`

計算給定特徵在滑動時間窗口內的熵值。

- **參數**:
  - `data` (pd.DataFrame): 包含特徵的輸入數據。
  - `feature` (str): 要計算熵的特徵列名。
  - `window_size` (int): 滑動時間窗口的大小。
- **返回**: pd.Series: 對應於時間窗口的熵值序列。

### `calculate_slope(data, feature)`

計算特徵隨時間的斜率。

- **參數**:
  - `data` (pd.DataFrame): 包含特徵的輸入數據。
  - `feature` (str): 要計算斜率的特徵列名。
- **返回**: pd.Series: 計算出的特徵斜率。

### `visualize_entropy_distribution(data, features, label_column, benign_label, attack_label)`

可視化正常和攻擊流量的多個特徵熵的分佈。

- **參數**:

  - `data` (pd.DataFrame): 包含特徵和標籤的數據。
  - `features` (list): 要可視化的特徵列表。
  - `label_column` (str): 標籤列名。
  - `benign_label` (str): 正常流量的標籤。
  - `attack_label` (str): 攻擊流量的標籤。

- **返回**: None: 顯示熵分佈的圖表。

### `visualize_entropy_and_slope(data, features)`

可視化多個特徵的熵和斜率動態。

- **參數**:

  - `data` (pd.DataFrame): 包含熵和斜率特徵的數據。
  - `features` (list): 要可視化的熵特徵列表。

- **返回**: None: 顯示熵和斜率動態的圖表。

### `visualize_normality(data, features)`

可視化熵特徵的正態性測試和正態分佈擬合。

- **參數**:

  - `data` (pd.DataFrame): 包含熵特徵的數據。
  - `features` (list): 要測試和可視化的熵特徵列表。

- **返回**: None: 顯示正態性測試和分佈擬合的圖表。
