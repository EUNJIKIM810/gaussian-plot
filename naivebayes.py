import pandas as pd

# 데이터셋 로딩
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
df = pd.read_csv(url, sep='\t', header=None, names=["label", "message"])

# 데이터 확인
print(df.head())
