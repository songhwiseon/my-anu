import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('employee_data.csv')
print(df)

print(df.head())
print(df.info())
print("--------------------------------")
print(df.describe())



# 이름 데이터 추출
names = df['이름']
print(names.head())

older_than_30 = df[df['나이'] > 30]
print(older_than_30.head())

print("--------------------------------")

grouped_df = df.groupby('부서')['나이'].mean()
print(grouped_df)

df['연령대'] = df['나이'].apply(lambda x: '30대' if 30 <= x < 40 else '30대 이하' if x < 30 else '40대 이상')

df.to_csv('modified_employee_data.csv', index=False)