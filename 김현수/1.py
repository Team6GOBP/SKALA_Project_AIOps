import pandas as pd

DATA_PATH = "Adult Census Income.csv"

df = pd.read_csv(DATA_PATH)

print("=== shape ===")
print(df.shape)

print("\n=== dtypes ===")
print(df.dtypes)

print("\n=== head ===")
print(df.head())

print("\n=== 결측치 (NaN) ===")
print(df.isna().sum())

print("\n=== 결측치 ('?' 표기) ===")
print(df.isin(["?"]).sum()[df.isin(["?"]).sum() > 0])

print("\n=== 수치형 컬럼 기술통계 ===")
print(df.describe())

print("\n=== 범주형 컬럼 고유값 개수 ===")
print(df.select_dtypes(include="object").nunique())

print("\n=== 타겟 분포 (income) ===")
print(df["income"].value_counts())
print(df["income"].value_counts(normalize=True))
