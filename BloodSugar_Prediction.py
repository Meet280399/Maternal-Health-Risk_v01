## Meet Trusarbhai Patel - 202407335
## Dhruvi Nilesh Thakkar - 202401915
## Jyotsna Joshy - 202404936


# Importing pandas library
import pandas as pd

# Loading the dataset for data preprocessing
df = pd.read_csv("processed_dataset.csv")
print(df.head())

# Dropping RiskLevel column for setting the target as BloodSugar
df_drop_RiskLevel = df.drop(columns=["RiskLevel"])
print(df_drop_RiskLevel.head())

# Removing the duplicate rows
df_unique = df.drop_duplicates()

# Getting output in CVS format and will be stored in the name as BloodSugar_Prediction.csv
df_unique.to_csv('BloodSugar_Prediction.csv', index=False)

# Dropping BloodSugar column for setting the target as RiskLevel
df_drop_BloodSugar = df.drop(columns=["BS"])
print(df_drop_BloodSugar.head())

# Getting output in CVS format and will be stored in the name as BloodSugar_Impact.csv
df_drop_BloodSugar.to_csv('BloodSugar_Impact.csv', index=False)