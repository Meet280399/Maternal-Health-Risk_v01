## Meet Trusarbhai Patel - 202407335
## Dhruvi Nilesh Thakkar - 202401915
## Jyotsna Joshy - 202404936


# Importing pandas library
import pandas as pd

# Loading the dataset for data preprocessing
df = pd.read_csv("processed_dataset.csv")
print(df.head())

# Dropping Systolic Blood Pressure column for setting the target as Risk Level
df_drop_SystolicBP = df.drop(columns=["SystolicBP"])
print(df_drop_SystolicBP.head())

# Getting output in CVS format and will be stored in the name as DiastolicBloodPressure_Prediction.csv
df_drop_SystolicBP.to_csv('DiastolicBloodPressure_Prediction.csv', index=False)

# Dropping Diastolic Blood Pressure column for setting the target as Risk Level
df_drop_DiastolicBP = df.drop(columns=["DiastolicBP"])
print(df_drop_DiastolicBP.head())

# Getting output in CVS format and will be stored in the name as SystolicBloodPressure_Impact.csv
df_drop_DiastolicBP.to_csv('SystolicBloodPressure_Impact.csv', index=False)

# Dropping Blood Pressure column for setting the target as Risk Level
df_drop_BP = df.drop(columns=["DiastolicBP", "SystolicBP"])
print(df_drop_BP.head())

# Getting output in CVS format and will be stored in the name as BloodPressure_Impact.csv
df_drop_BP.to_csv('BloodPressure_Impact.csv', index=False)
