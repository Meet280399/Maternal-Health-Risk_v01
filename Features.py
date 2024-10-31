## Meet Trusarbhai Patel - 202407335
## Dhruvi Nilesh Thakkar - 202401915
## Jyotsna Joshy - 202404936


# Importing pandas library
import pandas as pd

# Loading the dataset for data preprocessing
df = pd.read_csv("processed_dataset.csv")
# print(df.head())

# Selecting the mean for the feature Systolic BP
systolic_means = df["SystolicBP"].mean()

# Displaying the mean for the feature Systolic BP
print("Mean for Systolic Blood Pressure:- " + str(systolic_means))

# Selecting the mean for the feature Diastolic BP
diastolic_means = df["DiastolicBP"].mean()

# Displaying the mean for the feature Diastolic BP
print("Mean for Diastolic Blood Pressure:- " + str(diastolic_means))

# Selecting the mean for the feature Age
age_means = df["Age"].mean()

# Displaying the mean for the feature Age
print("Mean for Age:- " + str(age_means))

# Selecting the mean for the feature Heart Rate
heartrate_means = df["HeartRate"].mean()

# Displaying the mean for the feature Heart Rate
print("Mean for Heart Rate:- " + str(heartrate_means))

# Selecting the mean for the feature Body Temperature
bodytemp_means = df["BodyTemp"].mean()

# Displaying the mean for the feature Body Temperature
print("Mean for Body Temperature:- " + str(bodytemp_means))

# Selecting the mean for the feature Blood Sugar
bs_means = df["BS"].mean()

# Displaying the mean for the feature Blood Sugar
print("Mean for Blood Sugar:- " + str(bs_means))

# Selecting the standard deviation for the feature Systolic BP
systolic_std = df["SystolicBP"].std()

# Displaying the standard deviation for the feature Systolic BP
print("Standard Deviation for Systolic Blood Pressure:- " + str(systolic_std))

# Selecting the standard deviation for the feature Diastolic BP
diastolic_std = df["DiastolicBP"].std()

# Displaying the standard deviation for the feature Diastolic BP
print("Standard Deviation for Diastolic Blood Pressure:- " + str(diastolic_std))

# Selecting the standard deviation for the feature Age
age_std = df["Age"].std()

# Displaying the standard deviation for the feature Age
print("Standard Deviation for Age:- " + str(age_std))

# Selecting the standard deviation for the feature Heart Rate
heartrate_std = df["HeartRate"].std()

# Displaying the standard deviation for the feature Heart Rate
print("Standard Deviation for Heart Rate:- " + str(heartrate_std))

# Selecting the standard deviation for the feature Body Temperature
bodytemp_std = df["BodyTemp"].std()

# Displaying the standard deviation for the feature Body Temperature
print("Standard Deviation for Body Temperature:- " + str(bodytemp_std))

# Selecting the standard deviation for the feature Blood Sugar
bs_std = df["BS"].std()

# Displaying the standard deviation for the feature Blood Sugar
print("Standard Deviation for Blood Sugar:- " + str(bs_std))
