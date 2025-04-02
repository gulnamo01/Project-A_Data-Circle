print('Hello World')
import pandas as pd

#open the datasets
submission = pd.read_csv("C:\\Users\\Gulnamo\\Desktop\\Redi School\\Data Circle\\Project-A_Data-Circle\\submission_format.csv")
print(submission)
test_features = pd.read_csv("C:\\Users\\Gulnamo\\Desktop\\Redi School\\Data Circle\\Project-A_Data-Circle\\test_set_features.csv")
print(test_features)
train_features = pd.read_csv("C:\\Users\\Gulnamo\\Desktop\\Redi School\\Data Circle\\Project-A_Data-Circle\\training_set_features.csv")
print(train_features)
train_labels = pd.read_csv("C:\\Users\\Gulnamo\\Desktop\\Redi School\\Data Circle\\Project-A_Data-Circle\\training_set_labels.csv")
print (train_labels)

#merge the X and Y training datasets 
train_data = train_features.merge(train_labels, on='respondent_id')
train_data.info()
train_data.head()

pd.set_option("display.max_columns", None)

print("Dataset dimensions:", train_data.shape)
print("Column types:\n", train_data.dtypes)
print("Missing values:\n", train_data.isnull().sum())


##Ticket 1.1.3: Create summary statistics for all features
#Generate statistics for numerical and categorical features
print("Numerical Features Summary:")
print(train_data.describe())
print("\nCategorical Features Summary:")
print(train_data.describe(include="object"))

#Check target variable distributions and class balance
# class distribution
print("H1N1 Vaccine Distribution:")
print(train_data["h1n1_vaccine"].value_counts())

print("\nSeasonal Vaccine Distribution:")
print(train_data["seasonal_vaccine"].value_counts())
#percentage distribution
print("H1N1 Vaccine Percentage:")
print(train_data["h1n1_vaccine"].value_counts(normalize=True))

print("\nSeasonal Vaccine Percentage:")
print(train_data["seasonal_vaccine"].value_counts(normalize=True))
