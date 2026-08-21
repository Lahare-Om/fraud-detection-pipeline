import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/creditcard.csv')

print(df.shape)
print (df['Class'].value_counts())
print(df.isnull().sum().sum())


sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0= Legit, 1= Fraud)')
plt.savefig('notebooks/class_distribution.png')
plt.show()

print(df.groupby('Class')['Amount'].describe())