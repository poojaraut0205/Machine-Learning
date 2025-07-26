import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
data = pd.read_csv("used_cars_data.csv")
data['Brand'] = data.Name.str.split().str.get(0)
data['Model'] = data.Name.str.split().str.get(1) + data.Name.str.split().str.get(2)
print(data[['Name','Brand','Model']])
print(data.Brand.unique())
print(data.Brand.nunique())
searchfor = ['Isuzu' ,'ISUZU','Mini','Land']
print(data[data.Brand.str.contains('|'.join(searchfor))].head(5))
data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper","Land":"Land Rover"}, inplace=True)
print(data.describe().T)
print(data.describe(include='all').T)
cat_cols=data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)

#graphs
#for col in num_cols:
   # print(col)
    #print('Skew :', round(data[col].skew(), 2))
    #plt.figure(figsize = (15, 4))
    #plt.subplot(1, 2, 1)
    #data[col].hist(grid=False)
    #plt.ylabel('count')
    #plt.subplot(1, 2, 2)
    #sns.boxplot(x=data[col])
    #plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

for col in num_cols:
    print(col)
    print('Skew :', round(data[col].skew(), 2))
    sns.histplot(data[col], kde=True, bins=30, label=col, element='step', stat='density', common_norm=False, alpha=0.5)

plt.title('Histograms with KDE for Numeric Columns')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

skew_values = [round(data[col].skew(), 2) for col in num_cols]

plt.figure(figsize=(12, 6))
plt.bar(num_cols, skew_values, color='purple', alpha=0.7)
plt.title('Skewness of Numeric Columns')
plt.ylabel('Skewness')
plt.xlabel('Columns')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.show()
import matplotlib.pyplot as plt

# Calculate absolute skewness values
skew_values = [abs(data[col].skew()) for col in num_cols]

plt.figure(figsize=(8, 8))
plt.pie(skew_values, labels=num_cols, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Proportion of Absolute Skewness by Numeric Column')
plt.show()
