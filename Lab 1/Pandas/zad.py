import pandas as pd
import numpy as np

# 1 

# 1.1
# data = {
#     'A': pd.Series([1, 2, 3]),
#     'B': pd.Series([4, 5, 6])
# }

# df = pd.DataFrame(data)
# print(df)

# 1.2
# data2 = {
#     'A': np.array([1, 2, 3]),
#     'B': ([4, 5, 6])
# }

# df2 = pd.DataFrame(data2)
# print(df2)

# 1.3
# data = [
#     {'A': 1, 'B': 4},
#     {'A': 2, 'B': 5},
#     {'A': 3, 'B': 6}
# ]

# df = pd.DataFrame(data)
# print(df)


# 1.4

# data = {
#     'A': (1, 2, 3),
#     'B': (4, 5, 6)
# }

# df = pd.DataFrame(data)
# print(df)


# 2

index1 = pd.Index([1,2,3])
index2 = pd.Index([1,3,4])
# - difference,
# result = index1.difference(index2)
# print(result)

# - intersection,
# result = index1.intersection(index2)
# print(result)

# - union,
# result = index1.union(index2)
# print(result)

# - isin,
# result = index1.isin(index2)
# print(result)

# - delete,
# index_delete = pd.Index([1,2,3])

# result = index_delete.delete(2)
# print(result)

# - drop,
# index_drop = pd.Index([1,21,2,3,69])

# result = index_drop.drop(2)
# print(result)

# - insert,
# result = index1.insert(2,6)
# print(result)

# - is_monotonic,
# index = pd.Index([1, 2, 3, 4, 5])
# result = index.is_monotonic_increasing
# print(result)

# - is_unique,
# result = index1.is_unique
# print(result)

# - unique
# result = index1.unique()
# result.insert(0,60)
# print(result)

# 3
# s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
# new_index = ['s', 'c', 'd']

# s_reindexed = s.reindex(new_index)
# print(s_reindexed)


# 4
df = pd.read_csv("Book_Dataset_1.csv") 
if list(df.columns) == ['Unnamed']:
    df = df.drop(columns=['Unnamed'])


df.rename(columns={'Avilability' : 'Availability'}, inplace=True)


df = df[['Category', 'Title'] + [col for col in df.columns if col not in ['Category', 'Title']]]
same_type_columns = df.dtypes.groupby(df.dtypes).apply(list)
# print(same_type_columns)


df['Total_Price'] = df['Price'] * df['Availability']
total_price = df['Total_Price'].sum()
# print(f"Całkowita cena książek: {total_price}")

df = df.sort_values(by=['Category', 'Availability', 'Title'])

nan_check = df.isnull().sum()
# print(nan_check)


df.set_index(df['Title'].str[:3], inplace=True)
duplicates = df.index[df.index.duplicated()]
# print(duplicates.value_counts())

tax_values = {category: np.random.randint(0, 21) for category in df['Category'].unique()}
def adjust_tax(row):
    tax = tax_values[row['Category']]
    if row['Availability'] < 5:
        return tax // 2
    return tax

df['Tax_amount'] = df.apply(adjust_tax, axis=1)

df['Price after tax'] = (df['Price'] * (1 + df['Tax_amount'] / 100)).round(2)


historical_books = df[(df['Category'] == 'History') & (df['Price after tax'] > 35)]
# print(historical_books[['Title', 'Number_of_reviews', 'Price after tax']])
avg_price_history_books = historical_books['Price'].mean().round(2)
# print(avg_price_history_books)

historical_books['Price'] = avg_price_history_books
historical_books['Price after tax'] = (historical_books['Price'] * (1 + historical_books['Tax_amount'] / 100)).round(2)

df['Expected_Tax_Revenue'] = (df['Price'] * df['Availability'] * (df['Tax_amount'] / 100)).round(2)
df = df[~df['Category'].isin(['Horror', 'Mystery'])]

long_description_books = df[(df['Book_Description'].str.len() > 2000) & (df['Title'].str.len() <= 20)]
print(long_description_books[['Title', 'Book_Description']])

df.to_csv('new_books.csv', index=False)
