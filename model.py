import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('ld_clean')
print(df.head())

x = df['lyrics']  # features
y = df['genre']  # labels

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, stratify=y)
y_train.value_counts(normalize=True)  # imbalanced dataset (83% rock, 17% rap)

# Handling imbalanced dataset
oversampled = RandomOverSampler(sampling_strategy=0.5, random_state=1)  # minority oversampling
x_over, y_over = oversampled.fit_resample(x, y)

