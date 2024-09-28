#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
file_path = r'C:\Users\ramya\Downloads\perception_data_modified (1).csv'

try:
    data = pd.read_csv(file_path, sep=';')
    print("File loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")
except Exception as e:
    raise Exception(f"Error loading file: {e}")

print("Data preview:")
print(data.head())
print("Data shape:", data.shape)

# Inspect the last column specifically
print("Last column sample values:", data.iloc[:, -1].head())
features = data.iloc[:, 1:-1].values 
labels = data.iloc[:, -1].astype(str)

# Split the labels by semicolon and convert to integers
def convert_labels(label_str):
    return [int(label) for label in label_str.split(';')]

# Apply the conversion to each row in the label column
labels_split = labels.apply(convert_labels)

# Convert list of labels to a DataFrame
y = pd.DataFrame(labels_split.tolist())

# Ensure y is a numpy array and flatten if necessary
y = np.array(y)
if y.ndim > 1:
    y = y[:, 0]

# Convert labels to integers
y = y.astype(int)

# Determine number of classes
num_classes = np.max(y) + 1
print(f"Number of classes: {num_classes}")

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)
scaler = MinMaxScaler()
try:
    X = scaler.fit_transform(features)
except ValueError as ve:
    print(f"ValueError during scaling: {ve}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  
model.add(Dense(128, activation='relu'))  
model.add(Dense(64, activation='relu'))   
model.add(Dense(32, activation='relu'))   
model.add(Dense(num_classes, activation='softmax'))  

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.save('my_trained_model.h5')


history = model.fit(X_train, y_train, epochs=100, batch_size=250, validation_split=0.3)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')


# In[4]:


from tensorflow.keras.models import load_model

# Load the model
model = load_model('my_trained_model.h5')
print("Model loaded successfully.")


# In[5]:


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')


# In[7]:


predicted_classes = np.argmax(predictions, axis=1)


# In[8]:


import pandas as pd
import numpy as np

# Assuming X_test and y_test are already defined and preprocessed
# Example: X_test is a numpy array of shape (num_samples, num_features)
# Example: y_test is a numpy array of shape (num_samples, num_classes)

# Convert X_test to a DataFrame with appropriate column names
feature_columns = [f'feature_{i}' for i in range(X_test.shape[1])]
X_test_df = pd.DataFrame(X_test, columns=feature_columns)

# Convert y_test to a DataFrame
# If y_test is one-hot encoded, you can convert it to a single column of class labels
y_test_labels = np.argmax(y_test, axis=1)
y_test_df = pd.DataFrame(y_test_labels, columns=['label'])

# Combine the feature DataFrame and the label DataFrame
test_data_df = pd.concat([X_test_df, y_test_df], axis=1)

# Save to CSV
csv_file_path = 'test_data.csv'
test_data_df.to_csv(csv_file_path, index=False)

print(f'Test data saved to {csv_file_path}')


# In[ ]:




