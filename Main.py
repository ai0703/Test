import pandas as pd

# Assuming you have uploaded a CSV file named 'your_file.csv'
data = pd.read_csv('datast5.csv')

# Now, you can perform operations on the 'data' DataFrame, including the line you provided.
data.columns = data.columns.str.strip()  # Remove leading/trailing whitespaces from column names
from sklearn.model_selection import train_test_split

# Assuming you have features in your dataset (excluding the label column)
X = data.drop(columns=['Label'])  # Adjust this if your dataset structure is different

# Split your dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a maximum threshold for data truncation (adjust as needed)
max_threshold = 10000  # For example, you can set it to a suitable value

# Truncate the large values in the training and testing sets
X_train_truncated = X_train.copy()
X_train_truncated[X_train_truncated > max_threshold] = max_threshold

X_test_truncated = X_test.copy()
X_test_truncated[X_test_truncated > max_threshold] = max_threshold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Define a maximum threshold for data truncation (adjust as needed)
max_threshold = 10000  # For example, you can set it to a suitable value

# Truncate the large values in the training set
X_train_truncated = X_train.copy()
X_train_truncated[X_train_truncated > max_threshold] = max_threshold

# Truncate the large values in the test set
X_test_truncated = X_test.copy()
X_test_truncated[X_test_truncated > max_threshold] = max_threshold

# Initialize the Naive Bayes model
nb_classifier_truncated = GaussianNB()

# Perform 5-fold cross-validation on the truncated training set
cross_val_scores = cross_val_score(nb_classifier_truncated, X_train_truncated, y_train, cv=5)

# Train the Naive Bayes model on the truncated training set
nb_classifier_truncated.fit(X_train_truncated, y_train)

# Make predictions on the truncated test set
nb_predictions_truncated = nb_classifier_truncated.predict(X_test_truncated)

# Evaluate the Naive Bayes model with cross-validation
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", cross_val_scores.mean())

# Evaluate the Naive Bayes model with truncated data
nb_accuracy_truncated = accuracy_score(y_test, nb_predictions_truncated)
nb_classification_report_truncated = classification_report(y_test, nb_predictions_truncated)
print("Naive Bayes Accuracy (with Data Truncation):", nb_accuracy_truncated)
print("Naive Bayes Classification Report (with Data Truncation):\n", nb_classification_report_truncated)






import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np

# Ensure your labels are encoded as 0s and 1s
y_train_encoded = (y_train == 'Malicious').astype(int)
y_test_encoded = (y_test == 'Malicious').astype(int)

# Define a maximum threshold for data truncation (adjust as needed)
max_threshold = 10000

# Truncate the large values in the training and testing sets
X_train_truncated = X_train.copy()
X_train_truncated[X_train_truncated > max_threshold] = max_threshold

X_test_truncated = X_test.copy()
X_test_truncated[X_test_truncated > max_threshold] = max_threshold

# Reshape the input data for the CNN model (assuming 1D Convolution)
X_train_reshaped_truncated = np.array(X_train_truncated).reshape(X_train_truncated.shape[0], X_train_truncated.shape[1], 1)
X_test_reshaped_truncated = np.array(X_test_truncated).reshape(X_test_truncated.shape[0], X_test_truncated.shape[1], 1)

# Initialize and build the CNN model with truncated data
cnn_model_truncated = Sequential()
cnn_model_truncated.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_truncated.shape[1], 1)))
cnn_model_truncated.add(MaxPooling1D(pool_size=2))
cnn_model_truncated.add(Flatten())
cnn_model_truncated.add(Dense(128, activation='relu'))
cnn_model_truncated.add(Dense(2, activation='softmax'))  # Two output classes (0 and 1)

# Compile the model with sparse categorical cross-entropy loss
cnn_model_truncated.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model with truncated data
cnn_model_truncated.fit(X_train_reshaped_truncated, y_train_encoded, epochs=10, batch_size=64)

# Evaluate the CNN model with truncated data
cnn_loss_truncated, cnn_accuracy_truncated = cnn_model_truncated.evaluate(X_test_reshaped_truncated, y_test_encoded)
print("CNN Accuracy (with Data Truncation):", cnn_accuracy_truncated)


# Save the trained Naive Bayes model
joblib.dump(nb_classifier_truncated, 'naive_bayes_model.pkl')
import joblib
# Save the trained CNN model
cnn_model_truncated.save('cnn_model.h5')

