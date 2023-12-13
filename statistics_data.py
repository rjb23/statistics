#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a seed for reproducibility
np.random.seed(42)

# Scenario 1: Wage data
age = np.linspace(20, 70, 100)
wage_age = 10000 + 200 * age - 10 * (age - 60) ** 2 + np.random.normal(scale=5000, size=len(age))

years = np.arange(2003, 2010)
wage_year = 30000 + 1500 * (years - 2003) + np.random.normal(scale=5000, size=len(years))

education_levels = np.random.randint(1, 6, size=500)  # Assuming 500 data points
wage_education = 20000 + 5000 * education_levels + np.random.normal(scale=3000, size=len(education_levels))

# Scenario 2: S&P index data
sp_index_changes = np.random.normal(scale=1, size=1000)  # Assuming 1000 data points

# Calculate percentage changes for 2 and 3 days previous
sp_index_changes_2days = np.roll(sp_index_changes, 2)
sp_index_changes_3days = np.roll(sp_index_changes, 3)

# Create figures
plt.figure(figsize=(15, 5))

# Figure 1.1
plt.subplot(1, 3, 1)
sns.scatterplot(x=age, y=wage_age)
plt.title('Wage as a function of age')

plt.subplot(1, 3, 2)
sns.lineplot(x=years, y=wage_year)
plt.title('Wage as a function of year')

plt.subplot(1, 3, 3)
sns.boxplot(x=education_levels, y=wage_education)
plt.title('Wage as a function of education')

# Figure 1.2
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x=np.where(sp_index_changes > 0, 'Increase', 'Decrease'), y=sp_index_changes)
plt.title('Percentage change in S&P index (1 day)')

plt.subplot(1, 3, 2)
sns.boxplot(x=np.where(sp_index_changes_2days > 0, 'Increase', 'Decrease'), y=sp_index_changes_2days)
plt.title('Percentage change in S&P index (2 days previous)')

plt.subplot(1, 3, 3)
sns.boxplot(x=np.where(sp_index_changes_3days > 0, 'Increase', 'Decrease'), y=sp_index_changes_3days)
plt.title('Percentage change in S&P index (3 days previous)')

plt.tight_layout()
plt.show()


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set a seed for reproducibility
np.random.seed(42)

# Generate synthetic data for a binary classification task
X = np.random.normal(scale=1, size=(500, 2))
y = np.random.choice([0, 1], size=500, p=[0.5, 0.5])

# Ensure a balanced distribution of classes
while np.sum(y == 1) < 2 or np.sum(y == 0) < 2:
    X = np.random.normal(scale=1, size=(500, 2))
    y = np.random.choice([0, 1], size=500, p=[0.5, 0.5])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Quadratic Discriminant Analysis model
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, y_train)

# Predict probabilities for the test data
y_pred_prob = qda_model.predict_proba(X_test)[:, 1]

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred_prob > 0.5)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[18]:


import matplotlib.pyplot as plt
import numpy as np

# Create a list of the predicted probabilities
predicted_probabilities = [0.46, 0.48, 0.50, 0.52]

# Create a boxplot of the predicted probabilities
plt.boxplot(predicted_probabilities)

# Set the x-axis labels
plt.xticks([1, 2, 3, 4], ["0.46", "0.48", "0.50", "0.52"])

# Set the y-axis label
plt.ylabel("Predicted Probability")

# Set the title
plt.title("Predicted Probability of Market Decrease")

# Show the plot
plt.show()


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Set a seed for reproducibility
np.random.seed(42)

# Generate synthetic data with four clusters representing different types of cancer
X, y = make_blobs(n_samples=64, n_features=6830, centers=4, cluster_std=1.0, random_state=42)

# Apply PCA to reduce dimensionality to two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the data in a two-dimensional space
plt.figure(figsize=(12, 5))

# Left panel: Colors represent different clusters
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('Clusters Based on Synthetic Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Right panel: Colors represent different cancer types
cancer_types = np.random.choice(range(14), size=64)
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cancer_types, cmap='tab20')
plt.title('Cancer Types Based on Synthetic Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Create a legend for the right panel
legend_labels = [f'Cancer Type {i}' for i in range(14)]
plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title='Cancer Types', loc='upper right')

plt.tight_layout()
plt.show()


# In[ ]:




