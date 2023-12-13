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


# In[ ]:




