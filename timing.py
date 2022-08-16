# This script times the runtimes of

# Import modules
import os, time, pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statistics

# Linear regression
linear_runtimes = []
linear = LinearRegression()
for i in range(100):
    start = time.process_time()
    linear.fit(X_train, y_train)
    end = time.process_time()
    linear_runtimes.append(end - start)
statistics.mean(linear_runtimes)
statistics.stdev(linear_runtimes)
    
# Plot
plot = sns.kdeplot(linear_runtimes)
plot.set_title('Distribution of model runtimes')
plot.set_xlabel('Runtime (seconds)')

