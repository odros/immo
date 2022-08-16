import matplotlib.pyplot as plt

errors = [linear_rmse, poisson_rmse, tree_rmse, forest_rmse, support_rmse, perceptron_rmse, ensamble_rmse]
runtimes = [linear_runtime, poisson_time, tree_runtime, forest_runtime, support_runtime, perceptron_runtime, ensemble_runtime]
models = ['linear', 'poisson', 'tree', 'forest', 'support', 'perceptron', 'ensemble']

fig, ax = plt.subplots()
ax.scatter(errors, runtimes)
ax.annotate(models[0], (errors[0]-0.0007, runtimes[0]))
ax.annotate(models[1], (errors[1]+0.00008, runtimes[1]+20))
ax.annotate(models[2], (errors[2]-0.0005, runtimes[2]))
ax.annotate(models[3], (errors[3]+0.0001, runtimes[3]-40))
ax.annotate(models[4], (errors[4]+0.00007, runtimes[4]-30))
ax.annotate(models[5], (errors[5], runtimes[5]+40))
ax.annotate(models[6], (errors[6]+0.0001, runtimes[6]-30))
plt.title('Errors and runtimes for implemented models')
plt.ylabel('Runtime (seconds)')
plt.xlabel('RMSE')

results = pd.DataFrame(list(zip(models, errors, runtimes)), columns =['Model', 'RMSE', 'Runtime'])

os.chdir("/Users/aleph/Desktop/MDS/semestres/2/ml/final/data/")
results.to_csv("results.csv", index = False)

