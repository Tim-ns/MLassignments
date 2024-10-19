cor_matrix = raw_train_data.corr()

plt.imshow(cor_matrix, cmap='Oranges')
plt.colorbar()
variables = []
for i in cor_matrix.columns:
    variables.append(i)

plt.xticks(range(len(cor_matrix)), variables, rotation=45, ha='right')
plt.yticks(range(len(cor_matrix)), variables)

plt.show()

# Remove self correlations, drop dublicate (set to NaN value)

np.fill_diagonal(cor_matrix.values, np.nan)
sorted_matrix = cor_matrix.unstack().sort_values(ascending=False)
sorted_matrix = sorted_matrix.drop_duplicates()
print(sorted_matrix) 

# Sort by absolute value
pairs_cor = sorted_matrix.abs().sort_values(ascending=False)

# Search for weak correlations 

weak_cors = pairs_cor[pairs_cor < 0.15]

weak_cors_np = [[]]
weak_cors_np.pop(0)
weak_cors_np.append([weak_cors['v_x_1']])
weak_cors_np.append([weak_cors['v_y_1']])
weak_cors_np.append([weak_cors['v_x_2']])
weak_cors_np.append([weak_cors['v_y_2']])
weak_cors_np.append([weak_cors['v_x_3']])
weak_cors_np.append([weak_cors['v_y_3']])

print(weak_cors_np)
