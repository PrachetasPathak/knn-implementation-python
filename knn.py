import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from matplotlib.colors import ListedColormap

# Load dataset, select features, and split
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configurations
k_values, metrics = [5, 10], ['euclidean', 'cosine']
cmap_light, cmap_bold = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']), ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("KNN Classification on Iris Dataset", fontsize=16)

# Loop through combinations
for i, k in enumerate(k_values):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        if metric == 'cosine':
            X_train_dist, X_test_dist = cosine_distances(X_train), cosine_distances(X_test, X_train)
            knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed').fit(X_train_dist, y_train)
            y_pred, X_vis = knn.predict(X_test_dist), cosine_distances(X, X_train)
        else:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X_train, y_train)
            y_pred, X_vis = knn.predict(X_test), X

        # Metrics
        acc, prec, rec, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), recall_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='weighted')
        
        # Decision boundary
        xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1), np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1))
        Z = knn.predict(cosine_distances(np.c_[xx.ravel(), yy.ravel()], X_train) if metric == 'cosine' else np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_light)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

        # Confusion matrix inset
        cm = confusion_matrix(y_test, y_pred)
        inset_ax = fig.add_axes([ax.get_position().x1 - 0.12, ax.get_position().y1 - (0.18 if i else 0.12), 0.1, 0.1])
        inset_ax.matshow(cm, cmap='Blues')
        for (k_l, k_m), val in np.ndenumerate(cm):
            inset_ax.text(k_m, k_l, f'{val}', ha='center', va='center')
        inset_ax.set_xticks([]), inset_ax.set_yticks([])

        # Titles and labels
        ax.set_title(f'k={k}, Metric={metric}\nAcc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}, F1={f1:.2f}')
        ax.set_xlabel(iris.feature_names[0]), ax.set_ylabel(iris.feature_names[1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
