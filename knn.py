#K - nearest neighbours is a supervised learning algorithm (thus it is trained 
#on labeled data).
#It is used for classification/regression problems.
#HOW IT WORKS: It decides on the classification of a data point based on the classification of 
#its K-nearest points (neighbours)
#K is chosen by the user.
#It is based on distances.

import numpy as np
import sklearn
import sklearn.datasets as ds
import sklearn.model_selection as ms
import sklearn.neighbors as nb
import matplotlib.pyplot as plt
import matplotlib

digits = ds.load_digits()
X = digits.data
y = digits.target
print((X.min(), X.max()))
print(X.shape)

nrows, ncols = 2, 5
fig, axes = plt.subplots(nrows, ncols, figsize=(6,3))

for i in range(nrows):
	for j in range(ncols):
		# Image index
		k = j + i * ncols
		ax = axes[i,j]
		ax.matshow(digits.images[k, ...], cmap = plt.cm.gray)
		ax.set_axis_off()
		ax.set_title(digits.target[k])
plt.show()

(X_train, X_test, y_train, y_test) = \
    ms.train_test_split(X, y, test_size=.25)
knc = nb.KNeighborsClassifier()
knc.fit(X_train, y_train)

knc.score(X_test, y_test)

# Let's draw a 1.
one = np.zeros((8, 8))
one[1:-1, 4] = 16  # The image values are in [0, 16].
one[2, 3] = 16
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.imshow(one, interpolation='none',
          cmap=plt.cm.gray)
ax.grid(False)
ax.set_axis_off()
ax.set_title("One")
plt.show()
# We need to pass a (1, D) array.
knc.predict(one.reshape((1, -1)))
