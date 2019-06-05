import ember
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_dir = "./data/ember/"
# this function mmaps the data into memory
X_train, y_train = ember.read_vectorized_features(data_dir, subset="train")

unlabeled_start_idx = 300001
pos_start_idx = 600001
neg_start_idx = 0

num_unlabeled = 5000
num_pos = 1000
num_neg = 1000
X_unlabeled = []
X_pos = []
X_neg = []
y_unlabeled = []
y_pos = []
y_neg = []
for i in np.random.permutation(900000):
    data_point = X_train[i]
    label = y_train[i]
    if label == -1 and len(X_unlabeled)<num_unlabeled:
        X_unlabeled.append(data_point)
        y_unlabeled.append(label)
    elif label == 1 and len(X_pos)<num_pos:
        X_pos.append(data_point)
        y_pos.append(label)
    elif label == 0 and len(X_neg)<num_neg:
        X_neg.append(data_point)
        y_neg.append(label)
    elif len(X_unlabeled)==num_unlabeled and len(X_pos)==num_pos and len(X_neg)==num_neg:
        break

real_num_pos = len(y_pos)
real_num_neg = len(y_neg)
print("original # dimensions", len(X_unlabeled[0]))
print("# unlabeled",len(y_unlabeled))
print("# pos",real_num_pos)
print("# neg",real_num_neg)

print("scaling")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unlabeled)

pca_dimensions = 2
print("pca-ing to ", pca_dimensions, " dimensions")
pca = PCA(n_components=pca_dimensions)
X_pcad = pca.fit_transform(X_scaled)

print("training (kmeans on 2 centers)")
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_pcad)
print("label min", np.min(kmeans.labels_))
print("label max", np.max(kmeans.labels_))

print("testing")
total = real_num_pos+real_num_neg
confusion_matrix = np.zeros((2,2))
pos_scaled = scaler.transform(X_pos)
pos_pcad = pca.transform(pos_scaled)
pos_guess = kmeans.predict(pos_pcad)
neg_scaled = scaler.transform(X_neg)
neg_pcad = pca.transform(neg_scaled)
neg_guess = kmeans.predict(neg_pcad)
guess_matrix = []
guess_matrix.append([[dp[0] for dp in list(zip(pos_pcad,pos_guess)) if dp[1] == 0],[dp[0] for dp in list(zip(pos_pcad,pos_guess)) if dp[1] == 1]])
guess_matrix.append([[dp[0] for dp in list(zip(neg_pcad,neg_guess)) if dp[1] == 0],[dp[0] for dp in list(zip(neg_pcad,neg_guess)) if dp[1] == 1]])
confusion_matrix[0][0] += len(guess_matrix[0][0])
confusion_matrix[0][1] += len(guess_matrix[0][1])
confusion_matrix[1][0] += len(guess_matrix[1][0])
confusion_matrix[1][1] += len(guess_matrix[1][1])
correct = confusion_matrix[0][0]+confusion_matrix[1][1]

print("total",total)
print("correct",correct)
print("confusion_matrix\n",confusion_matrix)

mean = np.mean(X_pcad.transpose()[1])
mean_correct = 0
print("meany", mean)
for dp in pos_pcad:
    if dp[1]>mean:
            mean_correct += 1
for dp in neg_pcad:
    if dp[1]<mean:
            mean_correct += 1
print("y eval", mean_correct, total)

print("creating graphs")
cluster_data = np.array(list(zip(X_pcad,kmeans.labels_)))
group0 = [data_point[0].tolist() for data_point in cluster_data if data_point[1] == 0]
group1 = [data_point[0].tolist() for data_point in cluster_data if data_point[1] == 1]
group0x = np.array(group0).transpose()[0]
group0y = np.array(group0).transpose()[1]
group1x = np.array(group1).transpose()[0]
group1y = np.array(group1).transpose()[1]
plt.figure()
plt.scatter(group0x, group0y, color = 'red')
plt.scatter(group1x, group1y, color = 'blue')
path = './kmeans_clusters.png'
print("saving learning cluster to ", path)
plt.savefig(path)

plt.figure()
group00x = np.array(guess_matrix[0][0]).transpose()[0]
group00y = np.array(guess_matrix[0][0]).transpose()[1]
group01x = np.array(guess_matrix[0][1]).transpose()[0]
group01y = np.array(guess_matrix[0][1]).transpose()[1]
group10x = np.array(guess_matrix[1][0]).transpose()[0]
group10y = np.array(guess_matrix[1][0]).transpose()[1]
group11x = np.array(guess_matrix[1][1]).transpose()[0]
group11y = np.array(guess_matrix[1][1]).transpose()[1]
plt.scatter(group00x, group00y, color = 'red', marker='x', label='benign=0')
plt.scatter(group01x, group01y, color = 'blue', marker='x', label='benign=1')
plt.scatter(group10x, group10y, color = 'orange', marker='+', label='malware=0')
plt.scatter(group11x, group11y, color = 'purple', marker='+', label='malware=1')
plt.title('kmeans malware analysis, true=guessed')
plt.legend()
path = './kmeans_guesses.png'
print("saving guess cluster to ", path)
plt.savefig(path)
