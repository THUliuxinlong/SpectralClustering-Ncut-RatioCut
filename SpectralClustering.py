# Unnormalized spectral clustering --- Corresponding RatioCut
# Input: Similarity matrix S ∈ Rn×n, number k of clusters to construct.
# • Construct a similarity graph by one of the ways described in Section 2. Let W be its weighted adjacency matrix.
# • Compute the unnormalized Laplacian L.
# • Compute the first k eigenvectors u1;...;uk of L.
# • Let U ∈ Rn×k be the matrix containing the vectors u1;...;uk as columns.
# • For i = 1;...;n, let yi ∈ Rk be the vector corresponding to the i-th row of U.
# • Cluster the points yi in Rk with the k-means algorithm into clusters C1;...;Ck.
# Output: Clusters A1;...;Ak with Ai = {j| yj ∈ Ci}.

# Normalized spectral clustering according to Shi and Malik (2000) --- Corresponding Ncut
# Input: Similarity matrix S ∈ Rn×n, number k of clusters to construct.
# • Construct a similarity graph by one of the ways described in Section 2. Let W be its weighted adjacency matrix.
# • Compute the unnormalized Laplacian L.
# • Compute the first k generalized eigenvectors u1;...;uk of the generalized eigenproblem Lu = λDu.
# • Let U ∈ Rn×k be the matrix containing the vectors u1;...;uk as columns.
# • For i = 1;...;n, let yi ∈ Rk be the vector corresponding to the i-th row of U.
# • Cluster the points yi in Rk with the k-means algorithm into clusters C1;...;Ck.
# Output: Clusters A1;...;Ak with Ai = {j| yj ∈ Ci}.

# Normalized spectral clustering according to Ng, Jordan, and Weiss (2002) --- Corresponding Ncut
# Input: Similarity matrix S ∈ Rn×n, number k of clusters to construct.
# • Construct a similarity graph by one of the ways described in Section 2. Let W be its weighted adjacency matrix.
# • Compute the normalized Laplacian Lsym.
# • Compute the first k eigenvectors u1;...;uk of Lsym.
# • Let U ∈ Rn×k be the matrix containing the vectors u1;...;uk as columns.
# • Form the matrix T ∈ Rn×k from U by normalizing the rows to norm 1, that is set tij = uij / (Sum_k uik^2)^1/2.
# • For i = 1;...;n, let yi ∈ Rk be the vector corresponding to the i-th row of T.
# • Cluster the points yi with the k-means algorithm into clusters C1;...;Ck.
# Output: Clusters A1;...;Ak with Ai = {j| yj ∈ Ci}.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
# 聚类的评价指标
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# np.set_printoptions(threshold=np.inf)


class SpectralClustering:
    def __init__(self, n_cluster, method='normalized_Shi', criterion='k_nearest', sigma=2.0, epsilon=70, k=5):
        self.N = None  # 数据点的个数
        self.n_cluster = n_cluster  #聚类的数量
        self.method = method  # 规范化或非规范化的谱聚类算法
        self.criterion = criterion  # 相似度图的构建方法
        self.sigma = sigma  # 高斯相似度中的sigma参数
        self.epsilon = epsilon  # epsilon-近邻方法的参数
        self.k = k  # k近邻方法的参数

        self.W = None  # 图的相似性矩阵
        self.L = None  # 图的拉普拉斯矩阵
        self.Lrw = None  # 规范化后的拉普拉斯矩阵, Lrw
        self.Lsym = None  # 规范化后的拉普拉斯矩阵, Lsym
        self.D = None  # 图的度矩阵

        self.cluster = None  # 聚类的结果

    def init_param(self, data):
        self.N = data.shape[0]
        dis_mat = self.cal_sqare_dis_mat(data)
        self.cal_weight_mat(dis_mat)
        self.D = np.diag(self.W.sum(axis=1))
        self.L = self.D - self.W
        if self.method == 'normalized_Shi':
            self.Lrw = np.linalg.inv(self.D) @ self.L
        elif self.method == 'normalized_Jordan':
            D = np.linalg.inv(np.sqrt(self.D))
            self.Lsym = D @ self.L @ D
        return

    def cal_sqare_dis_mat(self, data):
        # 计算距离平方的矩阵
        dis_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dis_mat[i, j] = pow(int(data[i]) - int(data[j]), 2)  # ubyte_scalars 0~255,转成int
                dis_mat[j, i] = dis_mat[i, j]
        return dis_mat

    def cal_weight_mat(self, dis_mat):
        if self.criterion == 'full_connected':
            if self.sigma is None:
                raise ValueError('sigma is not set')
            self.W = np.exp(-dis_mat / (2 * pow(self.sigma, 2)))

        elif self.criterion == 'k_nearest':
            if self.k is None or self.sigma is None:
                raise ValueError('k or sigma is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                idx = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]  # 由于包括自身，所以+1
                self.W[i][idx] = np.exp(-self.sigma * dis_mat[i][idx])
            # 为了使W对称，转置后再运算一次
            self.W = self.W.T
            for i in range(self.N):
                idx = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]
                self.W[i][idx] = np.exp(-self.sigma * dis_mat[i][idx])
            self.W = self.W.T

        elif self.criterion == 'mutual_k_nearest':
            if self.k is None or self.sigma is None:
                raise ValueError('k or sigma is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                idx_i = np.argpartition(dis_mat[i], self.k + 1)[:self.k + 1]
                idx = []
                for j in idx_i:
                    idx_j = np.argpartition(dis_mat[j], self.k + 1)[:self.k + 1]
                    # 如果j的k近邻也包含包含i，则保留
                    if i in idx_j:
                        idx.append(j)
                self.W[i][idx] = np.exp(-self.sigma * dis_mat[i][idx])

        elif self.criterion == 'eps_nearest':  # 适合于较大样本集
            if self.epsilon is None:
                raise ValueError('epsilon is not set')
            self.W = np.zeros((self.N, self.N))
            for i in range(self.N):
                idx = np.where(dis_mat[i] <= self.epsilon)
                self.W[i][idx] = self.epsilon

        else:
            raise ValueError('the criterion is not supported')
        return

    def fit(self, data):
        self.init_param(data)

        if self.method == 'unnormalized':
            w, v = np.linalg.eig(self.L)
            idx = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, idx].real  # U

        elif self.method == 'normalized_Shi':
            w, v = np.linalg.eig(self.Lrw)
            idx = np.argsort(w)[:self.n_cluster]
            Vectors = v[:, idx].real  # U

        elif self.method == 'normalized_Jordan':
            w, v = np.linalg.eig(self.Lsym)
            idx = np.argsort(w)[:self.n_cluster]
            U = v[:, idx].real
            normalizer = np.linalg.norm(U, axis=1)
            normalizer = np.repeat(normalizer.reshape(-1, 1), self.n_cluster, axis=1)
            Vectors = U / normalizer  # T

        else:
            raise ValueError('the method is not supported')

        cluster = KMeans(n_clusters=self.n_cluster, random_state=42).fit_predict(Vectors)
        self.cluster = cluster
        return


def plot_cluster(shape, labels, title=None):
    # 绘制聚类结果
    img = labels.reshape(shape)
    plt.imshow(img, interpolation='nearest')
    plt.title(title)
    plt.show()


def cal_metrics(data, labels):
    # 计算聚类的内部评价指标
    S_score = silhouette_score(data.reshape(-1, 1), labels, metric='euclidean')  # 计算轮廓系数，[-1,1]越大越好
    CH_score = calinski_harabasz_score(data.reshape(-1, 1), labels)  # 计算CH score，越大越好
    DBI = davies_bouldin_score(data.reshape(-1, 1), labels)  # 计算 DBI，戴维森堡丁指数，最小是0，值越小越好。
    print('silhouette_score: {:.6f}    CH_score: {:.6f}    DBI: {:.6f}'.format(S_score, CH_score, DBI))


def main():
    img = cv2.imread('./image/baseball_game.jpg', 0)
    img_flatten = img.flatten()
    print(img.shape)
    print(img_flatten.shape)

    # sigma, epsilon, k的取值
    epsilon = 1  # 数据点上全连接图的最⼩⽣成树中最⻓边的⻓度
    k = 100  # 小数据集可以步进的尝试，大数据集用log(n)
    sigma = epsilon  # 一个点到它的第k个近邻的平均距离的顺序来选择, k ∼ log(n) + 1; 或者等于epsilon
    n_cluster = 7  # 7
    print('sigma:{}, k:{}'.format(sigma, k))

    # Ncut
    time_start = time.time()  # 计时
    Ncut_1 = SpectralClustering(n_cluster=n_cluster, method='normalized_Shi', criterion='k_nearest', sigma=sigma, k=k)
    Ncut_1.fit(img_flatten)
    plot_cluster(img.shape, Ncut_1.cluster, 'Ncut_1')  # 绘制聚类结果
    cal_metrics(img_flatten, Ncut_1.cluster)  # 计算聚类评价指标
    time_1 = time.time()
    print('time_Ncut_1: {:.6f} s'.format(time_1 - time_start))

    Ncut_2 = SpectralClustering(n_cluster=n_cluster, method='normalized_Jordan', criterion='k_nearest', sigma=sigma, k=k)
    Ncut_2.fit(img_flatten)
    plot_cluster(img.shape, Ncut_2.cluster, 'Ncut_2')  # 绘制聚类结果
    cal_metrics(img_flatten, Ncut_2.cluster)  # 计算聚类评价指标
    time_2 = time.time()
    print('time_Ncut_2: {:.6f} s'.format(time_2 - time_1))

    # RatioCut
    RatioCut = SpectralClustering(n_cluster=n_cluster, method='unnormalized', criterion='k_nearest', sigma=sigma, k=k)
    RatioCut.fit(img_flatten)
    plot_cluster(img.shape, RatioCut.cluster, 'RatioCut')
    cal_metrics(img_flatten, RatioCut.cluster)
    time_3 = time.time()
    print('time_RatioCut: {:.6f} s'.format(time_3 - time_2))

    # Kmeans
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(img_flatten.reshape(-1, 1))
    plot_cluster(img.shape, kmeans.labels_, 'kmeans')
    cal_metrics(img_flatten, kmeans.labels_)
    time_4 = time.time()
    print('time_Kmeans: {:.6f} s'.format(time_4 - time_3))


if __name__ == '__main__':
    main()
