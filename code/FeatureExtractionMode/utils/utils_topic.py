from pylab import zeros, random, log
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD


# TODO: LSA
def lsa(vectors, com_prop=0.8):
    # Component proportion 成分占比
    n_components = int(len(vectors[0]) * com_prop)
    lsa_vectors = TruncatedSVD(n_components).fit_transform(vectors)

    return lsa_vectors


# TODO: LDA and Label LDA
def lda(vectors, labels=None, com_prop=0.8):
    n_components = int(len(vectors[0]) * com_prop)
    if labels is not None:
        lda_vectors = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                                learning_method='batch',
                                                learning_offset=50.,
                                                random_state=0).fit_transform(vectors, labels)
    else:
        lda_vectors = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                                learning_method='batch',
                                                learning_offset=50.,
                                                random_state=0).fit_transform(vectors)
    return lda_vectors


# TODO: PLSA
class PLsa(object):
    def __init__(self, vectors, com_prop=0.8, max_iter=20):
        self.X = vectors
        self.N, self.M = vectors.shape  # document-word矩阵, N为文档个数，M为词表长
        self.K = int(self.M * com_prop)
        self.p = zeros([self.N, self.M, self.K])  # 定义隐变量的后验概率的矩阵表示
        self.max_iter = max_iter

    def init_lamda(self):
        lamda = random([self.N, self.K])
        for i in range(0, self.N):
            normalization = sum(lamda[i, :])
            for j in range(0, self.K):
                lamda[i, j] /= normalization
        return lamda

    def init_theta(self):
        theta = random([self.K, self.M])
        for i in range(0, self.K):
            normalization = sum(theta[i, :])
            for j in range(0, self.M):
                theta[i, j] /= normalization
        return theta

    # E-Step
    def e_step(self, theta, lamda):
        for i in range(0, self.N):
            for j in range(0, self.M):
                denominator = 0
                for k in range(0, self.K):
                    self.p[i, j, k] = theta[k, j] * lamda[i, k]
                    denominator += self.p[i, j, k]
                if denominator == 0:
                    for k in range(0, self.K):
                        self.p[i, j, k] = 0
                else:
                    for k in range(0, self.K):
                        self.p[i, j, k] /= denominator
        return theta, lamda

    # M-Step
    def m_step(self, theta, lamda):
        # 更新参数theta
        for k in range(0, self.K):
            denominator = 0
            for j in range(0, self.M):
                theta[k, j] = 0
                for i in range(0, self.N):
                    theta[k, j] += self.X[i, j] * self.p[i, j, k]
                denominator += theta[k, j]
            if denominator == 0:
                for j in range(0, self.M):
                    theta[k, j] = 1.0 / self.M
            else:
                for j in range(0, self.M):
                    theta[k, j] /= denominator

        # 更新参数lamda
        for i in range(0, self.N):
            for k in range(0, self.K):
                lamda[i, k] = 0
                denominator = 0
                for j in range(0, self.M):
                    lamda[i, k] += self.X[i, j] * self.p[i, j, k]
                    denominator += self.X[i, j]
                if denominator == 0:
                    lamda[i, k] = 1.0 / self.K
                else:
                    lamda[i, k] /= denominator
        return theta, lamda

    def log_likelihood(self, theta, lamda):
        loglikelihood = 0
        for i in range(0, self.N):
            for j in range(0, self.M):
                tmp = 0
                for k in range(0, self.K):
                    tmp += theta[k, j] * lamda[i, k]
                if tmp > 0:
                    loglikelihood += self.X[i, j] * log(tmp)
        print('log likelihood : ', loglikelihood)

    # EM algorithm
    # ==============================================================================
    def em_algorithm(self):
        theta = self.init_theta()  # topic-word matrix: [K, M]
        lamda = self.init_lamda()  # doc-topic matrix: [N, K] -> 相当于降维，也就是我们需要的
        self.log_likelihood(theta, lamda)
        for i in range(0, self.max_iter):
            theta, lamda = self.e_step(theta, lamda)
            theta, lamda = self.m_step(theta, lamda)
        return theta, lamda
