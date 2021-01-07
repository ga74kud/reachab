import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class causal_prob(object):
    def __init__(self, **kwargs):
        self.ax=None
        self.prob={"mu": None, "Sigma": None}
    def set_mu(self, mu):
        self.prob["mu"]=mu
    def set_Sigma(self, Sigma):
        self.prob["Sigma"]=Sigma
    def get_probabilities_position(self, coordinates):
        erg=[]
        for wlt in coordinates:
            x = np.matrix([wlt[0]], wlt[1])
            erg.append(np.float(self.multivariate_gaussian_distribution(x, self.prob["mu"], self.prob["Sigma"])))
        return erg
    def mahalabonis_dist(self, x, mu, Sigma):
        return -0.5*np.transpose(x-mu)*np.linalg.inv(Sigma)*(x-mu)
    def multivariate_gaussian_distribution(self, x, mu, Sigma):
        factor_A=1/np.sqrt((2*np.pi)**2*np.linalg.det(Sigma))
        factor_B=np.exp(self.mahalabonis_dist(x, mu, Sigma))
        erg=factor_A*factor_B
        return erg[0]
    def visualize_multivariate_gaussian(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(projection='3d')
        Z=np.zeros((np.size(self.X, 0), np.size(self.X, 1)))
        for idx_A in range(0, np.size(self.X, 0)):
            for idx_B in range(0, np.size(self.X, 1)):
                x = np.matrix([[self.X[idx_A, idx_B]], [self.Y[idx_A, idx_B]]])
                Z[idx_A, idx_B]=self.multivariate_gaussian_distribution(x, self.prob["mu"], self.prob["Sigma"])
        self.ax.plot_surface(self.X, self.Y, Z,  cmap='viridis',
                       linewidth=0, antialiased=False, alpha=.3)

        self.ax.contour(self.X, self.Y, Z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=0)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('p(x,y)')
    def show(self):
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        #self.ax.xaxis([-0.2, 2.3])
        #self.ax.yaxis([-0.2, 2.3])

        self.ax.view_init(10, 0)
        #self.ax.set_aspect('equal')
        plt.show()
    def set_fixed_domain(self):
        N = 16
        X = np.linspace(-3, 3, N)
        Y = np.linspace(-3, 4, N)
        self.X, self.Y = np.meshgrid(X, Y)
        self.x_rav = np.ravel(X)
        self.y_rav = np.ravel(Y)
    def plot_arrow(self, mu, w, v):
        Q=self.ax.quiver(mu[0], mu[1], 0, w*v[0,0], w*v[0, 1], 0, color="red", linewidth=2,
                         alpha=.5)
    def kullback_leibler(self, mu_B, Sigma_B):
        k=len(self.prob["mu"])
        sum_A=np.trace(np.linalg.inv(Sigma_B)*self.prob["Sigma"])
        dif_mu=mu_B-self.prob["mu"]
        sum_B=np.transpose(dif_mu)*np.linalg.inv(Sigma_B)*dif_mu-2
        sum_C=np.log(np.linalg.det(Sigma_B)/np.linalg.det(self.prob["Sigma"]))
        return 0.5*(sum_A+sum_B+sum_C)

    def plot_eigen_vectors_Sigma(self):
        w, v = np.linalg.eigh(self.prob["Sigma"])
        for idx, wlt in enumerate(v):
            self.plot_arrow(self.prob["mu"], w[idx], wlt)



if __name__ == '__main__':
    mlt.use('Qt5Agg')
    mu = np.matrix([[0.], [0.]])
    Sigma = np.matrix([[1., 0.], [0, 1.]])
    mu_B=np.matrix([[0.], [0.]])
    Sigma_B = np.matrix([[1., 0.], [0, 1.]])
    obj_causal=causal_prob()
    obj_causal.set_fixed_domain()
    obj_causal.set_mu(mu)
    obj_causal.set_Sigma(Sigma)

    obj_causal.visualize_multivariate_gaussian()
    obj_causal.plot_eigen_vectors_Sigma()
    obj_causal.show()

    print(obj_causal.kullback_leibler(mu_B, Sigma_B))

