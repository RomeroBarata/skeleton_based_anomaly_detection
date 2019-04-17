import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import scipy.stats as stats


class ScoreNormalization:
    def __init__(self, method="KDE", options=None):
        self.method = method
        self.options = {} if options is None else options
        self.name = "_".join([method]+["-".join([str(key), str(value)]) for key, value in self.options.items()])
        if self.method == "KDE":
            kernel = self.options.get("kernel", "gaussian")
            bandwidth = self.options.get("bandwidth", 0.75)
            self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        elif self.method == "gamma":
            self.k = self.options.get("k", 1)
            self.loc = self.options.get("loc", 0)
            self.theta = self.options.get("theta", 1.5)
        elif self.method == "chi2":
            self.df = self.options.get("df", 2)
            self.loc = self.options.get("loc", 0)
            self.scale = self.options.get("scale", 0.5)
        else:
            raise ("Invalid method {}".format(self.method))

    def fit(self, X):
        if self.method == "KDE":
            self.kde.fit(X)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        elif self.method == "gamma":
            self.fit_k, self.fit_loc, self.fit_theta = stats.gamma.fit(X, self.k, loc=self.loc, scale=self.theta)
        elif self.method == "chi2":
            self.fit_df, self.fit_loc, self.fit_scale = stats.chi2.fit(X, self.df, loc=self.loc, scale=self.scale)
            pass
        else:
            raise ("Invalid method {}".format(self.method))

    def score(self, x):
        if self.method == "KDE":
            return self.kde.score_samples(x)
        elif self.method == "gamma":
            return 1 - stats.gamma.cdf(x, self.fit_k, self.fit_loc, self.fit_theta)
        elif self.method == "chi2":
            return 1 - stats.chi2.cdf(x, self.fit_df, self.fit_loc, self.fit_scale)
        elif self.method == "histogram":
            raise ("Not implemented method {}".format(self.method))
        else:
            raise ("Invalid method {}".format(self.method))
    
    transform = score

    def get_fit_params_string(self):
        if self.method == "KDE":
            return ""
        elif self.method == "gamma":
            return "fit_k %3.1f fit_loc %3.1f fit_theta %3.1f" % (self.fit_k, self.fit_loc, self.fit_theta)
        elif self.method == "chi2":
            return "fit_df %s, fit_loc %s, fit_scale %s" % (self.fit_df, self.fit_loc, self.fit_scale)
        elif self.method == "histogram":
            return ""
        else:
            raise ("Invalid method {}".format(self.method))

    def get_fit_params(self):
        if self.method == "KDE":
            return ""
        elif self.method == "gamma":
            return self.fit_k, self.fit_loc, self.fit_theta
        elif self.method == "chi2":
            return self.fit_df, self.fit_loc, self.fit_scale
        elif self.method == "histogram":
            return ""
        else:
            raise ("Invalid method {}".format(self.method))


def normalizing_lstm_autoencoder(training_sample_rate=1):
    """
    models=[ ScoreNormalization(method="KDE", options={"kernel": "gaussian", "bandwidth": 0.01}),
             ScoreNormalization(method="KDE", options={"kernel": "gaussian", "bandwidth": 0.001}),
             ScoreNormalization(method="KDE", options={"kernel": "gaussian", "bandwidth": 0.0001})
             ]
    """
    models = [ScoreNormalization(method="gamma", options={})]

    # models = [ScoreNormalization(method="chi2", options={})]

    training_file = "/data/romero/ShanghaiTech/training/anomaly_model_output/reconstruction_errors_mse.npz"
    testing_file = "/data/romero/ShanghaiTech/testing/anomaly_model_output/reconstruction_errors_mse.npz"
    all_training_score = np.load(training_file)
    all_testing_score = np.load(testing_file)
    test_normed_score = {}
    train_normed_score = {}
    final_train_scores = {}
    final_test_scores = {}
    for cam_id in range(len(all_training_score.files)):
        training_score = all_training_score["arr_{}".format(cam_id)]
        testing_score = all_testing_score["arr_{}".format(cam_id)]

        is_non_zero_training_score = training_score > 0.0
        non_zero_training_score = training_score[is_non_zero_training_score].reshape(-1, 1)
        non_zero_training_score = non_zero_training_score[0::training_sample_rate]

        is_non_zero_testing_score = testing_score > 0.0
        non_zero_testing_score = testing_score[is_non_zero_testing_score].reshape(-1, 1)

        for model in models:
            model.fit(non_zero_training_score)
            train_normed_score[cam_id] = model.score(non_zero_training_score)
            test_normed_score[cam_id] = model.score(non_zero_testing_score)

            training_score = training_score.ravel()
            training_score[is_non_zero_training_score] = train_normed_score[cam_id].ravel()
            testing_score = testing_score.ravel()
            testing_score[is_non_zero_testing_score] = test_normed_score[cam_id].ravel()

            plot_title = 'camera ' + str(cam_id + 1) + ' ' + model.name + '_' + model.get_fit_params_string()
            visualize(non_zero_training_score, non_zero_testing_score,
                      train_normed_score[cam_id], test_normed_score[cam_id],
                      plot_title=plot_title)

        final_train_scores[cam_id] = training_score
        final_test_scores[cam_id] = testing_score


def normalizing_3Dconv():
    pass


def visualize(X, X_test, dens_train, dens, plot_title):
    #----------------------------------------------------------------------
    # Plot the progression of histograms to kernels
    num_bins = 50
    fig, ax = plt.subplots()
    # histogram 1
    # weights = np.ones_like(X) / float(num_bins)
    n, bins, patches = ax.hist(X, bins=num_bins, density=True, log=True, alpha=0.6, fc='#AAAAFF')
    ax.set_ylabel("Histogram")

    # p = X_test.ravel().argsort()
    # X_test = X_test.ravel()[p]
    # dens = dens[p]

    # p = X.ravel().argsort()
    # X = X.ravel()[p]
    # dens_train = dens_train[p]

    # ax.plot(X_test, dens, 'g-', label="test")
    # ax.plot(X, dens_train, 'k-', label="train")
    ax.scatter(X, dens_train, c='g', alpha=0.7, label='train')
    ax.scatter(X_test, dens, c='r', alpha=0.2, label='test')
    ax.legend(loc='upper left')
    ax.set_title(plot_title)
    plt.show()
    pass


if __name__ == '__main__':
    normalizing_lstm_autoencoder(training_sample_rate=1)
