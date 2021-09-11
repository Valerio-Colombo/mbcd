import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import datetime
import os


class DriftHandler:
    def __init__(self):
        self.save_path_root_dir = '/home/valerio/PycharmProjects/mbcd/drift/logs'
        date_time = datetime.datetime.now()
        sub_dir = 'Drift_Log-' + \
                  str(date_time.month) + '_' + \
                  str(date_time.day) + '_' + \
                  str(date_time.hour) + '_' + \
                  str(date_time.minute) + '_' + \
                  str(date_time.second)
        self.save_path = os.path.join(self.save_path_root_dir, sub_dir)
        os.mkdir(self.save_path)

        print('Saving drift logs to {}'.format(self.save_path))

        self.save_counter = 0

    def save_drift_log(self, log_prob_chunks, filename_suffix=""):
        if filename_suffix == "":
            filename = 'drift_log_' + str(self.save_counter)
        else:
            filename = 'drift_log_' + filename_suffix

        filename_path = os.path.join(self.save_path, filename)
        np.save(file=filename_path, arr=log_prob_chunks)

        self.save_counter += 1

    @staticmethod
    def predict_future_performance(l_arr):
        num_model = l_arr.shape[-1]
        num_chunks = l_arr.shape[0]
        num_features = 6

        x = np.arange(num_chunks)

        y_flip = np.empty_like(l_arr)
        for i in range(num_model):
            y_flip[:, i] = np.flip(l_arr[:, i])

        poly = PolynomialFeatures(num_features)
        phi = poly.fit_transform(x[:, np.newaxis])
        proto_H = np.matmul(np.linalg.inv(np.matmul(phi.transpose(), phi)), phi.transpose())

        w = np.zeros([num_features+1, num_model])

        fut_pred_p = np.zeros([y_flip.shape[-1]])

        """
        poly_model = make_pipeline(PolynomialFeatures(6),  # 6 is very good
                                   LinearRegression())

        # y_fit_p = np.zeros([1000, y_flip.shape[-1]])
        fut_pred_p = np.zeros([y_flip.shape[-1]])

        for i in range(y_flip.shape[-1]):
            poly_model.fit(x[:, np.newaxis], y_flip[:, i])
            # y_fit_p[:, i] = poly_model.predict(x_fit[:, np.newaxis])
            fut_step = np.array([x[-1] + 1])[None]
            fut_pred_p[i] = poly_model.predict(fut_step)
        """

        return fut_pred_p
