import numpy as np

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

import datetime
import os


class DriftHandler:
    def __init__(self, model_drift_chunk_size, model_drift_window_length):
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

        self.model_drift_chunk_size = model_drift_chunk_size
        self.model_drift_window_length = model_drift_window_length

        self.grad_coeff = self.calculate_gradient_rescaling_coeff()

    def save_drift_log(self, log_prob_chunks, filename_suffix=""):
        if filename_suffix == "":
            filename = 'drift_log_' + str(self.save_counter)
        else:
            filename = 'drift_log_' + filename_suffix

        filename_path = os.path.join(self.save_path, filename)
        np.save(file=filename_path, arr=log_prob_chunks)

        self.save_counter += 1

    def calculate_gradient_rescaling_coeff(self):
        num_data = self.model_drift_window_length/self.model_drift_chunk_size
        x = np.arange(num_data)

        poly = PolynomialFeatures(4)  # TODO hardcoded polynomial grade
        phi = poly.fit_transform(x[:, np.newaxis])
        proto_H = np.matmul(np.linalg.inv(np.matmul(phi.transpose(), phi)), phi.transpose())

        fut_step = np.array([num_data])[None]
        coeff = np.matmul(poly.fit_transform(fut_step), proto_H)

        return coeff

    @staticmethod
    def _find_outliers(x, log_like):
        # Outliers finder
        log_like_ensemble_mean = np.mean(log_like, axis=-1)  # log_like_ensemble_mean = (80,)

        poly_model = make_pipeline(PolynomialFeatures(4), Ridge())  # TODO hardcoded ridge regression order
        poly_model.fit(x[:, np.newaxis], log_like_ensemble_mean[:, np.newaxis])

        log_like_reg = poly_model.predict(x[:, np.newaxis])  # log_like_reg = (num_data,1)
        log_like_ensemble_mean = np.expand_dims(log_like_ensemble_mean, axis=-1)  # log_like_ensemble_mean = (num_data,1)
        log_like_ensemble_mean_normalized = log_like_ensemble_mean - log_like_reg  # log_like_ensemble_mean_normalized = (80,1)

        std_dev = np.std(log_like_ensemble_mean_normalized)
        z_score = log_like_ensemble_mean_normalized / std_dev

        outliers_mask = np.absolute(z_score) < 2
        # log_like_ensemble_mean_masked = log_like_ensemble_mean[outliers_mask]
        # x_masked = np.expand_dims(x, axis=-1)[outliers_mask]

        return outliers_mask, log_like_ensemble_mean

    def _find_change_point(self, num_data, mask, log_like_ensemble_mean):
        change_point_step = 1  # TODO maybe make these class attributes?
        change_point_int_min, change_point_int_max = 0.0, 1.0
        change_point_buf_pre, change_point_buf_post = 20, 10

        data_int_min = int(change_point_int_min * num_data)
        data_int_max = int(change_point_int_max * num_data)

        poly = PolynomialFeatures(1)

        coeff_max_diff = 0
        change_point = 0
        final_start_y_pre = 0
        final_end_y_pre = 0
        final_start_y_post = 0
        final_end_y_post = 0

        x = np.arange(num_data)

        for i in range(data_int_min + change_point_buf_pre, data_int_max - change_point_buf_post, change_point_step):
            x_capped = x[data_int_min:i, np.newaxis]
            phi_pre = poly.fit_transform(x_capped[mask[data_int_min:i, 0]])
            x_capped = x[i:data_int_max, np.newaxis]
            phi_post = poly.fit_transform(x_capped[mask[i:data_int_max, 0]])

            #phi_fit_pre = poly.fit_transform(xfit[fit_int_min:int(i * ratio), np.newaxis])
            #phi_fit_post = poly.fit_transform(xfit[int(i * ratio):fit_int_max, np.newaxis])

            proto_H_pre = np.matmul(np.linalg.inv(np.matmul(phi_pre.transpose(), phi_pre)), phi_pre.transpose())
            proto_H_post = np.matmul(np.linalg.inv(np.matmul(phi_post.transpose(), phi_post)), phi_post.transpose())

            w_pre = np.zeros([proto_H_pre.shape[0]])  # [M, 1]
            w_post = np.zeros([proto_H_post.shape[0]])  # [M, 1]

            y_mean_capped = log_like_ensemble_mean[data_int_min:i]
            w_pre = np.matmul(proto_H_pre, y_mean_capped[mask[data_int_min:i, 0]])

            y_mean_capped = log_like_ensemble_mean[i:data_int_max]
            w_post = np.matmul(proto_H_post, y_mean_capped[mask[i:data_int_max, 0]])

            #start_y_pre = np.matmul(phi_fit_pre[0], w_pre)
            #end_y_pre = np.matmul(phi_fit_pre[-1], w_pre)
            #start_y_post = np.matmul(phi_fit_post[0], w_post)
            #end_y_post = np.matmul(phi_fit_post[-1], w_post)

            start_y_pre = np.matmul(poly.fit_transform(np.array([data_int_min])[None]), w_pre)
            end_y_pre = np.matmul(poly.fit_transform(np.array([i-(1E-6)])[None]), w_pre)
            start_y_post = np.matmul(poly.fit_transform(np.array([i])[None]), w_post)
            end_y_post = np.matmul(poly.fit_transform(np.array([data_int_max-1])[None]), w_post)  # TODO !!!

            mean_start_y_pre = np.mean(start_y_pre)
            mean_end_y_pre = np.mean(end_y_pre)
            mean_start_y_post = np.mean(start_y_post)
            mean_end_y_post = np.mean(end_y_post)

            coeff_pre = (mean_end_y_pre - mean_start_y_pre) / (i - data_int_min)
            coeff_post = (mean_end_y_post - mean_start_y_post) / (data_int_max - i)

            angle = abs(np.arctan((coeff_pre - coeff_post) / (1 + coeff_pre * coeff_post)))
            # print("Iter: {}, Angle: {}".format(i, np.rad2deg(angle)))

            if angle > coeff_max_diff:
                coeff_max_diff = angle

                coeff_pre_change = coeff_pre
                coeff_post_change = coeff_post

                change_point = i
                final_start_y_pre = mean_start_y_pre
                final_end_y_pre = mean_end_y_pre
                final_start_y_post = mean_start_y_post
                final_end_y_post = mean_end_y_post

        print("Angle diff: {} rad, {} deg".format(coeff_max_diff, np.rad2deg(coeff_max_diff)))

        if coeff_max_diff >= 0.61:  # TODO set threshold
            elaborated_x = np.rint(self._get_intersect(
                [data_int_min, final_start_y_pre], [change_point, final_end_y_pre],
                [change_point, final_start_y_post], [data_int_max, final_end_y_post])[0])
            print("Intercept X: {}".format(elaborated_x))
        else:
            elaborated_x = 0

        return elaborated_x

    def _find_change_point_mse(self, num_data, mask, log_like_ensemble_mean):
        change_point_step = 1  # TODO maybe make these class attributes?
        change_point_int_min, change_point_int_max = 0.0, 1.0
        change_point_buf_pre, change_point_buf_post = int(num_data/4), int(num_data/8)

        data_int_min = int(change_point_int_min * num_data)
        data_int_max = int(change_point_int_max * num_data)

        poly_pre = make_pipeline(PolynomialFeatures(1), LinearRegression())
        poly_post = make_pipeline(PolynomialFeatures(1), LinearRegression())

        coeff_max_diff = np.Inf
        change_point = 0
        final_start_y_pre = 0
        final_end_y_pre = 0
        final_start_y_post = 0
        final_end_y_post = 0

        x = np.arange(num_data)

        for i in range(data_int_min + change_point_buf_pre, data_int_max - change_point_buf_post, change_point_step):
            x_capped_pre = x[data_int_min:i, np.newaxis][mask[data_int_min:i, 0]]
            x_capped_post = x[i:data_int_max, np.newaxis][mask[i:data_int_max, 0]]

            y_mean_capped_pre = log_like_ensemble_mean[data_int_min:i][mask[data_int_min:i, 0]]
            y_mean_capped_post = log_like_ensemble_mean[i:data_int_max][mask[i:data_int_max, 0]]

            poly_pre.fit(x_capped_pre, y_mean_capped_pre)
            poly_post.fit(x_capped_post, y_mean_capped_post)

            start_y_pre = poly_pre.predict(np.expand_dims([data_int_min], axis=-1))
            end_y_pre = poly_pre.predict(np.expand_dims([i - (1E-6)], axis=-1))
            start_y_post = poly_post.predict(np.expand_dims([i], axis=-1))
            end_y_post = poly_post.predict(np.expand_dims([data_int_max - 1], axis=-1))

            coeff_pre = (end_y_pre - start_y_pre) / (i - data_int_min)
            coeff_post = (end_y_post - start_y_post) / (data_int_max - i)

            y_pred_pre = poly_pre.predict(x_capped_pre)
            mse_pre = mean_absolute_error(y_pred_pre, y_mean_capped_pre)
            y_pred_post = poly_post.predict(x_capped_post)
            mse_post = mean_absolute_error(y_pred_post, y_mean_capped_post)
            # mse_tot = mse_pre + mse_post
            mse_tot = (mse_pre * y_pred_pre.shape[0] + mse_post * y_pred_post.shape[0]) / (y_pred_pre.shape[0] + y_pred_post.shape[0])

            angle = abs(np.arctan((coeff_pre - coeff_post) / (1 + coeff_pre * coeff_post)))

            if mse_tot < coeff_max_diff:
                coeff_max_diff = mse_tot
                angle_c = angle

                coeff_pre_change = coeff_pre
                coeff_post_change = coeff_post

                change_point = i
                final_start_y_pre = start_y_pre
                final_end_y_pre = end_y_pre
                final_start_y_post = start_y_post
                final_end_y_post = end_y_post

                final_x_capped_pre = x_capped_pre
                final_poly_pre = poly_pre
                final_y_mean_capped_pre = y_mean_capped_pre
                final_start_y_post = start_y_post
                final_end_y_pre = end_y_pre

                final_mean = poly_pre.predict(final_x_capped_pre)

        print("Angle diff: {} rad, {} deg".format(angle_c, np.rad2deg(angle_c)))

        if angle_c >= 0.61:  # TODO set threshold
            elaborated_x = np.rint(self._get_intersect([data_int_min,
                                                        final_start_y_pre[0, 0]],
                                                       [change_point, final_end_y_pre[0, 0]],
                                                       [change_point, final_start_y_post[0, 0]],
                                                       [data_int_max, final_end_y_post[0, 0]])[0])
            print("Intercept X: {}".format(elaborated_x))

            drift = True
        else:
            elaborated_x = change_point
            drift = False

        abrupt = self._recognize_abrupt(final_y_mean_capped_pre,
                                        final_mean,
                                        final_start_y_post,
                                        final_end_y_pre)

        if abrupt:
            drift = False

        return drift, elaborated_x

    def _recognize_abrupt(self, final_y_mean_capped_pre, final_mean, final_start_y_post, final_end_y_pre):
        final_y_norm = final_y_mean_capped_pre - final_mean

        sigma = np.std(final_y_norm)
        z_score = (final_start_y_post - final_end_y_pre) / sigma

        return z_score > 4


    @staticmethod
    def _get_intersect(a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return float('inf'), float('inf')
        return x / z, y / z

    """
    @staticmethod
    def _predict_future_performance(num_data, num_models, mask, log_like):
        # Initialization
        poly = PolynomialFeatures(4)  # TODO regression order is hardcoded!

        x = np.arange(num_data)
        x_masked = np.expand_dims(x, axis=-1)[mask]
        phi = poly.fit_transform(x_masked[:, np.newaxis])
        proto_H = np.matmul(np.linalg.inv(np.matmul(phi.transpose(), phi)), phi.transpose())
        w = np.zeros([proto_H.shape[0], num_models])

        fut_pred_p = np.zeros([num_models])

        for i in range(num_models):
            w[:, i] = np.matmul(proto_H, log_like[mask[:, 0], i])

            fut_step = np.array([x[-1] + 1])[None]
            fut_pred_p[i] = np.matmul(poly.fit_transform(fut_step), w[:, i])

        return fut_pred_p
    """

    def check_env_drift(self, l_arr):  # l_arr = (num_data, num_models)
        # Initialization
        num_data = l_arr.shape[0]
        num_models = l_arr.shape[-1]

        x = np.arange(num_data)
        log_like = np.empty_like(l_arr)

        for i in range(num_models):  # l_arr is flipped to have older data in lower indices and vice versa
            log_like[:, i] = np.flip(l_arr[:, i])

        # Outliers detection
        mask, log_like_ensemble_mean = self._find_outliers(x, log_like)

        # Change point search
        drift, change_point = self._find_change_point_mse(num_data, mask, log_like_ensemble_mean)
        print("Change point X: {}".format(change_point))

        return drift, change_point, mask
