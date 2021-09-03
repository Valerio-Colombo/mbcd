import numpy as np
from sklearn.linear_model import LinearRegression

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

    def save_drift_log(self, log_prob_chunks):
        filename = 'drift_log_' + str(self.save_counter)
        filename_path = os.path.join(self.save_path, filename)
        np.save(file=filename_path, arr=log_prob_chunks)

        self.save_counter += 1
'''
    def update_drift_model(self, log_prob_chunks):
        # TODO return loss function?/update model based on predictions


    def predict_future_performance(self, chunks):
        # TODO implement regression
'''
