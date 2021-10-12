import numpy as np


class Dataset:

    def __init__(self, obs_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr, self.size, = 0, 0

        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rews_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)
        
        self.input_mean, self.output_mean = None, None
        self.input_std, self.output_std = None, None
        
    def push(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def remove_last_n(self, n):
        self.ptr -= n

    def sample(self, batch_size, replace=True):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        return self.obs_buf[inds], self.acts_buf[inds], self.rews_buf[inds], self.next_obs_buf[inds], self.done_buf[inds]

    def sample_chunks(self, chunk_size, window_length):  # lower index are newer data
        chunk_num = int(min(self.size, window_length) / chunk_size)

        chunk_arr_input = np.zeros((chunk_num,
                                    chunk_size,
                                    self.obs_buf.shape[-1]+self.acts_buf.shape[-1]),
                                   dtype=np.float32)  # s,a
        chunk_arr_output = np.zeros((chunk_num,
                                     chunk_size,
                                     self.obs_buf.shape[-1]+1),
                                    dtype=np.float32)  # r,s'

        for c in range(chunk_num):
            chunk_start = self.ptr-1-chunk_size*c
            indices = range(chunk_start, chunk_start-chunk_size, -1)

            l_obs = np.take(self.obs_buf, indices, mode='wrap', axis=0)  # [chunk_size, obs_dim]
            l_act = np.take(self.acts_buf, indices, mode='wrap', axis=0)  # [chunk_size, act_dim]
            chunk_arr_input[c, 0:chunk_size] = np.concatenate((l_obs, l_act), axis=-1)

            l_rew = np.take(self.rews_buf, indices, mode='wrap')[None].T  # [chunk_size, 1]
            l_next_obs = np.take(self.next_obs_buf, indices, mode='wrap', axis=0)  # [chunk_size, obs_dim]
            chunk_arr_output[c, 0:chunk_size] = np.concatenate((l_rew, l_next_obs), axis=-1)

        return chunk_num, chunk_arr_input, chunk_arr_output

    def to_train_batch(self, normalization=False):
        inds = np.arange(self.size)

        X = np.hstack((self.obs_buf[inds], self.acts_buf[inds]))
        Y = np.hstack((self.rews_buf[inds], self.next_obs_buf[inds] - self.obs_buf[inds]))

        """ if normalization:
            self.input_mean = np.mean(X, axis=0)
            self.input_std = np.std(X, axis=0)
            self.output_mean = np.mean(Y, axis=0)
            self.output_std = np.std(Y, axis=0)
            X = normalize(X, self.input_mean, self.input_std)
            Y = normalize(Y, self.output_mean, self.output_std) """

        return X, Y

    def to_train_batch_separated(self, window_length, chunk_size):
        chunk_num = int(min(self.size, window_length) / chunk_size)

        X = []
        Y = []

        for c in range(chunk_num):
            chunk_start = self.ptr - 1 - chunk_size * c
            indices = range(chunk_start, chunk_start - chunk_size, -1)

            l_obs = np.take(self.obs_buf, indices, mode='wrap', axis=0)  # [chunk_size, obs_dim]
            l_act = np.take(self.acts_buf, indices, mode='wrap', axis=0)  # [chunk_size, act_dim]
            X_c = np.hstack((l_obs, l_act))
            X.append(X_c)

            l_rew = np.take(self.rews_buf, indices, mode='wrap', axis=0)  # [chunk_size, 1]
            l_next_obs = np.take(self.next_obs_buf, indices, mode='wrap', axis=0)
            Y_c = np.hstack((l_rew, l_next_obs - l_obs))
            Y.append(Y_c)

        return chunk_num, X, Y

    def __len__(self):
        return self.size

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-10)

def denormalize(data, mean, std):
    return data * (std + 1e-10) + mean