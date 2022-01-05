import os
import gym
import numpy as np
import pandas as pd
import random
import argparse

import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines.common.vec_env import VecNormalize
#from stable_baselines.common.policies import register_policy
from stable_baselines.sac.policies import FeedForwardPolicy

from mbcd.mbcd import MBCD
from mbcd.envs.non_stationary_wrapper import NonStationaryEnv

from mbcd.utils.util import evaluate
from mbcd.sac_mbcd import SAC

from experiments.experiments_enum import ExpType


class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs, layers=[256, 256], feature_extraction="mlp")  # 64,64
# register_policy('CustomSACPolicy', CustomSACPolicy)


parser = argparse.ArgumentParser()
parser.add_argument('-seed', dest='seed', required=False, type=int, help="Seed\n", default=0)      
parser.add_argument('-algo', dest='algo', required=True, type=str, help="Algo [mbcd, mbpo, sac]\n")
parser.add_argument('-env', dest='env', required=False, type=str, help="Env [halfcheetah, pusher]", default='halfcheetah')
parser.add_argument('-load', dest='load', required=False, type=bool, help="Load pre-trained [True, False]", default=False)
parser.add_argument('-gif', dest='gif', required=False, type=bool, help="Save gifs [True, False]", default=False)
parser.add_argument('-roll', dest='roll', required=False, type=str, help="Rollout type [mbcd, m2ac]\n", default='mbcd')
args = parser.parse_args()
assert args.algo in ['mbcd', 'mbpo', 'sac']
assert args.roll in ['mbcd', 'm2ac']
mbcd = args.algo == 'mbcd'
mbpo = args.algo != 'sac'

SEED = args.seed
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)


def main(config):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    model = SAC(CustomSACPolicy,
                env=config['env'],
                rollout_schedule=config['rollout_schedule'],
                verbose=1,
                batch_size=config['batch_size'],
                gradient_steps=config['gradient_steps'],
                target_entropy='auto',
                ent_coef='auto',
                mbpo=mbpo,
                mbcd=mbcd,
                max_std=config['max_std'],
                num_stds=config['num_stds'],
                n_hidden_units_dynamics=config['n_hidden_units_dynamics'],
                n_layers_dynamics=config['n_layers_dynamics'],
                dynamics_memory_size=config['dynamics_memory_size'],
                cusum_threshold=config['cusum_threshold'],
                run_id=config['run_id'],
                tensorboard_log='./logs/',
                seed=SEED,
                load_pre_trained_model=args.load,
                save_gifs=args.gif,
                rollout_mode=args.roll)

    tb_log_name = 'mbcd-test-' + args.roll
    model.learn(total_timesteps=config['total_timesteps'], tb_log_name=tb_log_name)
    if args.algo == 'sac':
        model.save('weights/'+'sacfinalpolicy')
    else:
        model.deepMBCD.save_current()
        model.deepMBCD.save_models()


if __name__ == '__main__':

    if args.env == 'halfcheetah':
        tasks = ExpType.Base_Drift_Switch_Test
        change_freq = tasks.value["change_freq"]
        if isinstance(change_freq, list):
            total_timesteps = sum(tasks.value["change_freq"])
        else:
            total_timesteps = change_freq * len(tasks.value["tasks"])

        if args.roll == 'm2ac':
            rollout_schedule = [5000, 20000, 1, 25]
        else:
            rollout_schedule = [5000, 10000, 1, 1]
        config = {
                'env': NonStationaryEnv(gym.envs.make('HalfCheetah-v2'), tasks=tasks),
                'rollout_schedule': rollout_schedule,
                'batch_size': 256,
                'gradient_steps': 5000,
                'target_entropy': 'auto',
                'ent_coef': 'auto',
                'max_std': 0.5,
                'num_stds': 2.0, 
                'n_hidden_units_dynamics': 200,
                'n_layers_dynamics': 4,
                'dynamics_memory_size': 100000,
                'cusum_threshold': 100,
                'run_id':'{}-halfcheetah-ns-paper{}'.format(args.algo, str(SEED)),
                'total_timesteps': total_timesteps
        }


    main(config)



   


