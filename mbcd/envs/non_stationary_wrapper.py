from gym import Wrapper
from collections import deque
from experiments.experiments_enum import ExpType
from mbcd.envs.envs_enum import SimType, Env


class NonStationaryEnv(Wrapper):
    def __init__(self, env, tasks=ExpType.Base_HC):
        super(NonStationaryEnv, self).__init__(env)
        self.tasks = deque(tasks.value["tasks"])
        self.sim_type = tasks.value["sim"]
        # self.change_freq = tasks.value["change_freq"]
        if isinstance(tasks.value["change_freq"], int):
            self.change_freq = deque([tasks.value["change_freq"]]*len(tasks.value["tasks"]))
        else:
            self.change_freq = deque(tasks.value["change_freq"])
        self.counter = 0
        self.fix_reward_vel = (self.sim_type == SimType.HalfCheetah) or (self.sim_type == SimType.Hopper)

        self.env_task = Env(self.sim_type, self.current_task, self.current_change_freq)  # Get parameters
        self.avg_vel = 0

    @property
    def current_task(self):
        return self.tasks[0]

    @property
    def current_change_freq(self):
        return self.change_freq[0]

    def step(self, action):
        if self.counter % self.current_change_freq == 0 and self.counter > 0:
            self.tasks.popleft()
            self.change_freq.popleft()
            self.env_task = Env(self.sim_type, self.current_task, self.current_change_freq)
            print("CHANGED TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
            self.counter = 0

        pos_before = self.unwrapped.sim.data.qpos[0]
        env_parameters = self.env_task.get_params()

        action = env_parameters["malfunction_mask"] * action

        for part in self.unwrapped.sim.model._body_name2id.values():
            self.unwrapped.sim.data.xfrc_applied[part, :] = env_parameters["wind"]

        next_obs, reward, done, info = self.env.step(action)
        # TODO check value of self.current_task for compatibility. Implement to_string()
        info['task'] = self.current_task

        if self.fix_reward_vel:
            pos_after = self.unwrapped.sim.data.qpos[0]
            forward_vel = (pos_after - pos_before) / self.unwrapped.dt

            reward -= forward_vel # remove this term
            reward += -1 * abs(forward_vel - env_parameters["target_velocity"])

        self.avg_vel += forward_vel
        if not(self.counter % 100):
            self.avg_vel = self.avg_vel / 100
            print("Step :{}".format(self.counter))
            print("Target velocity: {}".format(env_parameters["target_velocity"]))
            print("Actual velocity: {}".format(self.avg_vel))
            self.avg_vel = 0

        self.counter += 1

        return next_obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 0
