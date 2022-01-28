import enum
import numpy as np


class EnvType(enum.Enum):
    Normal = 0
    Joint_Malfunction = 1
    Wind = 2
    Velocity = 3
    Joint_Malfunction_Drift = 4
    Wind_Drift = 5
    Velocity_Drift = 6


class SimType(enum.Enum):
    HalfCheetah = {"action_dim": 6, "observation_dim": 17}
    Hopper = {"action_dim": 3, "observation_dim": 11}  # TODO: add real number of dim


class Env:
    def __init__(self, sim_type, env_type, change_freq):
        self.sim_type = sim_type
        self.env_type = env_type
        self.change_freq = change_freq

        self.target_velocity = 2.5
        self.target_velocity_drift = 3.5
        self.env_parameters, self.env_parameters_gen = self._init_parameters(sim_type, env_type)

    """
    Parameters:
        malfunction_mask:   array[] coefficient for every actuator. 1 don't modify action
        velocity:           scalar
        wind:               array[6] 3D force + 3D torque
    """
    def _init_parameters(self, sim_type, env_type):
        param_dict = {}
        env_parameters_gen = {}

        if sim_type == SimType.HalfCheetah:
            if env_type == EnvType.Normal:
                param_dict["malfunction_mask"] = np.ones(sim_type.value["action_dim"])
                param_dict["target_velocity"] = self.target_velocity
                param_dict["wind"] = [0, 0, 0, 0, 0, 0]

                env_parameters_gen["malfunction_mask_0"] = self.param_generator(1)
                env_parameters_gen["malfunction_mask_1"] = self.param_generator(1)
                env_parameters_gen["target_velocity"] = self.param_generator(self.target_velocity)
                env_parameters_gen["wind_0"] = self.param_generator(0)

            elif env_type == EnvType.Joint_Malfunction:
                param_dict["malfunction_mask"] = np.ones(sim_type.value["action_dim"])
                param_dict["malfunction_mask"][0] = -1
                param_dict["malfunction_mask"][1] = -1
                param_dict["target_velocity"] = self.target_velocity
                param_dict["wind"] = [0, 0, 0, 0, 0, 0]

                env_parameters_gen["malfunction_mask_0"] = self.param_generator(0)
                env_parameters_gen["malfunction_mask_1"] = self.param_generator(0)
                env_parameters_gen["target_velocity"] = self.param_generator(self.target_velocity)
                env_parameters_gen["wind_0"] = self.param_generator(0)

            elif env_type == EnvType.Velocity:
                param_dict["malfunction_mask"] = np.ones(sim_type.value["action_dim"])
                param_dict["target_velocity"] = 2.5
                param_dict["wind"] = [0, 0, 0, 0, 0, 0]

                env_parameters_gen["malfunction_mask_0"] = self.param_generator(1)
                env_parameters_gen["malfunction_mask_1"] = self.param_generator(1)
                env_parameters_gen["target_velocity"] = self.param_generator(2.5)
                env_parameters_gen["wind_0"] = self.param_generator(0)

            elif env_type == EnvType.Wind:
                param_dict["malfunction_mask"] = np.ones(sim_type.value["action_dim"])
                param_dict["target_velocity"] = self.target_velocity
                param_dict["wind"] = [-4, 0, 0, 0, 0, 0]

                env_parameters_gen["malfunction_mask_0"] = self.param_generator(1)
                env_parameters_gen["malfunction_mask_1"] = self.param_generator(1)
                env_parameters_gen["target_velocity"] = self.param_generator(self.target_velocity)
                env_parameters_gen["wind_0"] = self.param_generator(-6)

            else:  # Drifting environments
                param_dict["malfunction_mask"] = np.ones(sim_type.value["action_dim"])
                param_dict["target_velocity"] = self.target_velocity
                param_dict["wind"] = [0, 0, 0, 0, 0, 0]

                if env_type == EnvType.Joint_Malfunction_Drift:
                    env_parameters_gen["malfunction_mask_0"] = self.param_generator_drift(1, 0)
                    env_parameters_gen["malfunction_mask_1"] = self.param_generator_drift(1, 0)
                    env_parameters_gen["target_velocity"] = self.param_generator(self.target_velocity)
                    env_parameters_gen["wind_0"] = self.param_generator(0)

                if env_type == EnvType.Velocity_Drift:
                    env_parameters_gen["malfunction_mask_0"] = self.param_generator(1)
                    env_parameters_gen["malfunction_mask_1"] = self.param_generator(1)
                    env_parameters_gen["target_velocity"] = self.param_generator_drift(self.target_velocity, self.target_velocity+1)
                    env_parameters_gen["wind_0"] = self.param_generator(0)

                if env_type == EnvType.Wind_Drift:
                    env_parameters_gen["malfunction_mask_0"] = self.param_generator(1)
                    env_parameters_gen["malfunction_mask_1"] = self.param_generator(1)
                    env_parameters_gen["target_velocity"] = self.param_generator(self.target_velocity)
                    env_parameters_gen["wind_0"] = self.param_generator_drift(0, -4)

        return param_dict, env_parameters_gen

    def get_params(self):
        self.env_parameters["malfunction_mask"][0] = next(self.env_parameters_gen["malfunction_mask_0"])
        self.env_parameters["malfunction_mask"][1] = next(self.env_parameters_gen["malfunction_mask_1"])
        self.env_parameters["target_velocity"] = next(self.env_parameters_gen["target_velocity"])
        self.env_parameters["wind"][0] = next(self.env_parameters_gen["wind_0"])

        return self.env_parameters

    def param_generator_drift(self, start_val, end_val):
        d_val = end_val - start_val
        for internal_counter in range(self.change_freq):
            yield start_val + ((internal_counter / self.change_freq) * d_val)

    def param_generator(self, val):
        for internal_counter in range(self.change_freq):
            yield val
