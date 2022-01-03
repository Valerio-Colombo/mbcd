import enum
from mbcd.envs.envs_enum import EnvType, SimType


class ExpType(enum.Enum):
    Normal = {"sim": SimType.HalfCheetah,
              "tasks": [EnvType.Normal],
              "change_freq": [30000]}
    Scheduler_Test = {"sim": SimType.HalfCheetah,
                      "tasks": [EnvType.Normal, EnvType.Joint_Malfunction_Drift, EnvType.Joint_Malfunction],
                      "change_freq": [1000, 500, 1500]}
    Base_HC = {"sim": SimType.HalfCheetah,
               "tasks": [EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                         EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                         EnvType.Wind, EnvType.Normal, EnvType.Velocity, EnvType.Joint_Malfunction],
               "change_freq": 40000}
    Base_Switch_Test = {"sim": SimType.HalfCheetah,
                        "tasks": [EnvType.Normal, EnvType.Joint_Malfunction],
                        "change_freq": [2000, 10000]}
    Base_Drift_Switch_Test = {"sim": SimType.HalfCheetah,
                              "tasks": [EnvType.Normal, EnvType.Joint_Malfunction_Drift, EnvType.Joint_Malfunction],
                              "change_freq": [2000, 20000, 1000]}
    Base_Short_Drift_Switch_Test = {"sim": SimType.HalfCheetah,
                                    "tasks": [EnvType.Normal, EnvType.Joint_Malfunction_Drift, EnvType.Joint_Malfunction],
                                    "change_freq": [2000, 10000, 1000]}
    Base_Wind_Test = {"sim": SimType.HalfCheetah,
                      "tasks": [EnvType.Wind],
                      "change_freq": 40000}
    Drift_End_HC = {"sim": SimType.HalfCheetah,
                    "tasks": [EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                              EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                              EnvType.Wind_Drift, EnvType.Velocity_Drift, EnvType.Joint_Malfunction_Drift],
                    "change_freq": 40000}
    Drift_Mid_HC = {"sim": SimType.HalfCheetah,
                    "tasks": [EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                              EnvType.Wind_Drift, EnvType.Velocity_Drift, EnvType.Joint_Malfunction_Drift,
                              EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity],
                    "change_freq": 40000}
    Drift_Test_Velocity = {"sim": SimType.HalfCheetah,
                           "tasks": [EnvType.Velocity_Drift, EnvType.Velocity_Drift],
                           "change_freq": 1000}
    Drift_Test_All = {"sim": SimType.HalfCheetah,
                      "tasks": [EnvType.Joint_Malfunction_Drift, EnvType.Wind_Drift, EnvType.Velocity_Drift],
                      "change_freq": 1000}


