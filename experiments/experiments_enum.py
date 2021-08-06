import enum
from mbcd.envs.envs_enum import EnvType


class ExpType(enum.Enum):
    Base = [EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
            EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
            EnvType.Wind, EnvType.Normal, EnvType.Velocity, EnvType.Joint_Malfunction]
    Drift_End = [EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                 EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                 EnvType.Wind_Drift, EnvType.Velocity_Drift, EnvType.Joint_Malfunction_Drift]
    Drift_Mid = [EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity,
                 EnvType.Wind_Drift, EnvType.Velocity_Drift, EnvType.Joint_Malfunction_Drift,
                 EnvType.Normal, EnvType.Joint_Malfunction, EnvType.Wind, EnvType.Velocity]
