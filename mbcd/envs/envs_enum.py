import enum


class EnvType(enum.Enum):
    Normal = 0
    Joint_Malfunction = 1
    Wind = 2
    Velocity = 3
    Joint_Malfunction_Drift = 4
    Wind_Drift = 5
    Velocity_Drift = 6
