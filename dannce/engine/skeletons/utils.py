from logging import RootLogger
import os
import scipy.io as sio

ROOT = os.path.abspath(os.path.dirname(__file__))
_BODY_PROFILES = [file.split('.mat')[0] for file in os.listdir(ROOT) if file.endswith('.mat')]


# _BODY_PROFILES = {
#     ""
# }


def load_body_profile(name):
    assert name in _BODY_PROFILES, f"{name} not a valid skeleton profile"
    profile = sio.loadmat(os.path.join(ROOT, f"{name}.mat"))
    limbs = profile["joints_idx"] - 1
    return {'limbs': limbs}


SYMMETRY = {
    "mouse22": [
        [(1, 2), (0, 2)],
        [(0, 3), (1, 3)],
        [(9, 10), (13, 14)],
        [(10, 11), (14, 15)],
        [(11, 3), (15, 3)],
        [(12, 13), (8, 9)],
        [(16, 17), (19, 20)],
        [(17, 18), (20, 21)],
        [(18, 4), (21, 4)],
    ]
}
