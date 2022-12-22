from logging import RootLogger
import os
import scipy.io as sio

_BODY_PROFILES = ['mouse22', 'rat7m', 'rat16', 'rat23', 'jesse_skeleton']

ROOT = os.path.abspath(os.path.dirname(__file__))

def load_body_profile(name):
    assert name in _BODY_PROFILES
    profile = sio.loadmat(os.path.join(ROOT, f"{name}.mat"))
    limbs = profile["joints_idx"] - 1
    joints_name = [j[0][0] for j in profile["joint_names"]]
    return {'joints_name': joints_name, 'limbs': limbs}

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