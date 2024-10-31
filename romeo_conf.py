# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import os

import numpy as np
import pinocchio as pin
from example_robot_data.robots_loader import getModelPath
from pinocchio.visualize import MeshcatVisualizer
from pprint import pprint
np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

### TODO ###
### First set of trajectories (step length = 0.15): "romeo_walking_traj015.npz" ###
### Second set of trajectories (step length = 0.30): "romeo_walking_traj030.npz" ###
# DATA_FILE_TSID = "romeo_walking_traj030.npz"

DATA_FILE_LIPM = "romeo_walking_traj_lipm.npz"

# robot parameters
# ----------------------------------------------
urdf = "/romeo_description/urdf/romeo_small.urdf"
path = getModelPath(urdf)
urdf = path + urdf
srdf = path + '/romeo_description/srdf/romeo_small.srdf'
path = os.path.join(path, '../..')

# filename = str(os.path.dirname(os.path.abspath(__file__)))
# path = filename + "/../models/romeo"
# urdf = path + "/urdf/romeo.urdf"
# srdf = path + "/srdf/romeo_collision.srdf"

nv = 37
foot_scaling = 1.0
lxp = foot_scaling * 0.10  # foot length in positive x direction
lxn = foot_scaling * 0.05  # foot length in negative x direction
lyp = foot_scaling * 0.05  # foot length in positive y direction
lyn = foot_scaling * 0.05  # foot length in negative y direction
lz = 0.07  # foot sole height with respect to ankle joint
mu = 0.7  # friction coefficient
fMin = 0.0  # minimum normal force
fMax = 1e6  # maximum normal force
rf_frame_name = "RAnkleRoll"  # right foot frame name
lf_frame_name = "LAnkleRoll"  # left foot frame name
contactNormal = np.matrix(
    [0.0, 0.0, 1.0]
).T  # direction of the normal to the contact surface

# configuration for LIPM trajectory optimization
# ----------------------------------------------
wu = 1e1    # CoP error squared cost weight
wc = 0      # CoM position error squared cost weight
wdc = 1e-1  # CoM velocity error squared cost weight
h = 0.58    # fixed CoM height
g = 9.81    # norm of the gravity vector
foot_step_0 = np.array([0.0, -0.096])  # initial foot step position in x-y
dt_mpc = 0.1  # sampling time interval
T_step = 1.2  # time needed for every step
step_length = 0.15  # fixed step length
step_height = 0.05  # fixed step height
nb_steps = 6  # number of desired walking steps

# configuration for TSID
# ----------------------------------------------
dt = 0.002  # controller time step
T_pre = 1.5  # simulation time before starting to walk
T_post = 1.5  # simulation time after walking


print("Select step size (0, 1):")
print(" 0: 15cm")
print(" 1: 30cm")
input_data = input()
if input_data == "0":
    DATA_FILE_TSID = "Ass_1_orc/romeo_walking_traj015.npz"
    print("Using step size=15cm")
elif input_data == "1":
    DATA_FILE_TSID = "Ass_1_orc/romeo_walking_traj030.npz"
    print("Using step size=30cm")
else:
    print("Insert a valid choice")
    exit(1)



w_com = 1.0  # weight of center of mass task
w_cop = 0.0  # weight of center of pressure task
w_am = 1e-6  # weight of angular momentum task
w_foot = 1e0  # weight of the foot motion task
w_contact = 1e2  # weight of the foot in contact
w_posture = 0  # weight of joint posture task
w_forceRef = 1e-5  # weight of force regularization task
w_torque_bounds = 1.0  # weight of the torque bounds
w_joint_bounds = 1.0

kp_contact = 10.0  # proportional gain of contact constraint
kp_foot = 10.0  # proportional gain of contact constraint
kp_com = 10.0  # proportional gain of center of mass task
kp_am = 10.0  # proportional gain of angular momentum task
kp_posture = 1.0  # proportional gain of joint posture task

print("Select weights (0, 1, 2):")
print(" 0: default weights")
print(" 1: weight to answer question 1")
print(" 2: weight to answer question 2")
input_data = input()
if input_data == "0":
    print("Using default weights")
elif input_data == "1":
    w_posture = 1e-2  # weight of joint posture task
    print(f"Using {w_posture=}")
elif input_data == "2":
    w_com = 1.0
    w_cop = 0.0
    w_am = 1e-4
    w_foot = 1e0
    w_contact = 1e2
    w_posture = 1e-3
    w_forceRef = 1e-4
    w_torque_bounds = 1.0
    w_joint_bounds = 1.0
    kp_contact = 10.0
    kp_foot = 10.0
    kp_com = 10.0
    kp_am = 50.0
    kp_posture = 1.0
else:
    print("Insert a valid choice")
    exit(1)


gain_vector = kp_posture * np.ones(nv - 6)
masks_posture = np.ones(nv - 6)
tau_max_scaling = 1.45  # scaling factor of torque bounds
v_max_scaling = 0.8


# configuration for viewer
# ----------------------------------------------
viewer = pin.visualize.MeshcatVisualizer
PRINT_N = 500  # print every PRINT_N time steps
DISPLAY_N = 20  # update robot configuration in viwewer every DISPLAY_N time steps
CAMERA_TRANSFORM = [
    3.578777551651001,
    1.2937744855880737,
    0.8885031342506409,
    0.4116811454296112,
    0.5468055009841919,
    0.6109083890914917,
    0.3978860676288605,
]