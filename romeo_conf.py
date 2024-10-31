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

## CONFIGURATION TESTED #########################################################################
def select_weights_and_gains(option):
    if option == 0:
        # Configuration 0: Standard configuraiton 
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
        
    elif option == 1:
        # Configuration 1: First solution first point
        w_com = 1.0  # weight of center of mass task
        w_cop = 0.0  # weight of center of pressure task
        w_am = 1e-6  # weight of angular momentum task
        w_foot = 1e0  # weight of the foot motion task
        w_contact = 1e2  # weight of the foot in contact
        w_posture = 1e-2  # weight of joint posture task
        w_forceRef = 1e-5  # weight of force regularization task
        w_torque_bounds = 1.0  # weight of the torque bounds
        w_joint_bounds = 1.0
        
        kp_contact = 10.0  # proportional gain of contact constraint
        kp_foot = 10.0  # proportional gain of contact constraint
        kp_com = 10.0  # proportional gain of center of mass task
        kp_am = 10.0  # proportional gain of angular momentum task
        kp_posture = 1.0  # proportional gain of joint posture task

    elif option == 2:
        # Configuration 2: way ti achive solution second point
        # Type of the configuration for the second part
        w_com = 17.0 
        w_cop = 0.0
        w_am = 1e-4
        w_foot = 1e0
        w_contact = 1e2
        w_posture = 1e-1
        w_forceRef = 1e-7
        w_torque_bounds = -1.0
        w_joint_bounds = 1.0
        kp_contact = 10.0
        kp_foot = 10.0
        kp_com = 100.0
        kp_am = 10.0
        kp_posture = -1.0
        
    elif option == 3:
        # Configuration 3 (Selezionata come migliore)
        # best configuration for second point
        w_com = 1.0
        w_cop = 0.0
        w_am = 1e-4
        w_foot = 1e0
        w_contact = 1e2
        w_posture = 1e-3
        w_forceRef = 1e-4       #forza che al piede sul suolo
        w_torque_bounds = 1.0
        w_joint_bounds = 1.0
        kp_contact = 10.0
        kp_foot = 10.0
        kp_com = 10.0
        kp_am = 50.0
        kp_posture = 1.0
    else:
        raise ValueError("Opzione non valida! Scegliere tra 1, 2 o 3.")

    gain_vector = kp_posture * np.ones(nv - 6)
    masks_posture = np.ones(nv - 6)
    
    return {
        "w_com": w_com,
        "w_cop": w_cop,
        "w_am": w_am,
        "w_foot": w_foot,
        "w_contact": w_contact,
        "w_posture": w_posture,
        "w_forceRef": w_forceRef,
        "w_torque_bounds": w_torque_bounds,
        "w_joint_bounds": w_joint_bounds,
        "kp_contact": kp_contact,
        "kp_foot": kp_foot,
        "kp_com": kp_com,
        "kp_am": kp_am,
        "kp_posture": kp_posture,
        "gain_vector": gain_vector,
        "masks_posture": masks_posture
    }

def select_step_walk(val):
    if (val == 15):
        DATA_FILE_TSID = "romeo_walking_traj015.npz"
    elif(val == 30):
        DATA_FILE_TSID = "romeo_walking_traj030.npz"
    else:
        raise ValueError("Not valid option! choice between 15 or 30")
    return DATA_FILE_TSID

try:
    val = int(input("select step of walk (15 or 30): "))
except ValueError:
    print("not valid option! choice, Using default value --> step walk 15.")
    val = 15
    
try:
    option = int(input("Inserisci il valore desiderato per selezionare i pesi e i guadagni (es. 1, 2 o 3): "))
except ValueError:
    print("not valid option! choice, Using default value --> configuration 0.")
    option = 1 

config = select_weights_and_gains(option)
walk_step = select_step_walk(val)

print("setting value:")
for key, value in config.items():
    print(f"{key}: {value}")
    
    
DATA_FILE_TSID = walk_step

w_com = config["w_com"]
w_cop = config["w_cop"]
w_am = config["w_am"]
w_foot = config["w_foot"]
w_contact = config["w_contact"]
w_posture = config["w_posture"]
w_forceRef = config["w_forceRef"]
w_torque_bounds = config["w_torque_bounds"]
w_joint_bounds = config["w_joint_bounds"]
kp_contact = config["kp_contact"]
kp_foot = config["kp_foot"]
kp_com = config["kp_com"]
kp_am = config["kp_am"]
kp_posture = config["kp_posture"]
gain_vector = config["gain_vector"]
masks_posture = config["masks_posture"]


## END CODE ######################################################################### 

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