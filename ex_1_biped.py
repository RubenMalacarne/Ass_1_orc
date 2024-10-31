import time
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import plot_utils as plut
import talos_conf as conf
from numpy import nan
from numpy.linalg import norm as norm
from tsid_biped import TsidBiped
import time

# SET FREQUENCY BY USER DURING RUN TIME PROCESS
def set_feqeuncy(val):
    if (val == 1 or val == 1.0):
        f = np.array([0.0, 1.0, 0.0])
    elif(val == 0.5):
        f = np.array([0.0, 0.5, 0.0])
    else:
        raise ValueError("Not valid option! choice between 1 or 0.5")
    return f

try:
    freq = int(input("select frequency (0.5 or 1): "))
except ValueError:
    print("not valid option! Using default frequenxy --> 0.5.")
    freq = 0.5
    
f = set_feqeuncy(freq)

print("".center(conf.LINE_WIDTH, "#"))
print(" TSID - Biped Sin Tracking ".center(conf.LINE_WIDTH, "#"))
print("".center(conf.LINE_WIDTH, "#"), "\n")

tsid = TsidBiped(conf, conf.viewer)

N = conf.N_SIMULATION
com_pos = np.empty((3, N)) * nan
com_vel = np.empty((3, N)) * nan
com_acc = np.empty((3, N)) * nan

com_pos_ref = np.empty((3, N)) * nan
com_vel_ref = np.empty((3, N)) * nan
com_acc_ref = np.empty((3, N)) * nan
com_acc_des = np.empty((3, N)) * nan  # acc_des = acc_ref - Kp*pos_err - Kd*vel_err

offset = tsid.robot.com(tsid.formulation.data())

amp = np.array([0.0, 0.05, 0.00])
# f = np.array([0.0,   .5, 0.0]) #see  up

two_pi_f = 2 * np.pi * f
two_pi_f_amp = two_pi_f * amp
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp

sampleCom = tsid.trajCom.computeNext()
samplePosture = tsid.trajPosture.computeNext()

t = 0.0
q, v = tsid.q, tsid.v

for i in range(0, N):
    time_start = time.time()

    sampleCom.value(offset + amp * np.sin(two_pi_f * t))
    sampleCom.derivative(two_pi_f_amp * np.cos(two_pi_f * t))
    sampleCom.second_derivative(-two_pi_f_squared_amp * np.sin(two_pi_f * t))

    tsid.comTask.setReference(sampleCom)
    tsid.postureTask.setReference(samplePosture)

    HQPData = tsid.formulation.computeProblemData(t, q, v)

    sol = tsid.solver.solve(HQPData)
    if sol.status != 0:
        print("QP problem could not be solved! Error code:", sol.status)
        break

    tau = tsid.formulation.getActuatorForces(sol)
    dv = tsid.formulation.getAccelerations(sol)

    com_pos[:, i] = tsid.robot.com(tsid.formulation.data())
    com_vel[:, i] = tsid.robot.com_vel(tsid.formulation.data())
    com_acc[:, i] = tsid.comTask.getAcceleration(dv)
    com_pos_ref[:, i] = sampleCom.value()
    com_vel_ref[:, i] = sampleCom.derivative()
    com_acc_ref[:, i] = sampleCom.second_derivative()
    com_acc_des[:, i] = tsid.comTask.getDesiredAcceleration

    if i % conf.PRINT_N == 0:
        print("Time %.3f" % (t))
        if tsid.formulation.checkContact(tsid.contactRF.name, sol):
            f = tsid.formulation.getContactForce(tsid.contactRF.name, sol)
            print(
                "\tnormal force %s: %.1f"
                % (tsid.contactRF.name.ljust(20, "."), tsid.contactRF.getNormalForce(f))
            )

        if tsid.formulation.checkContact(tsid.contactLF.name, sol):
            f = tsid.formulation.getContactForce(tsid.contactLF.name, sol)
            print(
                "\tnormal force %s: %.1f"
                % (tsid.contactLF.name.ljust(20, "."), tsid.contactLF.getNormalForce(f))
            )

        print(
            "\ttracking err %s: %.3f"
            % (tsid.comTask.name.ljust(20, "."), norm(tsid.comTask.position_error, 2))
        )
        print("\t||v||: %.3f\t ||dv||: %.3f" % (norm(v, 2), norm(dv)))

    q, v = tsid.integrate_dv(q, v, dv, conf.dt)
    t += conf.dt

    if i % conf.DISPLAY_N == 0:
        tsid.display(q)

    time_spent = time.time() - time_start
    if time_spent < conf.dt:
        time.sleep(conf.dt - time_spent)

# PLOT POSIZIONE
time = np.arange(0.0, N * conf.dt, conf.dt)

(f, ax) = plut.create_empty_figure(3, 1)
for i in range(3):
    ax[i].plot(time, com_pos[i, :], label="CoM " + str(i))
    ax[i].plot(time, com_pos_ref[i, :], "r:", label="CoM Ref " + str(i))
    ax[i].set_xlabel("Time [s]")
    ax[i].set_ylabel("CoM [m]")
    leg = ax[i].legend()
    leg.get_frame().set_alpha(0.5)
f.suptitle("Figure 1 CoM Position")
f.tight_layout(rect=[0, 0, 1, 0.95])  # Aggiusta il layout per il titolo
f.savefig("Figure_1_f1_pos_wcom_pkcom_10-10.png")

# PLOT VELOCITÀ
(f, ax) = plut.create_empty_figure(3, 1)
for i in range(3):
    ax[i].plot(time, com_vel[i, :], label="CoM Vel " + str(i))
    ax[i].plot(time, com_vel_ref[i, :], "r:", label="CoM Vel Ref " + str(i))
    ax[i].set_xlabel("Time [s]")
    ax[i].set_ylabel("CoM Vel [m/s]")
    leg = ax[i].legend()
    leg.get_frame().set_alpha(0.5)
f.suptitle("Figure 2 CoM Velocity")
f.tight_layout(rect=[0, 0, 1, 0.95])
f.savefig("Figure_2_f1_vel_wcom_pkcom_10-10.png")

# PLOT ACCELERAZIONE
(f, ax) = plut.create_empty_figure(3, 1)
for i in range(3):
    ax[i].plot(time, com_acc[i, :], label="CoM Acc " + str(i))
    ax[i].plot(time, com_acc_ref[i, :], "r:", label="CoM Acc Ref " + str(i))
    ax[i].plot(time, com_acc_des[i, :], "g--", label="CoM Acc Des " + str(i))
    ax[i].set_xlabel("Time [s]")
    ax[i].set_ylabel("CoM Acc [m/s^2]")
    leg = ax[i].legend()
    leg.get_frame().set_alpha(0.5)
f.suptitle("Figure 3 CoM Acceleration")
f.tight_layout(rect=[0, 0, 1, 0.95])
f.savefig("Figure_3_f1_acc_wcom_pkcom_10-10.png")

# PLOT COMBINATO PER POSIZIONE, VELOCITÀ E ACCELERAZIONE SULL'ASSE Y
f, ax = plt.subplots(3, 1, figsize=(8, 12))  # Crea una finestra con 3 righe di grafici
y_index = 1  # Indice per l'asse y

# Posizione con traiettoria di riferimento
ax[0].plot(time, com_pos[y_index, :], label="CoM Position Y")
ax[0].plot(time, com_pos_ref[y_index, :], "r:", label="CoM Position Ref Y")
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Position [m]")
ax[0].legend()
ax[0].set_title("Position Y Axis")

# Velocità con traiettoria di riferimento
ax[1].plot(time, com_vel[y_index, :], label="CoM Velocity Y")
ax[1].plot(time, com_vel_ref[y_index, :], "r:", label="CoM Velocity Ref Y")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Velocity [m/s]")
ax[1].legend()
ax[1].set_title("Velocity Y Axis")

# Accelerazione con traiettoria di riferimento e accelerazione desiderata
ax[2].plot(time, com_acc[y_index, :], label="CoM Acceleration Y")
ax[2].plot(time, com_acc_ref[y_index, :], "r:", label="CoM Acceleration Ref Y")
ax[2].plot(time, com_acc_des[y_index, :], "g--", label="CoM Desired Acceleration Y")
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Acceleration [m/s^2]")
ax[2].legend()
ax[2].set_title("Acceleration Y Axis")

# Layout e salvataggio
f.suptitle("Position, Velocity, and Acceleration on Y Axis with References")
f.tight_layout(rect=[0, 0, 1, 0.95])  # Aggiusta il layout per il titolo
f.savefig("Figure_Y_axis_combined_position_velocity_acceleration.png")  # Salva la figura


plt.show()
