from gekko import GEKKO
import numpy as np
import math
import matplotlib.pyplot as plt

class coordinate_system(object):
    def __init__(self, **kwargs):
        self.set_coord(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def set_coord(self, coord):
        self.coord=coord
    def visualize(self, **kwargs):
        ax1.arrow(self.coord[0,2], self.coord[1, 2], self.coord[0, 0], self.coord[1, 0], color=kwargs["color"], fc=kwargs["color"], ec="black", alpha=.6, width=.1,
                  head_width=.3, head_length=.3)
        ax1.arrow(self.coord[0,2], self.coord[1, 2], self.coord[0, 1], self.coord[1, 1], color=kwargs["color"], fc=kwargs["color"], ec="black", alpha=.6, width=.1,
                  head_width=.3, head_length=.3)
        circle1 = plt.Circle((self.coord[0,2], self.coord[1, 2]), 1, color=kwargs["color"], alpha=.2)
        ax1.add_patch(circle1)
    def transform(self, mat):
        dummy=self
        dummy.coord=np.dot(self.coord, mat)
        return dummy
    def get_information(self):
        return {"coord": self.coord}
class control_over_manifold(object):
    def __init__(self, **kwargs):
        self.m = GEKKO()  # initialize gekko
        nt = 101
        p = np.zeros(nt)  # mark final time point
        p[-1] = 1.0
        self.final = self.m.Param(value=p)
        self.m.time = np.linspace(0, 2, nt)

        self.control_input = {"u_0": self.m.Var(value=0, lb=-10, ub=10),
                      "u_1": self.m.Var(value=0, lb=-10, ub=10)}
        self.cost=self.m.Var(value=0)
    def get_velocity(self):
        vx = np.array(self.state["x1_2"])
        vy = np.array(self.state["x1_3"])
        v = np.sqrt(vx ** 2 + vy ** 2)
        return v, vx, vy
    def set_initial_state(self, init_state):
        self.state= {"x1_0": self.m.Var(value=init_state[0]),
                     "x1_1": self.m.Var(value=init_state[1]),
                     "x1_2": self.m.Var(value=init_state[2]),
                     "x1_3": self.m.Var(value=init_state[3])}

    def control(self, target):
        # Equations
        self.m.Equation(self.state["x1_0"].dt() == self.state["x1_2"])
        self.m.Equation(self.state["x1_1"].dt() == self.state["x1_3"])
        self.m.Equation(self.state["x1_2"].dt() == self.control_input["u_0"])
        self.m.Equation(self.state["x1_3"].dt() == self.control_input["u_1"])
        self.m.Equation((self.state["x1_0"] - target[0]+0.1) * self.final >= 0)
        self.m.Equation((self.state["x1_1"] - target[1]+0.1) * self.final >= 0)
        self.m.Equation((self.state["x1_0"] - target[0]-0.1) * self.final <= 0)
        self.m.Equation((self.state["x1_1"] - target[1]-0.1) * self.final <= 0)
        self.m.Equation(self.cost.dt() == 0.5 * self.control_input["u_0"] ** 2 + 0.5 * self.control_input["u_1"] ** 2)
        self.m.Obj(self.cost * self.final)  # Objective function
        self.m.options.IMODE = 6  # optimal control mode
        self.m.solve(disp=False)  # solve
    def get_information(self):
        return {"state": self.state, "time": self.m.time, "cost": self.cost}

angle_b=np.radians(30)
angle_c=np.radians(-10)
angle_d=np.radians(50)
transf_mat_b=np.array([[np.cos(angle_b), -np.sin(angle_b), 2], [np.sin(angle_b), np.cos(angle_b), 1], [0, 0, 1]])
transf_mat_c=np.array([[np.cos(angle_c), -np.sin(angle_c), 1.3], [np.sin(angle_c), np.cos(angle_c), 3.5], [0, 0, 1]])
transf_mat_d=np.array([[np.cos(angle_d), -np.sin(angle_d), 4], [np.sin(angle_d), np.cos(angle_d), 3], [0, 0, 1]])

transf_mat_no_rot_b=np.array([[np.cos(angle_b), -np.sin(angle_b), 0], [np.sin(angle_b), np.cos(angle_b), 0], [0, 0, 1]])
transf_mat_no_rot_c=np.array([[np.cos(angle_c), -np.sin(angle_c), 0], [np.sin(angle_c), np.cos(angle_c), 0], [0, 0, 1]])
transf_mat_no_rot_d=np.array([[np.cos(angle_d), -np.sin(angle_d), 0], [np.sin(angle_d), np.cos(angle_d), 0], [0, 0, 1]])

inv_transf_mat_b=np.linalg.inv(transf_mat_b)
inv_transf_mat_c=np.linalg.inv(transf_mat_c)
no_rot_inv_transf_mat_b=np.linalg.inv(transf_mat_no_rot_b)
no_rot_inv_transf_mat_c=np.linalg.inv(transf_mat_no_rot_c)

fig, (ax1, ax2) = plt.subplots(2, 1)
coord_a=coordinate_system()
coord_b=coordinate_system()
coord_c=coordinate_system()
coord_d=coordinate_system()
coord_a.visualize(**{"color": "green"})
coord_b=coord_b.transform(transf_mat_b)
coord_b.visualize(**{"color": "orange"})
coord_c=coord_c.transform(transf_mat_c)
coord_c.visualize(**{"color": "red"})
coord_d=coord_d.transform(transf_mat_d)
coord_d.visualize(**{"color": [.2, 0, 1]})


abc=control_over_manifold()
abc.set_initial_state([0, 0, .3, -.5])
abc.control([2, 0.3])
rt=abc.get_information()
v,vx,vy=abc.get_velocity()
time=rt["time"]
cost=rt["cost"].value
traj=np.vstack((rt["state"]["x1_0"].value, rt["state"]["x1_1"].value, np.ones((1, len(rt["state"]["x1_1"].value))), vx, vy, np.ones((1, len(rt["state"]["x1_1"].value)))))




end_pos_new_coord=np.dot(inv_transf_mat_b, traj[0:3, -1])
last_vel=traj[3:, -1]
end_vel_new_coord=np.dot(no_rot_inv_transf_mat_b, traj[3:, -1])

new_state=np.vstack((end_pos_new_coord[0], end_pos_new_coord[1], end_vel_new_coord[0], end_vel_new_coord[1]))

abc=control_over_manifold()
abc.set_initial_state(new_state)
abc.control([.3,2.3])
rt=abc.get_information()
traj_b=np.vstack((rt["state"]["x1_0"].value, rt["state"]["x1_1"].value, np.ones((1, len(rt["state"]["x1_1"].value))), vx, vy, np.ones((1, len(rt["state"]["x1_1"].value)))))
end_pos_new_coord=np.dot(transf_mat_b, traj_b[0:3, :])
time_b=rt["time"]+time[-1]
cost_b=rt["cost"].value
vb,vx,vy=abc.get_velocity()





#####################
### Visualization ###
#####################
ax1.plot(traj[0, :], traj[1, :],'-',label="position ref 1", color="blue", linewidth=2.3)
ax1.plot(end_pos_new_coord[0, :], end_pos_new_coord[1, :],'-',label="position ref 2", color="cyan", linewidth=2.3)

#ax1.legend(loc='lower right')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid()
ax1.set(xlim=(-.3, 8.4), ylim=(-1.3, 5.4))
ax1.set_aspect('equal', adjustable='box')
ax3 = ax2.twinx()
ax2.plot(time, cost,'k-',label="cost ref1", color="red", alpha=.7)
ax3.plot(time, v,'k-',label="velocity ref1", color="blue", alpha=.7)
ax2.plot(time_b, cost_b,'k-',label="cost ref2", color="orange", alpha=.7)
ax3.plot(time_b, vb,'k-',label="velocity ref2", color="cyan", alpha=.7)
ax2.legend(loc='upper left')
ax3.legend(loc='lower right')
ax2.set_xlabel('t')
ax2.set_ylabel('u')
ax3.set_ylabel('v')
ax2.grid()
plt.show()

