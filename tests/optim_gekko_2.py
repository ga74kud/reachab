from gekko import GEKKO
import numpy as np
import math
import matplotlib.pyplot as plt

class coordinate_system(object):
    def __init__(self, **kwargs):
        self.coord=None
    def set_coord(self, coord):
        self.coord=coord
    def visualize(self, **kwargs):
        ax1.arrow(self.coord[0,2], self.coord[1, 2], self.coord[0, 0], self.coord[1, 0], color=kwargs["color"])
        ax1.arrow(self.coord[0,2], self.coord[1, 2], self.coord[0, 1], self.coord[1, 1], color=kwargs["color"])
    def transform(self, mat):
        self.coord=np.dot(self.coord, mat)
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

        self.control_input = {"u_0": self.m.Var(value=0, lb=-1, ub=1),
                      "u_1": self.m.Var(value=0, lb=-1, ub=1)}
        self.cost=self.m.Var(value=0)
    def set_initial_state(self, init_state):
        self.state= {"x1_0": self.m.Var(value=init_state[0]),
                     "x1_1": self.m.Var(value=init_state[1]),
                     "x1_2": self.m.Var(value=init_state[2]),
                     "x1_3": self.m.Var(value=init_state[3])}

    def control(self):
        # Equations
        self.m.Equation(self.state["x1_0"].dt() == self.state["x1_2"])
        self.m.Equation(self.state["x1_1"].dt() == self.state["x1_3"])
        self.m.Equation(self.state["x1_2"].dt() == self.control_input["u_0"])
        self.m.Equation(self.state["x1_3"].dt() == self.control_input["u_1"])
        self.m.Equation((self.state["x1_0"] - 2) * self.final >= 0)
        self.m.Equation((self.state["x1_1"] - 1.3) * self.final >= 0)
        self.m.Equation(self.cost.dt() == 0.5 * self.control_input["u_0"] ** 2 + 0.5 * self.control_input["u_1"] ** 2)
        self.m.Obj(self.cost * self.final)  # Objective function
        self.m.options.IMODE = 6  # optimal control mode
        self.m.solve(disp=False)  # solve
    def get_information(self):
        return {"state": self.state, "time": self.m.time, "cost": self.cost}

angle=np.radians(30)
transf_mat=np.array([[np.cos(angle), -np.sin(angle), 2], [np.sin(angle), np.cos(angle), 1], [0, 0, 1]])
inv_transf_mat=np.linalg.inv(transf_mat)
fig, (ax1, ax2) = plt.subplots(2, 1)
coord_a=coordinate_system()
coord_a.set_coord(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
coord_a.visualize(**{"color": "green"})
coord_a.transform(transf_mat)
coord_a.visualize(**{"color": "orange"})
coord=coord_a.get_information()
abc=control_over_manifold()
abc.set_initial_state([0, 0, .3, -.2])
abc.control()
rt=abc.get_information()
traj=np.vstack((rt["state"]["x1_0"].value, rt["state"]["x1_1"].value, np.ones((1, len(rt["state"]["x1_1"].value)))))
end_pos_new_coord=np.dot(inv_transf_mat, traj[:, -1])
ax1.plot(rt["state"]["x1_0"].value, rt["state"]["x1_1"].value,'k-',label="position", color="blue")
ax1.legend(loc='best')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set(xlim=(-.3, 3), ylim=(-.3, 3))
ax1.set_aspect('equal', adjustable='box')
vx=np.array(rt["state"]["x1_2"])
vy=np.array(rt["state"]["x1_3"])
v=np.sqrt(vx**2+vy**2)
ax3 = ax2.twinx()
ax2.plot(rt["time"], rt["cost"].value,'k-',label="control input", color="red", alpha=.7)
ax3.plot(rt["time"], v,'k-',label="velocity", color="blue", alpha=.7)
ax2.legend(loc='upper left')
ax3.legend(loc='lower right')
ax2.set_xlabel('t')
ax2.set_ylabel('u')
ax3.set_ylabel('v')
plt.show()

