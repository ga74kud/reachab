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
class control_over_manifold(object):
    def __init__(self, **kwargs):
        self.m = GEKKO()  # initialize gekko
        nt = 101
        p = np.zeros(nt)  # mark final time point
        p[-1] = 1.0
        self.final = self.m.Param(value=p)
        self.m.time = np.linspace(0, 2, nt)
        self.state= {"x1_0": self.m.Var(value=0),
                     "x1_1": self.m.Var(value=0),
                     "x1_2": self.m.Var(value=0.2),
                     "x1_3": self.m.Var(value=-.3)}
        self.control_input = {"u_0": self.m.Var(value=0, lb=-1, ub=1),
                      "u_1": self.m.Var(value=0, lb=-1, ub=1)}
        self.cost=self.m.Var(value=0)
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
fig, (ax1, ax2) = plt.subplots(2, 1)
a=coordinate_system()
a.set_coord(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
a.visualize(**{"color": "green"})
a.transform(transf_mat)
a.visualize(**{"color": "orange"})

abc=control_over_manifold()
abc.control()
rt=abc.get_information()
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

