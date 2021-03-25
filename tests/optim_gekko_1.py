from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# a = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
# b = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
# c = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
# d = np.array([[0, 0], [0, 0]])
# cont_sys=signal.StateSpace(a, b, c, d)
# disc_sys=cont_sys.to_discrete(0.1)

m = GEKKO() # initialize gekko
nt = 101
m.time = np.linspace(0,2,nt)
# Variables
x1_0 = m.Var(value=0)
x1_1 = m.Var(value=0)
x1_2 = m.Var(value=.2)
x1_3 = m.Var(value=-.3)
x2 = m.Var(value=0)
u_0 = m.Var(value=0,lb=-1,ub=1)
u_1 = m.Var(value=0,lb=-1,ub=1)
p = np.zeros(nt) # mark final time point
p[-1] = 1.0
final = m.Param(value=p)
# Equations
m.Equation(x1_0.dt()==x1_2)
m.Equation(x1_1.dt()==x1_3)
m.Equation(x1_2.dt()==u_0)
m.Equation(x1_3.dt()==u_1)
m.Equation((x1_0-1)*final >= 0)
m.Equation((x1_1-.3)*final >= 0)
m.Equation(x2.dt()==0.5*u_0**2+0.5*u_1**2)
m.Obj(x2*final) # Objective function
#m.Obj(np.sum(x2)) # Objective function
m.options.IMODE = 6 # optimal control mode
m.solve(disp=False) # solve
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x1_0.value, x1_1.value,'k-',label="position")
#plt.plot(m.time,x2.value,'b-',label=r'$x_2$')
#plt.plot(m.time,u.value,'r--',label=r'$u$')
ax1.legend(loc='best')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
vx=np.array(x1_2)
vy=np.array(x1_3)
v=np.sqrt(vx**2+vy**2)
ax3 = ax2.twinx()
ax2.plot(m.time, x2.value,'k-',label="control input", color="red", alpha=.7)
ax3.plot(m.time, v,'k-',label="velocity", color="blue", alpha=.7)
ax2.legend(loc='upper left')
ax3.legend(loc='lower right')
ax2.set_xlabel('t')
ax2.set_ylabel('u')
ax3.set_ylabel('v')
plt.show()

