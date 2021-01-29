# -------------------------------------------------------------
# code developed by Michael Hartmann during his Ph.D.
# Reachability Analysis
#
# (C) 2020 Michael Hartmann, Graz, Austria
# Released under GNU GENERAL PUBLIC LICENSE
# email michael.hartmann@v2c2.at
# -------------------------------------------------------------
import reachab.src.reachability as rb
import matplotlib.pyplot as plt
'''
    Test me function. A user should see if the program is installed and a simple plot is available 
'''
def test_me():
    obj_reachability = rb.reachability()
    obj_reachability.test_function()

'''
    Reachability Analysis. 
    Omega_0: Initial state set
    U: Fixed control input set
    Params: Parameters
'''
def reach(Omega_0, U, params):
    ##################################
    ## REACHABILITY ANALYSIS PARAMS ##
    ##################################
    ra_params = {
        'T': params['time_horizon'],
        'N': params['steps'],
    }
    erg=[]
    obj_reach = rb.reachability(**ra_params)
    program = ['without_box', 'with_box']
    if (program[0] == params['box_function']):
        R, X = obj_reach.approximate_reachable_set_without_box(Omega_0, U)
    elif (program[1] == params['box_function']):
        R, X = obj_reach.approximate_reachable_set_with_box(Omega_0, U)
    for act_zono in R:
        zonoset = obj_reach.get_points_of_zonotype(act_zono)
        if (params['visualization'] == 'y'):
            obj_reach.obj_visual.filled_polygon(zonoset, params['face_color'], .2)
        erg.append(zonoset)
    return erg

'''
    Show the plots
'''
def show_all():
    plt.axis('equal')
    plt.grid("on")
    plt.show()




