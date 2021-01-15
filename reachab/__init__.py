import reachab.src.reachability as rb
def test_me():
    obj_reachability = rb.reachability()
    obj_reachability.test_function()

def reach(Omega_0, U, program_select = 0):
    obj_reach = rb.reachability()
    program = ['without_box', 'with_box']
    if (program[0] == program[program_select]):
        R, X = obj_reach.approximate_reachable_set_without_box(Omega_0, U)
    elif (program[1] == program[program_select]):
        R, X = obj_reach.approximate_reachable_set_with_box(Omega_0, U)
    for act_zono in R:
        zonoset = obj_reach.get_points_of_zonotype(act_zono)
    for act_zono in X:
        zonoset = obj_reach.get_points_of_zonotype(act_zono)
    return zonoset
