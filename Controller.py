"""
Vehicle controller
"""
import CLF_CBF_QP

class Controller():
    """
    Class for vehicle controller
    """
    def __init__(self, cont_goal, cont_flag, method_param, veh_param, lane, other_vehicles) -> None:
        self.goal = cont_goal
        # 1 for CBF-CLF-QP controller (ego vehicle),
        # 2 for CLF-QP controller (surrounding vehicle)
        self.flag = cont_flag
        self.param_opt = method_param
        self.param_sys = veh_param
        self.lane = lane
        self.other_vehicles = other_vehicles

    def control(self, state, last_input, lane_id, input_log, init_lane_id, dir_flag, acc_flag):
        """
        Computes optimal control input 
        """
        if self.flag == 1:  # CLF-CBF-QP (ego)
            cbf = CLF_CBF_QP.CLF_CBF_QP(self.param_opt, self.param_sys, self.goal,
                                        self.lane, self.other_vehicles)
            [acc_flag, optimal_input, e] = cbf.get_optimal_input(state, last_input, lane_id,
                                                                     input_log, init_lane_id,
                                                                     dir_flag, acc_flag)
        # elif self.flag == 2:    # CLF-QP (surrounding)
        #     clf = CLF_QP.CLF_QP(self.param_opt, self.param_sys, self.goal, self.lane)
            # [acc_flag, optimal_input, e] = clf.get_optimal_input(state, last_input,
            #                                                      lane_id, input_log,
            #                                                      init_lane_id, dir_flag)
        return [acc_flag, optimal_input, e]
