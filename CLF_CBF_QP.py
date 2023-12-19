"""
CLF-CBF-QP controller 
"""
import numpy as np
import qp

class CLF_CBF_QP():
    """
    CLF-CBF-QP controller class
    """
    def __init__(self, cbf_param, veh_param, cont_goal, lane, other_vehicles) -> None:
        self.param_opt = cbf_param  # qp parameters (dict)
        self.param_sys = veh_param  # vehicle parameters (dict)
        self.goal = cont_goal       # control objective
        self.lane = lane            # driving lanes
        self.other_vehicles = other_vehicles

    def get_optimal_input(self, state, last_input, lane_id,
                          input_log, current_lane_id, dir_flag, acc_flag):
        """
        Solve QP to compute optimal control input 
        """
        # Load parameter and control object
        safety_factor = self.goal.safety_factor
        lim_speed = self.goal.lim_speed
        l_rc = self.param_sys['l_rc']
        l_fc = self.param_sys['l_fc']
        l_r = self.param_sys['l_r']
        l_f = self.param_sys['l_f']
        width = self.param_sys['width']
        dt = self.param_opt['dt']
        [x,y,psi,v] = state
        [acc, beta] = last_input
        alpha_y = self.param_opt['alpha_y']
        alpha_v = self.param_opt['alpha_y']
        alpha_yaw = self.param_opt['alpha_y']
        gamma_1 = self.param_opt['gamma_1']
        gamma_2 = self.param_opt['gamma_2']
        gamma_3 = self.param_opt['gamma_3']
        gamma_4 = self.param_opt['gamma_4']
        gamma_5 = self.param_opt['gamma_5']
        gamma_6 = self.param_opt['gamma_6']
        target_y = self.goal.target_y
        alpha_u = self.goal.grade_uncertainty   # road grade uncertainty parameter

        # Load surrounding vehicles
        current_lane_vehicles = []
        target_lane_vehicles = []
        nb_veh = len(self.other_vehicles)
        # Sort surrounding vehicles according to lane id information
        for idx in range(nb_veh):
            other_veh_lane = self.other_vehicles[idx].lane_id
            if other_veh_lane == current_lane_id or other_veh_lane == current_lane_id - dir_flag * 0.5:
                # Vehicle is in current lane
                if self.other_vehicles[idx].state[0] >= x:
                    # If vehicle is in front of ego vehicle
                    current_lane_vehicles.append(self.other_vehicles[idx])
            elif other_veh_lane == current_lane_id + dir_flag:
                # Vehicle is across dividing line
                if self.other_vehicles[idx].state[0] >= x:
                    current_lane_vehicles.append(self.other_vehicles[idx])
                target_lane_vehicles.append(self.other_vehicles[idx])
            elif other_veh_lane == current_lane_id + dir_flag:
                target_lane_vehicles.append(self.other_vehicles[idx])
            elif other_veh_lane == current_lane_id + 1.5 * dir_flag:
                target_lane_vehicles.append(self.other_vehicles)

        # closest leading vehicle in current lance
        car_fc = []
        car_fc_range = x + 100   # tuned parameter for range in which car_fc would be considered
        for _, current_lane_veh in enumerate(current_lane_vehicles):
            if current_lane_veh.state[0] <= car_fc_range:
                car_fc.append(current_lane_veh)
                car_fc_range = current_lane_veh.state[1]

        car_bt = []     # closest behind ego vehicle in target lane
        car_ft = []     # closest in front of ego vehicle in target lane
        car_bt_range = x - 100  # closest vehicle in target lane that is behind ego
        car_ft_range = x + 100  # closest vehicle in target lane that is in front of ego
        for _, target_lane_veh in enumerate(target_lane_vehicles):
            if target_lane_veh.state[0] <= x and target_lane_veh.state[0] >= car_bt_range:
                car_bt.append(target_lane_veh)
                car_bt_range = target_lane_veh.state[0]
            if target_lane_veh.state[0] >= x and target_lane_veh.state[0] <= car_ft_range:
                car_ft.append(target_lane_veh)
                car_ft_range = target_lane_veh.state[0]

        if lane_id == current_lane_id + dir_flag:
            acc_flag = 0   # indicates if vehicle is accelerating
        if acc_flag == 0:
            target_speed = self.goal.target_speed
        else:
            target_speed = lim_speed

        # CLF-CBF-QP formulation

        # Lateral position CLF
        h_y = y - target_y
        V_y = h_y**2
        phi0_y = 2 * h_y * (v * np.sin(psi) ) + alpha_y * V_y
        phi1_y = np.hstack(( 0, 2 * h_y * v * np.cos(psi) ))

        # Velocity CLF
        h_v = v - target_speed
        V_v = h_v**2
        phi0_v = alpha_v * V_v
        phi1_v = np.hstack((2*h_v, 0))

        # Yaw angle CLF
        h_yaw = psi
        V_yaw = psi**2
        phi0_yaw = alpha_yaw * V_yaw
        phi1_yaw = np.hstack((0, 2 * h_yaw * V_yaw * l_r))

        # Complete CLF
        Aclf = np.vstack((
            np.hstack((phi1_y, -1, 0, 0)),
            np.hstack((phi1_v, 0, -1, 0)),
            np.hstack((phi1_yaw, 0, 0, -1))
        ))
        bclf = np.array([
            [-phi0_y],
            [-phi0_v],
            [-phi0_yaw]
        ]).reshape((3,1))

        # Car_fc relevant CBFs
        if len(car_fc) == 0:
            Acbf1 = np.array([0,0,0,0,0])
            bcbf1 = np.array([0])
            Acbf2 = np.array([0,0,0,0,0])
            bcbf2 = np.array([0])
            h_cbf1 = []
            h_cbf2 = []
        else:
            # time-varying longitudinal distance between vehicle k and ego vehicle
            deltax_fc = abs(car_fc[0].state[0] - x) - l_rc - l_fc
            v_fc = car_fc[0].state_log[-1][3]
            a_fc = car_fc[0].input[0]

            # Distance based CBF
            h_cbf1 = deltax_fc
            h1dot = v_fc
            Lfh1 = np.array([-np.cos(psi) * v])
            Lgh1 = np.hstack((0, v * np.sin(psi)))
            Acbf1 = np.hstack((-1*Lgh1, 0, 0, 0))
            bcbf1 = np.array([Lfh1 + gamma_1 * h_cbf1 + h1dot])

            # Force based CBF
            if v > v_fc:
                h_cbf2 = deltax_fc - (1 + safety_factor) * (v - v_fc)**2 / (2 * self.goal.lim_acc)
                h2dot = v_fc - 1 / (2 * self.goal.lim_acc) * 2 * (v_fc - v) * a_fc
                Lfh2 = np.array([-np.cos(psi) * v])
                Lgh2 = alpha_u * np.hstack(( (-(1 + safety_factor) + (v_fc - v) / self.goal.lim_acc),
                                 v * np.sin(psi)))
                Acbf2 = np.hstack((-1*Lgh2, 0, 0, 0))
                bcbf2 = np.array([Lfh2 + gamma_2 * h_cbf2 + h2dot])
            else:
                h_cbf2 = deltax_fc - (1 + safety_factor) * v
                h2dot = v_fc
                Lfh2 = np.array([-np.cos(psi) * v])
                Lgh2 = alpha_u * np.hstack((-(1 + safety_factor), v * np.sin(psi)))
                Acbf2 = np.hstack((-Lgh2, 0, 0, 0))
                bcbf2 = Lfh2 + gamma_2 * h_cbf2 + h2dot


        # Car_bt relevant CBFs
        if len(car_bt) == 0:
            Acbf3 = np.array([0,0,0,0,0])
            bcbf3 = np.array([0])
            Acbf4 = np.array([0,0,0,0,0])
            bcbf4 = np.array([0])
            h_cbf3 = []
            h_cbf4 = []
        else:
            deltax_bt = abs(x - car_bt[0].state[0]) - l_fc - l_rc
            v_bt = car_bt[0].state_log[-1][3]
            a_bt = car_bt[0].input[0]
            if car_bt[0].state[0] <= (x - l_fc - l_rc):    # ego has lateral clearance with car_bt
                # Distance based CBF
                if v_bt <= v:
                    h_cbf3 = deltax_bt
                    h3dot = - v_bt
                    Lfh3 = np.array([np.cos(psi) * v])
                    Lgh3 = np.hstack((0, -v * np.sin(psi)))
                    Acbf3 = np.hstack((-1*Lgh3, 0, 0, 0))
                    bcbf3 = np.array([Lfh3 + gamma_3 * h_cbf3 + h3dot])
                else:
                    h_cbf3 = deltax_bt - 0.5 * (v_bt - v)**2 / self.goal.lim_acc
                    h3dot = - v_bt - (v_bt - v)/(self.goal.lim_acc) * a_bt
                    Lfh3 = np.cos(psi) * v
                    Lgh3 = np.hstack(( (v - v_bt)/(self.goal.lim_acc), - v * np.sin(psi) ))
                    Acbf3 = np.hstack((-1*Lgh3, 0, 0, 0))
                    bcbf3 = Lfh3 + gamma_3 * h_cbf3 + h3dot
            else:   # ego and car_bt overlap laterally
                h_cbf3 = dir_flag * (car_bt[0].state[1] - y - width - safety_factor)
                Lfh3 = - dir_flag * v * np.sin(psi)
                Lgh3 = np.array([0, -dir_flag * v * np.cos(psi)])
                Acbf3 = np.array([-Lgh3, 0, 0, 0])
                bcbf3 = np.array([Lfh3 + gamma_3 * h_cbf3])

            if v_bt > v:
                h_cbf4 = deltax_bt - (1 + safety_factor) * v_bt - 0.5 * (v_bt - v)**2 / (self.goal.lim_acc)
                h4_dot = - v_bt - (1 + safety_factor) * a_bt - (v_bt - v)/(self.goal.lim_acc) * a_bt
                Lfh4 = np.cos(psi) * v
                Lgh4 = alpha_u * np.hstack(( (v - v_bt)/(self.goal.lim_acc), - v * np.sin(psi) ))
                Acbf4 = np.hstack((-Lgh4, 0, 0, 0))
                bcbf4 = Lfh4 + gamma_4 * h_cbf4 + h4_dot
            else:
                h_cbf4 = deltax_bt - (1 + safety_factor) * v_bt
                h4_dot = - v_bt - (1 + safety_factor) * a_bt
                Lfh4 = np.array([np.cos(psi) * v])
                Lgh4 = alpha_u * np.hstack((0, - v * np.sin(psi)))
                Acbf4 = np.hstack((-Lgh4, 0, 0, 0))
                bcbf4 = Lfh4 + gamma_4 * h_cbf4 + h4_dot


        # Car_ft relevant CBFs
        if len(car_ft) == 0:
            Acbf5 = np.array([0,0,0,0,0])
            bcbf5 = np.array([0])
            Acbf6 = np.array([0,0,0,0,0])
            bcbf6 = np.array([0])
            h_cbf5 = []
            h_cbf6 = []
        else:
            deltax_ft = abs(x - car_ft[0].state[0]) - l_fc - l_rc
            v_ft = car_ft[0].state_log[-1][3]
            a_ft = car_ft[0].input[0]
            if car_ft[0].state[0] - x > l_fc + l_rc: # lateral clearance with car ft
                # Distance based CBF
                if v_ft > v:
                    h_cbf5 = deltax_ft
                    h5dot = v_ft
                    Lfh5 = np.array([-np.cos(psi) * v])
                    Lgh5 = np.hstack((0, v*np.sin(psi)))
                    Acbf5 = np.hstack((-1*Lgh5, 0, 0, 0))
                    bcbf5 = np.array([Lfh5 + gamma_5 * h_cbf5 + h5dot])
                else:
                    h_cbf5 = deltax_ft - 0.5 * (v_ft - v)**2 / (self.goal.lim_acc)
                    h5dot = v_ft - ( (v_ft - v) * a_ft) / (self.goal.lim_acc)
                    Lfh5 = -np.cos(psi) * v
                    Lgh5 = np.array([(v_ft - v)/(self.goal.lim_acc), v * np.sin(psi)]).reshape((2,))
                    Acbf5 = np.hstack((-Lgh5, 0, 0, 0))
                    bcbf5 = np.array([Lfh5 + gamma_5 * h_cbf5 + h5dot])
            else:
                h_cbf5 = dir_flag * (car_ft[0].state[1] - y - width - safety_factor)
                Lfh5 = np.array([- dir_flag * v * np.sin(psi)])
                Lgh5 = np.array([0, -dir_flag * v * np.cos(psi)])
                Acbf5 = np.array([-Lgh5, 0, 0, 0])
                bcbf5 = np.array([Lfh5 + gamma_5 * h_cbf5])
            # Force based CBF
            if v_ft < v:
                h_cbf6 = deltax_ft - (1 + safety_factor) * v - ( (v_ft - v) ** 2)/(2 * self.goal.lim_acc)
                h6dot = v_ft - a_ft * (v_ft - v)/(self.goal.lim_acc)
                Lfh6 = np.array([- np.cos(psi) * v])
                Lgh6 = alpha_u * np.array([-(1 + safety_factor) * (v_ft - v)/(self.goal.lim_acc), v*np.sin(psi)])
                Acbf6 = np.hstack((-Lgh6.reshape((2,)), 0, 0, 0))
                bcbf6 = Lfh6 + gamma_6 * h_cbf6 + h6dot
            else:
                h_cbf6 = deltax_ft - (1 + safety_factor) * v
                h6dot = v_ft
                Lfh6 = np.array([- np.cos(psi) * v])
                Lgh6 = alpha_u * np.hstack((-(1 + safety_factor), v * np.sin(psi)))
                Acbf6 = np.hstack((-Lgh6, 0, 0, 0))
                bcbf6 = Lfh6 + gamma_6 * h_cbf6 + h6dot

        # Adjust CBFs based on state (0 = keep current lane, do ACC,
        # 1 = change to left, -1 = change to right)
        if dir_flag == 0: # adaptive cruise control (only care about car_fc)
            Acbf3 = np.array([0,0,0,0,0])
            bcbf3 =np.array([0])
            Acbf4 = np.array([0,0,0,0,0])
            bcbf4 =np.array([0])
            Acbf5 = np.array([0,0,0,0,0])
            bcbf5 =np.array([0])
            Acbf6 = np.array([0,0,0,0,0])
            bcbf6 =np.array([0])
        else: # ego vehicle change lane
            if lane_id == current_lane_id + dir_flag:
                # ego is already in it's target lane, don't need car_fc, car_bt
                Acbf1 = np.array([0,0,0,0,0])
                bcbf1 =np.array([0])
                Acbf2 = np.array([0,0,0,0,0])
                bcbf2 =np.array([0])
                Acbf3 = np.array([0,0,0,0,0])
                bcbf3 =np.array([0])
                Acbf4 = np.array([0,0,0,0,0])
                bcbf4 =np.array([0])

        # Input constraint (accel limit, slip angle limit,
        # lateral accel limit, slip angle rate limit)
        A_u = np.array([
            [1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0],
            np.hstack((np.cos(psi + beta), 0, 0, 0, 0)),
            np.hstack((-np.cos(psi + beta), 0, 0, 0, 0))
        ])
        A_u0 = np.array([
            [0,1,0,0,0],
            [0,-1,0,0,0]
        ])

        b_u = np.array([
            [self.goal.lim_acc],
            [self.goal.lim_acc],
            [self.goal.lim_beta],
            [self.goal.lim_beta],
            [0.5 * 0.9 * 9.81],
            [0.5 * 0.9 * 9.81]
        ])

        b_u0 = np.array([
            [beta + 1 * self.goal.lim_beta_rate * dt],
            [-beta + 1 * self.goal.lim_beta_rate * dt],
        ]).reshape((2,1))

        Acbf = np.vstack((Aclf,
                          Acbf2,
                          Acbf4,
                          Acbf6,
                          A_u,
                          A_u0))

        b_cbf = np.vstack((bclf,
                           bcbf2.reshape((1,1)),
                           bcbf4.reshape((1,1)),
                           bcbf6.reshape((1,1)),
                           b_u,
                           b_u0))

        h_mat = self.param_opt['H']
        f_mat = self.param_opt['F']

        u = qp.qp_solver(h_mat, f_mat, Acbf, b_cbf)
        if u.size != 0:
            opt_control = np.array([
                [u[0]], # acceleration
                [u[1]]  # beta
            ]).reshape((2,1))
            e = 1
            if v >= target_speed + 0.5:
                acc_flag = 1
        else:
            print("No optimal input, QP not solvable")

        return [acc_flag, opt_control, e]
        