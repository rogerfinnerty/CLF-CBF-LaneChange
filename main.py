"""
Implementation of CLF-CBF-QP method for autonomous lane change with 
uncertainty in road grade, based off of 
https://github.com/HybridRobotics/Lane-Change-CBF by 
Suiyi He, Jun Zeng, Bike Zhang, Koushil Sreenath
"""
import Lanes
import Vehicle
import EgoControllerGoal as ego
import Controller
import simulator as sim

import matplotlib.pyplot as plt
import numpy as np

def load_veh_params():
    """
    Load geometric vehicle parameters 
    """
    params = {
        'l_r': 1.74,
        'l_f': 1.11,
        'width': 1.86,
        'dt': 0.01,
        'l_fc': 2.15,
        'l_rc': 2.77
    }
    return params

def load_surrounding_veh_params(delta_t):
    """
    Load parameters for surrounding vehicle optimization problem 
    """
    surr_opt_params = {
        'alpha': 2,
        'dt': delta_t,
        'H': np.array([[0,0,0], [0,0.01,0],[0,0,1e-6]]),
        'F': np.array([[0],[0],[0]])
    }
    return surr_opt_params

def load_ego_params(delta_t):
    """
    Load parameters for ego vehicle optimization problem
    """
    ego_opt_params = {
        'alpha_y': 0.8,
        'alpha_v': 1.7,
        'alpha_yaw': 12,
        'gamma_1': 1,
        'gamma_2': 1,
        'gamma_3': 1,
        'gamma_4': 1,
        'gamma_5': 1,
        'gamma_6': 1,
        'dt': delta_t,
        'H': np.array([
            [0.01,0,0,0,0],
            [0,0,0,0,0],
            [0,0,15,0,0],
            [0,0,0,0.1,0],
            [0,0,0,0,400]
        ]),
        'F': np.array([[0],[0],[0],[0],[0]])
    }
    return ego_opt_params

def get_movement_log(state_log, input_log, param_sys):
    """
    Returns yaw log, velocity log, and steering angle log 
    """
    nb_steps = len(state_log)
    yaw_log = np.zeros((nb_steps, 1))
    velocity_log = np.zeros((nb_steps, 1))
    steering_log = np.zeros((nb_steps, 1))
    for idx in range(nb_steps):
        yaw_log[idx,:] = round(float(state_log[idx][2]), 3)
        velocity_log[idx, :] = round(float(state_log[idx][3]), 3)
        slip_angle = input_log[idx][1]
        steering_angle = np.arctan((param_sys['l_r']+param_sys['l_f'])/(param_sys['l_r']) * np.tan(slip_angle))
        steering_log[idx,:] = round(float(steering_angle), 3)

    return yaw_log, velocity_log,  steering_log

def test_simulation():
    """
    Single test simulation with ego, fc, ft, and bt cars
    """
    # Define simulation parameters
    dt = 0.01
    simulation_time = 6
    veh_params = load_veh_params()
    scenario = 1    # 1 for highway (wider roads), 2 for urban road
    if scenario == 1:
        width = 3.6
    else:
        width = 3

    # Load roadway
    road = Lanes.Road(num_lanes=3, lane_width=width, max_length=750)

    # Define driving scenario with three other normal cars
    init_input = np.array([[0],[0]])
    car1_state = np.array([[150], [0.5 * width], [0], [24]])
    car2_state = np.array([[170], [1.5 * width], [0], [28]])
    car3_state = np.array([[50], [1.5 * width], [0], [25]])
    car1 = Vehicle.Vehicle(0, veh_params, car1_state, init_input, [], 1, dt, road, 0, 0, 1, scenario)
    car2 = Vehicle.Vehicle(0, veh_params, car2_state, init_input, [], 1, dt, road, 0, 0, 1, scenario)
    car3 = Vehicle.Vehicle(0, veh_params, car3_state, init_input, [], 1, dt, road, 0, 0, 1, scenario)
    other_vehicles = [car1, car2, car3]

    # Define clf-cbf-qp
    ego_acc_flag = 0
    ego_initial_lane_id = 1 # the initial lane id of ego vehicle
    ego_direction_flag = 1  # the direction of the lane change process of ego vehicle
    ego_desired_speed = 27.5
    ego_initial_state = np.array([[100], [0.5 * width], [0], [ego_desired_speed]])
    ego_initial_input = np.array([[0],[0]])
    ego_lim_slip_angle = 15 * np.pi / 180
    ego_lim_acc = 0.3 * 9.81
    ego_lim_slip_rate = 15 * np.pi / 180
    ego_controller_flag = 1
    ego_safety_factor = 0.5         # range: 0.1~1, 1 means maximum safety
    # parameter accounting for uncertainty in road grade
    alpha_u = 1.0
    controller_goal = ego.EgoControllerGoal(ego_initial_lane_id, ego_direction_flag,
                                            ego_desired_speed, road,
                                            alpha_u, ego_lim_slip_angle,
                                            ego_lim_acc, ego_lim_slip_rate,
                                            ego_safety_factor, scenario)
    param_opt = load_ego_params(dt)
    controller0 = Controller.Controller(controller_goal, ego_controller_flag, param_opt,
                            veh_params, road, other_vehicles)


    # Define ego vehicle
    ego_car = Vehicle.Vehicle(1, veh_params, ego_initial_state, ego_initial_input, controller0,
                    ego_initial_lane_id, dt, road, ego_direction_flag,
                    ego_acc_flag, alpha_u, scenario)

    # Define simulator, start simulation
    simulator = sim.Simulator(road, ego_car, other_vehicles, dt)
    [state_log, input_log] = simulator.start_sim(simulation_time)

    # Plot
    yaw_log, velocity_log, steering_log = get_movement_log(state_log, input_log, veh_params)
    _, (ax1, ax2, ax3) = plt.subplots(3,1)
    steps = int(simulation_time/dt)+1
    time = np.linspace(0, simulation_time, steps).reshape((steps,1))
    ax1.plot(time, velocity_log)
    ax1.set_title('Velocity history')
    ax1.set_ylabel('m/s')
    ax1.set_xlabel('s')

    ax2.plot(time, steering_log)
    ax2.set_title('Steering history')
    ax2.set_ylabel('rad')
    ax2.set_xlabel('s')

    ax3.plot(time, yaw_log)
    ax3.set_title('Yaw history')
    ax3.set_ylabel('rad')
    ax3.set_xlabel('s')

    plt.show()

def alpha_test():
    """
    Computes the time to complete a lane change for a range of values
    for alpha_u, the road grade uncertainty parameter. 
    """
    # Define simulation parameters
    dt = 0.01
    simulation_time = 6
    veh_params = load_veh_params()
    scenario = 1    # 1 for highway (wider roads), 2 for urban road
    if scenario == 1:
        width = 3.6
    else:
        width = 3

    # Load roadway
    road = Lanes.Road(num_lanes=3, lane_width=width, max_length=750)

    # Define driving scenario with three other normal cars
    init_input = np.array([[0],[0]])
    car1_state = np.array([[150], [0.5 * width], [0], [24]])
    car2_state = np.array([[170], [1.5 * width], [0], [28]])
    car3_state = np.array([[50], [1.5 * width], [0], [25]])
    car1 = Vehicle.Vehicle(0, veh_params, car1_state, init_input, [], 1, dt, road, 0, 0, 1, scenario)
    car2 = Vehicle.Vehicle(0, veh_params, car2_state, init_input, [], 1, dt, road, 0, 0, 1, scenario)
    car3 = Vehicle.Vehicle(0, veh_params, car3_state, init_input, [], 1, dt, road, 0, 0, 1, scenario)
    other_vehicles = [car1, car2, car3]

    # Define clf-cbf-qp
    ego_acc_flag = 0
    ego_initial_lane_id = 1 # the initial lane id of ego vehicle
    ego_direction_flag = 1  # the direction of the lane change process of ego vehicle
    ego_desired_speed = 27.5
    ego_initial_state = np.array([[100], [0.5 * width], [0], [ego_desired_speed]])
    ego_initial_input = np.array([[0],[0]])
    ego_lim_slip_angle = 15 * np.pi / 180
    ego_lim_acc = 0.3 * 9.81
    ego_lim_slip_rate = 15 * np.pi / 180
    ego_controller_flag = 1
    ego_safety_factor = 0.5         # range: 0.1~1, 1 means maximum safety

    alpha_u_list = np.linspace(0.85, 1.15, 10)
    lane_change_times = np.zeros((len(alpha_u_list),))
    for idx, alpha_u in enumerate(alpha_u_list):
        # Create ego vehicle controller goal object using current
        # alpha_u value
        controller_goal = ego.EgoControllerGoal(ego_initial_lane_id, ego_direction_flag,
                                            ego_desired_speed, road, alpha_u,
                                            ego_lim_slip_angle,ego_lim_acc,
                                            ego_lim_slip_rate,ego_safety_factor,
                                            scenario)
        param_opt = load_ego_params(dt)
        controller0 = Controller.Controller(controller_goal, ego_controller_flag, param_opt,
                            veh_params, road, other_vehicles)


        # Define ego vehicle
        ego_car = Vehicle.Vehicle(1, veh_params, ego_initial_state, ego_initial_input, controller0,
                    ego_initial_lane_id, dt, road, ego_direction_flag,
                    ego_acc_flag, alpha_u, scenario)

        # Define simulator, start simulation
        simulator = sim.Simulator(road, ego_car, other_vehicles, dt)
        lane_change_times[idx] = simulator.lane_change_time(simulation_time)

    # Plot results
    plt.figure()
    plt.plot(alpha_u_list, lane_change_times)
    plt.xlabel('Uncertainty parameter value')
    plt.ylabel('Time to lane change completion (s)')
    plt.show()

if __name__ == '__main__':
    test_simulation()
