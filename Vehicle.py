"""
Class representing vehicles in lane-change scenario
"""
import numpy as np
import matplotlib.pyplot as plt

class Vehicle():
    """
    Vehicle class
    """
    def __init__(self, dynamics_flag, veh_param, state_initial, initial_input, controller, initial_lane_id, dt, lanes, dir_flag, acc_flag, alpha_u, scenario):
        # 0 = normal surrounding vehicle (constant speed/acceleration) (blue)
        # 1 = ego vehicle (red)
        # 2 = surrounding vehicle that will change lanes (blue)
        self.dynamics_flag = dynamics_flag
        self.param = veh_param
        self.controller = controller
        self.state = state_initial  # current state of vehicle (x, y, phi, speed)
        self.input = initial_input  # current input of vehicle (accel, steering angle)
        self.state_log = [state_initial]  # history of states (list)
        self.input_log = [initial_input]  # history of inputs (list)
        self.initial_lane_id = initial_lane_id
        self.lane_id = initial_lane_id  # current lane id of vehicle
        self.dt = dt    # time step for simulation
        self.lanes = lanes
        self.dir_flag = dir_flag
        self.acc_flag = acc_flag
        self.alpha_u = alpha_u
        self.scenario = scenario
        self.other_log = [(initial_lane_id, 1)] #idk what this is for

    def update(self):
        """
        Update vehicle during each time step
        """
        if self.dynamics_flag == 0:
            self.normal_car_update()
        elif self.dynamics_flag == 1:
            self.ego_car_update()
        # elif self.dynamics_flag == 2:
        # do lane change vehicle car update

    def normal_car_update(self):
        """
        Update position of a normal car (constant speed or acceleration)
        """
        # Get lane id
        self.get_lane_id()
        speed = self.state[3]
        accel = self.input[0]
        # Update speed
        self.state[3] = self.state[3] + accel * self.dt

        # Speed limit according to scenario
        if self.scenario == 1:  # highway
            upper_lim = 33.33
            lower_lim = 23
        else:   # urban
            upper_lim = 16.67
            lower_lim = 12

        # Adjust speed to be within limits
        if self.state[3] >= upper_lim:
            self.state[3] = upper_lim
        elif self.state[3] <= lower_lim:
            self.state[3] = upper_lim

        # dx = vdt + 0.5*a*dt^2
        dx = speed*self.dt + 0.5 * accel * self.dt**2
        # Update x position
        self.state[0] = self.state[0] + dx
        # Update logs
        self.state_log.append(self.state)
        self.input_log.append(self.input)
        # update other_log (with lane id?)

    def ego_car_update(self):
        """
        Update state vector of ego vehicle
        """
        self.get_lane_id()
        # Compute optimal input of the vehicle
        [self.acc_flag, u, e] = self.controller.control(self.state, self.input,
                                                        self.lane_id, self.input_log,
                                                        self.initial_lane_id,
                                                        self.dir_flag, self.acc_flag)
        # Calculate dX using nonlinear bicycle model,
        # here input is the vector [v; beta]
        bicycle_input = np.hstack( (self.state[3] + self.dt * u[0], u[1]) ).reshape((2,))
        dx = self.bicycle(self.state[0:3].reshape((3,)), bicycle_input, self.alpha_u )
        # Update vehicle state
        self.state = self.state + self.dt * np.vstack((dx, u[0]))
        # Update the input
        self.input = u
        # Update state history
        self.state_log.append(self.state)
        self.input_log.append(self.input)
        # other_log update

    def bicycle(self, state, input, alpha_u):
        """
        Nonlinear vehicle dynamics model, from kinematic bicycle model; 
        assuming small angle (beta) approximation: 
        cos(beta) = 1, sin(beta) = beta
        """
        l_f = self.param['l_f']
        l_r = self.param['l_r']
        l = l_f + l_r

        phi = state[2]
        v = input[0]
        beta = input[1]
        # Steering angle
        delta_f = np.arctan(l * np.tan(beta)/l_r)
        # x velocity, x_dot = v * np.cos(phi + beta)
        x_dot = v * (np.cos(phi) - alpha_u * beta * np.sin(phi))
        # y velocity, y_dot = v * np.sin(phi + beta)
        y_dot = v * (np.sin(phi) + alpha_u * beta * np.cos(phi))
        # yaw rate
        phi_dot = alpha_u * (v / l_r) * np.sin(beta)
        dx = np.array([
            [x_dot],
            [y_dot],
            [phi_dot]
        ])
        return dx

    def plot_vehicle(self):
        """
        Plot vehicles in simulation
        """
        length = self.param['l_fc'] + self.param['l_rc']
        self.draw_vehicle(self.state, length, self.param['width'], self.dynamics_flag)

    def draw_vehicle(self, state, length, height, dynamics_flag):
        """
        Draw vehicle 
        """
        x_center = state[0]
        y_center = state[1]
        theta = state[2]

        # 2d rotation matrix
        rot2d = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        # Corners of rectangle
        pts = np.array([
            [-length/2, length/2, length/2, -length/2],
            [-height/2, -height/2, height/2, height/2]
        ])
        # Rotated points
        pts_rot = np.matmul(rot2d.reshape((2,2)), pts)

        pts_rot = pts_rot + np.hstack((x_center, y_center)).reshape(2,1)

        # Plot vehicle
        for i in range(4):
            x1 = pts_rot[0,i]
            y1 = pts_rot[1,i]
            if i == 3:
                x2 = pts_rot[0,0]
                y2 = pts_rot[1,0]
            else:
                x2 = pts_rot[0,i+1]
                y2 = pts_rot[1,i+1]
            if dynamics_flag == 0:  # normal vehicle
                plt.plot([x1,x2], [y1,y2], color='b')
            else:   # ego vehicle
                plt.plot([x1,x2], [y1,y2], color='r')

    def get_lane_id(self):
        """
        Returns current lane of vehicle (doesn't count half lanes)
        """
        y = self.state[1]
        width = self.param['width']
        lane_width = self.lanes.width

        if y <= lane_width - 0.5 * width:
            self.lane_id = 1
        elif y <= lane_width + 0.5 * width:
            self.lane_id = 1.5
        elif y <= 2 * lane_width - 0.5 * width:
            self.lane_id = 2
        elif y <= 2 * lane_width + 0.5 * width:
            self.lane_id = 2.5
        elif y <= 3 * lane_width - 0.5 *width:
            self.lane_id = 3
        elif y <= 3 * lane_width + 0.5 * width:
            self.lane_id = 3.5
        elif y <= 4 * lane_width - 0.5 * width:
            self.lane_id = 4
