"""
Class for controlling ego vehicle 
"""

class EgoControllerGoal():
    """
    Define target parameters for ego vehicle
    """
    def __init__(self, init_lane_id, dir_flag, desired_speed, lanes, alpha_u,
                 lim_slip, lim_acc, lim_slip_rate, safety_factor, scenario) -> None:
        yt = ((init_lane_id + dir_flag) - 1) * lanes.width + 0.5 * lanes.width
        self.target_y = yt                      # target lateral position
        self.target_speed = desired_speed       # target speed
        self.lim_beta = lim_slip                # target slip angle
        self.lim_acc = lim_acc                  # target acceleration
        self.lim_beta_rate = lim_slip_rate      # target slip angle rate
        self.safety_factor = safety_factor      # factor of safety epsilon
        self.grade_uncertainty = alpha_u        # uncertainty parameter for road grade
        if scenario == 1:
            self.lim_speed = 33.33  # highway
        else:
            self.lim_speed = 16.67  # urban road
