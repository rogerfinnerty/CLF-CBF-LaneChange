"""
Class for visualizing automated lane change simulation
"""
import matplotlib.pyplot as plt

class Simulator():
    """
    Simulator
    """
    def __init__(self, lanes, ego, other_vehicles, dt) -> None:
        self.lanes = lanes
        self.ego = ego
        self.other_vehicles = other_vehicles
        self.dt = dt

    def start_sim(self, sim_time):
        """
        Simulated lane change for set amount of time 
        """
        num_steps = int(sim_time / self.dt)
        for _ in range(num_steps):
            plt.clf()
            plt.ylim(-50, 50)
            self.lanes.plot_lanes()

            # Plot + update other vehicles
            if len(self.other_vehicles) > 0:
                for _, veh in enumerate(self.other_vehicles):
                    # Plot vehicle
                    veh.plot_vehicle()
                    # Update vehicle state
                    veh.update()

            # Plot + update ego
            self.ego.plot_vehicle()
            self.ego.update()
            plt.pause(self.dt)

            # Update state, input history
            state_log = self.ego.state_log
            input_log = self.ego.input_log

        plt.show()

        return [state_log, input_log]

    def lane_change_time(self, sim_time):
        """
        Computes time taken for vehicle to make complete lane change; the 
        criteria for a complete lane change is being in the target 
        lane for 1 second
        """
        nb_steps = int(sim_time / self.dt)
        lane_count = 0      # variable to track how long ego has been in target lane

        # Lane change procedure
        for i in range(nb_steps):
            if len(self.other_vehicles) > 0:
                # Update other vehicles
                for _, veh in enumerate(self.other_vehicles):
                    veh.update()

            # Update ego vehicle
            self.ego.update()
            # Check if in target lane
            if self.ego.lane_id == self.ego.initial_lane_id + self.ego.dir_flag:
                lane_count += 1
            else:
                lane_count = 0
            if lane_count >= (1.5 / self.dt):
                break
        complete_time = i * self.dt

        return complete_time
