"""
Class representing a straight roadway consisting of one or more lanes, 
assuming that vehicles drive from left to right
"""
import matplotlib.pyplot as plt

class Road():
    """
    Straight horizontal roadway with multiple lanes
    """
    def __init__(self, num_lanes, lane_width, max_length) -> None:
        self.num_lanes = num_lanes
        self.width = lane_width
        self.length = max_length

    def plot_lanes(self):
        """
        Plot roadway
        """
        plt.plot([0,self.length],[0,0],'k-')
        for idx in range(self.num_lanes):
            mid = (idx+0.5)*self.width    # lane midpoint
            top = (idx+1)*self.width        # lane top
            plt.plot([0,self.length], [mid, mid],'k--')
            plt.plot([0,self.length], [top, top],'k-')
