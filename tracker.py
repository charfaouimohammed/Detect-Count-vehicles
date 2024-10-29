import math
from filterpy.kalman import KalmanFilter
import numpy as np

class Tracker:
    def __init__(self):
        # Store object Kalman filters and center points
        self.trackers = {}
        self.id_count = 0

    def create_kalman_filter(self, cx, cy):
        """Initialize a Kalman filter for a new object."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # Define the initial state (position and velocity)
        kf.x = np.array([cx, cy, 0, 0])  # initial state vector [x, y, vx, vy]
        
        # Define the transition matrix
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        
        # Define the measurement matrix
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        
        # Define the covariance matrices
        kf.P *= 1000  # initial uncertainty
        kf.R *= 5     # measurement noise
        kf.Q = np.eye(4) * 0.1  # process noise

        return kf

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            
            same_object_detected = False

            # Update the Kalman filter for each tracked object
            for object_id, tracker in self.trackers.items():
                # Predict the next position
                tracker.predict()
                kf_cx, kf_cy = tracker.x[0], tracker.x[1]
                
                # Calculate distance between prediction and detection
                dist = math.hypot(cx - kf_cx, cy - kf_cy)
                
                if dist < 35:
                    # Update the Kalman filter with the new measurement
                    tracker.update(np.array([cx, cy]))
                    
                    # Update the tracked position
                    self.trackers[object_id] = tracker
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                # Create a new tracker for the new object
                kf = self.create_kalman_filter(cx, cy)
                self.trackers[self.id_count] = kf
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean up trackers by keeping only those that were updated
        self.trackers = {obj_id: tracker for obj_id, tracker in self.trackers.items() if any(obj_id == obj[4] for obj in objects_bbs_ids)}

        return objects_bbs_ids
