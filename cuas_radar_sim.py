import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Target model ("the drone")
class Target:
    def __init__(self, x0, y0, vx, vy):
        """
        x0, y0 : the initial position
        vx, vy : velocity components (units per second)
        """
        self.x = x0
        self.y = y0
        self.vx = vx
        self.vy = vy

    def update(self, dt): # Move the target forward in time by dt seconds
        self.x += self.vx * dt
        self.y += self.vy * dt

    @property
    def position(self): # return current position as a NumPy array
        return np.array([self.x, self.y], dtype=float)
    

# 2. Radar measurement (adds noise)

def measure_position(true_position, noise_std=0.5):
    """
    Simulate a noisy radar measurement of the target's position.

    true_position : np.array([x, y])
    noise_std : standard deviation of Guassian noise in both x and y
    """
    noise = np.random.normal(loc=0.0, scale=noise_std, size=2)
    measurement = true_position + noise
    return measurement


# 3. Simple alpha-beta tracker (1 target, 2D state)
class AlphaBetaTracker:
    def __init__(self, initial_position, alpha=0.3, beta=0.05, dt=1.0):
        """
        initial_position : first measured position (np.array([x, y]))
        alpha, beta : tracking gains (0 < alpha, beta < 1)
        dt  : time step (seconds)
        """
        self.dt = dt

        self.position_est = initial_position.astype(float) # Initial state : we start with position from measurement
        
        self.velocity_est = np.array([0.0, 0.0], dtype=float) # Assume the starting velocity (will be learned overtime)

        self.alpha = alpha
        self.beta = beta

    def update(self, measurement):
        """
        Perform one predict + update step of the alpha-beta filter.
        measurement : np.array([x_meas, y_meas])
        """
        # prediction step - Predict the next position based on previous estimate and velocity
        predicted_position = self.position_est + self.velocity_est * self.dt
        predicted_velocity = self.velocity_est # constant velocity model

        # ---- Update Step ---- Residual = what measurement says minus prediction
        residual = measurement - predicted_position

        # Correct the predicted position using fraction alpha of residual
        # BUG FIX 1: Changed self.prediction_est to self.position_est
        self.position_est = predicted_position + self.alpha * residual 

        # Correct the predicted velocity using fraction beta of residual
        self.velocity_est = predicted_velocity + (self.beta / self.dt) * residual

        return self.position_est.copy(), self.velocity_est.copy()
    
from enum import Enum
from datetime import datetime, timezone
from math import sqrt

class State(str, Enum):
    TENTATIVE = "target.tentative" # I might be real, but im not proven
    CONFIRMED = "target.confirmed" # Im not real enough to act on
    COASTING = "target.coasting" # used to be real, but hasn't been seen recently; predicting forward
    DROPPED = "target.dropped" # Dead; Remove the track

class State_Threshold(float, Enum):
    CONFIRM_CONFIDENCE = 0.6
    COAST_MISSES = 2 # 2 misses in a row starts coasting
    DROP_MISSES_TENTATIVE = 2 # tentative dies quickly
    DROP_MISSES_CONFIRMED = 5 # gets more grace
    DROP_AGE = 5 | 10

class Candidate:
        measurement: np.ndarray
        distance: float

        def __init__(self, measurement, distance):
            self.measurement = measurement
            self.distance = distance
    
class Track:
    track_id: int
    state: State 
    confidence: int | float
    tracker: AlphaBetaTracker
    hits: int
    misses: int
    last_update_time: datetime
    last_seen_time: datetime
    threshold: State_Threshold
    hit_streak: int | float # consecutive hits for confidence & state determination
    miss_streak: int | float # consecutive misses for confidence & state determination

    # everything for the most part is set to None or 0 by default, which means the state must be Dropped by default as well
    def __init__(self, track_id, state: State = State.DROPPED, confidence=None, tracker=None, hits=0, misses=0, last_update_time=None, last_seen_time=None, threshold=None, hit_streak=0, miss_streak=0):
        self.track_id=track_id
        self.state=state 
        self.confidence=confidence
        self.tracker=tracker
        self.hits=hits
        self.misses=misses
        self.last_update_time=last_update_time
        self.last_seen_time=last_seen_time
        self.threshold=threshold
        self.hit_streak=hit_streak
        self.miss_streak=miss_streak
        
    

    def predict(self): # kinematics under the hood -> "Given how this object was moving before, where should it be now if nothing unexpected happened?"

        # belief not truth
        velo_est = self.tracker.velocity_est
        pos_est = self.tracker.position_est
        dt = self.tracker.dt

        position_pred = pos_est + velo_est * dt
        velocity_pred = velo_est

        return(position_pred, velocity_pred)
    

    def try_update(self, measurements: list, gate_radius: float) -> bool:
        # Pick the best measurement for this track (if any) using predicted_state + gating; return True if a hit was used.
        now = datetime.now(timezone.utc)
        P_pred, V_pred  = self.predict()
        candidates = List[Candidate] = []
        x_p = P_pred[0] # position x-coord
        y_p = P_pred[1] # position y-coord

        for meas in measurements: # for every measurement access the position & measurement coordinates and calculate distance

            x_m = meas[0] # measurement x-coord
            y_m = meas[1] # measurement y-coord

            dx = x_m - x_p
            dy = y_m - y_p

            distance = sqrt(dx**2 + dy**2) # distance formula
            if distance <= gate_radius:
                candidate = Candidate(measurement=meas, distance=distance) # Candidate objects
                candidates.append(candidate) # add a list of Candidate objects to candidates

        if not candidates: # if no hits
            self.register_miss(now=now) # increment misses
            return False
        
        # this is now only assuming there was a hit
        best_candidate = min(candidates, key=lambda c: c.distance)
        best_measurement = best_candidate.measurement

        self.register_hit(measurement=best_measurement, now=now)
        return True



    def register_hit(self, measurement, now: datetime) -> None:
        # Record that this track was observed this step: increment hits, reset misses, update last_seen/last_update time, and raise confidence.
        self.last_seen_time = now
        self.last_update_time = now

        self.hits += 1 # increment hits by 1
        self.misses = 0 # reset to 0

        self.hit_streak += 1 # adds to or starts hit streak
        self.miss_streak = 0 # ends miss streak

        self.update_confidence(hit=True) 
        self.tracker.update(measurement=measurement) # updating predicted measurement to realistic measurement so it doesnt go off track

    def register_miss(self, now: datetime) -> None:
        # Record that this track was NOT observed this step: increment misses, update last_update time, and decay confidence (coasting behavior starts here).
        self.last_update_time = now
        self.misses += 1

        self.miss_streak += 1 # adds to / starts miss streak
        self.hit_streak = 0 # ends hit streak

        self.update_confidence(hit=False)


    def update_confidence(self, hit: bool) -> None:
        # Apply your confidence rule: increase on hit, decrease on miss, clamp into [0.0, 1.0].

        step = 0.2 # since the max window is 5, if you take 5 steps that are each 0.2 you be a 1
        max_confidence = 1.0
        min_confidence = 0
        if hit:
            self.confidence += step

        else:
            self.confidence -= step

        if self.confidence > max_confidence:
            self.confidence = 1.0

        elif self.confidence < min_confidence:
            self.confidence = 0.0

    def evaluate_state_transition(self) -> None:
        # Use hits/misses/confidence/time-since-seen to transition between TENTATIVE ↔ CONFIRMED ↔ COASTING → DROPPED.

        if 




    def is_dropped(self) -> bool:
        # Return True if the track is in DROPPED state (or otherwise considered dead and removable).
        if self.state is not State.DROPPED:
            return False
        
        return True
    
    def is_confirmed(self) -> bool:
        # Return True if the track is in CONFIRMED state (trusted enough for threat logic/alerts).
        if self.state is not State.CONFIRMED:
            return False
        
        return True
    
    def is_coasting(self) -> bool:
        # Return True if the track is currently COASTING (no recent measurement, running on prediction).
        if self.state is not State.COASTING:
            return False
        
        return True
    
    def age(self, now: datetime) -> float:
        # Return how long it has been since the last measurement hit (or since creation), used for timeouts/decay.
        
        


# 4. Simulation Loop
def run_simulation(total_time=60.0, dt=1.0, noise_std=5.0, alpha=0.3, beta=0.05):
    """
    Run a full simulation of a single moving target and a tracker.

    total_time : total duration of the simulation in seconds
    dt : time step in seconds
    noise_std : radar noise level
    alpha,beta : tracking gains
    """
    num_steps = int(total_time / dt)
    # times = np.arange(0, total_time, dt) # Not used, but fine to keep

    # Create a target with some initial position and velocity
    target = Target(
        x0=0.0,
        y0=0.0,
        vx=2.0, # move 2 units per second in x
        vy=1.0 # move 1 unit per second in y
        )
    
    # Storage for plotting
    true_positions = []
    measurements = []
    estimates = []

    tracker = None

    for step in range(num_steps):
        # 1. Update the target position
        target.update(dt)
        true_pos = target.position

        # 2. Simulate radar measurement of that position
        meas = measure_position(true_position=true_pos, noise_std=noise_std)

        # 3. Initialize tracker on first measurement
        if tracker is None:
            tracker = AlphaBetaTracker(
                initial_position=meas,
                alpha=alpha, 
                beta=beta,
                dt=dt
            )

        # 4. Update tracker with current measurement
        est_position, est_velocity = tracker.update(measurement=meas)

        # 5. Save for plotting
        true_positions.append(true_pos)
        # BUG FIX 2: Appending 'meas' instead of the 'measurements' list itself
        measurements.append(meas) 
        estimates.append(est_position)

    # Convert lists to NumPy arrays for easier slicing
    true_positions = np.vstack(true_positions)
    measurements = np.vstack(measurements)
    estimates = np.vstack(estimates)

    # 6. Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(true_positions[:, 0], true_positions[:, 1], label="True path", linewidth=2)
    plt.scatter(measurements[:, 0], measurements[:, 1], label="Radar measurements", s=10, alpha=0.4)
    plt.plot(estimates[:, 0], estimates[:, 1], label="Tracker estimate", linestyle="--", linewidth=2)

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("C-UAS Style 2D Target Tracking (Alpha-Beta Filter)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42) # for reproducible noise
    run_simulation()