import numpy as np


class SimpleIntersection:
    """A minimal, deterministic-ish intersection environment.


    State: [q_ns, q_ew, phase, elapsed]
    Actions: 0 = extend current, 1 = switch (subject to min green)
    """


    def __init__(self, max_steps=3600, seed=None):
        self.max_steps = max_steps
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.reset()


    def reset(self):
        self.step_count = 0
        self.phase = 0 # 0 = NS green, 1 = EW green
        self.elapsed = 0
        self.queues = np.zeros(2, dtype=float)
        return self._obs()


    def _obs(self):
        return np.array([self.queues[0], self.queues[1], float(self.phase), float(self.elapsed)], dtype=np.float32)


    def step(self, action):
        # arrivals: Poisson with time-varying mean (simulate rush hour waves)
        base = 1.2
        wave = 1.0 + 0.8 * np.sin(self.step_count / 50.0)
        lam = np.array([base * wave, base * (2.0 - wave)])
        arrivals = np.random.poisson(lam)
        self.queues += arrivals


        # discharge up to capacity per step in green approach
        capacity = 3
        served = min(capacity, int(self.queues[self.phase]))
        self.queues[self.phase] -= served


        # enforce min green
        MIN_GREEN = 5
        if action == 1 and self.elapsed >= MIN_GREEN:
            self.phase = 1 - self.phase
            self.elapsed = 0
        else:
            self.elapsed += 1


        reward = -float(self.queues.sum()) # negative total queue
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._obs(), reward, done, {}