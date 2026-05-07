import sys
import json
import subprocess
import tempfile
import os

import gym
import gym_dssat_pdi


class DSSATWorker:
    """
    DSSAT worker that wraps gym-dssat-pdi in a JSON I/O interface.

    This class is used by AgriManager to communicate with DSSAT via
    a Python 3.10 environment, allowing AgriManager to run in Python 3.12.
    """

    def __init__(self, worker_script=None, dssat_bin_path=None):
        """
        Args:
            worker_script: Not used — compatibility placeholder.
            dssat_bin_path: Optional DSSAT binary path.
        """
        # Create gym-DSSAT environment
        self.env = gym.make("DSSAT-v0")

        # Store DSSAT binary path if needed
        self.dssat_bin_path = dssat_bin_path

    def reset(self):
        """Reset environment and return raw obs."""
        obs = self.env.reset()
        return {"obs": obs}

    def step(self, action):
        """Take a step and return raw info dict."""
        obs, reward, done, info = self.env.step(action)
        return {
            "obs": obs,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }

    def close(self):
        """Close env."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
