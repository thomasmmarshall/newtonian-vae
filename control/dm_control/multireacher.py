# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Reacher domain."""

from __future__ import absolute_import, division, print_function

import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.utils import rewards

SUITE = containers.TaggedTasks()
_DEFAULT_TIME_LIMIT = 10
_BIG_TARGET = .02

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(os.path.dirname(__file__), 'multireacher.xml')), common.ASSETS

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, 
         fixed_targets=False, fixed_start=False, test_mode=False, environment_kwargs=None):
    """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = MultiReacher(target_size=_BIG_TARGET, fixed_targets=fixed_targets, 
                        fixed_start=fixed_start, random=random, test_mode=test_mode)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Reacher domain."""
    def finger_to_target(self):
        """Returns the vector from target to finger in global coordinates."""
        return (self.named.data.geom_xpos['target1', :2] -
                self.named.data.geom_xpos['finger', :2])

    def finger_to_target_dist(self):
        """Returns the signed distance between the finger and target surface."""
        return np.linalg.norm(self.finger_to_target())


class MultiReacher(base.Task):
    """A reacher `Task` to reach the target."""
    def __init__(self, target_size, fixed_targets, fixed_start, random=None, test_mode=False):
        """Initialize an instance of `Reacher`.

        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        
        super(MultiReacher, self).__init__(random=random)
        self._target_size = target_size
        self._fixed_targets = fixed_targets
        self._fixed_start = fixed_start
        self._target_pos = {}
        self._curr_target_idx = 1
        self._reached = {1: False, 2: False, 3: False}
        self.test_mode = test_mode

        if self._fixed_targets:
            # If fixed targets, set random targets position from the start
            for i in range(1,4):
                angle = self.random.uniform(0, 2*np.pi)
                radius = self.random.uniform(.02, .24)
                self._target_pos["%d"%i] = radius*np.array([np.sin(angle), np.cos(angle)])

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self._curr_target_idx = 1
        self._reached = {1: False, 2: False, 3: False}
        for i in range(1,4):
            physics.named.model.geom_size['target%d'%i, 0] = self._target_size

        if self._fixed_start:
            physics.named.data.qpos['wrist'] = 0.5 + (np.random.rand()-0.5)
            physics.named.data.qpos['shoulder'] = -np.pi +0.3+ np.random.rand()*0.5
        else:
            randomizers.randomize_limited_and_rotational_joints(physics, self.random)

        for i in range(1,4):
            if self._fixed_targets:
                physics.named.model.geom_pos['target%d'%i, 'x'] = self._target_pos["%d"%i][0]
                physics.named.model.geom_pos['target%d'%i, 'y'] = self._target_pos["%d"%i][1]
            else:
                angle = self.random.uniform(0, 2*np.pi)
                radius = self.random.uniform(.02, .24)
                physics.named.model.geom_pos['target%d'%i, 'x'] = radius*np.sin(angle)
                physics.named.model.geom_pos['target%d'%i, 'y'] = radius*np.cos(angle)

        super(MultiReacher, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the state and the target position."""
        obs = collections.OrderedDict()
        obs['curr_target'] = physics.named.model.geom_pos['target%d'%self._curr_target_idx, :2]

        # angle and angular vels of the joints
        obs['joint_angle'] = physics.position()
        obs['joint_vel'] = physics.velocity()

        obs['finger'] = physics.named.data.geom_xpos['finger', :2]
        for i in range(1,4):
            obs['target%d'%i] = physics.named.model.geom_pos['target%d'%i, :2]
        return obs

    def get_reward(self, physics):
        dist = np.linalg.norm(physics.named.data.geom_xpos['finger', :2]-\
                              physics.named.model.geom_pos['target%d'%self._curr_target_idx, :2])
        vel = np.linalg.norm(physics.velocity())

        mult = 0.5 if self.test_mode else 1
        vel_mult = 20 if self.test_mode else 1
        if dist <= mult*3e-2 and vel <= vel_mult*1e-2 and self._reached[self._curr_target_idx] == False:
            reward = 1.0
            self._reached[self._curr_target_idx] = True
            self._curr_target_idx = min(self._curr_target_idx+1, 3)
        else:
            reward = 0.0
        return reward
