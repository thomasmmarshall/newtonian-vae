# Copyright 2020 The dm_control Authors.
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

"""Panda Domain."""

import collections
import os

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base, common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.utils import rewards, xml_tools
from lxml import etree

_DEFAULT_TIME_LIMIT = 15
_CONTROL_TIMESTEP = .015

SUITE = containers.TaggedTasks()

_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'panda_assets')

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the easy panda task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Panda(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
physics, task, time_limit=time_limit, **environment_kwargs)


def make_model():
  """Sets floor size, removes ball and walls (Stand and Move tasks)."""
  xml_string = resources.GetResource(os.path.join(os.path.dirname(__file__), 'panda.xml'))
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)

  return etree.tostring(mjcf, pretty_print=True)


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  assets = common.ASSETS.copy()

  _, _, filenames = next(resources.WalkResources(_ASSET_DIR))
  for filename in filenames:
    assets[filename] = resources.GetResource(os.path.join(_ASSET_DIR, filename))
  return make_model(), assets


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Panda domain."""


class Panda(base.Task):
    """A dog stand task generating upright posture."""

    def __init__(self, random=None):
        """Initializes an instance of `Stand`.

        Args:
        random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        observe_reward_factors: Boolean, whether the factorised reward is a
            key in the observation dict returned to the agent.
        """
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Randomizes initial root velocities and actuator states.

        Args:
        physics: An instance of `Physics`.

        """

        # Reset physics engine
        physics.reset()



        # Randomizes the joints of the robot for the starting position
        #randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    def get_observation(self, physics):
        """Returns the observations for the Stand task."""
        obs = collections.OrderedDict()
        obs['pos'] = physics.position()
        obs['vel'] = physics.velocity()
        
        return obs

    def get_reward(self, physics):
        return 0.0