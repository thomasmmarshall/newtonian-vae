# Import dependencies
import os
import sys

# Add root file system to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import pathlib

import numpy as np
import PIL.Image
from dm_control.suite.wrappers import pixels
from tqdm import tqdm

from control.dm_control import suite
from lib.utils import dict_dataset_split


def unit_vector(vector):
    """Returns the unit vector of the vector"""
    return vector / np.linalg.norm(vector)


def angle_between(vector1, vector2):
    """Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    if minor == 0:
        raise NotImplementedError("Too odd vectors =(")
    return np.sign(minor) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# Main thread
if __name__ == "__main__":

    # Parse commmandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    # Load the domain for this robot using dm_control suite

    env = suite.load(
        domain_name="multireacher", task_name="easy", task_kwargs=dict(random=3)
    )
    env = pixels.Wrapper(
        env, pixels_only=False, render_kwargs=dict(camera_id=0, height=240, width=240)
    )

    action_spec = env.action_spec()

    # Initialise list to store dataset information
    states_episodes = []
    actions_episodes = []
    frames_episodes = []

    # Run episodes in the environment
    for episode in tqdm(range(args.episodes + 200)):

        # Reset the environment
        time_step = env.reset()

        # Initialise list to store episode information
        states = []
        actions = []
        frames = []
        images = []

        for t in range(args.steps):

            # Observe a frame of pixels from the environment
            frame = time_step.observation["pixels"]
            im = PIL.Image.fromarray(frame)

            # Get the state by concatenating the position and velocity of the robot
            state = np.concatenate(
                [time_step.observation["joint_angle"], time_step.observation["joint_vel"]]
            )

            # Sample a random action vector uniformly
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape
            )

            # Take the action
            time_step = env.step(action)

            # Collect sates and actions
            states.append(state)

            # Normalise action to [-1, 1]
            normalised_action = action / (action_spec.maximum - action_spec.minimum)
            actions.append(action[:])

            # Collect image
            im = PIL.Image.fromarray(frame)
            frame = np.array(im.resize((64, 64), PIL.Image.Resampling.BICUBIC))
            frames.append(frame)

        # Collect data from this episode
        states_episodes.append(states)
        actions_episodes.append(actions)
        frames_episodes.append(frames)

    # Save dataset of states, actions and images
    states_episodes = np.array(states_episodes)
    actions_episodes = np.array(actions_episodes)

    data = {"state": states_episodes, "act": actions_episodes, "img": frames_episodes}

    dest = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "../datasets/reacher%s_%deps_%dsteps_%s.npz"
        % ("_rand", args.episodes, args.steps, args.name),
    )

    dataset = dict_dataset_split(args.episodes, 100, 100, data)
    np.savez_compressed(dest, **dataset)
    print("Saved to file %s" % dest)
