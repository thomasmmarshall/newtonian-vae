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

# Main thread
if __name__ == "__main__":

    # Parse commmandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    # Load the domain for this robot using dm_control suite
    env_ = suite.load(domain_name="panda", task_name="easy")
    env = pixels.Wrapper(
        env_, pixels_only=False, render_kwargs=dict(camera_id=0, height=240, width=240)
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

        # Joints we are keeping
        joints = [0, 3]

        for t in range(args.steps + 25):  # +25 is for increased config space coverage

            # Observe a frame of pixels from the environment
            frame = time_step.observation["pixels"]

            # Get the state by concatenating the position and velocity of the robot
            state = np.concatenate(
                [time_step.observation["pos"], time_step.observation["vel"]]
            )

            # Sample a random action vector uniformly
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape
            )

            # Action freezing
            action[4:] = 0
            action[[1, 2, 4]] = 0

            # Tell the physics engine what the pos/vel of joints must be
            # Set state of physics simulator freezing joint 0
            # phys_state = env_.physics.get_state()
            # phys_state[2:7] = np.pi/4; phys_state[7+2:] = 0
            # env_.physics.set_state(phys_state)

            # Take the action
            time_step = env.step(action)

            if t > 25:
                # Collect sates and actions
                states.append(
                    state[joints]
                )  # Save only particular elements of the state

                # Normalise action to [-1, 1]
                normalised_action = action / (action_spec.maximum - action_spec.minimum)
                actions.append(
                    normalised_action[joints]
                )  # Save only particular elements of the state

                # Collect image
                im = PIL.Image.fromarray(frame)
                frame = np.array(im.resize((64, 64), PIL.Image.Resampling.BICUBIC))
                frames.append(frame)

                # Save frames for a video of the first episode for inspection
                if episode == 0:
                    images.append(np.array(frame))

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
        "../datasets/panda%s_%deps_%dsteps_%s.npz"
        % ("_rand", args.episodes, args.steps, args.name),
    )

    dataset = dict_dataset_split(args.episodes, 100, 100, data)
    np.savez_compressed(dest, **dataset)
    print("Saved to file %s" % dest)
