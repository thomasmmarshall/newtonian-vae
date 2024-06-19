# Variational State Space Models for Visual Robotics

## Introduction
This is a version of the repo I worked on during my masters. It has several notebooks which should just work if you run them. 

To get started:
- Clone the repo
- Run the command `pip install -r requirements.txt` to install all dependencies
- Generate datasets (see following section)
- Explore the notebooks 

## Generating datasets
There are three `dm_control` environments running on `mujoco` here: Linearmass, Reacher and Panda. In the folder `/data-generators/` there are files for generating datasets. 

They are used as follows: `python3 generate_linearmass.py --episodes 100 --steps 100`

This example generates 100 episodes of 100 steps in the linearmass simulated environment. Actions are sampled uniformly and are normalised `[-1, 1]` before they saved (with the image frames and true states) to a `.npz` `numpy` archive. 

Make sure the datasets you want to use are in the datasets folder before running a notebook.

### The Panda
The Panda requires a special mention here. This robot has 7 joints (we are using the gripper-less version). However, its helpful to use only a few of them at a time, usually two. 

In the first case we want to merely freeze some set of joints in the default position. This is easily done by setting those joints' actions to zero during dataset generation. 

In the second case, we want to freeze joints in a non-default position. For this we need to modify the state vector of the physics engine directly. This state is made up of 7 values for position and 7 more for velocity. We do this in the file `generate_panda.py`.
There are several places where you have to take note of this:
- Line 54: Specify the joints that will be moving
- Line 72: Set actions to zero to freeze joints
- Line 74: Mess with physics engine if necessary

### Expanding this framework
New environments can be added to use with `dm_control` very easily. There are a number of things one must do:
- Create a `new_environment.py` file and put it in `/control/dm_control/` with the `.xml` and its assets (if any). Follow the examples to make new ones. The panda has assets so refer to that for robots with assets.
- Add this to the `__init__.py` file in `/control/dm_control/` to match the others. 
- Write a data generator script and just replace `domain_name="new_environment"`.


## Additional documentation
See the folder `/docs/` for the file `msc-project.pdf` for my masters research project for more information. Otherwise, send me an email at `marshallthomasm@gmail.com`.