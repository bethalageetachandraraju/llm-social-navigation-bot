# llm-social-navigation-bot

This repository contains instructions and code for simulating a social navigation robot using Gazebo and ROS. The robot is simulated in different environments and utilizes leg tracking for social interaction. Below, you will find information on setting up the environment, running the simulation, and accessing the code.

## Environment Requirements

- ROS version: Noetic
- Nvidia driver version: RTX 3050 Ti (Not mandatory)
- Gazebo version: Gazebo 11
- Python version: Python 3.8.10

## GitHub Links and Download

- Gazebo Environment Code Link: [https://github.com/bethalageetachandraraju/llm-social-navigation-bot](https://github.com/bethalageetachandraraju/llm-social-navigation-bot)

- Leg Tracker Code Link: [https://github.com/bethalageetachandraraju/llm-social-navigation-bot](https://github.com/bethalageetachandraraju/llm-social-navigation-bot)

To clone this repository and install the leg tracker, use the following command:

```bash
git clone https://github.com/bethalageetachandraraju/llm-social-navigation-bot
```

For additional information about the leg_tracker, refer to the following link: [https://github.com/angusleigh/leg_tracker](https://github.com/angusleigh/leg_tracker)

## Gazebo Configuration

### Robot Configuration

### Environment Configuration

## Run Commands

### Compile Command

To set up your ROS workspace, use the following commands:

```bash
mkdir catkin_ws && cd catkin_ws && mkdir src
cd src
git clone https://github.com/bethalageetachandraraju/llm-social-navigation-bot
cd ..
catkin_make
source devel/setup.bash
```

### Start Gazebo Command

To start the Gazebo simulation, use the following command:

```bash
roslaunch gpt_robot robot.launch
```

### Start Leg Tracker Command

To start the leg tracker, use the following command in a different terminal:

```bash
roslaunch leg_tracker joint_leg_tracker.launch
```

### Run Python Code

In a separate terminal, run the Python code using the following command:

```bash
python mix.py
```

Note: Near line 220 in `mix.py`, uncomment the line corresponding to the world that was launched in Gazebo.

Remember to speak after running the Python code to interact with the social navigation robot in the simulation. Enjoy exploring the different environments and social interactions!
