# llm-social-navigation-bot

This repository contains instructions and code for simulating a social navigation robot using Gazebo and ROS. The robot is simulated in different environments and utilizes leg tracking for social interaction. Below, you will find information on setting up the environment, running the simulation, and accessing the code.

## Citation

If you use this work, please cite:

```bibtex
@misc{wen2024enhancingsociallyawarerobotnavigation,
  title={Enhancing Socially-Aware Robot Navigation through Bidirectional Natural Language Conversation}, 
  author={Congcong Wen and Yifan Liu and Geeta Chandra Raju Bethala and Zheng Peng and Hui Lin and Yu-Shen Liu and Yi Fang},
  year={2024},
  eprint={2409.04965},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2409.04965}
}
```

## Environment Requirements

- ROS version: Noetic
- Nvidia driver version: RTX 3050 Ti (Not mandatory)
- Gazebo version: Gazebo 11
- Python version: Python 3.8.10
- CUDA version: 10.1
- Torch version: 2.0.1+cu117

  And for **text to speech** i used  **pyttsx3** 2.90 for more documentation https://pypi.org/project/pyttsx3/

  For **Speech Recognition** i used python speech recognition **(SpeechRecognition 3.10.0)** library version 3.10  documentation: https://pypi.org/project/SpeechRecognition/

## GitHub Links and Download

- Gazebo Environment Code Link: [https://github.com/bethalageetachandraraju/llm-social-navigation-bot](https://github.com/bethalageetachandraraju/llm-social-navigation-bot)

- Leg Tracker Code Link: [https://github.com/bethalageetachandraraju/llm-social-navigation-bot](https://github.com/bethalageetachandraraju/llm-social-navigation-bot)

To clone this repository and install the leg tracker, use the following command:

```bash
git clone https://github.com/bethalageetachandraraju/llm-social-navigation-bot
```

For additional information about the leg_tracker, refer to the following link: [https://github.com/angusleigh/leg_tracker](https://github.com/angusleigh/leg_tracker)



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
### Turtlebot setup
After creating thw workspace, follow this lonk for downloading the turtlebot packages
https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation
https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/
and download the packages into **src folder of the workspace**
## Parameters config on Turtloebot
Tuttlebot files are found here
```bash
/opt/ros/noetic/share/turtlebot3_description
```
go to this directory there we can find the gazebo files for the respective turtlebot model that you are using, in my case im using **waffel_pi** so im changing modifing the **turtlebot3_waffle_pi.gazebo.xacro**
add this part of code in place of base_scan or change the parameters
and also dont forget to turn of the visualization here 

  **<xacro:arg name="laser_visual"  default="true"/>**

```bash
  <gazebo reference="base_scan">
    <material>Gazebo/FlatBlack</material>
    <sensor type="gpu_ray" name="lds_lfcd_sensor">
      <pose>0 0 0 0 0 0</pose>
      <visualize>$(arg laser_visual)</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>1440</samples>
            <resolution>1</resolution>
            <min_angle>-1.57079633</min_angle>
            <max_angle>1.57079633</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.120</min>
          <max>10</max>
          <resolution>0.015</resolution>
        </range>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.01</stddev>
        </noise>
      </ray>
      <plugin name="gazebo_ros_lds_lfcd_controller" filename="libgazebo_ros_gpu_laser.so">
        <topicName>scan</topicName>
        <frameName>base_scan</frameName>
      </plugin>
    </sensor>
  </gazebo>
```

### Gazebo actor setup
use the following link _https://github.com/blackcoffeerobotics/gazebo-ros-actor-plugin_ for setting up the actor control, So that we can control the actor(human) with twist messages.
Follow the above git hub link and clone the repo in the src folder of workspace and follow the instruction and build it with catkin_make. 
After running the below command for lunching the _robot.launch_ in rostopic list we will see the topic **cmd_velp** to control the actor in gazebo
```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py remap cmd_vel:= cmd_velp
```
with this command we can control the motion of the actor.


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

### Remember to speak after running the Python code to interact with the social navigation robot in the simulation. Enjoy exploring the different environments and social interactions!








### Leg Tracker Dependencies

Before running the leg tracker code, make sure you have the following dependencies installed:

To add the installation instructions for SciPy and pykalman to your README file, you can include the following section:

```markdown
## Leg Tracker Dependencies

Before running the leg tracker code, make sure you have the following dependencies installed:

### SciPy

```bash
sudo apt install python-scipy
```

### pykalman

You can install pykalman by following one of these methods:

- Official pykalman Installation Guide: [http://pykalman.github.io/#installation](http://pykalman.github.io/#installation)

or

- Using pip:

```bash
sudo pip install pykalman
```

Ensure that you have these dependencies installed to successfully run the leg tracker code.
```

This section provides clear instructions on how to install SciPy and pykalman, either via apt or pip, before running the leg tracker code. Users can follow these instructions to meet the required dependencies.

