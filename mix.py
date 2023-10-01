import rospy
import numpy as np
import math
import cv2
import tensorflow as tf
import torch
import threading
import torch.nn as nn
import torch.nn.functional as F
import speech_recognition as sr
import pyttsx3
import time
from collections import deque
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from typing import List, Tuple
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point, Quaternion
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from leg_tracker.msg import PersonArray
from cv_bridge import CvBridge

import openai
openai.api_key = "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"

input_shape = 512
out_c = 8
# no interact, up, right, down, left, stop
out_d = 4
LOG_STD_MAX = 0.0
LOG_STD_MIN = -5.0

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

speech_queue = []
speech_queue.append("None")
speak_engine = pyttsx3.init()
if_replied = False




def mlp(input_mlp: List[Tuple[int, int]]) -> nn.Sequential:
    if not input_mlp:
        return nn.Sequential()
    mlp_list = []
    for input_dim, out_put_dim, af in input_mlp:
        mlp_list.append(nn.Linear(input_dim, out_put_dim, bias=True))
        if af == "relu":
            mlp_list.append(nn.ReLU())
        if af == 'sigmoid':
            mlp_list.append(nn.Sigmoid())
    return nn.Sequential(*mlp_list)


class PreNet(torch.nn.Module):
    def __init__(self):
        super(PreNet, self).__init__()


class NavPedPreNet(PreNet):
    def __init__(self,
                 image_channel=4,
                 last_output_dim=512,
                 ):
        super(NavPedPreNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(image_channel, 64, 3, stride=1, padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1))
        # self.fc0 = nn.Linear(256 * 6 * 6, 512)
        self.fc0 = mlp([(256 * 6 * 6, 512, "relu")])
        # self.fc1 = nn.Linear(512 + 9, 512)
        self.fc1 = mlp([(512 + 10, 512, "relu")])
        self.fc2 = nn.Linear(512, 512)

        self.conv1.to("cuda")
        self.conv2.to("cuda")
        self.conv3.to("cuda")
        self.fc0.to("cuda")
        self.fc1.to("cuda")
        self.fc2.to("cuda")
        # self.fc3 = nn.Linear(512, 512)
        # assert self.fc2.out_features == last_output_dim

    def _encode_image(self, image):
        image_x = F.max_pool2d(F.relu(self.conv1(image)), 2, stride=2)
        image_x = F.max_pool2d(F.relu(self.conv2(image_x)), 2, stride=2)
        image_x = F.max_pool2d(F.relu(self.conv3(image_x)), 2, stride=2)
        image_x = image_x.view(image_x.size(0), -1)
        return image_x

    def forward(self, state):
        env_img = torch.from_numpy(state[0]).float().to("cuda")
        ped_img = torch.from_numpy(state[2]).float().to("cuda")
        encoded_image = self._encode_image(torch.cat([env_img, ped_img], axis=1).to("cuda"))
        # print(encoded_image)
        x = self.fc0(encoded_image)
        # x = torch.cat((x, torch.from_numpy(state[1]).float().to("cuda"), dim=1)
        x = torch.cat((x, torch.from_numpy(state[1]).float().to("cuda"), torch.from_numpy(state[3]).int().to("cuda")),
                      dim=1)
        # x = state[1]
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x


def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        torch.nn.init.constant_(layer.bias, bias_const)


class Policy(nn.Module):
    def __init__(self, input_shape, out_c, out_d):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)  # Better result with slightly wider networks.
        self.mean = nn.Linear(128, out_c)
        self.logstd = nn.Linear(128, out_c)
        self.pi_d = nn.Linear(128, out_d)

        self.apply(layer_init)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)

        x = F.relu(self.fc1(x))
        mean = torch.tanh(self.mean(x))
        log_std = torch.tanh(self.logstd(x))
        pi_d = self.pi_d(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std, pi_d

    def get_action(self, x, device):
        mean, log_std, pi_d = self.forward(x, device)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action_c = torch.tanh(x_t)
        all_log_prob_c = normal.log_prob(x_t)
        all_log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)
        log_prob_c = torch.cat([all_log_prob_c[:, :2].sum(1, keepdim=True), all_log_prob_c[:, 2:4].sum(1, keepdim=True),
                                all_log_prob_c[:, 4:6].sum(1, keepdim=True),
                                all_log_prob_c[:, 6:].sum(1, keepdim=True)], 1)

        dist = Categorical(logits=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-8)

        return action_c, action_d, log_prob_c, log_prob_d, prob_d

    def to(self, device):
        return super(Policy, self).to(device)


def to_gym_action(action_c, action_d):
    # assuming both are torch tensors
    ac = action_c.tolist()[0]
    ad = action_d.squeeze().item()
    return [(ac[0], ac[1], ac[2], ac[3], ac[4], ac[5], ac[6], ac[7], ad)]


def wrapAction(action):
    if action[0][8] == 0.0:
        v = 0.6 * (action[0][0] + 1) / 2.
        w = 1.8 * (action[0][1] + 1) / 2. - 0.9
    elif action[0][8] == 1.0:
        v = 0.6 * (action[0][2] + 1) / 2.
        w = 1.8 * (action[0][3] + 1) / 2. - 0.9
    elif action[0][8] == 2.0:
        v = 0.6 * (action[0][4] + 1) / 2.
        w = 1.8 * (action[0][5] + 1) / 2. - 0.9
    elif action[0][8] == 3.0:
        v = 0.6 * (action[0][6] + 1) / 2.
        w = 1.8 * (action[0][7] + 1) / 2. - 0.9

    return [(v, w, action[0][8])]


def getResponse(prompt):

    response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    model="gpt-4",
    messages=[
        {"role": "system", "content": "you are a robot walking on the road, you may speak to the pedestrains and pedestrians may speak to you to clear their ways. Reply to them and make your reply concise. your reply command will also be generated based on numbers, ranging from (1, 2, 3, 4, 5, 6). 1 means ask pedestrians to stop, 2 means you will stop. 3 means ask pedestrians to margin-right, 4 means you will go margin right.  5 means ask pedestrians to margin-left, and 6 means you will go margin left. At the end of the reply, classify your reply into code (2 means you will stop, 4 means you will go margin-right, 6 means you will go margin-left) For example: 1. Input: Can I go first, You will reply to something like: Sure, go ahead! 2; 2. Input: On your left, You will reply something like I will right. 4; 3. Input: 1, You will reply something like: Could you let me pass first? 1; 4. Input: 3, You will reply something like: On your right. 3; 5.Input: 5, You will reply something like: passing on your left! 5. Your reply could not be exactly the same as the example but it should mean the same thing."},
        {"role": "user", "content": "Can I go first"},
        {"role": "assistant", "content": "Sure, go head! 2"},
        {"role": "user", "content": "On your left"},
        {"role": "assistant", "content": "I will move towards my right! 4"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "Could you let me go first? 1"},
        {"role": "user", "content": "3"},
        {"role": "assistant", "content": "Could you margin right? 3"},
        {"role": "user", "content": "5"},
        {"role": "assistant", "content": "On your left! 5"},
        {"role": "user", "content": prompt}
    ]
  )

    print("Input: {0}".format(prompt))
    print("Output: {0}".format(response.choices[0]["message"]["content"]))
    print('\n')

    return response.choices[0]["message"]["content"]


# load model
torch.manual_seed(2)
pre_net = NavPedPreNet(image_channel=4, last_output_dim=512)

pg = Policy(input_shape, out_c, out_d).to(device)
#pg.load_state_dict(torch.load("/home/geeta/catkin_cb/src/c_step_120192_llm.pth")) #cross_walk
pg.load_state_dict(torch.load("/home/geeta/catkin_cb/src/h_step_128000_llm.pth")) #hall
#pg.load_state_dict(torch.load("/home/geeta/catkin_cb/src/b_step_128000_llm.pth")) #box
pg.eval()


def distance(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def _draw_ped_map(ped_info) -> np.ndarray:
    """
        draw the pedestrian map, which consisted of 3 channels [X veloicty, Y velocity, pos]
        detail information, see paper:
        self.ped_info = [x_position, y_position, orientation, relative_speed_x, relative_speed_y]
    """
    ped_dia = 0.6
    robot_dia = 0.17
    max_ped = 10
    ped_vec_dim = 7
    ped_image_r = 0.3  # the radius of pedestrians in ped_image, paper:
    resolution = 6 / 48
    ped_tmp = np.zeros([max_ped * ped_vec_dim + 1], dtype=np.float32)
    ped_tmp[0] = 1
    ped_image = np.zeros([3, 48, 48], dtype=np.float32)
    for j in range(int(ped_tmp[0])):

        ped_tmp[j * ped_vec_dim + 1] = ped_info[0]
        ped_tmp[j * ped_vec_dim + 2] = ped_info[1]
        ped_tmp[j * ped_vec_dim + 3] = ped_info[3]
        ped_tmp[j * ped_vec_dim + 4] = ped_info[4]
        ped_r = round(ped_dia, 2)
        ped_tmp[j * ped_vec_dim + 5] = ped_r
        ped_tmp[j * ped_vec_dim + 6] = ped_r + robot_dia
        ped_tmp[j * ped_vec_dim + 7] = math.sqrt(ped_info[0] ** 2 + ped_info[1] ** 2)

        # need to confirm
        if ped_info[0] > 3 or ped_info[0] < -3 or ped_info[1] > 3 or ped_info[1] < -3:
            continue

        tmx, tmy = -ped_info[0] + 3, -ped_info[1] + 3

        # below this need change in future.
        # draw grid which midpoint in circle
        coor_tmx = (tmx - ped_image_r) // resolution, (tmx + ped_image_r) // resolution
        coor_tmy = (tmy - ped_image_r) // resolution, (tmy + ped_image_r) // resolution
        coor_tmx = list(map(int, coor_tmx))
        coor_tmy = list(map(int, coor_tmy))

        for jj in range(*coor_tmx):
            for kk in range(*coor_tmy):
                if jj < 0 or jj >= 48 or kk < 0 or kk >= 48:
                    pass
                else:
                    if distance((jj + 0.5) * resolution, (kk + 0.5) * resolution, tmx,
                                tmy) < ped_image_r ** 2:
                        # Put 3 values per pixel
                        ped_image[:, jj, kk] = 1.0, ped_info[3], ped_info[4]
    # imageio.imsave("{}ped.png".format(self.index), ped_image[0])
    return ped_image


def trans_cv2_sensor_map(image_path):
    # Load the PNG image using OpenCV
    cv_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image using cubic interpolation
    img_data = cv2.resize(cv_image, (48, 48), interpolation=cv2.INTER_CUBIC)

    # Normalize the pixel values to the range [0, 1]
    img_data_normalized = img_data.astype('float16') / 255.0

    return img_data_normalized

def txt_to_voice(txt):
    speak_engine.say(txt)
    speak_engine.runAndWait()


def voice_to_txt():    
    r = sr.Recognizer()
    # Reading Microphone as source
    # listening the speech and store in audio_text variable
    with sr.Microphone() as source:
        for i in range(10):
            print("Talk")
            audio_text = r.listen(source)
            print("Time over, thanks")
            # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling

            try:
                # using google speech recognition
                print("Text: " + r.recognize_google(audio_text))
                speech_queue.append(r.recognize_google(audio_text))
                time.sleep(2)
            except:
                speech_queue.append("None")
                print("Sorry, I did not get that")

def observation_to_state(processed_image, relative_goal, ped_info, ped_reply_string):
    # state_shape nd.array (1, 9) (t-2x, t-2y, t-2yaw, t-1x, t-1y, t-1yaw, tx, ty, tyaw)
    # yaw is robot's yaw from start
    # real velocity and angular speed is 0.4 * calculated speed
    
    if ped_reply_string == "None":
        ped_reply = np.array([[0.0]])
    else:
        # ped_reply = np.array([[0.0]])
        
        gptResponse = getResponse(ped_reply_string)
        reply_code = float(gptResponse[-1])
        ped_reply = np.array([[reply_code]])

        print(ped_reply_string)
        print(reply_code)

        global if_replied

        if (if_replied == False):
            txt_to_voice(gptResponse[0:-2])

            if_replied = True
        

    sensor_maps = processed_image
    sensor_maps = sensor_maps.reshape(1, 1, 48, 48)

    ped_image = _draw_ped_map(ped_info)
    ped_image = ped_image.reshape(1, 3, 48, 48)

    vector_state = np.array([relative_goal])

    return [sensor_maps, vector_state, ped_image, ped_reply]

class GridMapVisualizer:
    def __init__(self):
        rospy.init_node('grid_map_visualizer')
        # rospy.init_node('cmd_vel_random_publisher', anonymous=True)
        # Set parameters
        self.map_size = 400
        self.resolution = 0.05
        self.goal_x = 3.0  # Replace with your desired x goal
        self.goal_y = 0.0  # Replace with your desired y goal

        self.relative_pub = rospy.Publisher('/relative_goal', Odometry, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.ped_sub = rospy.Subscriber('/people_tracked', PersonArray, self.people_tracked_callback)

        # Create a CvBridge
        self.bridge = CvBridge()

        # Create a blank image with a bit lighter gray color
        self.blank_image = np.ones((self.map_size, self.map_size), dtype=np.uint8) * 200

        # keep the most recent three states
        self.fixed_size_queue = deque(maxlen=9)
        for i in range(9):
            self.fixed_size_queue.append(0.0)

        # variables to store previous data
        self.prev_positions_ped = []
        self.ped_info = [0, 0, 0, 0, 0]
        self.prev_time_ped = None

        # Store max lidar range
        self.max_lidar_range = 10  # Adjust this value as needed

        # Subscribe to LaserScan topic
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # Create a TensorFlow session
        self.session = tf.compat.v1.Session()

        self.image_path = 'sensor_map.png'
        self.tensor_data_filename = 'tensor_data.npy'


        # Set the timer to run the data processing for 1 second
        rospy.Timer(rospy.Duration(1), self.timer_callback)

        # Main loop
        rospy.spin()



    def people_tracked_callback(self, data):

        self.ped_info = [0, 0, 0, 0, 0]

        # Get the current time
        current_time = rospy.get_time()

        # Extract positions and orientations from the PersonArray message
        current_positions = []
        current_orientations = []

        for person in data.people:
            current_positions.append((person.pose.position.x, person.pose.position.y))
            current_orientations.append((
                person.pose.orientation.x, person.pose.orientation.y, person.pose.orientation.z,
                person.pose.orientation.w))

        # Check if this is the first message received
        if not self.prev_positions_ped:
            self.prev_positions_ped = current_positions
            self.prev_time_ped = current_time  # Initialize prev_time
            return

        # Calculate relative speeds
        relative_speeds = []

        for i in range(len(current_positions)):
            dx = current_positions[i][0] - self.prev_positions_ped[i][0]
            dy = current_positions[i][1] - self.prev_positions_ped[i][1]

            # Calculate the time difference
            time_diff = current_time - self.prev_time_ped

            # Check for zero division before calculating relative speed
            if time_diff > 0:
                relative_speed_x = dx / time_diff
                relative_speed_y = dy / time_diff
            else:
                relative_speed_x = 0.0
                relative_speed_y = 0.0

            # Save the position and orientation data
            x_position, y_position = current_positions[i]
            orientation = current_orientations[i]

            # Append the data to the relative_speeds list
            relative_speeds.append((x_position, y_position, orientation, relative_speed_x, relative_speed_y))

        # Print the relative speeds, positions, and orientations
        # for i, (x, y, orientation, speed_x, speed_y) in enumerate(relative_speeds):
        #     rospy.loginfo("Person %d - Relative Speed in X: %.4f m/s, Y: %.4f m/s", i + 1, speed_x, speed_y)
        #     rospy.loginfo("Person %d - X Position: %.4f m, Y Position: %.4f m", i + 1, x, y)
        #     rospy.loginfo("Person %d - Orientation (x, y, z, w): %.4f, %.4f, %.4f, %.4f", i + 1, orientation[0],
        #                   orientation[1], orientation[2], orientation[3])

        # Update the previous positions and time for the next callback
        self.prev_positions_ped = current_positions
        self.prev_time_ped = current_time
        self.ped_info = [x_position, y_position, orientation, relative_speed_x, relative_speed_y]

    def odom_callback(self, odom_msg):
        current_position = odom_msg.pose.pose.position
        current_orientation = odom_msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([
            current_orientation.x, current_orientation.y,
            current_orientation.z, current_orientation.w
        ])

        relative_goal_msg = Odometry()
        relative_goal_msg.header.stamp = rospy.Time.now()
        relative_goal_msg.header.frame_id = "odom"

        relative_goal_msg.pose.pose.position = Point(
            self.goal_x - current_position.x,
            self.goal_y - current_position.y,
            0.0  # Assuming z remains constant
        )
        relative_goal_msg.pose.pose.orientation = yaw

        self.relative_goal = relative_goal_msg
        self.relative_pub.publish(relative_goal_msg)


    def scan_callback(self, scan_msg):
        self.scan_data = scan_msg

    def timer_callback(self, event):
        if hasattr(self, 'scan_data'):
            scan = self.scan_data
            maxAngle = scan.angle_max
            minAngle = scan.angle_min
            angleInc = scan.angle_increment
            ranges = scan.ranges
            num_pts = len(ranges)

            disc_factor = self.map_size / (2 * self.max_lidar_range)

            image_copy = np.copy(self.blank_image)
            previous_dis = None

            # Initialize closest_edge_x and closest_edge_y
            closest_edge_x = -1
            closest_edge_y = -1

            for i in range(num_pts - 1, 0, -1):
                angle = minAngle + float(i) * angleInc
                distance = ranges[i]

                if 0 <= distance <= self.max_lidar_range:
                    x = int((distance * math.cos(angle + math.pi / 2) + self.max_lidar_range) * disc_factor)
                    y = int((-distance * math.sin(angle + math.pi / 2) + self.max_lidar_range) * disc_factor)

                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        cv2.line(image_copy, (self.map_size // 2, self.map_size // 2), (x, y), 255, 1)

                        if previous_dis is not None and (previous_dis - distance) > 0.5:
                            closest_edge_x = x
                            closest_edge_y = y
                previous_dis = distance

            # Draw the LiDAR square at the center
            self.draw_lidar_circle(image_copy)

            # Check if closest_edge_x and closest_edge_y have been assigned values
            # if closest_edge_x != -1 and closest_edge_y != -1:
            # Draw a black dot at the closest laser edge
            # cv2.circle(image_copy, (closest_edge_x, closest_edge_y), 5, 0, -1)

            # Save the image as a PNG file

            # Check if closest_edge_x and closest_edge_y have been assigned values
            if closest_edge_x != -1 and closest_edge_y != -1:
                # Draw a black dot at the closest laser edge
                cv2.circle(image_copy, (closest_edge_x, closest_edge_y), 6, 0, -1)

            # Zoom in the visualization by cropping a smaller region
            zoom_factor = 0.6  # Adjust this value to zoom in/out
            crop_size = int(self.map_size * zoom_factor)
            cropped_image = image_copy[
                            (self.map_size - crop_size) // 2: (self.map_size + crop_size) // 2,
                            (self.map_size - crop_size) // 2: (self.map_size + crop_size) // 2
                            ]

            # Resize the cropped image back to the original window size
            resized_image = cv2.resize(cropped_image, (self.map_size, self.map_size))

            # Flip the visualization horizontally to the left and then mirror
            resized_image = np.flip(resized_image, axis=1)
            resized_image = np.flip(resized_image, axis=1)

            cv2.imwrite(self.image_path, resized_image)

            # Process the image using trans_cv2_sensor_map
            processed_image = trans_cv2_sensor_map(self.image_path)
            processed_image[23][23] = 100.0 / 255.0
            processed_image[23][24] = 100.0 / 255.0

            processed_image[24][23] = 100.0 / 255.0
            processed_image[24][24] = 100.0 / 255.0

            # Save the processed tensor data as .npy file
            np.save(self.tensor_data_filename, processed_image)

            # Print tensor shape
            print("Processed tensor shape:", processed_image.shape)
            print("Processed single tensor element:", processed_image[0, 0])
            print(self.relative_goal.pose.pose.position)
            print("here1:", self.relative_goal.pose.pose.orientation)
            # print(self.ped_info)

            # Display the map
            self.display_map(image_copy)
            relative_x = self.relative_goal.pose.pose.position.x * np.cos(
                -self.relative_goal.pose.pose.orientation) - self.relative_goal.pose.pose.position.y * np.sin(
                -self.relative_goal.pose.pose.orientation)
            relative_y = self.relative_goal.pose.pose.position.x * np.sin(
                -self.relative_goal.pose.pose.orientation) + self.relative_goal.pose.pose.position.y * np.cos(
                -self.relative_goal.pose.pose.orientation)

            self.fixed_size_queue.append(relative_x)
            self.fixed_size_queue.append(relative_y)
            # self.fixed_size_queue.append(self.relative_goal.pose.pose.position.x)
            # self.fixed_size_queue.append(self.relative_goal.pose.pose.position.y)

            self.fixed_size_queue.append(-self.relative_goal.pose.pose.orientation)
            print("here2:", self.relative_goal.pose.pose.orientation)
            # self.fixed_size_queue.append(self.relative_goal.pose.pose.orientation[2])

            print(self.fixed_size_queue)

            ped_reply_string = speech_queue[-1]
            print("ped_reply:", ped_reply_string, speech_queue)
            # ped_reply_string = "None"

            state = observation_to_state(processed_image, self.fixed_size_queue, self.ped_info, ped_reply_string)
            # do step
            action_c, action_d, _, _, _ = pg.get_action([pre_net(state).cpu().detach().numpy()[0]], device)
            action = to_gym_action(action_c, action_d)

            # modify later
            ped_reply = 0.0
            env_action = action
            if ped_reply == 2.0 or ped_reply == 4.0 or ped_reply == 6.0:
                env_action = [(action[0][0], action[0][1], action[0][2], action[0][3], action[0][4], action[0][5],
                               action[0][6], action[0][7], 0.0)]

            v = wrapAction(env_action)[0][0]
            w = wrapAction(env_action)[0][1]
            # v = 0
            # w = 0
            robot_voice_choice = wrapAction(env_action)[0][2]

            min_ped_dist = self.ped_info[0] * self.ped_info[0] + self.ped_info[1] * self.ped_info[1]
            print("ped dis: ", min_ped_dist)
            global if_replied
            #if robot_voice_choice != 0.0 and min_ped_dist > 0.0 and min_ped_dist <= 10.0 and (if_replied == False):
            if robot_voice_choice != 0.0 and min_ped_dist > 0.0 and min_ped_dist <= -1.0 and (if_replied == False):
                robot_reply_string = getResponse(str(int(robot_voice_choice)))
                txt_to_voice(robot_reply_string[0:-2])
                print("Replied: ", robot_reply_string)
                if_replied = True

            print(v, w, robot_voice_choice)
            cmd_msg = Twist()
            # 0.6, 0.6
            cmd_msg.linear.x = 0.4 * v  # Random linear velocity in the range [-1, 1]
            cmd_msg.linear.y = 0.0  # Random linear velocity in the range [-1, 1]
            cmd_msg.linear.z = 0.0  # Random linear velocity in the range [-1, 1]
            cmd_msg.angular.z = 0.4 * w  # Random angular velocity in the range [-1, 1]

            # Publish the Twist message
            self.cmd_vel_pub.publish(cmd_msg)

    def draw_lidar_circle(self, image):
        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the center coordinates of the image
        center_x = width // 2
        center_y = height // 2

        # Calculate the radius of the circle (adjust the value as needed)
        radius = 6

        # Set the color of the circle (white in this example)
        color = 100

        # Draw the circle on the image
        cv2.circle(image, (center_x, center_y), radius, color, -1)

    def draw_lidar_square(self, image):
        # Reduce the size of the square at the center to represent the LiDAR
        lidar_size = 12  # Adjust this value for the size of the square
        center_x = self.map_size // 2
        center_y = self.map_size // 2
        start_x = center_x - lidar_size // 2
        start_y = center_y - lidar_size // 2
        image[start_y:start_y + lidar_size, start_x:start_x + lidar_size] = 100

    def display_map(self, image):
        # Zoom in the visualization by cropping a smaller region
        zoom_factor = 1  # Adjust this value to zoom in/out
        crop_size = int(self.map_size * zoom_factor)
        cropped_image = image[
                        (self.map_size - crop_size) // 2: (self.map_size + crop_size) // 2,
                        (self.map_size - crop_size) // 2: (self.map_size + crop_size) // 2
                        ]

        # Resize the cropped image back to the original window size
        resized_image = cv2.resize(cropped_image, (self.map_size, self.map_size))

        # Flip the visualization horizontally to the left and then mirror
        resized_image = np.flip(resized_image, axis=1)
        resized_image = np.flip(resized_image, axis=1)

        # Show the map
        cv2.imshow('Map', resized_image)
        cv2.waitKey(3)


if __name__ == '__main__':
    background_thread = threading.Thread(target=voice_to_txt, args=())
    background_thread.start()
    visualizer = GridMapVisualizer()
    # print("Main")



