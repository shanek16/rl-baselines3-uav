import os
import numpy as np
from . import rendering

from gym import Env
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gym.utils import seeding
from .mdp import Actions, States
from numpy import arctan2, array, cos, pi, sin
from PIL import Image, ImageDraw, ImageFont


class Rand_cycle(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        r_max=40,
        r_min=0,
        dt=0.05,
        v=1.0,
        d=10.0,
        l=3,  # noqa
        m=3,
        n=2,
        r_c=3,
        d_min=4.5,
        max_step=3600, # one circle 1200 time steps
    ):  # m: # of target n: # of uavs
        self.observation_space = Dict(
            {  # r, beta, theta
                # only r, alpha does not imply information of abs position of uav
                "uav1_state": Box(
                    low=np.float32([r_min, -pi, -pi]),
                    high=np.float32([r_max, pi, pi]),
                    dtype=np.float32,
                ),
                "uav2_state": Box(
                    low=np.float32([r_min, -pi, -pi]),
                    high=np.float32([r_max, pi, pi]),
                    dtype=np.float32,
                ),
                "target1_position": Box(
                    low=np.float32([r_min, -pi]), high=np.float32([r_max, pi]), dtype=np.float32
                ),
                "target2_position": Box(
                    low=np.float32([r_min, -pi]), high=np.float32([r_max, pi]), dtype=np.float32
                ),
                "target3_position": Box(
                    low=np.float32([r_min, -pi]), high=np.float32([r_max, pi]), dtype=np.float32
                ),
                "battery": Box(
                    low=np.float32([0, 0]),
                    high=np.float32([3000, 3000]),
                    dtype=np.float32,
                ),
                "surveillance": MultiBinary(
                    array([3])
                ),  # 1 or 0 [Target1, Target2, Target3]
                "charge_station_occupancy": Discrete(
                    3
                ),  # 0:free, 1:uav1 docked, 2:uav2 docked
            }
        )
        self.action_space = MultiDiscrete(
            [4, 4]
        )  # 0: charge, n: surveillance target n-1
        self.dt = dt
        self.v = v
        self.vdt = v * dt
        self.d = d  # target distance
        self.l = l  # coverage gap: coverage: d-l ~ d+l # noqa
        self.m = m  # of targets
        self.n = n  # of uavs
        self.r_c = r_c  # charge station radius
        self.omega_max = v / d_min
        self.step_count = None
        self.uav1_state = None
        self.uav2_state = None
        self.target1_state = None
        self.target2_state = None
        self.target3_state = None
        self.battery = array([3000, 3000])  # 3rounds*1200steps/round
        self.charge_station_occupancy = None
        self.surveillance = None
        # for debugging
        self.uav1_in_charge_station = 0
        self.uav2_in_charge_station = 0
        self.uav1docked_time = 0
        self.uav2docked_time = 0
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
        self.num2str={0:'charge', 1:'target_1', 2:'target_2', 3:'target_3'}

        self.seed()
        self.max_step = max_step
        self.viewer = None
        self.SAVE_FRAMES_PATH = "rand_frames_01/"
        self.episode_counter = 0
        self.frame_counter = 0
        self.save_frames = False

        # initialization for Dynamic Programming
        self.n_r = 800
        self.n_alpha = 360
        self.n_u = 21
        self.distance_keeping_result00 = np.load("./rl_zoo3/80_dkc_result_0.0.npz")
        self.time_optimal_result00 = np.load("./rl_zoo3/80_toc_result_0.0.npz")

        self.distance_keeping_straightened_policy00 = self.distance_keeping_result00[
            "policy"
        ]
        self.time_optimal_straightened_policy00 = self.time_optimal_result00["policy"]

        self.states = States(
            np.linspace(0.0, 80.0, self.n_r, dtype=np.float32),
            np.linspace(
                -np.pi + np.pi / self.n_alpha,
                np.pi - np.pi / self.n_alpha,
                self.n_alpha,
                dtype=np.float32,
            ),
            cycles=[None, np.pi * 2],
        )

        self.actions = Actions(
            np.linspace(-1.0 / 4.5, 1.0 / 4.5, self.n_u, dtype=np.float32).reshape(
                (-1, 1)
            )
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
        self,
        uav1_pose=None,
        uav2_pose=None,
        target1_pose=None,
        target2_pose=None,
        target3_pose=None,
    ):
        self.episode_counter += 1
        self.step_count = 0
        if self.save_frames:
            os.makedirs(
                os.path.join(self.SAVE_FRAMES_PATH, f"{self.episode_counter:03d}"),
                exist_ok=True,
            )
            self.frame_counter = 0
        if uav1_pose is None:
            uav1_r = self.np_random.uniform(0, 40)  # D=40
            uav1_beta = self.np_random.uniform(-pi, pi)
            self.uav1_state = array(
                (
                    uav1_r * cos(uav1_beta),
                    uav1_r * sin(uav1_beta),
                    self.np_random.uniform(-pi, pi),  # = theta
                )
            )
        elif uav1_pose is not None:
            self.uav1_state = uav1_pose

        if uav2_pose is None:
            uav2_r = self.np_random.uniform(0, 40)  # D=40
            uav2_beta = self.np_random.uniform(-pi, pi)
            self.uav2_state = array(
                (
                    uav2_r * cos(uav2_beta),
                    uav2_r * sin(uav2_beta),
                    self.np_random.uniform(-pi, pi),  # = theta
                )
            )
        elif uav2_pose is not None:
            self.uav2_state = uav2_pose

        if target1_pose is None:
            target1_r = self.np_random.uniform(0, 30)  # 0~ D-d
            target1_beta = self.np_random.uniform(-pi, pi)
            self.target1_state = array(
                (target1_r * cos(target1_beta), target1_r * sin(target1_beta))
            )
        elif target1_pose is not None:
            self.target1_state = target1_pose

        if target2_pose is None:
            target2_r = self.np_random.uniform(0, 30)  # 0~ D-d
            target2_beta = self.np_random.uniform(-pi, pi)
            self.target2_state = array(
                (target2_r * cos(target2_beta), target2_r * sin(target2_beta))
            )
        elif target2_pose is not None:
            self.target2_state = target2_pose

        if target3_pose is None:
            target3_r = self.np_random.uniform(0, 30)  # 0~ D-d
            target3_beta = self.np_random.uniform(-pi, pi)
            self.target3_state = array(
                (target3_r * cos(target3_beta), target3_r * sin(target3_beta))
            )
        elif target3_pose is not None:
            self.target3_state = target3_pose

        self.battery = array([3000, 3000])
        self.uav1_in_charge_station = 0
        self.uav2_in_charge_station = 0
        self.uav1docked_time = 0
        self.uav2docked_time = 0
        self.charge_station_occupancy = 0
        self.surveillance = array([0,0,0])
        return self.observation

    def uav1kinematics(self, action):
        dtheta = action[0] * self.dt
        _lambda = dtheta / 2
        if _lambda == 0.0:
            self.uav1_state[0] += self.vdt * cos(self.uav1_state[-1])
            self.uav1_state[1] += self.vdt * sin(self.uav1_state[-1])
        else:
            ds = self.vdt * sin(_lambda) / _lambda
            self.uav1_state[0] += ds * cos(self.uav1_state[-1] + _lambda)
            self.uav1_state[1] += ds * sin(self.uav1_state[-1] + _lambda)
            self.uav1_state[2] += dtheta
            self.uav1_state[2] = wrap(self.uav1_state[2])

    def uav2kinematics(self, action):
        dtheta = action[0] * self.dt
        _lambda = dtheta / 2
        if _lambda == 0.0:
            self.uav2_state[0] += self.vdt * cos(self.uav2_state[-1])
            self.uav2_state[1] += self.vdt * sin(self.uav2_state[-1])
        else:
            ds = self.vdt * sin(_lambda) / _lambda
            self.uav2_state[0] += ds * cos(self.uav2_state[-1] + _lambda)
            self.uav2_state[1] += ds * sin(self.uav2_state[-1] + _lambda)
            self.uav2_state[2] += dtheta
            self.uav2_state[2] = wrap(self.uav2_state[2])

    def toc_get_action(self, state):
        S, P = self.states.computeBarycentric(state)
        action = 0
        for s, p in zip(S, P):
            action += p * self.actions[int(self.time_optimal_straightened_policy00[s])]
        return action

    def dkc_get_action(self, state):
        S, P = self.states.computeBarycentric(state)
        action = 0
        for s, p in zip(S, P):
            action += (
                p * self.actions[int(self.distance_keeping_straightened_policy00[s])]
            )
        return action

    def step(self, action):
        terminal = False
        # action clipping is done in dp already
        # uav1
        battery1, battery2 = self.battery
        if action[0] == 0:  # go to charge uav1
            if (
                self.observation1[0] < self.r_c and self.uav2_in_charge_station == False
            ):  # uav1_in_charge_station
                # uav1 no move
                self.uav1_in_charge_station = 1
                if self.uav1docked_time ==0: #landing
                    self.uav1docked_time += 1
                    # battery stays the same(docking time)
                else: # uav1docked_time > 0
                    battery1 = min(battery1 + 10, 3000) 
                    self.uav1docked_time += 1
            else: # not able to land on charge station(too far/uav2 is in)
                self.uav1_in_charge_station = 0
                self.uav1docked_time = 0
                battery1 -= 1
                w1_action = self.toc_get_action(self.observation1[:2])
                self.uav1kinematics(w1_action)
        elif action[0] == 1:  # surveil target1
            self.uav1_in_charge_station = 0
            self.uav1docked_time = 0
            battery1 -= 1
            w1_action = self.dkc_get_action(self.rel_observation(uav=1, target=1))
            self.uav1kinematics(w1_action)
        elif action[0] == 2:  # surveil target2
            self.uav1_in_charge_station = 0
            self.uav1docked_time = 0
            battery1 -= 1
            w1_action = self.dkc_get_action(self.rel_observation(uav=1, target=2))
            self.uav1kinematics(w1_action)
        else:  # surveil target3
            self.uav1_in_charge_station = 0
            self.uav1docked_time = 0
            battery1 -= 1
            w1_action = self.dkc_get_action(self.rel_observation(uav=1, target=3))
            self.uav1kinematics(w1_action)

        # uav2
        if action[1] == 0:  # go to charge uav2
            if (
                self.observation2[0] < self.r_c and self.uav1_in_charge_station == False
            ):  # uav2_in_charge_station
                # uav2 no move
                self.uav2_in_charge_station = 2
                if self.uav2docked_time ==0: #landing
                    self.uav2docked_time += 1
                    # battery stays the same(docking time)
                else: # uav2docked_time > 0
                    battery2 = min(battery2 + 10, 3000) 
                    self.uav2docked_time += 1
            else: # not able to land on charge station(too far/uav1 is in)
                self.uav2_in_charge_station = 0
                self.uav2docked_time = 0
                battery2 -= 1
                w2_action = self.toc_get_action(self.observation2[:2])
                self.uav2kinematics(w2_action)
        elif action[1] == 1:  # surveil target1
            self.uav2_in_charge_station = 0
            self.uav2docked_time = 0
            battery2 -= 1
            w2_action = self.dkc_get_action(self.rel_observation(uav=2, target=1))
            self.uav2kinematics(w2_action)
        elif action[1] == 2:  # surveil target2
            self.uav2_in_charge_station = 0
            self.uav2docked_time = 0
            battery2 -= 1
            w2_action = self.dkc_get_action(self.rel_observation(uav=2, target=2))
            self.uav2kinematics(w2_action)
        else:  # surveil target3
            self.uav2_in_charge_station = 0
            self.uav2docked_time = 0
            battery2 -= 1
            w2_action = self.dkc_get_action(self.rel_observation(uav=2, target=3))
            self.uav2kinematics(w2_action)
        self.charge_station_occupancy = max(self.uav1_in_charge_station, self.uav2_in_charge_station)
        self.surveillance = self.cal_surveillance()
        self.battery = array([battery1, battery2])
        # print('battery: ', self.battery)

        # reward = (2 * L1(self.surveillance) - self.n) / self.n  # -1~1
        if self.charge_station_occupancy > 0:
            reward = -1
        reward = -2
        reward = reward + 3*L1(self.surveillance)/self.n # -2~1
        if self.step_count > self.max_step:
            terminal = True
            # print("terminal True: over max_step")
        if min(self.battery) == 0:  # either one of uav falls
            reward = -3600*2 # - max_timestep*2
            terminal = True
            # print("terminal True: run out of battery")

        if self.save_frames and int(self.step_count) % 6 == 0:
            image = self.render(action, mode="rgb_array")
            path = os.path.join(
                self.SAVE_FRAMES_PATH,
                f"{self.episode_counter:03d}",
                f"{self.frame_counter+1:04d}.bmp",
            )
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            text_1 = "uav1_in_charge_station: {}".format(self.uav1_in_charge_station)
            text0 = "uav2_in_charge_station: {}".format(self.uav2_in_charge_station)
            text1 = "charge station occupancy: {}".format(self.charge_station_occupancy)
            text2 = "surveillance : {}".format(self.surveillance)
            text3 = "uav1docked_time: {}".format(self.uav1docked_time)
            text4 = "uav2docked_time: {}".format(self.uav2docked_time)
            text5 = "uav1 action: {}".format(self.num2str[action[0]])
            text6 = "uav2 action: {}".format(self.num2str[action[1]])
            text7 = "uav1 battery: {}".format(self.battery[0])
            text8 = "uav2 battery: {}".format(self.battery[1])
            draw.text((0, 0), text_1, color=(200, 200, 200), font=self.font)
            draw.text((0, 20), text0, color=(200, 200, 200), font=self.font)
            draw.text((0, 40), text1, color=(200, 200, 200), font=self.font)
            draw.text((0, 60), text2, color=(200, 200, 200), font=self.font)
            draw.text((0, 80), text3, color=(200, 200, 200), font=self.font)
            draw.text((0, 100), text4, color=(200, 200, 200), font=self.font)
            draw.text((0, 120), text5, color=(255, 255, 0), font=self.font)
            draw.text((0, 140), text6, color=(255, 255, 255), font=self.font)
            draw.text((770, 0), text7, color=(255, 255, 255), font=self.font)
            draw.text((770, 20), text8, color=(255, 255, 255), font=self.font)
            image.save(path)
            self.frame_counter += 1
        self.step_count += 1
        return self.observation, reward, terminal, {}

    def render(self, action, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000, 1000)
            bound = int(40 * 1.05)
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        # target1
        target1_x, target1_y = self.target1_state
        # draw donut
        outer_donut = self.viewer.draw_circle(
            radius=self.d + self.l, x=target1_x, y=target1_y, filled=True
        )
        if self.surveillance[0] == 1:
            outer_donut.set_color(0.6, 0.6, 1.0, 0.3) # lighter
        else:
            outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
        inner_donut = self.viewer.draw_circle(
            radius=self.d - self.l, x=target1_x, y=target1_y, filled=True
        )
        inner_donut.set_color(0, 0, 0)  # erase inner part
        # target1 & circle
        circle = self.viewer.draw_circle(
            radius=self.d, x=target1_x, y=target1_y, filled=False
        )
        circle.set_color(1, 1, 1)
        # draw target
        target1 = self.viewer.draw_circle(
            radius=2, x=target1_x, y=target1_y, filled=True
        )
        if action[0] == 1:
            target1.set_color(1,1,0) # yellow
        elif action[1] == 1:
            target1.set_color(0.9,0.9,0.9) # white
        else:
            target1.set_color(1, 0.6, 0)  # orange

        # target2
        target2_x, target2_y = self.target2_state
        # draw donut
        outer_donut = self.viewer.draw_circle(
            radius=self.d + self.l, x=target2_x, y=target2_y, filled=True
        )
        if self.surveillance[1] == 1:
            outer_donut.set_color(0.6, 0.6, 1.0, 0.3) # lighter
        else:
            outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
        inner_donut = self.viewer.draw_circle(
            radius=self.d - self.l, x=target2_x, y=target2_y, filled=True
        )
        inner_donut.set_color(0, 0, 0)  # erase inner part
        # target2 & circle
        circle = self.viewer.draw_circle(
            radius=self.d, x=target2_x, y=target2_y, filled=False
        )
        circle.set_color(1, 1, 1)
        # draw target
        target2 = self.viewer.draw_circle(
            radius=2, x=target2_x, y=target2_y, filled=True
        )
        if action[0] == 2:
            target2.set_color(1,1,0) # yellow
        elif action[1] == 2:
            target2.set_color(0.9,0.9,0.9) # white
        else:
            target2.set_color(1, 0.6, 0)  # orange
        # target3
        target3_x, target3_y = self.target3_state
        # draw donut
        outer_donut = self.viewer.draw_circle(
            radius=self.d + self.l, x=target3_x, y=target3_y, filled=True
        )
        if self.surveillance[2] == 1:
            outer_donut.set_color(0.6, 0.6, 1.0, 0.3) # lighter
        else:
            outer_donut.set_color(0.3, 0.3, 0.9, 0.3)  # transparent blue
        inner_donut = self.viewer.draw_circle(
            radius=self.d - self.l, x=target3_x, y=target3_y, filled=True
        )
        inner_donut.set_color(0, 0, 0)  # erase inner part
        # target3 & circle
        circle = self.viewer.draw_circle(
            radius=self.d, x=target3_x, y=target3_y, filled=False
        )
        circle.set_color(1, 1, 1)
        # draw target
        target3 = self.viewer.draw_circle(
            radius=2, x=target3_x, y=target3_y, filled=True
        )
        if action[0] == 3:
            target3.set_color(1,1,0) # yellow
        elif action[1] == 3:
            target3.set_color(0.9,0.9,0.9) # white
        else:
            target3.set_color(1, 0.6, 0)  # orange

        # charge station @ origin
        charge_station = self.viewer.draw_circle(radius=self.r_c, filled=True)
        if self.charge_station_occupancy == 0:
            if action[0]==0:
                charge_station.set_color(1, 1, 0)  # yellow
            elif action[1]==0:
                charge_station.set_color(0.9, 0.9, 0.9)  # white
            else:
                charge_station.set_color(0.1, 0.9, 0.1)  # green
        else:
            charge_station.set_color(0.9, 0.1, 0.1)  # red

        # uav1 (yellow)
        uav1_x, uav1_y, uav1_theta = self.uav1_state
        uav1_tf = rendering.Transform(translation=(uav1_x, uav1_y), rotation=uav1_theta)
        uav1_tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
        uav1_tri.set_color(1, 1, 0)  # yellow
        uav1_tri.add_attr(uav1_tf)
        # uav2 (white)
        uav2_x, uav2_y, uav2_theta = self.uav2_state
        uav2_tf = rendering.Transform(translation=(uav2_x, uav2_y), rotation=uav2_theta)
        uav2_tri = self.viewer.draw_polygon([(-0.8, 0.8), (-0.8, -0.8), (1.6, 0)])
        uav2_tri.set_color(0.9, 0.9, 0.9)  # white
        uav2_tri.add_attr(uav2_tf)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    # relative position
    @property
    def observation1(self):
        x, y = self.uav1_state[:2]
        r = (x**2 + y**2) ** 0.5
        alpha = wrap(arctan2(y, x) - wrap(self.uav1_state[-1]) - pi)
        return array([r, alpha, self.uav1_state[2]])  # beta

    @property
    def observation2(self):
        x, y = self.uav2_state[:2]
        r = (x**2 + y**2) ** 0.5
        alpha = wrap(arctan2(y, x) - wrap(self.uav2_state[-1]) - pi)
        return array([r, alpha, self.uav2_state[2]])  # beta

    @property
    def target1_obs(self):
        x, y = self.target1_state[:2]
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        return array([r, beta])  # beta

    @property
    def target2_obs(self):
        x, y = self.target2_state[:2]
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        return array([r, beta])  # beta

    @property
    def target3_obs(self):
        x, y = self.target3_state[:2]
        r = (x**2 + y**2) ** 0.5
        beta = arctan2(y, x)
        return array([r, beta])  # beta

    # absolute position
    def rel_observation(self, uav, target):
        if uav == 1:
            uav_x, uav_y = self.uav1_state[:2]
            beta = self.uav1_state[2]
        else:
            uav_x, uav_y = self.uav2_state[:2]
            beta = self.uav2_state[2]

        if target == 1:
            target_x, target_y = self.target1_state
        elif target == 2:
            target_x, target_y = self.target2_state
        else:
            target_x, target_y = self.target3_state

        x = uav_x - target_x
        y = uav_y - target_y
        r = (x**2 + y**2) ** 0.5
        alpha = wrap(arctan2(y, x) - wrap(beta) - pi)
        return array([r, alpha])

    def cal_surveillance(self):
        # is any uav surveilling target 1?
        if (
            (
                self.d - self.l < self.rel_observation(uav=1, target=1)[0]
                and self.rel_observation(uav=1, target=1)[0] < self.d + self.l
                and self.uav1_in_charge_station == 0
            )
            or (
                self.d - self.l < self.rel_observation(uav=2, target=1)[0]
                and self.rel_observation(uav=2, target=1)[0] < self.d + self.l
                and self.uav2_in_charge_station == 0
            )
        ):
            s1 = 1
        else:
            s1 = 0
        # is any uav surveilling target 2?
        if (
            (
                self.d - self.l < self.rel_observation(uav=1, target=2)[0]
                and self.rel_observation(uav=1, target=2)[0] < self.d + self.l
                and self.uav1_in_charge_station == 0
            )
            or (
                self.d - self.l < self.rel_observation(uav=2, target=2)[0]
                and self.rel_observation(uav=2, target=2)[0] < self.d + self.l
                and self.uav2_in_charge_station == 0
            )
        ):
            s2 = 1
        else:
            s2 = 0
        # is any uav surveilling target 3?
        if (
            (
                self.d - self.l < self.rel_observation(uav=1, target=3)[0]
                and self.rel_observation(uav=1, target=3)[0] < self.d + self.l
                and self.uav1_in_charge_station == 0
            )
            or (
                self.d - self.l < self.rel_observation(uav=2, target=3)[0]
                and self.rel_observation(uav=2, target=3)[0] < self.d + self.l
                and self.uav2_in_charge_station == 0
            )
        ):
            s3 = 1
        else:
            s3 = 0
        return array([s1, s2, s3])

    @property
    def observation(self):
        dictionary_obs = {
            "uav1_state": self.observation1,
            "uav2_state": self.observation2,
            "target1_position": self.target1_obs,
            "target2_position": self.target2_obs,
            "target3_position": self.target3_obs,
            "battery": self.battery,
            "surveillance": self.surveillance,
            "charge_station_occupancy": self.charge_station_occupancy,
        }
        return dictionary_obs

def wrap(theta):
    if theta > pi:
        theta -= 2 * pi
    elif theta < -pi:
        theta += 2 * pi
    return theta


def L1(x):
    return sum(abs(x))


if __name__ == "__main__":
    # Number of actions
    uav_env = Rand_cycle()
    action_sample = uav_env.action_space.sample()
    print("action_sample: ", action_sample)

    # Number of features
    state_sample = uav_env.observation_space.sample()
    print("state_sample: ", state_sample)

    print(uav_env.observation_space.spaces["uav1_state"])
    print(uav_env.reset()["uav1_state"])
