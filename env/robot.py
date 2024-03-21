import math
import warnings

import os
import random
from collections.abc import Iterable

import numpy as np
from gym.vector.utils import spaces
import pybullet as p
import pybullet_data
import gym
import cv2


class Robot(gym.Env):
    def __init__(self, render=False):
        self._render = render

        # 定义动作空间
        self.action_space = spaces.Box(
            low=np.array(np.float32([-100., -100., -100., -100.])),
            high=np.array(np.float32([100., 100., 100., 100.])),
            dtype=np.float32
        )

        # 定义状态空间
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(224, 224, 3), dtype=np.uint8
        )

        # 连接引擎
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        self.wheel_link_tuples = [(2, 'right_front_wheel_joint'),
                                  (3, 'right_back_wheel_joint'),
                                  (6, 'left_front_wheel_joint'),
                                  (7, 'left_back_wheel_joint')]

        # BASE_RADIUS: 是机器人底盘的半径。BASE_THICKNESS: 是机器人的厚度。
        self.BASE_RADIUS = 0.2
        self.BASE_THICKNESS = 0.6

        # 获取BASE路径
        self.BASE_DIR = os.path.dirname(__file__)

        # 计数器
        self.step_num = 0

        self.last_r = 0
        self.hit = False
        self.sum_r = 0
        self.last_b = np.array([0, 0])

    def reset(self):
        self.step_num = 0
        self.last_r = 0
        self.hit = False
        self.sum_r = 0
        self.last_b = np.array([0, 0])

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.8)

        angle = np.pi / (random.choice([1, -1])*random.randint(1, 8))
        # angle = 3 * np.pi / 4

        startPos = [0, 0, 0.6]
        startOrientation = p.getQuaternionFromEuler([0, 0, angle])
        self.robot = p.loadURDF(os.path.join(self.BASE_DIR, "urdf/r2d2.urdf"), startPos, startOrientation)
        self.sphere2red = p.loadURDF("sphere2red.urdf", basePosition=[-2, 2, 0.5], physicsClientId=self._physics_client_id)
        self.plane = p.loadURDF("plane100.urdf", physicsClientId=self._physics_client_id, useMaximalCoordinates=True)
        available_joints_indexes = [i for i in range(p.getNumJoints(self.robot)) if
                                    p.getJointInfo(self.robot, i)[2] != p.JOINT_FIXED]

        return self._get_observation()

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation(physicsClientId=self._physics_client_id)
        self.step_num += 1

        done = False
        reward, d = self._get_reward()

        self.sum_r += reward
        if self.step_num > 3600:
            if self.sum_r >= -0.0001 and self.sum_r <= 0.0001:
                reward -= 5
            done = True

        done |= d
        obs = self._get_observation()
        info = {}
        return obs, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect()
        self._physics_client_id = -1

    def _is_in_eyes(self, matrix, robotBasePos, distance):
        """
        用于判断，当前红球是否在视野内
        """
        # 用4个激光，确保在视野内的时候，reward都是true
        rayLength = 15  # 激光长度
        rayNum = 2  # 激光数量
        if distance > 3:
            rayNum = 4

        hitRayColor = [0, 1, 0]
        missRayColor = [1, 0, 0]

        if distance >= 3:
            # 这里的坐标轴有些偏倚，我在这里做一下修正
            vector = np.array([matrix[0], matrix[1]])
            angle_deg = 80 * np.pi / 180
            rotation_matrix = np.array([[np.cos(angle_deg), -np.sin(angle_deg)],
                                        [np.sin(angle_deg), np.cos(angle_deg)]])
            vector = np.dot(rotation_matrix, vector)
            norm = np.linalg.norm(vector)
            vector80 = vector / norm

            vector = np.array([matrix[0], matrix[1]])
            angle_deg = 110 * np.pi / 180
            rotation_matrix = np.array([[np.cos(angle_deg), -np.sin(angle_deg)],
                                        [np.sin(angle_deg), np.cos(angle_deg)]])
            vector = np.dot(rotation_matrix, vector)
            norm = np.linalg.norm(vector)
            vector110 = vector / norm

            vector = np.array([matrix[0], matrix[1]])
            angle_deg = 90 * np.pi / 180
            rotation_matrix = np.array([[np.cos(angle_deg), -np.sin(angle_deg)],
                                        [np.sin(angle_deg), np.cos(angle_deg)]])
            vector = np.dot(rotation_matrix, vector)
            norm = np.linalg.norm(vector)
            vector90 = vector / norm

            vector = np.array([matrix[0], matrix[1]])
            angle_deg = 100 * np.pi / 180
            rotation_matrix = np.array([[np.cos(angle_deg), -np.sin(angle_deg)],
                                        [np.sin(angle_deg), np.cos(angle_deg)]])
            vector = np.dot(rotation_matrix, vector)
            norm = np.linalg.norm(vector)
            vector100 = vector / norm

            rayFroms = [robotBasePos for _ in range(rayNum)]
            rayTos = [
                [
                    robotBasePos[0] + rayLength * vector80[1],
                    robotBasePos[1] + rayLength * vector80[0],
                    robotBasePos[2]
                ],
                [
                    robotBasePos[0] + rayLength * vector110[1],
                    robotBasePos[1] + rayLength * vector110[0],
                    robotBasePos[2]
                ],
                [
                    robotBasePos[0] + rayLength * vector90[1],
                    robotBasePos[1] + rayLength * vector90[0],
                    robotBasePos[2]
                ],
                [
                    robotBasePos[0] + rayLength * vector100[1],
                    robotBasePos[1] + rayLength * vector100[0],
                    robotBasePos[2]
                ],
            ]

        else:
            vector = np.array([matrix[0], matrix[1]])
            angle_deg = 90 * np.pi / 180
            rotation_matrix = np.array([[np.cos(angle_deg), -np.sin(angle_deg)],
                                        [np.sin(angle_deg), np.cos(angle_deg)]])
            vector = np.dot(rotation_matrix, vector)
            norm = np.linalg.norm(vector)
            vector90 = vector / norm

            vector = np.array([matrix[0], matrix[1]])
            angle_deg = 100 * np.pi / 180
            rotation_matrix = np.array([[np.cos(angle_deg), -np.sin(angle_deg)],
                                        [np.sin(angle_deg), np.cos(angle_deg)]])
            vector = np.dot(rotation_matrix, vector)
            norm = np.linalg.norm(vector)
            vector100 = vector / norm

            rayFroms = [robotBasePos for _ in range(rayNum)]
            rayTos = [
                [
                    robotBasePos[0] + rayLength * vector90[1],
                    robotBasePos[1] + rayLength * vector90[0],
                    robotBasePos[2]
                ],
                [
                    robotBasePos[0] + rayLength * vector100[1],
                    robotBasePos[1] + rayLength * vector100[0],
                    robotBasePos[2]
                ],
            ]

        # 调用激光探测函数
        results = p.rayTestBatch(rayFroms, rayTos)

        # 染色前清楚标记
        p.removeAllUserDebugItems()

        # 根据results结果给激光染色
        hit = False
        for index, result in enumerate(results):
            if result[0] == -1:
                # 这里就不显示激光了，要不然运行速度太慢了
                # p.addUserDebugLine(rayFroms[index], rayTos[index], missRayColor)
                hit = False
            else:
                # p.addUserDebugLine(rayFroms[index], rayTos[index], hitRayColor)
                hit = True
                break

        return hit


    def _get_reward(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")

        reward = 0

        # 检测碰撞信息
        P_min, P_max = p.getAABB(self.robot)

        # 得到所有与这个AABB包围盒重叠的模型对象的ID
        # 计算角度与欧式距离获取reward
        # R = ||p - p0||_2 + alpha * ||vd - vp||_2
        #       p:      机器人坐标
        #       p0:     目标的坐标
        #       vd:     机器人的前进方向
        #       vp:     小球的坐标
        robotBasePos, robotBOri = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        sphereBasePos, sphereBOri = p.getBasePositionAndOrientation(self.sphere2red,
                                                                    physicsClientId=self._physics_client_id)
        d2 = np.linalg.norm(np.array(robotBasePos)[:2] - np.array(self.last_b)[:2])
        self.last_b = robotBasePos[:2]
        id_tuple = p.getOverlappingObjects(P_min, P_max)
        # 先获取距离
        distance = np.linalg.norm(np.array(robotBasePos)[:2] - np.array(sphereBasePos)[:2])
        matrix = p.getMatrixFromQuaternion(robotBOri, physicsClientId=self._physics_client_id)
        hit = self._is_in_eyes(matrix, robotBasePos, distance)

        if len(id_tuple) > 1:
            for ID, _ in id_tuple:
                if ID == self.robot:
                    continue

                elif ID == self.sphere2red and hit is True:
                    # print(f"you win!")
                    reward += 5
                    return reward, True

                elif ID == self.sphere2red and hit is False:
                    continue

                else:
                    # 如果箱体内的ID有不是robot_id的，则说明出现了碰撞
                    # print(f"hi happen! hit object is {p.getBodyInfo(ID)}")
                    reward -= 5
                    return reward, True

        # 如果距离大于6米，则结束，同时返回reward
        if distance > 6:
            reward -= 5
            return reward, True

        ty_vec = np.array([matrix[0], matrix[3]])
        sphere_vec = np.array(sphereBasePos)[:2]     # 只拿x与y

        p_L2 = np.linalg.norm(sphere_vec)
        d_L2 = np.linalg.norm(ty_vec)
        dot_product = np.dot(ty_vec / d_L2, sphere_vec / p_L2)
        dot_product = np.clip(dot_product, -1, 1)
        angle = np.arccos(dot_product)

        # 检测是否是钝角
        if dot_product < 0:
            angle += np.pi

        # 角度不是主要考虑的目标，因为可能也可以绕一圈
        R = np.abs(10 - distance) + 0.1*angle
        reward += R
        reward /= 10
        reward = round(reward, 3)
        custum_reward = reward - self.last_r
        self.last_r = reward
        custum_reward *= 10
        custum_reward = round(custum_reward, 3)

        self.hit |= hit

        if self.hit is False:
            return 0, False

        # 见过红球了，但是后面红球不在视线内
        if self.hit is True and hit is False:
            # 太小回一直转圈圈
            return -1e-3, False

        return custum_reward, False

    def _apply_action(self, action):
        assert isinstance(action, list) or isinstance(action, np.ndarray)
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")

        right_front_wheel_joint, right_front_wheel_joint, left_front_wheel_joint, left_back_wheel_joint = action
        # 裁剪防止输入动作超出动作空间
        right_front_wheel_joint = np.clip(right_front_wheel_joint, -100., 100.)
        right_front_wheel_joint = np.clip(right_front_wheel_joint, -100., 100.)
        left_front_wheel_joint = np.clip(left_front_wheel_joint, -100., 100.)
        left_back_wheel_joint = np.clip(left_back_wheel_joint, -100., 100.)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[i for i, _ in self.wheel_link_tuples],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[right_front_wheel_joint, right_front_wheel_joint, left_front_wheel_joint, left_back_wheel_joint],
            forces=[10 for _ in self.wheel_link_tuples]
        )

    def _get_observation(self, width: int = 224, height: int = 224):
        """
        给合成摄像头设置图像并返回robot_id对应的图像
        摄像头的位置为miniBox前头的位置
        """
        basePos, baseOrientation = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=self._physics_client_id)
        tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
        ty_vec = np.array([matrix[1], matrix[4], matrix[7]])  # 变换后的y轴
        tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴

        basePos = np.array(basePos)
        # BASE_RADIUS: 是机器人底盘的半径。BASE_THICKNESS: 是机器人底盘的厚度。
        cameraPos = basePos + self.BASE_RADIUS * tx_vec + 0.8 * self.BASE_THICKNESS * tz_vec
        targetPos = cameraPos + 1 * tx_vec

        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tz_vec,
            physicsClientId=self._physics_client_id
        )
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=50.0,  # 摄像头的视线夹角
            aspect=1.0,
            nearVal=0.01,  # 摄像头焦距下限
            farVal=20,  # 摄像头能看上限
            physicsClientId=self._physics_client_id
        )

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=width, height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            physicsClientId=self._physics_client_id
        )
        # 将元组转换为NumPy数组
        image_array = np.array(rgbImg, dtype=np.uint8)

        # 返回的是RGBA 4个维度
        image_array = image_array.reshape((width, height, -1))

        # 只取 RBG 3个维度
        rgbImg = image_array[:, :, :3]
        return rgbImg

    def debug(self):
        self.step_num += 1
        maxV = 100
        t = 2  # 左前或右前的轮子的速度差的倍数

        p.stepSimulation()
        key_dict = p.getKeyboardEvents()

        if len(key_dict) != 0:
            if p.B3G_UP_ARROW in key_dict and p.B3G_LEFT_ARROW in key_dict:  # 左前
                p.setJointMotorControlArray(  # 2,3为左 6,7为右
                    bodyUniqueId=self.robot,
                    jointIndices=[6, 2],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[-maxV, -maxV],
                    forces=[10, 10],
                )
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot,
                    jointIndices=[7, 3],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[-maxV / t, -maxV / t],
                    forces=[10, 10],
                )

            elif p.B3G_UP_ARROW in key_dict and p.B3G_RIGHT_ARROW in key_dict:  # 右前
                p.setJointMotorControlArray(  # 2,3为左 6,7为右
                    bodyUniqueId=self.robot,
                    jointIndices=[7, 3],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[-maxV, -maxV],
                    forces=[-maxV, -maxV],
                )
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot,
                    jointIndices=[6, 2],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[-maxV / t, -maxV / t],
                    forces=[10, 10],
                )

            elif p.B3G_DOWN_ARROW in key_dict:  # 向后
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot,
                    jointIndices=[2, 3, 6, 7],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[-maxV, -maxV, -maxV, -maxV],
                    forces=[10, 10, 10, 10],
                )

            elif p.B3G_UP_ARROW in key_dict:  # 向前
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot,
                    jointIndices=[2, 3, 6, 7],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[maxV, maxV, maxV, maxV],
                    forces=[10, 10, 10, 10],
                )

            elif p.B3G_LEFT_ARROW in key_dict:
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot,
                    jointIndices=[2, 3, 6, 7],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[-maxV, -maxV, maxV, maxV],
                    forces=[10, 10, 10, 10],
                )

            elif p.B3G_RIGHT_ARROW in key_dict:
                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot,
                    jointIndices=[2, 3, 6, 7],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocities=[maxV, maxV, -maxV, -maxV],
                    forces=[10, 10, 10, 10],
                )

        else:  # 没有按键，则停下
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot,
                jointIndices=[6, 2, 7, 3],
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[0, 0, 0, 0],
                forces=[0, 0, 0, 0]
            )

        done = False
        if self.step_num > 36000:
            done = True

        reward, d = self._get_reward()
        done |= d
        info = {}

        obs = self._get_observation()
        return obs, reward, done, info

    def get_test_info(self, sign):
        basePos, baseOrientation = p.getBasePositionAndOrientation(self.robot, physicsClientId=self._physics_client_id)
        print(f"{sign}: {{{self._physics_client_id}: [{basePos}, {baseOrientation}]}}")


class NormalizedEnv(object):
    def __init__(self, env: Robot, skip=4):
        self.env = env
        self.skip = skip

    def process_frame(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # frame = cv2.resize(frame, (84, 84))
            # cv2show(frame)
            frame = frame[None, :, :] / 255.
            return frame
        else:
            return np.zeros((1, 224, 224))

    def step(self, action):
        total_reward = 0
        states = []
        done = False
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            state = self.process_frame(state)

            if not done:
                total_reward += reward
                states.append(state)
            else:
                for _ in range(self.skip - len(states)):
                    states.append(np.zeros((1, 224, 224)))
                states.append(state)
                total_reward += reward
                break

        states = np.concatenate(states, 0)[None, :, :, :]
        info = {}
        return states.astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.process_frame(state)
        states = []
        for _ in range(self.skip):
            states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), 0, False, {}

    def seed(self, seed=None):
        return self.env.seed(seed)


def cv2show(frame):
    cv2.imshow('Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_env(args, train=True):
    render = (not train) and args.render
    env = Robot(render=render)
    env = NormalizedEnv(env=env, skip=args.skip_frame)
    env.reset()

    return env, env.env.action_space.shape[0], args.skip_frame


#########################################################################################
def NormalizedEnv__TEST():
    import pickle

    env = Robot(render=True)
    env = NormalizedEnv(env=env, skip=4)
    env.reset()
    # p.setRealTimeSimulation(0)
    while True:
        action = np.random.uniform(-100, 100, size=(4,))
        # obs, reward, done, _ = env.step(action)
        # obs, reward, done, _ = env.step()
        # with open('../temp.pickle', 'wb') as f:
        #     pickle.dump(obs, f)
        # break
        obs, reward, done, _ = env.env.debug()
        if reward != 0:
            print(f"reward : {reward}, step: {env.env.step_num}")

        if done:
            break

def Robot__TEST():
    env = Robot(render=True)
    env.reset()
    # p.setRealTimeSimulation(0)
    while True:
        action = np.random.uniform(-100, 100, size=(4,))
        # obs, reward, done, _ = env.step(action)
        # obs, reward, done, _ = env.step()
        obs, reward, done, _ = env.debug()
        print(f"reward : {reward}")

        if done:
            break
#########################################################################################


if __name__ == '__main__':
    NormalizedEnv__TEST()
    # Robot__TEST()

