"""
@author Bruce Hou
@brief 
@version 1.0
@date 2022-01-14

@copyright Copyright (c) 2022
"""
import os
import pickle
import traceback
import numpy as np
from collections import deque


class Buffer:
    """
    @brief 经验池
    @details 包含各个数据的储存，保存，获取，采样等
    """

    def __init__(
        self,
        capacity=10000,
        batch_size=16,
        nstep=10,
        width = 128, 
        height = 128,
        action_dim=5,
        file_path="Datasets/",
    ):
        self.capacity = capacity  # << 单个正激励储存池的容量
        self.batch_size = batch_size
        self.nstep = nstep  # << 环境运行的步数
        self.width = width
        self.height = height
        self.action_dim = action_dim

        self.file_path = file_path

        self.temp_buffer = {
            "img": deque(maxlen=nstep),
            "a": deque(maxlen=nstep),
            "r": deque(maxlen=nstep),
            "img_": deque(maxlen=nstep),
            "d": deque(maxlen=nstep),
        }
        # 经验池
        try:
            if os.path.exists(self.file_path + "buffer.pkl"):
                with open(self.file_path + "buffer.pkl", "rb") as f:
                    self.buffer = pickle.load(f)
            else:
                self.buffer = {
                    "img": deque(maxlen=capacity),
                    "a": deque(maxlen=capacity),
                    "r": deque(maxlen=capacity),
                    "img_": deque(maxlen=capacity),
                    "d": deque(maxlen=capacity),
                }
        except Exception as e:
            traceback.print_exc()
            print("文件保存时出错，重新创建经验池")
            self.buffer = {
                "img": deque(maxlen=capacity),
                "a": deque(maxlen=capacity),
                "r": deque(maxlen=capacity),
                "img_": deque(maxlen=capacity),
                "d": deque(maxlen=capacity),
            }

        # 保存珍贵的成功轨迹
        if os.path.exists(self.file_path + "p_file/info.pkl"):
            with open(self.file_path + "p_file/info.pkl", "rb") as info:
                self.p_file_id = pickle.load(info)
            p_file = open(
                self.file_path + "p_file/p_file_" + str(self.p_file_id) + ".pkl", "rb"
            )
            self.p_buffer = pickle.load(p_file)
        else:
            self.p_file_id = 0
            self.p_buffer = {
                "img": deque(maxlen=1000),
                "a": deque(maxlen=1000),
                "r": deque(maxlen=1000),
                "img_": deque(maxlen=1000),
                "d": deque(maxlen=1000),
            }

    def store(self, state, action, reward, next_states, done):
        buffer_ready, p_data = self.preProcessData(
            state, action, reward, next_states, done
        )
        # 正激励数据储存
        if buffer_ready:
            if p_data:
                self.store2PositivePool()

        buffer_size = len(self.buffer["r"])
        temp_buffer_size = len(self.temp_buffer["r"])
        if buffer_size + temp_buffer_size <= self.capacity:
            # 添加数据
            for key in self.buffer:
                self.buffer[key].extend(self.temp_buffer[key])
            # 清空temp_buffer
            for key in self.temp_buffer:
                self.temp_buffer[key].clear()
        else:
            # 剔除元素
            for _ in range(temp_buffer_size):
                for key in self.buffer:
                    self.buffer[key].popleft()
            # 储存数据
            for key in self.buffer:
                self.buffer[key].extend(self.temp_buffer[key])
            # 清空temp_buffer
            for key in self.temp_buffer:
                self.temp_buffer[key].clear()

        return buffer_ready

    def store2PositivePool(self):
        """
        @brief 储存数据到正激励池
        """
        temp_buffer_size = len(self.buffer["r"])
        p_size = len(self.p_buffer["r"])
        if temp_buffer_size + p_size <= self.capacity:
            for key in self.p_buffer:
                self.p_buffer[key].extend(self.temp_buffer[key])
        else:
            # 先储存为文件
            p_file = open(
                self.file_path + "p_file/p_file" + str(self.p_file_id) + ".pkl", "wb"
            )
            pickle.dump(self.p_buffer, p_file)
            p_file.close()
            self.p_file_id += 1
            # 在建立一个空的新文件
            temp_p_buffer = {
                "img": deque(maxlen=1000),
                "a": deque(maxlen=1000),
                "r": deque(maxlen=1000),
                "img_": deque(maxlen=1000),
                "d": deque(maxlen=1000),
            }
            new_file = open(
                self.file_path + "p_file/p_file_" + str(self.p_file_id) + ".pkl", "wb"
            )
            pickle.dump(temp_p_buffer, new_file)
            new_file.close()
            # 再清空内存数据
            for key in self.p_buffer:
                self.p_buffer[key].clear()
            # 储存数据
            for key in self.p_buffer:
                self.p_buffer[key].extend(self.temp_buffer[key])

    def preProcessData(self, obs, action, reward, next_obs, done):
        """
        @brief 对数据进行预处理，如果到最后一步没有抓取成功，则之前的所有动作的奖励都设为-0.01
        @param 
        @return
        """
        # 先存储数据
        self.temp_buffer["img"].append(obs)
        self.temp_buffer["a"].append(action)
        self.temp_buffer["r"].append(reward)
        self.temp_buffer["img_"].append(next_obs)
        if done:
            self.temp_buffer["d"].append(1)
        else:
            self.temp_buffer["d"].append(0)
        # 更新奖励，储存池是否已经完成收集
        if done:
            # 已经是最后一个动作了
            buffer_ready = True
            # 判断奖励是否为1
            if reward != 1:
                p_data = False
                # 最后一步奖励为0,则之前所有奖励设置为-0.01
                for i in range(len(self.temp_buffer["r"])):
                    self.temp_buffer["r"][i] -= 0.01
            else:
                p_data = True
        else:
            buffer_ready = False
            p_data = False
            # buffer还没有采集完程数据
        return buffer_ready, p_data

    def save(self):
        # 先保存珍贵的成功数据
        p_file = open(
            self.file_path + "p_file/p_file_" + str(self.p_file_id) + ".pkl", "wb"
        )
        pickle.dump(self.p_buffer, p_file)
        p_file.close()
        # 保存成功数据文件id信息
        p_file_id = open(self.file_path + "p_file/info.pkl", "wb")
        pickle.dump(self.p_file_id, p_file_id)
        p_file_id.close()
        # 保存经验池
        buffer_file = open(self.file_path + "buffer.pkl", "wb")
        pickle.dump(self.buffer, buffer_file)
        buffer_file.close()

    def sample(self,):
        """
        @brief 均匀采样
        """
        # 收集采样数据空间
        samples_img = np.empty((self.batch_size, 1, self.width, self.height), dtype=np.float32)
        samples_a = np.empty((self.batch_size, self.action_dim), dtype=np.float32)
        samples_r = np.empty((self.batch_size), dtype=np.float32)
        samples_img_ = np.empty((self.batch_size, 1, self.width, self.height), dtype=np.float32)
        samples_d = np.empty((self.batch_size), dtype=np.uint8)

        buffer_size = len(self.buffer["r"])
        # 均匀采样
        indices = np.random.choice(
            buffer_size, self.batch_size, replace=False
        )  # << sp选取的索引
        for item in enumerate(indices):
            samples_img[item[0], :, :, :] = self.buffer["img"][item[1]]
            samples_a[item[0], :] = self.buffer["a"][item[1]]
            samples_r[item[0]] = self.buffer["r"][item[1]]
            samples_img_[item[0], :, :, :] = self.buffer["img_"][item[1]]
            samples_d[item[0]] = self.buffer["d"][item[1]]

        return (
            samples_img,
            samples_a,
            samples_r,
            samples_img_,
            samples_d,
        )

    @property
    def size(self):
        return len(self.buffer["s"])
