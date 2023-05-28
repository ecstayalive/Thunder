# Thunder

[中文](./README-ZH.md)

**Thunder** is a deep reinforcement learning package (platform), which still has a lot of shortcomings. The project will gradually improve with my learning.

And more important, this project welcome everyone's ideas, improvements and contributions.

## Graduation Project

My graduation project, a vision-based robotic grasp task by using the deep reinforcement learning, also uses this platform. Due to time constraints and hardware limits, it did not achieve a good result in visual servo based grasp task without any prior knowledge.

### The DRL Algorithm
The graduation project use the SAC(soft actor-critic) algorithm which is implemented in **Thunder** to train a cnn model. Actually, because of the limitation of hardware, the default architecture of the cnn model has small parameters.

### non-vision servo based grasping task

After training for 300,000 steps, the model achieved a grasping rate of 64% on non-visual servo grasping tasks using the jaco robotic arm.The following is the change curve of its crawl rate.
![jaco_non_vision_servo_grasping_rate](./docs/pictures/jaco_non_vision_servo_grasping_rate.png)

But the model achieved a grasping rate of 77% only after training for 160,000 steps if using the Kuka robotic arm. The following is the change curve of its crawl rate.
![kuka_non_vision_servo_grasping_rate](./docs/pictures/kuka_non_vision_servo_grasping_rate.png)

### vision servo based grasping task

For this small model, the complete vision servo grasp problem seems too hard and the model fails to converge. But after reducing the amount of control to learn, the model can still learn something. The following is the change curve of its crawl rate on vision servo grasping task.
![jaco_half_vision_servo_grasping_rate](./docs/pictures/jaco_half_vision_servo_grasping_rate.png)

### Demo

#### non-vision servo based grasping task
##### jaco
![jaco_non_vision_servo_grasp](./docs/pictures/jaco_non_vision_servo_grasp.gif)

##### kuka
![kuka_non_vision_servo_grasp](./docs/pictures/kuka_non_vision_servo_grasp.gif)

#### vision servo based grasping task
![jaco_vision_servo_grasp](./docs/pictures/jaco_half_vision_servo_grasp.gif)
