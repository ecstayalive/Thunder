from .j2n6s200 import J2n6s200
from .j2n6s300 import J2n6s300
from .j2s7s200 import J2s7s200
from .j2s7s300 import J2s7s300


class Jaco2:
    """Jaco2 robotic arm control system interface"""

    def __init__(
        self,
        jaco_model="j2n6s200",
        base_position=[0, 0, 0],
        jaco_end_effector_position=[0, 0, 0],
        gripper_original_angle=0,
        use_quaternion=True,
        time_step=1.0 / 240,
    ):
        """Initialization

        Args:
            jaco_model: The model of the jaco robotic arm
            base_position: the initial position of the jaco robotic arm
            jaco_end_effector_position: jaco's end effector position
            gripperAngle: the gripper's close angle
            time_step: one step time for simulation
            action_apply_time: the total time for applying action

        RaiseError:
            KeyError: when program can not find the input robotic model

        """
        if jaco_model == "j2n6s200":
            self.robotic_arm = J2n6s200(
                base_position,
                jaco_end_effector_position,
                gripper_original_angle,
                use_quaternion,
                time_step,
            )
        elif jaco_model == "j2n6s300":
            self.robotic_arm = J2n6s300(
                base_position,
                jaco_end_effector_position,
                gripper_original_angle,
                use_quaternion,
                time_step,
            )
        elif jaco_model == "j2s7s200":
            self.robotic_arm = J2s7s200(
                base_position,
                jaco_end_effector_position,
                gripper_original_angle,
                use_quaternion,
                time_step,
            )
        elif jaco_model == "j2s7s300":
            self.robotic_arm = J2s7s300(
                base_position,
                jaco_end_effector_position,
                gripper_original_angle,
                use_quaternion,
                time_step,
            )
        else:
            raise KeyError(f"There is no jaco's model named{jaco_model}")

    def reset(self):
        self.robotic_arm.reset()

    def get_end_effector_state(self):
        """Get end effector height

        Returns: [pos, euler_angle]

        """
        return self.robotic_arm.get_end_effector_state()

    def apply_action(self, action):
        """The robotic arm performs action"""
        self.robotic_arm.apply_action(action)

    def set_original_position(self):
        """Set the original position of Jaco arm"""
        self.robotic_arm.set_original_position()
