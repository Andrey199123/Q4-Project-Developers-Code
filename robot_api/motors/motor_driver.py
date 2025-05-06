import time
from enum import Enum

from robot_api.motors.PCA9685 import PCA9685


# Motor driver class
class MotorDriver:
    # Define constants for PWM & motors
    __PWMA = 0
    __AIN1 = 1
    __AIN2 = 2
    __PWMB = 5
    __BIN1 = 3
    __BIN2 = 4

    # Direction enum
    class Direction(Enum):
        FORWARD = 0
        REVERSE = 1
        OFF = 2

    def __init__(self):
        # Create instance of PCA9685 library
        self.__pwm = PCA9685(0x40, debug=False)
        self.__pwm.setPWMFreq(100)

        # Define variables used to determine whether or not a function
        # should run as well as if it should be stopped mid-run
        self.__command_running = False
        self.__command_interrupt = False

    def stop(self):
        # Kill all power to motors
        self.__pwm.setDutycycle(self.__PWMA, 0)
        self.__pwm.setDutycycle(self.__PWMB, 0)

        # Interrupt current command
        if self.__command_running:
            self.__command_interrupt = True

    def __drive_motors(self, duration, power, a, b):
        # Don't run if there is already a command running
        if self.__command_running:
            return

        self.__command_running = True

        # If a isn't set to be disabled, set its direction based on the input
        # argument provided
        if a != self.Direction.OFF:
            self.__pwm.setLevel(
                self.__AIN1, 1 if a == self.Direction.FORWARD else 0
            )
            self.__pwm.setLevel(
                self.__AIN2, 0 if a == self.Direction.FORWARD else 1
            )

        # If b isn't set to be disabled, set its direction based on the input
        # argument provided
        if b != self.Direction.OFF:
            self.__pwm.setLevel(
                self.__BIN1, 0 if b == self.Direction.FORWARD else 1
            )
            self.__pwm.setLevel(
                self.__BIN2, 1 if b == self.Direction.FORWARD else 0
            )

        # Enable motors and set power to what is provided
        self.__pwm.setDutycycle(self.__PWMA, power - 5)
        self.__pwm.setDutycycle(self.__PWMB, power)

        end_time = time.time() + duration

        # Wait until the duration has passed
        while time.time() < end_time and not self.__command_interrupt:
            pass

        # Stop the motors
        self.stop()

        # Reset state to allow another command to run
        self.__command_running = False
        self.__command_interrupt = False

    # Wrappers over __drive_motors for each direction
    # hard-coded to 1/4 of a second at 50% power
    def forward(self):
        self.__drive_motors(
            500, 40, self.Direction.FORWARD, self.Direction.FORWARD
        )

    def left(self):
        self.__drive_motors(
            500, 40, self.Direction.REVERSE, self.Direction.FORWARD
        )

    def right(self):
        self.__drive_motors(
            500, 40, self.Direction.FORWARD, self.Direction.REVERSE
        )

    def reverse(self):
        self.__drive_motors(
            500, 40, self.Direction.REVERSE, self.Direction.REVERSE
        )
