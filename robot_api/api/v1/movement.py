import inspect
import signal
import sys
import threading
import time

from flask import Blueprint, jsonify

# from robot_api.PCA9685 import PCA968
from robot_api.motors.motor_driver import MotorDriver

motor_driver = MotorDriver()


def signal_handler(sig, frame):
    motor_driver.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Create blueprint that will be registered to the top-level app
movement = Blueprint("movement", __name__)


# This route will miove the robot forwards, and returns a JSON that contains whether or not the request was successful,
# and the name of the method that was called.
@movement.route("/forward", methods=["POST"])
def forward():
    response = {}
    response["success"] = True
    response["methodName"] = inspect.currentframe().f_code.co_name

    # threading.Thread(target=motor_forward, args=(3,)).start()
    # motor_forward(0.25)
    threading.Thread(target=motor_driver.forward).start()

    return jsonify(response)


# This route will move the robot backwards (in reverse), and returns a JSON that contains whether or not the request was successful,
# and the name of the method that was called.
@movement.route("/backward", methods=["POST"])
def backward():
    response = {}
    response["success"] = True
    response["methodName"] = inspect.currentframe().f_code.co_name

    # threading.Thread(target=motor_backward, args=(3,)).start()
    # motor_backward(0.25)
    threading.Thread(target=motor_driver.reverse).start()

    return jsonify(response)


# This route will turn the robot to the left, and returns a JSON that contains whether or not the request was successful,
# and the name of the method that was called.
@movement.route("/left", methods=["POST"])
def left():
    response = {}
    response["success"] = True
    response["methodName"] = inspect.currentframe().f_code.co_name

    # threading.Thread(target=motor_left, args=(3,)).start()
    # motor_left(0.25)
    threading.Thread(target=motor_driver.left).start()

    return jsonify(response)


# This route will turn the robot to the right, and returns a JSON that contains whether or not the request was successful,
# and the name of the method that was called.
@movement.route("/right", methods=["POST"])
def right():
    response = {}
    response["success"] = True
    response["methodName"] = inspect.currentframe().f_code.co_name

    # threading.Thread(target=motor_right, args=(3,)).start()
    # motor_right(0.25)
    threading.Thread(target=motor_driver.right).start()

    return jsonify(response)


# This route will stop the robot, and returns a JSON that contains whether or not the request was successful,
# and the name of the method that was called.
@movement.route("/stop", methods=["POST"])
def stop():
    response = {}
    response["success"] = True
    response["methodName"] = inspect.currentframe().f_code.co_name

    # motor_stop()
    threading.Thread(target=motor_driver.stop).start()

    return jsonify(response)


# Bring in top-level app
from robot_api import app

# Register blueprint to top-level app
app.register_blueprint(movement, url_prefix="/api/v1/move")

print("Successfully registered movement blueprint")
