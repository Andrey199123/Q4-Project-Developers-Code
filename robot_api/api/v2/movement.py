import signal
import sys
import threading

from flask import Blueprint, jsonify

from robot_api.motors.motor_driver import MotorDriver

motor_driver = MotorDriver()


def signal_handler(*_):
    motor_driver.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Create blueprint that will be registered to the top-level app
movement = Blueprint("movement", __name__)


# Start Route
def start():
    return threading.Thread(target=motor_driver.forward).start()


@movement.route("/start", methods=["POST"])
def _start():
    start()
    return jsonify()


# Forward route
def forward():
    return threading.Thread(target=motor_driver.forward).start()


@movement.route("/forward", methods=["POST"])
def _forward():
    forward()
    return jsonify()


def backward():
    return threading.Thread(target=motor_driver.reverse).start()


# Sends a move backwards command and then returns an HTTP 200 (OK) status code
@movement.route("/backwards", methods=["POST"])
def _backward():
    backward()
    return jsonify()


def left():
    return threading.Thread(target=motor_driver.left).start()


# Sends a turn left command and then returns an HTTP 200 (OK) status code
@movement.route("/left", methods=["POST"])
def _left():
    left()
    return jsonify()


def right():
    return threading.Thread(target=motor_driver.right).start()


# Sends a turn right command and then returns an HTTP 200 (OK) status code
@movement.route("/right", methods=["POST"])
def _right():
    right()
    return jsonify()


# Sends a stop command (this will interrupt whatever command is currently running) and then returns an HTTP 200 (OK) status code
def stop():
    return threading.Thread(target=motor_driver.stop).start()


@movement.route("/stop", methods=["POST"])
def _stop():
    stop()
    return jsonify()


# Import the top-level app
from robot_api import app

# Register blueprint to top-level app
app.register_blueprint(movement, url_prefix="/api/v2/move")

print("Successfully registered the API V2 Movement blueprint")
