import time
import threading
import robot_api.api.v2.movement


def execute_obstacle_avoidance(action_flags):
    """
    Executes the obstacle avoidance maneuver (backup).
    Modifies action_flags['CURRENTLY_RUNNING_PROCEDURE'] and action_flags['object_detected'].
    """
    if not action_flags.get('CURRENTLY_RUNNING_PROCEDURE', False):
        action_flags['CURRENTLY_RUNNING_PROCEDURE'] = True
        print("Obstacle detected! Backing up.")
        robot_api.api.v2.movement.stop()
        robot_api.api.v2.movement.backward()
        time.sleep(5) 
        robot_api.api.v2.movement.stop()
        action_flags['CURRENTLY_RUNNING_PROCEDURE'] = False
        action_flags['object_detected'] = False # Reset detection flag
        print("Obstacle avoidance maneuver complete.")

def execute_turn(direction, action_flags, turn_params):
    """
    Executes a turn based on the given direction.
    Modifies action_flags like 'CURRENTLY_RUNNING_PROCEDURE', 'turned_left', 'turned_right'.
    turn_params is a dict like {'left_fwd_time': 6.5, 'left_turn_time': 1.4, ...}
    """
    action_flags['CURRENTLY_RUNNING_PROCEDURE'] = True
    robot_api.api.v2.movement.stop()

    if direction == "Left" and not action_flags.get('turned_left', False):
        print("Left Turn")
        robot_api.api.v2.movement.forward()
        time.sleep(6)
        robot_api.api.v2.movement.stop()
        robot_api.api.v2.movement.left()
        time.sleep(1.7) 
        robot_api.api.v2.movement.stop()
        print("Left turn sequence complete.")

    elif direction == "Right" and not action_flags.get('turned_right', False):
        print("Right Turn")
        robot_api.api.v2.movement.forward()
        time.sleep(6)
        robot_api.api.v2.movement.stop()
        robot_api.api.v2.movement.right()
        time.sleep(1.9)
        robot_api.api.v2.movement.stop()
        print("Right turn complete")
    
    elif direction == "Middle":
         robot_api.api.v2.movement.forward()
         time.sleep(1)
         robot_api.api.v2.movement.stop()
    else:
        robot_api.api.v2.movement.forward()
        pass

    action_flags['CURRENTLY_RUNNING_PROCEDURE'] = False
