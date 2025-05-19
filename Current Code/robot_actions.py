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
        time.sleep(2) 
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
        print("Executing left turn sequence.")
        robot_api.api.v2.movement.forward()
        time.sleep(turn_params.get('left_fwd_time', 1.0)) 
        robot_api.api.v2.movement.stop()
        robot_api.api.v2.movement.left()
        time.sleep(turn_params.get('left_turn_time', 1.0)) 
        robot_api.api.v2.movement.stop()
        robot_api.api.v2.movement.forward()
        print("Left turn sequence complete.")

    elif direction == "Right" and not action_flags.get('turned_right', False):
        print("Executing right turn sequence.")
        robot_api.api.v2.movement.forward()
        time.sleep(turn_params.get('right_fwd_time', 1.0))
        robot_api.api.v2.movement.stop()
        robot_api.api.v2.movement.right()
        time.sleep(turn_params.get('right_turn_time', 1.5))
        robot_api.api.v2.movement.stop()
        robot_api.api.v2.movement.forward() 
        time.sleep(1.5)
        robot_api.api.v2.movement.stop()
        print("Right turn sequence complete.")
    
    elif direction == "Middle":
         print("Path is Middle. Moving forward.")
         robot_api.api.v2.movement.forward()
         time.sleep(1)
         robot_api.api.v2.movement.stop()
    else:
        robot_api.api.v2.movement.forward() 
        pass

    action_flags['CURRENTLY_RUNNING_PROCEDURE'] = False


def execute_intersection_turn(action_flags, turn_params, default_turn_direction="Right"):
    """
    Executes a turn at an intersection (e.g., T-junction).
    The original code had commented out sections for turn_right/turn_left at horizontal line.
    """
    if action_flags.get('CURRENTLY_RUNNING_PROCEDURE', False):
        return

    action_flags['CURRENTLY_RUNNING_PROCEDURE'] = True
    print(f"Intersection detected. Executing {default_turn_direction} turn sequence.")
    robot_api.api.v2.movement.stop()
    robot_api.api.v2.movement.forward()
    time.sleep(turn_params.get('intersection_fwd_time', 2.0)) # Move into intersection
    robot_api.api.v2.movement.stop()

    if default_turn_direction == "Right":
        robot_api.api.v2.movement.right()
        time.sleep(turn_params.get('intersection_turn_time', 1.5)) # Turn duration
    elif default_turn_direction == "Left":
        robot_api.api.v2.movement.left()
        time.sleep(turn_params.get('intersection_turn_time', 1.5))
    
    robot_api.api.v2.movement.stop()
    robot_api.api.v2.movement.forward() # Move out of intersection
    time.sleep(turn_params.get('intersection_exit_fwd_time', 1.0))
    robot_api.api.v2.movement.stop()
    
    action_flags['CURRENTLY_RUNNING_PROCEDURE'] = False
