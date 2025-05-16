from gpiozero import DistanceSensor

import time

ultrasonic = DistanceSensor(trigger=17, echo=27)



while True:

    if (ultrasonic.wait_for_in_range):

        print('Distance: ', ultrasonic.distance * 100)

        time.sleep(1)

