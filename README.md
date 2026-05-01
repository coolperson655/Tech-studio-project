awe yeah we love documentation


# Python Environment
uv for python environment my goat (https://docs.astral.sh/uv/getting-started/installation/)[(here's the install)]

# Demos
## Multi_servo_control.py
`Multi_servo_control.py` contains code to set the positions of the servo.

| :warning: WARNING           |
|:----------------------------|
| The servos will spin all over when you first start the program (This seems to come from firmata).\ndisconnect the servos from knobs on start up or things may crash. |

`set_intensity()` takes a value from 0-100 as the amount to turn the knob, and it also takes the pin that you want to send the command to (so like which servo, you get it). The intensity is mapped to the actual servo degrees turned using the `SERVO_RANGE` values. these are just defined in the code so do whatever ya want

