awe yeah we love documentation


# Python Environment
uv for python environment my goat [https://docs.astral.sh/uv/getting-started/installation/](here's the install)]

# Overview
This project uses an off the shelf electroacupucture machine contoled by servos to give feedback to a simulated ball game running on pygame. The hand data is from a quest 3 running hand tracking streamer which is converted to finger curl values for the simulated ball. The TENS electrodes require some trail and error to get in the correct spot to pull back on the fingers, but once in place will give feedback proportional to the simulated ball's "squeeze level".

# Demos
## Multi_servo_control.py
`Multi_servo_control.py` contains code to set the positions of the servo.
you will need to add a `.env` file to your project which defines your com port
ex.
```env
COM_PORT=COM5
```

| :warning: WARNING           |
|:----------------------------|
| The servos will spin all over when you first start the program (This seems to come from firmata). **Disconnect the servos from knobs on start up or things may break.** |

`set_intensity()` takes a value from 0-100 as the amount to turn the knob, and it also takes the pin that you want to send the command to (so like which servo, you get it). The intensity is mapped to the actual servo degrees turned using the `SERVO_RANGE` values. these are just defined in the code so do whatever ya want

