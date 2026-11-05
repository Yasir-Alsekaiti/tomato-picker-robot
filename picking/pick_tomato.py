# Detect a red (ripe) tomato with the camera and pick it with the arm.
# Simple version using HSV color + servo angles mapped from the pixel position.

import time

import cv2
import numpy as np
import RPi.GPIO as GPIO

SERVO_BASE = 18
SERVO_VERTICAL = 26
SERVO_GRIPPER = 21

MIN_AREA = 500


def setup_servos():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    servos = {}
    for name, pin in [("base", SERVO_BASE), ("vertical", SERVO_VERTICAL), ("gripper", SERVO_GRIPPER)]:
        GPIO.setup(pin, GPIO.OUT)
        pwm = GPIO.PWM(pin, 50)
        pwm.start(0)
        servos[name] = pwm
    return servos


def set_angle(servo, angle):
    servo.ChangeDutyCycle(2.5 + angle / 18.0)
    time.sleep(0.3)
    servo.ChangeDutyCycle(0)  # stop jitter


def find_ripe_tomato(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # red wraps around in HSV so we need two ranges
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA:
        return None

    m = cv2.moments(biggest)
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def clamp(v, lo=45, hi=135):
    return max(lo, min(hi, v))


def move_arm_to(servos, cx, cy, w, h):
    base = clamp(90 + (cx - w // 2) * 0.1)      # factor tuned by hand for our arm
    vertical = clamp(90 + (cy - h // 2) * 0.1)
    print(f"base {base:.0f}, vertical {vertical:.0f}")
    set_angle(servos["base"], base)
    set_angle(servos["vertical"], vertical)


def pick(servos):
    set_angle(servos["gripper"], 0)    # open
    time.sleep(0.5)
    set_angle(servos["gripper"], 90)   # close
    time.sleep(0.5)


def home(servos):
    set_angle(servos["base"], 90)
    set_angle(servos["vertical"], 90)
    set_angle(servos["gripper"], 0)


def main():
    servos = setup_servos()
    cap = cv2.VideoCapture(0)
    home(servos)
    print("looking for ripe tomatoes... (ctrl+c to stop)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera error")
                break

            target = find_ripe_tomato(frame)
            if target is None:
                time.sleep(0.2)
                continue

            h, w = frame.shape[:2]
            print("ripe tomato at", target)
            move_arm_to(servos, *target, w, h)
            pick(servos)
            home(servos)
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
