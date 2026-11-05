# Grid navigation for the tomato robot
# A* path planning + L298N motors + 4 ultrasonic sensors

import heapq
import signal
import sys
import time

import RPi.GPIO as GPIO

# L298N motor driver
IN1, IN2, IN3, IN4 = 24, 23, 25, 22
ENA, ENB = 12, 13

# ultrasonic sensors: front, left, right, back
TRIG_PINS = [4, 17, 5, 19]
ECHO_PINS = [18, 27, 6, 16]
SENSOR_NAMES = ["front", "left", "right", "back"]

PWM_FREQ = 1000
OBSTACLE_CM = 20
FORWARD_SPEED = 40
TURN_SPEED = 75

pwm_a = None
pwm_b = None


def setup():
    global pwm_a, pwm_b
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
        GPIO.setup(pin, GPIO.OUT)

    pwm_a = GPIO.PWM(ENA, PWM_FREQ)
    pwm_b = GPIO.PWM(ENB, PWM_FREQ)
    pwm_a.start(0)
    pwm_b.start(0)

    for trig in TRIG_PINS:
        GPIO.setup(trig, GPIO.OUT)
        GPIO.output(trig, GPIO.LOW)
    for echo in ECHO_PINS:
        GPIO.setup(echo, GPIO.IN)
    time.sleep(0.5)


def set_motors(in1, in2, in3, in4, speed):
    GPIO.output(IN1, in1)
    GPIO.output(IN2, in2)
    GPIO.output(IN3, in3)
    GPIO.output(IN4, in4)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)


def stop():
    set_motors(0, 0, 0, 0, 0)


def forward(speed=FORWARD_SPEED, duration=0.5):
    set_motors(1, 0, 1, 0, speed)
    time.sleep(duration)
    stop()


def turn_right(speed=TURN_SPEED, duration=0.8):
    set_motors(0, 1, 1, 0, speed)
    time.sleep(duration)
    stop()


def read_distance(trig, echo, timeout=0.03):
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig, GPIO.LOW)

    start = time.time()
    t0 = start
    while GPIO.input(echo) == 0:
        start = time.time()
        if start - t0 > timeout:
            return 999  # no echo, treat as far away
    stop_t = start
    while GPIO.input(echo) == 1:
        stop_t = time.time()
        if stop_t - start > timeout:
            return 999

    return (stop_t - start) * 34300 / 2


def read_sensors(samples=5):
    result = {}
    for name, trig, echo in zip(SENSOR_NAMES, TRIG_PINS, ECHO_PINS):
        values = []
        for _ in range(samples):
            values.append(read_distance(trig, echo))
            time.sleep(0.02)
        values.sort()
        result[name] = values[len(values) // 2]  # median is less noisy than average
    return result


# ---------- A* on a simple grid (0 = free, 1 = blocked) ----------

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_list = [(heuristic(start, goal), start)]
    came_from = {}
    g = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        x, y = current
        for nxt in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            nx, ny = nxt
            if not (0 <= nx < rows and 0 <= ny < cols) or grid[nx][ny] == 1:
                continue
            new_g = g[current] + 1
            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                came_from[nxt] = current
                heapq.heappush(open_list, (new_g + heuristic(nxt, goal), nxt))
    return None


def follow_path(path):
    # one grid cell = one forward step (calibrated on our test floor)
    for cell in path[1:]:
        while read_sensors()["front"] < OBSTACLE_CM:
            print("obstacle ahead, turning")
            turn_right()
        print("moving to", cell)
        forward()
    print("goal reached")


def shutdown(*_):
    stop()
    GPIO.cleanup()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    setup()

    grid = [[0] * 13 for _ in range(12)]
    start, goal = (10, 1), (1, 11)

    path = astar(grid, start, goal)
    if path:
        print("path:", path)
        follow_path(path)
    else:
        print("no path found")
    shutdown()
