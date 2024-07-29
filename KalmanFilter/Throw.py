import math
import numpy as np

def throw(launch_pos_x,
          launch_pos_y,
          launch_speed,
          launch_angle,
          dt,
          dropout=None,
          error_range_x=(0, 0),
          error_range_y=(0, 0),
          gravity=9.81):
    """
    Function simulating the trajectory of a ball
    :param launch_pos_x: the horizontal position in meters the ball is launched off
    :param launch_pos_y: the vertical position in meters the ball is launched off
    :param launch_speed: velocity of the ball throw
    :param launch_angle: in which angle the ball is thrown
    :param dt: timesteps between each measurement/ computation
    :param dropout: a tuple with two entries: The begin of the dropout and the end. begin and end included
    :param error_range_x: for the x values, a tuple consisting of two values the mean and the scale of the normal distribution to choose a random error value of
    :param error_range_y: for the y values, a tuple consisting of two values the mean and the scale of the normal distribution to choose a random error value of
    :param gravity: gravitiy in m/s, defaults to 9.81. Dont change it unless you move to the moon
    :return: 3 numpy arrays:
    1) the true values of the throw
    2) the simulated measurements
    3) true values that fits the measurements in case there was a dropout, otherwise it is the same as the first array but only containing the first two columns
    """
    # make sure the dropout is reasonably chosen
    if dropout:
        if dropout[0] >= dropout[1]:
            raise ValueError("invalid dropout values")

    # initialization as well as first values for the balls movement-properties
    x = launch_pos_x # horizontal position
    y = launch_pos_y # vertical position
    vx = launch_speed * math.cos(math.radians(launch_angle)) # horizontal velocity
    vy = launch_speed * math.sin(math.radians(launch_angle)) # vertical velocity

    # initialize two arrays for the values
    true_values = [(x, y, vx, vy)]
    measurements = []

    # compute trajectory
    while y > 0:
        x = x + vx * dt
        y = y + vy * dt - 0.5 * gravity * dt**2
        vy = vy - gravity * dt
        true_values.append((x, y, vx, vy))

    # simulate measurement error as well as dropout
    for t, value in enumerate(true_values):
        if dropout and dropout[0] <= t <= dropout[1]:
            pass
        else:
            error_x = np.random.normal(error_range_x[0], error_range_x[1])
            x_e = value[0] + error_x
            error_y = np.random.normal(error_range_y[0], error_range_y[1])
            y_e = value[1] + error_y

            measurements.append((x_e, y_e))

    # convert to np arrays, initialize aligned true values for error measurement afterwards
    true_values = np.asarray(true_values)
    measurements = np.asarray(measurements)
    aligned_true_values = true_values[:,:2]

    # delete dropout from the aligned values
    if dropout:
        aligned_true_values = np.delete(aligned_true_values, [i for i in range(dropout[0], dropout[1] + 1)], axis=0)

    return true_values, measurements, aligned_true_values
