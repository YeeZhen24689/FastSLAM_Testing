import pygame
import pygame.gfxdraw
from python_ugv_sim.utils import vehicles, environment
import math
import numpy as np
from copy import deepcopy

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)])**2
R = np.diag([1.0, np.deg2rad(20.0)])**2

#  Simulation parameter
Qsim = np.diag([0.3, np.deg2rad(2.0)])**2
Rsim = np.diag([0.5, np.deg2rad(10.0)])**2
OFFSET_YAWRATE_NOISE = 0.01

# ~ Simulation Set Up ~
landmark_radius = 0.2
robot_fov = 2

DT = 0.01  # time tick [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 3.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM srate size [x,y]
N_PARTICLE = 50  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

STARTING_X = 1
STARTING_Y = 10.5
STARTING_YAW = 0

# ~ Declare our Particle Agent ~
class Particle:

    def __init__(self, N_LM):
        self.w = 1.0 / N_PARTICLE # Particle Weight
        self.x = STARTING_X # Particle Location X
        self.y = STARTING_Y # Particle Location Y
        self.yaw = STARTING_YAW # Particle Yaw
        # landmark x-y positions
        self.lm = np.zeros((N_LM, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((N_LM * LM_SIZE, LM_SIZE))

# ~ Landmark Parameters ~
#landmarks = [(4,2),(3,9),(4,9),(8,8),(12,10),(16,8),(20,2),(12,3),(8,5)]

# Test Track A
#landmarks = [(0,-1.5),(0.5,-1.5),(3,-1.5),(6,-1.5),(9,-1.5),(0,1.5),(0.5,1.5),(3,1.5),(6,1.5),(9,1.5)]

# Test Track B
sf = 0.9
landmarks = [(0*sf,9),(0.5*sf,9),(3*sf,9),(6*sf,9),(9*sf,9),(12*sf,9),(15*sf,12),(15*sf,8.5),(0*sf,12),(0.5*sf,12),
             (3*sf,12),(6*sf,12),(9*sf,12),(12*sf,12),(15*sf,12),(17.5*sf,11.5),(19.5*sf,10.5),(20.5*sf,9),(21*sf,7.5),(21*sf,6.5),
             (20.5*sf,5),(19.5*sf,3.5)]

n_landmarks = len(landmarks) # Number of landmarks in space

# ~ Setting Up the sensor and odometry for the vehicle ~

# ~ Measurement Simulation Function
def sim_measurements(robot_pose,landmarks):
    rx, ry, rtheta = robot_pose[0], robot_pose[1], robot_pose[2]
    zs=np.zeros((3,0))
    for (lidx,landmark) in enumerate(landmarks): # Iterate over landamrks and inveces
        lx,ly = landmark
        dist = np.linalg.norm(np.array([lx-rx,ly-ry])) # Distance between robot and landmark
        phi = np.arctan2(ly-ry,lx-rx) - rtheta # Angle between robot heading and landmark
        phi = np.arctan2(np.sin(phi),np.cos(phi)) # Normalize the angle between pi and -pi
        if dist < robot_fov: # Only append if observation is in the robot's field of view
            zs_add = np.array([dist,phi,lidx]).reshape(3,1)
            zs = np.hstack((zs,zs_add))
    return zs

def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])
    x = F @ x + B @ u

    x[2, 0] = pi_2_pi(x[2, 0])
    return x

# <--- Compute Weight --->
# Compute the weight of the particles
def compute_weight(particle, z, Q):
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)
    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = pi_2_pi(dx[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print("singuler")
        return 1.0

    num = math.exp(-0.5 * dx.T @ invS @ dx)
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))
    w = num / den

    return w

# Compute the jacobian of the movement (For the weight)
def compute_jacobians(particle, xf, Pf, Q):
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    d2 = dx**2 + dy**2
    d = math.sqrt(d2)

    zp = np.array(
        [d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

    Hv = np.array([[-dx / d, -dy / d, 0.0],
                   [dy / d2, -dx / d2, -1.0]])

    Hf = np.array([[dx / d, dy / d],
                   [-dy / d2, dx / d2]])

    Sf = Hf @ Pf @ Hf.T + Q

    return zp, Hv, Hf, Sf
# <--- END Compute Weight --->

# Add the landmark to the particle association (Local Particle Frame)
def add_new_lm(particle, z, Q):

    r = z[0]
    b = z[1]
    lm_id = int(z[2])

    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.y + r * s

    # covariance
    Gz = np.array([[c, -r * s],
                   [s, r * c]])

    particle.lmP[2 * lm_id:2 * lm_id + 2] = Gz @ Q @ Gz.T

    return particle

# <--- Update Particle Kalman Filters --->
def update_KF_with_cholesky(xf, Pf, v, Q, Hf):
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q

    S = (S + S.T) * 0.5
    SChol = np.linalg.cholesky(S).T
    SCholInv = np.linalg.inv(SChol)
    W1 = PHt @ SCholInv
    W = W1 @ SCholInv.T

    x = xf + W @ v
    P = Pf - W1 @ W1.T

    return x, P

def update_landmark(particle, z, Q):

    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])

    xf, Pf = update_KF_with_cholesky(xf, Pf, dz, Q, Hf)

    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return particle
# <--- END Update Particle Kalman Filters --->

def normalize_weight(particles):

    sumw = sum([p.w for p in particles])

    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sumw
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE

        return particles

    return particles

# ~ PARTICLE FILTER ~

# Simulate particles from given motion model
def prediction_step(particles, u):
    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE, 1))
        px[0, 0] = particles[i].x
        px[1, 0] = particles[i].y
        px[2, 0] = particles[i].yaw
        ud = u + (np.random.randn(1, 2) @ R).T  # add noise
        px = motion_model(px, ud)
        particles[i].x = px[0, 0]
        particles[i].y = px[1, 0]
        particles[i].yaw = px[2, 0]

    return particles

# Update the particles with the observation
def update_step(particles, zs):
    for iz in range(len(zs[0, :])):

        lmid = int(zs[2, iz])

        for ip in range(N_PARTICLE):
            # new landmark
            if abs(particles[ip].lm[lmid, 0]) <= 0.01:
                particles[ip] = add_new_lm(particles[ip], zs[:, iz], Q)
            # known landmark
            else:
                w = compute_weight(particles[ip], zs[:, iz], Q)
                particles[ip].w *= w
                particles[ip] = update_landmark(particles[ip], zs[:, iz], Q)

    return particles

# Resample by removing redundant particles
def resampling_step(particles):
    """
    low variance re-sampling
    """

    particles = normalize_weight(particles)

    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)

    pw = np.array(pw)

    Neff = 1.0 / (pw @ pw.T)  # Effective particle number
    # print(Neff)

    if Neff < NTH:  # resampling
        wcum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resampleid = base + np.random.rand(base.shape[0]) / N_PARTICLE

        inds = []
        ind = 0
        for ip in range(N_PARTICLE):
            while ((ind < wcum.shape[0] - 1) and (resampleid[ip] > wcum[ind])):
                ind += 1
            inds.append(ind)

        tparticles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tparticles[inds[i]].x
            particles[i].y = tparticles[inds[i]].y
            particles[i].yaw = tparticles[inds[i]].yaw
            particles[i].w = 1.0 / N_PARTICLE

    return particles

def pi_2_pi(angle): # Normalizer
    return (angle + math.pi) % (2 * math.pi) - math.pi

# ~ Plotting Functions ~ 

def show_robot_sensor_range(robot_pose,env):
    rx,ry,rtheta= robot_pose[0],robot_pose[1],robot_pose[2]
    sensor_radius = env.dist2pixellen(robot_fov-landmark_radius) # Take the actual robot radius
    rx_pix,ry_pix = env.position2pixel((rx,ry)) # Take the size of the map, figure out x and y based on the map
    #pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,int(rx_pix+np.cos(rtheta)),int(ry_pix+(sensor_radius**2)*np.cos(rtheta)),(255,255,153))
    pygame.gfxdraw.filled_circle(env.get_pygame_surface(),rx_pix,ry_pix,sensor_radius,(255,255,153)) # Last statement probably colour
    pygame.gfxdraw.circle(env.get_pygame_surface(),rx_pix,ry_pix,sensor_radius,(0,0,153)) # Last statement probably colour

# ~ Plot Landmarks ~
def show_landmarks(landmarks,env):
    for landmark in landmarks:
        lx_pixel, ly_pixel = env.position2pixel(landmark)
        r_pixel = env.dist2pixellen(landmark_radius) # Radius of the circle
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,255,255)) # Last statement probably colour

def show_robot_estimate_as_particles(history,env):
    for i in range(len(history)):
        pos = (history[i].x,history[i].y)
        lx_pixel, ly_pixel = env.position2pixel(pos)
        pygame.gfxdraw.pixel(env.get_pygame_surface(),lx_pixel,ly_pixel,(255,0,0))

def show_robot_estimate_as_point(history,env):
    rx_total = 0; ry_total = 0
    for p in history:
        rx_total = rx_total + p.x
        ry_total = ry_total + p.y
    rx_avg = np.divide(rx_total,N_PARTICLE)
    ry_avg = np.divide(ry_total,N_PARTICLE)
    pos = (rx_avg,ry_avg)
    lx_pixel, ly_pixel = env.position2pixel(pos)
    pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,env.dist2pixellen(0.08),(255,0,0))

def show_landmark_estimate_as_particles(particles,env):
    for p in particles:
        for i in range(N_LM):
            pos = (p.lm[i][0],p.lm[i][1])
            lx_pixel, ly_pixel = env.position2pixel(pos)
            pygame.gfxdraw.pixel(env.get_pygame_surface(),lx_pixel,ly_pixel,(255,0,0))

def show_landmark_estimate_as_point(particles,env):
    lm_total = np.zeros((N_LM, LM_SIZE))
    for p in particles:
        lm_total = np.add(p.lm,lm_total)
    lm_avg = np.divide(lm_total,N_PARTICLE)
    for lm in lm_avg:
        pos = (lm[0],lm[1])
        lx_pixel, ly_pixel = env.position2pixel(pos)
        pygame.gfxdraw.pixel(env.get_pygame_surface(),lx_pixel,ly_pixel,(255,0,0))
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,env.dist2pixellen(0.08),(0,0,0))

# ~ Plot Sensor Measurement ~
def show_measurements(robot_pose,zs,env):
    rx, ry, theta = robot_pose[0], robot_pose[1], robot_pose[2]
    rx_pix, ry_pix = env.position2pixel((rx,ry)) # Pixel measurement conversion
    for i in range(len(zs[0, :])):
        dist = zs[0][i] # Unpack the sensor data for each measurement
        phi = zs[1][i]
        lidx = zs[2][i]
        lx,ly = rx+dist*np.cos(phi+theta),ry+dist*np.sin(phi+theta) # Convert robot frame to global frame
        lx_pix, ly_pix = env.position2pixel((lx,ly)) # Pixel Unit Conversion
        pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,lx_pix,ly_pix,(155,155,155)) # Draw the landmarks

# <--- End Plotting Functions --->

if __name__=='__main__':

    N_LM = len(landmarks)
    particles = [Particle(N_LM) for i in range(N_PARTICLE)]
    history = []

    # Initialize pygame
    pygame.init()

    # Initialize robot and time step
    x_init = np.array([STARTING_X,STARTING_Y,STARTING_YAW])
    robot = vehicles.DifferentialDrive(x_init)

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")

    running = True
    u = np.array([0.,0.]) # Controls
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states
        robot.move_step(u,DT) # Integrate EOMs forward, i.e., move robot

        zs = sim_measurements(robot.get_pose(),landmarks) # Get measurements

        particles = prediction_step(particles, u.reshape(2,1)) 
        history = deepcopy(particles) # Update my current history
        particles = update_step(particles,zs)
        particles = resampling_step(particles)

        env.show_map() # Re-blit map
        show_robot_sensor_range(robot.get_pose(),env) # Show the range of robot sensor
        env.show_robot(robot) # Re-blit robot
        show_measurements(robot.get_pose(),zs,env) # Draw a line to illustrate that the robot has "Seen" the landmark
        show_landmarks(landmarks,env) # Display the landmarks

        show_robot_estimate_as_point(history,env)
        show_landmark_estimate_as_point(particles,env)

        # THIS SHOWS EVERY INDIVIDUAL PARTICLE, LAG WARNING #
        #show_robot_estimate_as_particles(history,env)
        #show_landmark_estimate_as_particles(particles,env)

        pygame.display.update() # Update display