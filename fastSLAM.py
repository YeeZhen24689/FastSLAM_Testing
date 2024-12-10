import pygame
import pygame.gfxdraw
import math
import numpy as np
from copy import deepcopy
import os
import random
from PIL import Image,ImageDraw

clear = lambda : os.system('cls')

##----------------------------------##
##-----------Conventions------------##
##----------------------------------##
#  Blue Cone - 0
#  Yellow Cone - 1
#  Orange -2
#  Big Orange - 3
#  Unknown - 4

# Fast SLAM covariance
Q = np.diag([3.0, np.deg2rad(10.0)])**2
R = np.diag([0.1, np.deg2rad(1.0)])**2

#  Simulation parameter
Qsim = np.diag([0.3, np.deg2rad(2.0)])**2
Rsim = np.diag([0.5, np.deg2rad(10.0)])**2
OFFSET_YAWRATE_NOISE = 0.01

# ~ Simulation Set Up ~
landmark_radius = 0.2
robot_fov = 3
landmark_counter = 0

DT = 0.1  # time tick [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 3.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM srate size [x,y]
N_PARTICLE = 50 # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling
STARTING_X = 37.5
STARTING_Y = 3
STARTING_YAW = 0
SENSOR_FOV_ANGLE = np.pi/3
LIDAR_FOV_ANGLE = 2*np.pi/3

# <--- SIM NOISE & LOOP CLOSURE CONTROLS --->
LOOP_CLOSURE_THRESHOLD = 0.5
DIVISION_FACTOR = 1000
SKEW = 3
SKEW_ON = 0; RANDOM_NOISE_ON = 0
RANDOM_NOISE_LB = -1; RANDOM_NOISE_UB = 1
# <--- END SIM NOISE & LOOP CLOSURE CONTROLS --->

# ~ Declare our Particle Agent ~
class Particle:

    def __init__(self):
        self.w = 1.0 / N_PARTICLE # Particle Weight
        self.x = STARTING_X # Particle Location X
        self.y = STARTING_Y # Particle Location Y
        self.yaw = STARTING_YAW # Particle Yaw
        # landmark x-y positions
        self.lm = np.zeros((0, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((0 * LM_SIZE, LM_SIZE))

# ~ Setting Up the sensor and odometry for the vehicle ~

# ~ Measurement Simulation Function
def sim_measurements(robot_pose,landmarks):
    rx, ry, rtheta = robot_pose[0], robot_pose[1], robot_pose[2]
    zs=np.zeros((3,0))
    #print("Next \/")
    for landmark in landmarks: # Iterate over landamrks and inveces
        lx,ly,colour = landmark
        dist = np.linalg.norm(np.array([lx-rx,ly-ry])) # Distance between robot and landmark
        phi = np.arctan2(ly-ry,lx-rx) - rtheta # Angle between robot heading and landmark
        phi = np.arctan2(np.sin(phi),np.cos(phi)) # Normalize the angle between pi and -pi
        if dist <= robot_fov and abs(phi) <= SENSOR_FOV_ANGLE: # Only append if observation is in the ZED Cam's field of view
            zs_add = np.array([dist,phi,colour]).reshape(3,1)
            zs = np.hstack((zs,zs_add))
            #zs = add_noise(zs,SKEW,RANDOM_NOISE_LB,RANDOM_NOISE_UB)
        elif dist <= robot_fov and abs(phi) <= LIDAR_FOV_ANGLE: # Only append if observation is in the robot'LIDAR's field of view
            zs_add = np.array([dist,phi,4]).reshape(3,1)
            zs = np.hstack((zs,zs_add))
    return zs

def detect_index(robot_estm_pose,lm_estm,zs):
    rex, rey, reyaw = robot_estm_pose[0], robot_estm_pose[1], robot_estm_pose[2]
    lidx_append = np.zeros((1,len(zs[0,:])))
    new_landmarks_this_iter = 0 # Reset amount of landmarks I see this iteration
    for id,landmark_read in enumerate(np.transpose(zs)):
        dist = landmark_read[0]; phi = landmark_read[1]; colour = landmark_read[2]
        cur_lm = norm2globalframe(rex,rey,reyaw,dist,phi)
        lidx,new_landmarks_this_iter = loop_closure(lm_estm,cur_lm,new_landmarks_this_iter)
        lidx_append[0,id] = lidx
    zs = np.vstack((zs,lidx_append[0]))
    return zs

def get_cone_colours(zs):
    colours_I_see = []
    for i in zs[2,:]:
        if i not in colours_I_see:
            colours_I_see.append(i)
    return colours_I_see

def norm2globalframe(rex,rey,reyaw,dist,phi):

    s = math.sin(pi_2_pi(reyaw + phi))
    c = math.cos(pi_2_pi(reyaw + phi))

    lm_xy = np.array([rex + dist * c, rey + dist * s])

    #print(reyaw+phi)
    #print(lm_xy)

    return lm_xy

def loop_closure(seen_landmarks,current_landmark,new_landmarks_this_iter):
    bx = np.ones((len(seen_landmarks),1))*current_landmark[0]
    by = np.ones((len(seen_landmarks),1))*current_landmark[1]
    b = np.hstack((bx,by))

    norm_seen_landmarks = np.power(seen_landmarks - b,2) # Normalise the seen landmarks to look for correspondences
    threshold = LOOP_CLOSURE_THRESHOLD
    relatexy = norm_seen_landmarks[:,0] + norm_seen_landmarks[:,1] # Mark the correspondences between x and y as the sum
    
    #print(seen_landmarks[np.where(relatexy == np.min(relatexy))[0][0]][0],seen_landmarks[np.where(relatexy == np.min(relatexy))[0][0]][1],np.min(relatexy),str(current_landmark[0]),str(current_landmark[1]))
    try:
        if np.min(relatexy) < threshold: # Landmark is in "acceptable range"
            #index = np.where(relatexy == np.min(relatexy))[0][0] # Substitute for the row below
            final_index = np.argmin(relatexy) # Which row is this landmark in?
        else:
            index = np.nan # Some default if its a new landmark
            enumeratevar = new_landmarks_this_iter
            new_landmarks_this_iter += 1
            final_index = len(seen_landmarks)+enumeratevar
    except ValueError:
        index = np.nan # Some default if its a new landmark
        enumeratevar = new_landmarks_this_iter
        new_landmarks_this_iter += 1
        final_index = len(seen_landmarks)+enumeratevar
        pass

    return final_index,new_landmarks_this_iter

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
    lm_id = int(z[3])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
    zp, Hv, Hf, Sf = compute_jacobians(particle, xf, Pf, Q)
    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = pi_2_pi(dx[1, 0])

    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print("singular")
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
    c = z[2]
    lm_id = int(z[3])

    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))

    particle.lm = np.vstack([particle.lm,[particle.x + r * c, particle.y + r * s]])

    #print("Add new lm -> [",particle.x + r * c, particle.y + r * s,"]")
    #print("(",particle.x + r * c,",",particle.y + r * s,")")
    #print("r = ",r," | b = ",b," | Particle Yaw = ",particle.yaw," | pyaw + b = ",particle.yaw + b)

    # covariance
    Gz = np.array([[c, -r * s],
                   [s, r * c]])

    particle.lmP = np.vstack([particle.lmP,Gz @ Q @ Gz.T])

    return particle

# <--- Update Particle Kalman Filters --->
def update_KF_with_cholesky(xf, Pf, v, Q, Hf):
    PHt = Pf.dot(np.transpose(Hf))
    S = Hf.dot(PHt) + Q

    S = (S + np.transpose(S)) * 0.5
    SChol = np.transpose(np.linalg.cholesky(S))
    SCholInv = np.linalg.inv(SChol)
    W1 = PHt.dot(SCholInv)
    W = W1.dot(np.transpose(SCholInv))

    x = xf + W.dot(v)
    P = Pf - W1.dot(np.transpose(W1))

    return x, P

def update_landmark(particle, z, Q):

    lm_id = int(z[3])
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
def update_step(particles, zs, colour_matrix):
    new_landmark = 0
    for iz in range(len(zs[0, :])):

        lmid = int(zs[3, iz])
        lmc = int(zs[2, iz])
        #print("Landmark_read")
        for ip in range(N_PARTICLE):
            #print(lmid,len(particles[ip].lm))
            # new landmark
            if lmid >= len(particles[ip].lm):
                particles[ip] = add_new_lm(particles[ip], zs[:, iz], Q)
                new_landmark = 1
            else:
                w = compute_weight(particles[ip], zs[:, iz], Q)
                particles[ip].w *= w
                particles[ip] = update_landmark(particles[ip], zs[:, iz], Q)
                new_landmark = 0
        if new_landmark == 1:
            colour_matrix.append(lmc)
        elif colour_matrix[lmid] == 4 and lmc != 4:
            colour_matrix[lmid] = lmc
    return particles,colour_matrix

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

# Drawing the mean of the robot pose as mean basis
def compute_lm_and_robot_estm(particles):
    # Special line of code to extract average landmark position represented by every particle
    lm_total = np.zeros((len(particles[0].lm), LM_SIZE))

    rx_total = 0; ry_total = 0; ryaw_total = 0
    for p in particles:
        lm_total = np.add(p.lm,lm_total)
        rx_total += p.x; ry_total += p.y; ryaw_total += p.yaw
    lm_estm = np.divide(lm_total,N_PARTICLE)    

    robot_estm_pose = [rx_total/N_PARTICLE,ry_total/N_PARTICLE,ryaw_total/N_PARTICLE]
    '''
    rx_total = 0; ry_total = 0
    rx_particle = 0; ry_particle = 0
    for p in particles:
        if robot_estm_pose[0]*0.5 < p.x < robot_estm_pose[0]*1.5:
            rx_total += p.x
            rx_particle += 1
        if robot_estm_pose[1]*0.5 < p.y < robot_estm_pose[1]*1.5:
            ry_total += p.y
            ry_particle += 1

    robot_estm_pose = [rx_total/rx_particle,ry_total/ry_particle,ryaw_total/N_PARTICLE]
    '''
    return lm_estm,robot_estm_pose

# <--- End Key Particle Filter Functions --->

# ~ Conversion Functions

def pi_2_pi(angle): # Normalizer
    return (angle + math.pi) % (2 * math.pi) - math.pi

# <--- Noise Machine --->and abs(phi) < SENSOR_FOV_ANGLE:
def add_noise(array,skew,lower_bound,upper_bound):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]): 
            if RANDOM_NOISE_ON == 1:
                array[i][j] += random.randint(lower_bound,upper_bound)/DIVISION_FACTOR
            elif SKEW_ON == 1:
                array[i][j] += skew/DIVISION_FACTOR
            else:
                break
    return array
#<--- END NOISE MACHINE --->

# <--- Plotting Functions --->

def show_robot_sensor_range(robot_pose,env):
    range = (robot_fov/3)*80
    rx,ry,rtheta= robot_pose[0],robot_pose[1],robot_pose[2]
    sensor_radius = env.dist2pixellen(robot_fov-landmark_radius) # Take the actual robot radius
    rx_pix,ry_pix = env.position2pixel((rx,ry)) # Take the size of the map, figure out x and y based on the map
    #pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,int(rx_pix+np.cos(rtheta)),int(ry_pix+(sensor_radius**2)*np.cos(rtheta)),(255,255,153))
    rect = pygame.rect.Rect(rx_pix-range/2,ry_pix-range/2,range,range)
    #pygame.gfxdraw.filled_circle(env.get_pygame_surface(),rx_pix,ry_pix,sensor_radius,(255,255,153)) # Last statement probably colour
    #pygame.gfxdraw.circle(env.get_pygame_surface(),rx_pix,ry_pix,sensor_radius,(0,0,153)) # Last statement probably colour
    pygame.draw.arc(env.get_pygame_surface(),(255,255,255),rect,pi_2_pi(rtheta-SENSOR_FOV_ANGLE),pi_2_pi(rtheta+SENSOR_FOV_ANGLE))

# ~ Plot Landmarks ~
def show_landmarks(landmarks,env):
    for id,landmark in enumerate(landmarks):
        lx_pixel, ly_pixel = env.position2pixel(landmark[0:2])
        c_pixel = landmark[2]
        r_pixel = env.dist2pixellen(landmark_radius) # Radius of the circle
        if c_pixel == 0:
            pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,191,255)) # Last statement probably colour
        elif c_pixel == 1:
            pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(238,210,2)) # Last statement probably colour
        elif c_pixel == 2:
            pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(255,165,0)) # Last statement probably colour
        elif c_pixel == 3:
            pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(255,165,0)) # Last statement probably colour
        else:
            pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,r_pixel,(0,0,0)) # Last statement probably colour

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

def show_robot_estimate_as_phantom(history,env):
    TRI_SIZE = 2
    rx_total = 0; ry_total = 0; ryaw_total = 0

    for p in history:
        rx_total = rx_total + p.x
        ry_total = ry_total + p.y
        ryaw_total = ryaw_total + p.yaw

    rx_avg = np.divide(rx_total,N_PARTICLE)
    ry_avg = np.divide(ry_total,N_PARTICLE)
    ryaw_total = np.divide(ryaw_total,N_PARTICLE)

    #Triangle Calculations
    centroid_dist = TRI_SIZE/(2*np.cos(np.deg2rad(22.5)))
    pos = (rx_avg,ry_avg)
    triangle_mat = np.array([[rx_avg,rx_avg-(TRI_SIZE*np.cos(np.deg2rad(22.5))),rx_avg+(TRI_SIZE*np.cos(np.deg2rad(22.5)))],
                             [ry_avg+centroid_dist,ry_avg-(TRI_SIZE*np.sin(np.deg2rad(22.5))),ry_avg-(TRI_SIZE*np.sin(np.deg2rad(22.5)))]])
    triangle_mat = np.array([[centroid_dist,-(TRI_SIZE*np.sin(np.deg2rad(22.5))),-(TRI_SIZE*np.sin(np.deg2rad(22.5)))],
                             [0,-(TRI_SIZE*np.cos(np.deg2rad(22.5))),+(TRI_SIZE*np.cos(np.deg2rad(22.5)))]])
    rotation = np.array([[np.cos(ryaw_total),-np.sin(ryaw_total)],[np.sin(ryaw_total),np.cos(ryaw_total)]])
    rtm = rotation.dot(triangle_mat)
    rtm[0,:] += rx_avg
    rtm[1,:] += ry_avg
    pos1 = (rtm[0][0],rtm[1][0]); pos2 = (rtm[0][1],rtm[1][1]); pos3 = (rtm[0][2],rtm[1][2])
    lx_pixel1, ly_pixel1 = env.position2pixel(pos1); lx_pixel2, ly_pixel2 = env.position2pixel(pos2); lx_pixel3, ly_pixel3 = env.position2pixel(pos3)
    pygame.gfxdraw.trigon(env.get_pygame_surface(),lx_pixel1,ly_pixel1,lx_pixel2,ly_pixel2,lx_pixel3,ly_pixel3,(255,0,0))

def show_landmark_estimate_as_particles(particles,env):
    for p in particles:
        for i in range(len(particles[0].lm)):
            pos = (p.lm[i][0],p.lm[i][1])
            lx_pixel, ly_pixel = env.position2pixel(pos)
            pygame.gfxdraw.pixel(env.get_pygame_surface(),lx_pixel,ly_pixel,(255,0,0))

def show_landmark_estimate_as_point(particles,env,c_matrix):
    # Definition Dictionary
    colour_code = [(144,213,255),(255,255,0),(255,165,0),(0,128,0),(255,255,255)]
    lm_total = np.zeros((len(particles[0].lm), LM_SIZE))
    for p in particles:
        lm_total = np.add(p.lm,lm_total)
    lm_avg = np.divide(lm_total,N_PARTICLE)
    for i,lm in enumerate(lm_avg):
        pos = (lm[0],lm[1])
        lx_pixel, ly_pixel = env.position2pixel(pos)
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(),lx_pixel,ly_pixel,env.dist2pixellen(0.08),colour_code[c_matrix[i]])

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

# <--- Running the Simulation --->

def loadmap(file_name):
    # Definition Dictionary
    cones = {"blue"       : 0,
             "yellow"     : 1,
             "orange"     : 2,
             "big_orange" : 3,
             "midpoint"   : 4,}

    # ~ Landmark Parameters ~"big_orange" : 2,
    file_extension = ".csv"

    # Load the full data into separate arrays
    full_data = np.loadtxt("Tracks/" + file_name + file_extension, delimiter=',', skiprows=1, usecols=(0, 1, 2), dtype=str)

    # Controlling how the landmarks are displayed
    sfx, sfy = 1, 0.95
    disp_x, disp_y = 20,16

    # Load the landmarks
    landmarks = []
    for row in full_data:
        if row[0] != "car_start":
            landmarks.append((float(row[1])*sfx+disp_x, float(row[2])*sfy+disp_y,cones[row[0]]))

    #landmarks = [(11,10),(14,10),(20,30)]

    return landmarks

def show_estimate(show_particles,show_point,history,particles,env,c_matrix):
    if show_particles == True:
        # THIS SHOWS EVERY INDIVIDUAL PARTICLE, LAG WARNING #
        show_robot_estimate_as_particles(history,env)
        show_landmark_estimate_as_particles(particles,env)
    if show_point == True:
        show_robot_estimate_as_phantom(history,env)
        show_landmark_estimate_as_point(particles,env,c_matrix)
    return
