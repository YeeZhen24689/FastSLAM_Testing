import numpy as np

# ~ Landmark Parameters ~
file_name = "comp_2021"
file_extension = ".csv"

# Load the full data, as well as loading the labels and positions into separate arrays
full_data = np.loadtxt("Tracks/" + file_name + file_extension, delimiter=',', skiprows=1, usecols=(0, 1, 2), dtype=str)
labels = np.loadtxt("Tracks/" + file_name + file_extension, delimiter=',', skiprows=1, usecols=(0), dtype=str)
positions = np.loadtxt("Tracks/" + file_name + file_extension, delimiter=',', skiprows=1, usecols=(1, 2))

# Convert an array to our data format
mapping = {"blue" : 0,
           "yellow" : 1,
           "orange" : 2,
           "big_orange" : 3,
           "unknown" : 4}

# Controlling how the landmarks are displayed
sfx, sfy = 1, 1
disp_x, disp_y = 20,17

# Load the landmarks
landmarks = []
for row in full_data:
  if row[0] in mapping:
    landmarks.append((float(row[1])*sfx+disp_x, float(row[2])*sfy+disp_y))

N_LM = len(landmarks)
STARTING_X = 7.5
STARTING_Y = 10.5
STARTING_YAW = np.pi/2

#from fastSLAM import *
from fastSLAM import *
from python_ugv_sim.utils import vehicles, environment

if __name__=='__main__':

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
        #show_measurements(robot.get_pose(),zs,env) # Draw a line to illustrate that the robot has "Seen" the landmark
        show_landmarks(landmarks,env) # Display the landmarks

        show_estimate(show_particles=False,show_point=True,history=history,particles=particles,env=env)

        pygame.display.update() # Update display