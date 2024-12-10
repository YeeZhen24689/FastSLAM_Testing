import numpy as np
from fastSLAM import *
from python_ugv_sim.utils import vehicles, environment

landmarks = loadmap("comp_2021")

if __name__=='__main__':
    particles = [Particle() for i in range(N_PARTICLE)]
    history = []
    lm_estm = np.zeros((1,LM_SIZE))
    robot_estm_pose = [STARTING_X, STARTING_Y,STARTING_YAW]
    c_matrix = []

    # Initialize pygame
    pygame.init()

    # Initialize robot and time step
    x_init = np.array([STARTING_X,STARTING_Y,STARTING_YAW])
    robot = vehicles.DifferentialDrive(x_init)

    # Initialize and display environment
    env = environment.Environment(map_image_path="./python_ugv_sim/maps/map_blank.png")

    running = True
    u = np.array([0.,0.]) # Controls
    show = 0

    while running:

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show = not show
                    break
                else: 
                    u = robot.update_u(u,event)
            else: 
                u # Update controls based on key states

        robot.move_step(u,DT) # Integrate EOMs forward, i.e., move robot

        zs = sim_measurements(robot.get_pose(),landmarks) # Get measurementsbreakout,orange)
        cone_colours = get_cone_colours(zs)
        zs = detect_index(robot_estm_pose,lm_estm,zs)
        particles = prediction_step(particles, u.reshape(2,1))
        history = deepcopy(particles) # Update my current history
        particles,c_matrix = update_step(particles,zs,c_matrix)
        particles = resampling_step(particles)
        lm_estm,robot_estm_pose = compute_lm_and_robot_estm(particles)

        env.show_map() # Re-blit map
        show_robot_sensor_range(robot.get_pose(),env) # Show the range of robot sensor
        env.show_robot(robot) # Re-blit robot
        show_measurements(robot.get_pose(),zs,env) # Draw a line to illustrate that the robot has "Seen" the landmark
        if show == 1:
            show_landmarks(landmarks,env) # Display the landmarks

        show_estimate(show_particles=False,show_point=True,history=history,particles=particles,env=env,c_matrix=c_matrix)

        pygame.display.update() # Update display