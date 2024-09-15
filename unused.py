import numpy as np

N_PARTICLE = 50
LM_SIZE = 2

# Drawing the mean of the robot pose as highest-frequency basis
def compute_lm_and_robot_estm_2(particles):
    # Special line of code to extract average landmark position represented by every particle
    lm_total = np.zeros((len(particles[0].lm), LM_SIZE))

    rx_total = []; ry_total = []; ryaw_total = []
    for p in particles:
        lm_total = np.add(p.lm,lm_total)
        rx_total.append(p.x),ry_total.append(p.y),ryaw_total.append(p.yaw)

    lm_estm = np.divide(lm_total,N_PARTICLE)

    rx_hist,rx_bin = np.histogram(rx_total); ry_hist,ry_bin = np.histogram(rx_total); ryaw_hist,ryaw_bin = np.histogram(rx_total)
    rx_max_index = np.where(rx_hist == np.max(rx_hist))[0][0]; ry_max_index = np.where(ry_hist == np.max(ry_hist))[0][0]; ryaw_max_index = np.where(ryaw_hist == np.max(ryaw_hist))[0][0]
    rx_value = (rx_bin[rx_max_index] + rx_bin[rx_max_index+1])/2
    ry_value = (ry_bin[ry_max_index] + ry_bin[ry_max_index+1])/2
    ryaw_value = (ryaw_bin[ryaw_max_index] + ryaw_bin[ryaw_max_index+1])/2
    
    robot_estm_pose = [rx_value,ry_value,ryaw_value]

    return lm_estm,robot_estm_pose

def norm2globalframe(rex,rey,dist,phi):
    #print("[",dist,phi,"]")
    new_phi = normalize_2_polframe_lidar(phi)
    HB_R = np.array([[1            ,0             ,rex],
                        [0            ,1             ,rey],
                        [0            ,0             ,1  ]])
    HR_L = np.array([[np.cos(new_phi),-np.sin(new_phi),dist],
                        [np.sin(new_phi),np.cos(new_phi) ,0   ],
                        [0            ,0                 ,1   ]])
    lm_final_shape = np.array([[0],
                                [0],
                                [1]])
    lm_final_shape = HB_R.dot(HR_L).dot(lm_final_shape) # Normalize from local robot frame to global frame
    #print("Next \___/")
    #print("[",lm_final_shape[0][0], lm_final_shape[1][0],"]")
    lm_xy = [lm_final_shape[0][0], lm_final_shape[1][0]]
    return lm_xy

def normalize_2_polframe_lidar(angle):
    phase_shift_by_90 = angle + np.pi/2
    if phase_shift_by_90 < 0:
        phase_shift_by_90 = 2*np.pi + phase_shift_by_90
    return phase_shift_by_90

def normalize_2_polframe_robot(angle):
    if angle < 0:
        angle =  angle + 2*np.pi 
    return angle

def pol2cart(dist,theta):
    x,y = dist*np.cos(theta),dist*np.sin(theta)
    return x,y

# Important print statement for loop closure
        #print(seen_landmarks[np.where(relatexy == np.min(relatexy))[0][0]][0],seen_landmarks[np.where(relatexy == np.min(relatexy))[0][0]][1],np.min(relatexy),str(current_landmark[0]),str(current_landmark[1]))
        #print(relatexy)
        #print("---")
        #print(seen_landmarks)
        #print("-----")
        #print(norm_seen_landmarks)
        #nums = input("Continue ->")
