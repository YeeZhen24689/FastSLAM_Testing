# FastSLAM_Testing
The Goal of this Repo is to implement FastSLAM 1.0 on a differential drive robot to simulate its usage for mapping on a racecar.

Part of the implementation is taken from https://atsushisakai.github.io/PythonRobotics/modules/slam/FastSLAM1/FastSLAM1.html.
Main pseudocode can be found in Probabilistic Robotics. Below are a few examples of the simulator in action:

<p align="center">
  <src=![Screenshot from 2024-12-10 11-24-09](https://github.com/user-attachments/assets/6077afee-4d27-4af9-8013-1abb758d9c12)>
  <src=https://github.com/user-attachments/assets/484d1241-41bb-4379-b037-595ea18fe597>
</p>

![Screenshot from 2024-12-10 11-28-17](https://github.com/user-attachments/assets/484d1241-41bb-4379-b037-595ea18fe597)
![Screenshot from 2024-12-10 11-24-09](https://github.com/user-attachments/assets/6077afee-4d27-4af9-8013-1abb758d9c12) 

# Using the Code
The differential drive robot simulation environment used is: https://github.com/jacobhiggins/python_ugv_sim <br />
For the simulation: <br />
1) Go to python_ugv_sim/utils/environment.py and change METER_PER_PIXEL variable to 0.07 (Approx. Line 11) <br />
2) Go to python_ugv_sim/utils/vehicles.py and change max_v to 30m/s (Approx. Line 56), change max_omega to 15m/s (Approx. Line 57) <br />

# Setting up the environment
The main code file should be outside of the folder of which you downloaded the differential drive robot repo. <br />
Environment folder: <br />
|_ fastSLAM.py <br />
|_ venv <br />
|_ python_ugv_sim (You need to edit a part of it - Check Using the code) <br />
|_ main.py <br />
|_ Tracks <br />
&ensp;|_ You will see different tracks stored in here with .csv <br />
