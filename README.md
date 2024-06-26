# FastSLAM_Testing
The Goal of this Repo is to implement FastSLAM 1.0 on a differential drive robot to simulate its usage for mapping on a racecar.
Part of the implementation is taken from https://atsushisakai.github.io/PythonRobotics/modules/slam/FastSLAM1/FastSLAM1.html.
Main pseudocode can be found in Probabilistic Robotics.

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
