import numpy as np

from scene import scene, get_Flashpoint
from evaluate import get_max_shallow_time_pro1
from sample_dots import sample_dots

if __name__ == "__main__":
    # 无人机初始位置坐标
    FY_coordinates_ini = np.array([[17800, 0, 1800],
                                   [12000, 1400, 1400],
                                   [6000, -3000, 700],
                                   [11000, 2000, 1800],
                                   [13000, -2000, 1300]])
    # 导弹初始位置坐标
    M_coordinates_ini = np.array([[20000, 0, 2000],
                                  [19000, 600, 2100],
                                  [18000, -600, 1900]])
    
    FY_velocity_ini = np.array([[-120,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]])

    scene_test_1 = scene(FY_coordinates_ini, M_coordinates_ini, FY_velocity_ini, True)
    evolve_time = 1.5
    scene_test_1.load_current_state(evolve_time, True)

    falsh_coordinate = get_Flashpoint(horizontal_speed = FY_velocity_ini[0],
                                      interval_time = 3.6,
                                      init_coordinate = scene_test_1.FY_coordinates[0])
   
    sample_dots_pro1 = sample_dots()
    
    max_shallow_time = get_max_shallow_time_pro1(falsh_coordinate, 
                                                 scene_test_1,
                                                 time_step_rate = 0.05,
                                                 sample_dots = sample_dots_pro1)
    
    

    