import numpy as np
import copy 

class scene:
    '''
    t: 当前系统时间
    FY_coordinates: 无人机当前位置坐标，类型为二维numpy数组: [(x1, y1, z1), (x2, y2, z2), ...]
    M_coordinates_ini: 导弹当前位置坐标，类型二维numpy数组: [(x1, y1, z1), (x2, y2, z2), ...]
    M_velocity: 导弹速度，类型二维numpy数组: [(vx1, vy1, vz1), (vx2, vy2, vz2), ...]
    FY_velocity: 无人机当前速度类型二维numpy数组: [(vx1, vy1, vz1), (vx2, vy2, vz2), ...]
    '''
    def __init__(self, FY_coordinates_ini, 
                 M_coordinates_ini, 
                 FY_velocity_ini,
                 verbose = False):
        '''
        verbose: 决定是否构造完成后打印信息
        '''
        self.t = 0  # 当前系统时间
        self.FY_coordinates = copy.deepcopy(FY_coordinates_ini) # 当前无人机坐标
        self.M_coordinates = copy.deepcopy(M_coordinates_ini) # 当前导弹坐标
        self.FY_velocity = copy.deepcopy(FY_velocity_ini) # 当前无人机速度

        self.M_velocity = np.zeros((3,3))

        # 初始化当前导弹速度
        for i in range(3):
            self.M_velocity[i] = self.calculate_M_velocity(self.M_coordinates[i])

        if verbose:
            print("[info][debug]=========== scene 信息初始化成功 ^_^! ===========")
            self.show_current_state()

    def calculate_M_velocity(self, coordinates): # 根据导弹位置坐标计算导弹速度，仅在构造函数中调用
        modulus = np.linalg.norm(coordinates) # 计算导弹到原点的距离
        velocity = []
        for i in range(3):
            velocity.append(-coordinates[i]/modulus * 300)
        return np.array(velocity)
    
    def load_current_state(self, verbose = False):
        '''
        verbose 决定是否在加载当前状态时打印出信息
        '''
        # 加载当前导弹位置坐标
        self.M_coordinates = self.M_coordinates + self.t * self.M_velocity

        # 加载当前无人机位置坐标
        self.FY_coordinates = self.FY_coordinates + self.t * self.FY_coordinates

        if verbose == True:
            self.show_current_state()

    def show_current_state(self):
        '''
        打印当前场景状态
        '''
        try:
            print(f"[info][debug]=========== t = {self.t} s时的状态信息 ===========")
            print(f"[info][debug]导弹坐标 :\n {self.M_coordinates}")
            print(f"[info][debug]导弹速度 :\n {self.M_velocity}")
            print(f"[info][debug]无人机坐标:\n {self.FY_coordinates}")
            print(f"[info][debug]无人机速度:\n {self.FY_velocity}")
            print(f"[info][debug]=========== 状态信息打印成功 ^_^ ===========")
            

        except Exception as e:
            print(f"[info][error]=========== 状态信息打印失败 ^_^ ===========\n {e}")

        


        
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
    
    FY_velocity_ini = np.array([[30, 30, 30] for _ in range(5)])

    scene_test_1 = scene(FY_coordinates_ini, M_coordinates_ini, FY_velocity_ini, True)

    scene_test_1.t = 5

    scene_test_1.load_current_state(True)
    
    

    
