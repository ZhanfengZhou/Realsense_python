
import numpy as np
import math

def euler_angle_2_trans(goals_value):

    #change ZYZ Euler angle to Trans Matrix for end effector (ee)
    phi = math.radians(goals_value[0])
    theta = math.radians(goals_value[1])
    psi = math.radians(goals_value[2])
    x = goals_value[3]
    y = goals_value[4]
    z = goals_value[5]
        
    sp = math.sin(phi)
    cp = math.cos(phi)
    st = math.sin(theta)
    ct = math.cos(theta)
    ss = math.sin(psi)
    cs = math.cos(psi)
        
    nx = cp * ct * cs - sp * ss
    ny = sp * ct * cs + cp * ss
    nz = - st * cs
    ox = - cp * ct * ss - sp * cs
    oy = - sp * ct * ss + cp * cs
    oz = st * ss
    ax = cp * st
    ay = sp * st
    az = ct
        
    Trans_R = [[nx, ox, ax],[ny, oy, ay],[nz, oz, az]]
    zcamera_6 = -0.175    #camera: z: 175mm
    
    Trans_T = [1000*(x+ax*zcamera_6), 1000*(y+ay*zcamera_6), 1000*(z+az*zcamera_6)]
    
    return Trans_R, Trans_T
    
    
    
    
    
