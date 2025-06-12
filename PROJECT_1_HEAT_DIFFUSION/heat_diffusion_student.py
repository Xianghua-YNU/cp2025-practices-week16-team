"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000 #
# 任务1: 基本热传导模拟
def basic_heat_diffusion():
    """任务1: 基本热传导模拟"""

    r = D*dt/(dx**2)
    print(f"任务1 - 稳定性参数 r = {r}")
    
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100
    u[0, :] = 0
    u[-1, :] = 0
    
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    return u

# 任务2: 解析解与数值解比较
def analytical_solution(n_terms=100):
    """解析解函数"""
    x = np.linspace(0, dx*(Nx-1), Nx)
    t = np.linspace(0, dt*Nt, Nt)
    x, t = np.meshgrid(x, t)
    s = 0
    for i in range(n_terms):
        j = 2*i + 1
        s += 400/(j*np.pi) * np.sin(j*np.pi*x/L) * np.exp(-(j*np.pi/L)**2 * t * D)
    return s.T

# 任务3: 数值解稳定性分析
def stability_analysis():
    """任务3: 数值解稳定性分析"""
    dx = 0.01
    dt = 0.6  # 使r>0.5
    r = D*dt/(dx**2)
    print(f"任务3 - 稳定性参数 r = {r} (r>0.5)")
    
    Nx = int(L/dx) + 1
    Nt = 2000
    
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100
    u[0, :] = 0
    u[-1, :] = 0
    
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    # 可视化不稳定解
    plot_3d_solution(u, dx, dt, Nt, title='Task 3: Unstable Solution (r>0.5)')

# 任务4: 不同初始条件模拟
def different_initial_condition():
    """任务4: 不同初始条件模拟"""
    dx = 0.01
    dt = 0.5
    r = D*dt/(dx**2)
    print(f"任务4 - 稳定性参数 r = {r}")
    
    Nx = int(L/dx) + 1
    Nt = 1000
    
    u = np.zeros((Nx, Nt))
    u[:51, 0] = 100  # 左半部分初始温度100K
    u[50:, 0] = 50   # 右半部分初始温度50K
    u[0, :] = 0
    u[-1, :] = 0
    
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    # 可视化
    plot_3d_solution(u, dx, dt, Nt, title='Task 4: Temperature Evolution with Different Initial Conditions')
    return u

# 任务5: 包含牛顿冷却定律的热传导
def heat_diffusion_with_cooling():
    """任务5: 包含牛顿冷却定律的热传导"""
    r = D*dt/(dx**2)
    h = 0.1  # 冷却系数
    print(f"任务5 - 稳定性参数 r = {r}, 冷却系数 h = {h}")
    
    Nx = int(L/dx) + 1
    Nt = 100
    
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100
    u[0, :] = 0
    u[-1, :] = 0
    
    for j in range(Nt-1):
        u[1:-1, j+1] = (1-2*r-h*dt)*u[1:-1, j] + r*(u[2:, j] + u[:-2, j])
    
    # 可视化
    plot_3d_solution(u, dx, dt, Nt, title='Task 5: Heat Diffusion with Newton Cooling')

def plot_3d_solution(u, dx, dt, Nt, title):
    """Plot 3D surface of temperature distribution"""
    Nx = u.shape[0]
    x = np.linspace(0, dx*(Nx-1), Nx)
    t = np.linspace(0, dt*Nt, Nt)
    X, T = np.meshgrid(x, t)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u.T, cmap='rainbow')
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Time t (s)')
    ax.set_zlabel('Temperature T (K)')
    ax.set_title(title)
    plt.show()
    
if __name__ == "__main__":
    print("=== 铝棒热传导问题参考答案 ===")
    print("1. 基本热传导模拟")
    u = basic_heat_diffusion()
    plot_3d_solution(u, dx, dt, Nt, title='Task 1: Heat Diffusion Solution')

    print("\n2. 解析解")
    s = analytical_solution()
    plot_3d_solution(s, dx, dt, Nt, title='Analytical Solution')

    print("\n3. 数值解稳定性分析")
    stability_analysis()
    
    print("\n4. 不同初始条件模拟")
    different_initial_condition()
    
    print("\n5. 包含牛顿冷却定律的热传导")
    heat_diffusion_with_cooling()
