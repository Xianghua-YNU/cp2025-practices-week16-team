#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        u = np.zeros(self.nx)
        # 设置初始条件（10 <= x <= 11 区域为1）
        u[(self.x >= 10) & (self.x <= 11)] = 1.0
        # 应用边界条件
        u[0] = 0.0
        u[-1] = 0.0
        return u
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算稳定性参数
        r = self.alpha * dt / (self.dx**2)
        
        # 检查稳定性条件
        if r > 0.5:
            print(f"警告: 显式方法稳定性条件不满足 (r={r:.4f} > 0.5)")
        
        # 初始化解数组
        u = self.u_initial.copy()
        t = 0.0
        
        # 结果存储字典
        result = {
            'times': [],
            'solutions': [],
            'method': 'Explicit FTCS',
            'dt': dt,
            'r': r,
            'time_elapsed': 0.0
        }
        
        # 存储初始条件
        if 0 in plot_times:
            result['times'].append(t)
            result['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 计算下一个时间步
            if t + dt > self.T_final:
                dt = self.T_final - t
                r = self.alpha * dt / (self.dx**2)
            
            # 使用laplace计算空间二阶导数
            d2u = laplace(u, mode='nearest') / (self.dx**2)
            
            # 更新解
            u += r * d2u
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t += dt
            
            # 存储指定时间点的解
            for pt in plot_times:
                if abs(t - pt) < dt/2 and pt not in result['times']:
                    result['times'].append(t)
                    result['solutions'].append(u.copy())
        
        end_time = time.time()
        result['time_elapsed'] = end_time - start_time
        
        return result
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数
        r = self.alpha * dt / (self.dx**2)
        
        # 构建三对角矩阵
        main_diag = np.ones(self.nx - 2) * (1 + 2*r)
        off_diag = np.ones(self.nx - 3) * (-r)
        
        # 创建带状矩阵表示
        A = np.zeros((3, self.nx - 2))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag   # 主对角线
        A[2, :-1] = off_diag  # 下对角线
        
        # 初始化解数组
        u = self.u_initial.copy()
        t = 0.0
        
        # 结果存储字典
        result = {
            'times': [],
            'solutions': [],
            'method': 'Implicit BTCS',
            'dt': dt,
            'r': r,
            'time_elapsed': 0.0
        }
        
        # 存储初始条件
        if 0 in plot_times:
            result['times'].append(t)
            result['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            if t + dt > self.T_final:
                dt = self.T_final - t
                r = self.alpha * dt / (self.dx**2)
            
            # 构建右端项（内部节点）
            rhs = u[1:-1].copy()
            
            # 求解三对角系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t += dt
            
            # 存储指定时间点的解
            for pt in plot_times:
                if abs(t - pt) < dt/2 and pt not in result['times']:
                    result['times'].append(t)
                    result['solutions'].append(u.copy())
        
        end_time = time.time()
        result['time_elapsed'] = end_time - start_time
        
        return result
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数
        r = self.alpha * dt / (self.dx**2)
        
        # 构建左端矩阵 A（内部节点）
        main_diag = np.ones(self.nx - 2) * (1 + r)
        off_diag = np.ones(self.nx - 3) * (-r/2)
        
        # 创建带状矩阵表示
        A = np.zeros((3, self.nx - 2))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag  # 主对角线
        A[2, :-1] = off_diag  # 下对角线
        
        # 初始化解数组
        u = self.u_initial.copy()
        t = 0.0
        
        # 结果存储字典
        result = {
            'times': [],
            'solutions': [],
            'method': 'Crank-Nicolson',
            'dt': dt,
            'r': r,
            'time_elapsed': 0.0
        }
        
        # 存储初始条件
        if 0 in plot_times:
            result['times'].append(t)
            result['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            if t + dt > self.T_final:
                dt = self.T_final - t
                r = self.alpha * dt / (self.dx**2)
            
            # 构建右端向量
            rhs = np.zeros(self.nx - 2)
            rhs = (r/2) * u[:-2] + (1 - r) * u[1:-1] + (r/2) * u[2:]
            
            # 求解线性系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            t += dt
            
            # 存储指定时间点的解
            for pt in plot_times:
                if abs(t - pt) < dt/2 and pt not in result['times']:
                    result['times'].append(t)
                    result['solutions'].append(u.copy())
        
        end_time = time.time()
        result['time_elapsed'] = end_time - start_time
        
        return result
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        
        参数:
            t (float): 当前时间
            u_internal (np.ndarray): 内部节点温度
            
        返回:
            np.ndarray: 内部节点的时间导数
        """
        # 重构包含边界条件的完整解
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        
        # 计算空间二阶导数
        d2u = laplace(u_full, mode='nearest') / (self.dx**2)
        
        # 返回内部节点的时间导数
        return self.alpha * d2u[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        
        参数:
            method (str): 积分方法（'RK45', 'BDF', 'Radau'等）
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 提取内部节点初始条件
        u0_internal = self.u_initial[1:-1].copy()
        
        # 结果存储字典
        result = {
            'times': [],
            'solutions': [],
            'method': f'solve_ivp ({method})',
            'dt': 'adaptive',
            'r': 'adaptive',
            'time_elapsed': 0.0
        }
        
        start_time = time.time()
        
        # 调用 solve_ivp 求解
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times
        )
        
        end_time = time.time()
        result['time_elapsed'] = end_time - start_time
        
        # 重构包含边界条件的完整解
        for i, t in enumerate(sol.t):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            result['times'].append(t)
            result['solutions'].append(u_full)
        
        return result
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        
        参数:
            dt_explicit (float): 显式方法时间步长
            dt_implicit (float): 隐式方法时间步长
            dt_cn (float): Crank-Nicolson方法时间步长
            ivp_method (str): solve_ivp积分方法
            plot_times (list): 比较时间点
            
        返回:
            dict: 所有方法的结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        print(f"\n{'='*50}")
        print("热传导方程数值解法比较")
        print(f"空间点数: {self.nx}, 最终时间: {self.T_final}")
        print(f"显式方法时间步长: {dt_explicit}")
        print(f"隐式方法时间步长: {dt_implicit}")
        print(f"Crank-Nicolson方法时间步长: {dt_cn}")
        print(f"solve_ivp方法: {ivp_method}")
        print(f"比较时间点: {plot_times}")
        print('='*50)
        
        # 调用四种求解方法
        results = {}
        results['explicit'] = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)
        results['implicit'] = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)
        results['crank_nicolson'] = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)
        results['solve_ivp'] = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        
        # 打印每种方法的计算时间和稳定性参数
        print("\n方法性能比较:")
        print("-"*70)
        print(f"{'方法':<25} | {'时间步长':<10} | {'稳定性参数(r)':<15} | {'计算时间(秒)':<15}")
        print("-"*70)
        
        for key, res in results.items():
            method_name = res['method']
            dt_val = str(res['dt'])
            r_val = str(res['r'])
            time_val = f"{res['time_elapsed']:.6f}"
            print(f"{method_name:<25} | {dt_val:<10} | {r_val:<15} | {time_val:<15}")
        
        print('='*50)
        
        return results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
        """
        plt.figure(figsize=(15, 10))
        
        # 获取所有时间点（以第一个方法为准）
        times = list(methods_results.values())[0]['times']
        
        # 创建2x2子图
        for i, t in enumerate(times):
            plt.subplot(2, 2, i+1)
            
            # 绘制每种方法在该时间点的解
            for method_name, result in methods_results.items():
                # 找到最接近的时间索引
                idx = np.argmin(np.abs(np.array(result['times']) - t))
                u = result['solutions'][idx]
                plt.plot(self.x, u, label=f"{result['method']} (t={result['times'][idx]:.2f})")
            
            plt.title(f"时间 t = {t:.2f}")
            plt.xlabel('位置 x')
            plt.ylabel('温度 u')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(filename, dpi=300)
            print(f"图像已保存为 {filename}")
        
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        
        参数:
            methods_results (dict): compare_methods的结果
            reference_method (str): 参考方法
            
        返回:
            dict: 精度分析结果
        """
        # 验证参考方法存在
        if reference_method not in methods_results:
            print(f"错误: 参考方法 '{reference_method}' 不存在!")
            return None
        
        # 获取参考解
        ref_result = methods_results[reference_method]
        
        # 精度分析结果字典
        accuracy = {}
        
        print("\n精度分析 (以 {} 为参考):".format(ref_result['method']))
        print("-"*70)
        print(f"{'方法':<25} | {'最大误差':<15} | {'平均绝对误差':<15} | {'均方根误差':<15}")
        print("-"*70)
        
        for key, result in methods_results.items():
            if key == reference_method:
                continue  # 跳过参考方法本身
                
            max_errors = []
            mae_errors = []
            rmse_errors = []
            
            # 对每个时间点计算误差
            for t in result['times']:
                # 找到参考解中最接近的时间点
                ref_idx = np.argmin(np.abs(np.array(ref_result['times']) - t))
                u_ref = ref_result['solutions'][ref_idx]
                
                # 找到当前方法中对应时间点的解
                idx = np.argmin(np.abs(np.array(result['times']) - t))
                u = result['solutions'][idx]
                
                # 计算误差指标
                error = np.abs(u - u_ref)
                max_errors.append(np.max(error))
                mae_errors.append(np.mean(error))
                rmse_errors.append(np.sqrt(np.mean(error**2)))
            
            # 计算平均误差
            avg_max_error = np.mean(max_errors)
            avg_mae = np.mean(mae_errors)
            avg_rmse = np.mean(rmse_errors)
            
            # 存储结果
            accuracy[key] = {
                'max_error': avg_max_error,
                'mae': avg_mae,
                'rmse': avg_rmse
            }
            
            print(f"{result['method']:<25} | {avg_max_error:<15.6f} | {avg_mae:<15.6f} | {avg_rmse:<15.6f}")
        
        print('='*50)
        return accuracy


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=201, T_final=25.0)
    
    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.001,  # 显式方法需要较小的时间步长
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF',
        plot_times=[0, 1, 5, 15, 25]
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度（以solve_ivp为参考）
    accuracy = solver.analyze_accuracy(results, reference_method='solve_ivp')
    
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
