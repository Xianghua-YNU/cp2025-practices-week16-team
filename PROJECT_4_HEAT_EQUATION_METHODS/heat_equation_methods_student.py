#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！
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
        mask = (self.x >= 10) & (self.x <= 11)
        u[mask] = 1.0
        # 确保边界条件
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
            print(f"警告: 显式方法不稳定 (r = {r:.4f} > 0.5)")
        
        # 初始化
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': []}
        
        # 存储初始条件
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 时间步进
        while t < self.T_final:
            # 应用显式更新
            u_new = u.copy()
            # 内部节点更新
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            
            # 应用边界条件
            u_new[0] = 0.0
            u_new[-1] = 0.0
            
            u = u_new
            t += dt
            
            # 检查是否到达绘图时间点
            for plot_time in plot_times:
                if t >= plot_time and plot_time not in results['times']:
                    results['times'].append(plot_time)
                    results['solutions'].append(u.copy())
        
        # 确保最终时间被记录
        if self.T_final not in results['times']:
            results['times'].append(self.T_final)
            results['solutions'].append(u.copy())
        
        return results
    
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
        n = self.nx - 2  # 内部节点数
        
        # 构建三对角矩阵
        main_diag = np.full(n, 1 + 2*r)
        off_diag = np.full(n-1, -r)
        
        # 转换为带状存储格式
        A = np.zeros((3, n))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag  # 主对角线
        A[2, :-1] = off_diag  # 下对角线
        
        # 初始化
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': []}
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 时间步进
        while t < self.T_final:
            # 右端项（内部节点）
            b = u[1:-1].copy()
            
            # 求解三对角系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, b)
            
            # 更新解
            u[1:-1] = u_internal
            t += dt
            
            # 存储结果
            for plot_time in plot_times:
                if t >= plot_time and plot_time not in results['times']:
                    results['times'].append(plot_time)
                    results['solutions'].append(u.copy())
        
        # 确保最终时间被记录
        if self.T_final not in results['times']:
            results['times'].append(self.T_final)
            results['solutions'].append(u.copy())
        
        return results
    
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
        r = self.alpha * dt / (2 * self.dx**2)
        n = self.nx - 2  # 内部节点数
        
        # 构建左端矩阵A（三对角）
        main_diag = np.full(n, 1 + 2*r)
        off_diag = np.full(n-1, -r)
        
        # 转换为带状存储格式
        A = np.zeros((3, n))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag  # 主对角线
        A[2, :-1] = off_diag  # 下对角线
        
        # 初始化
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': []}
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 时间步进
        while t < self.T_final:
            # 构建右端向量
            rhs = np.zeros(n)
            rhs[0] = r*u[0] + (1 - 2*r)*u[1] + r*u[2] + r*u[0]  # 包含左边界
            rhs[-1] = r*u[-3] + (1 - 2*r)*u[-2] + r*u[-1] + r*u[-1]  # 包含右边界
            
            # 内部节点
            if n > 2:
                rhs[1:-1] = r*u[1:-3] + (1 - 2*r)*u[2:-2] + r*u[3:-1]
            
            # 求解线性系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            t += dt
            
            # 存储结果
            for plot_time in plot_times:
                if t >= plot_time and plot_time not in results['times']:
                    results['times'].append(plot_time)
                    results['solutions'].append(u.copy())
        
        # 确保最终时间被记录
        if self.T_final not in results['times']:
            results['times'].append(self.T_final)
            results['solutions'].append(u.copy())
        
        return results
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        
        参数:
            t (float): 当前时间
            u_internal (np.ndarray): 内部节点温度
            
        返回:
            np.ndarray: 内部节点的时间导数
        """
        # 重构完整解向量（包含边界条件）
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        
        # 计算空间二阶导数
        d2u = laplace(u_full, mode='constant', cval=0.0) / (self.dx**2)
        
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
        
        # 调用solve_ivp
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times
        )
        
        # 重构包含边界条件的完整解
        solutions = []
        for i in range(sol.y.shape[1]):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            solutions.append(u_full)
        
        # 返回结果
        return {
            'times': sol.t,
            'solutions': solutions
        }
    
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
        
        print("="*60)
        print(f"热传导方程数值解法比较 (T_final={self.T_final}, nx={self.nx})")
        print("="*60)
        
        results = {}
        
        # 显式方法
        start = time.time()
        results['explicit'] = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)
        explicit_time = time.time() - start
        r_explicit = self.alpha * dt_explicit / (self.dx**2)
        print(f"显式方法: dt={dt_explicit:.4f}, r={r_explicit:.4f}, 时间={explicit_time:.4f}s")
        
        # 隐式方法
        start = time.time()
        results['implicit'] = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)
        implicit_time = time.time() - start
        r_implicit = self.alpha * dt_implicit / (self.dx**2)
        print(f"隐式方法: dt={dt_implicit:.4f}, r={r_implicit:.4f}, 时间={implicit_time:.4f}s")
        
        # Crank-Nicolson方法
        start = time.time()
        results['crank_nicolson'] = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)
        cn_time = time.time() - start
        r_cn = self.alpha * dt_cn / (self.dx**2)
        print(f"Crank-Nicolson: dt={dt_cn:.4f}, r={r_cn:.4f}, 时间={cn_time:.4f}s")
        
        # solve_ivp方法
        start = time.time()
        results['solve_ivp'] = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        ivp_time = time.time() - start
        print(f"solve_ivp({ivp_method}): 时间={ivp_time:.4f}s")
        
        return results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
        """
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('热传导方程数值解法比较', fontsize=16)
        
        # 为每种方法创建子图
        methods = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        titles = ['显式方法 (FTCS)', '隐式方法 (BTCS)', 'Crank-Nicolson', f'solve_ivp']
        
        for i, method in enumerate(methods):
            ax = axs[i//2, i%2]
            results = methods_results[method]
            
            for j, t in enumerate(results['times']):
                u = results['solutions'][j]
                ax.plot(self.x, u, label=f't={t}s')
            
            ax.set_title(titles[i])
            ax.set_xlabel('位置 (x)')
            ax.set_ylabel('温度 (u)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_figure:
            plt.savefig(filename, dpi=300)
            print(f"图像已保存为: {filename}")
        
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
            raise ValueError(f"参考方法 '{reference_method}' 不存在")
        
        # 获取参考解
        ref_results = methods_results[reference_method]
        ref_solutions = {}
        for i, t in enumerate(ref_results['times']):
            ref_solutions[t] = ref_results['solutions'][i]
        
        # 分析其他方法
        accuracy = {}
        methods = ['explicit', 'implicit', 'crank_nicolson']
        
        for method in methods:
            if method not in methods_results:
                continue
                
            results = methods_results[method]
            errors = []
            
            for i, t in enumerate(results['times']):
                if t in ref_solutions:
                    u_ref = ref_solutions[t]
                    u_method = results['solutions'][i]
                    # 计算绝对误差
                    error = np.abs(u_ref - u_method)
                    errors.append(error)
            
            if errors:
                # 计算平均误差和最大误差
                avg_error = np.mean([np.mean(e) for e in errors])
                max_error = np.max([np.max(e) for e in errors])
                
                accuracy[method] = {
                    'avg_error': avg_error,
                    'max_error': max_error,
                    'num_points': len(errors)
                }
        
        # 打印结果
        print("\n精度分析 (参考方法: solve_ivp):")
        print("-"*50)
        for method, data in accuracy.items():
            print(f"{method}:")
            print(f"  平均绝对误差: {data['avg_error']:.6f}")
            print(f"  最大绝对误差: {data['max_error']:.6f}")
            print(f"  比较点数: {data['num_points']}")
            print()
        
        return accuracy


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=101, T_final=25.0)
    
    # 比较所有方法
    plot_times = [0, 1, 5, 15, 25]
    results = solver.compare_methods(
        dt_explicit=0.001,
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF',
        plot_times=plot_times
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
