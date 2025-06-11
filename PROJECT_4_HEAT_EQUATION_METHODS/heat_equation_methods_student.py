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
        r = self.alpha * dt / (self.dx ** 2)
        if r > 0.5:
            print(f"警告: 显式方法可能不稳定 (r={r:.3f} > 0.5)")

        # 初始化解数组和时间变量
        u = self.u_initial.copy()
        t = 0.0
        current_time = 0.0

        # 创建结果存储字典
        results = {
            'method': 'Explicit FTCS',
            'times': [],
            'solutions': [],
            'r': r,
            'dt': dt,
            'execution_time': 0.0
        }

        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(t)
            results['solutions'].append(u.copy())

        # 记录开始时间
        start_time = time.time()

        # 时间步进循环
        while current_time < self.T_final:
            # 确保不会超过最终时间
            if current_time + dt > self.T_final:
                dt = self.T_final - current_time
                r = self.alpha * dt / (self.dx ** 2)

            # 使用laplace算子计算空间二阶导数
            lap = laplace(u, mode='constant', cval=0.0)  # 边界条件已固定为0
            # 更新解
            u += r * lap

            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0

            # 更新时间
            current_time += dt
            t += dt

            # 在指定时间点存储解
            if any(abs(current_time - pt) < dt / 2 for pt in plot_times):
                results['times'].append(current_time)
                results['solutions'].append(u.copy())

        # 记录执行时间
        results['execution_time'] = time.time() - start_time

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

        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)

        # 构建三对角矩阵（内部节点）
        n = self.nx - 2  # 内部节点数
        main_diag = (1 + 2 * r) * np.ones(n)
        off_diag = -r * np.ones(n - 1)

        # 创建带状矩阵
        A = np.zeros((3, n))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag  # 主对角线
        A[2, :-1] = off_diag  # 下对角线

        # 初始化解数组
        u = self.u_initial.copy()
        current_time = 0.0

        # 创建结果存储
        results = {
            'method': 'Implicit BTCS',
            'times': [],
            'solutions': [],
            'r': r,
            'dt': dt,
            'execution_time': 0.0
        }

        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(current_time)
            results['solutions'].append(u.copy())

        # 记录开始时间
        start_time = time.time()

        # 时间步进循环
        while current_time < self.T_final:
            # 确保不会超过最终时间
            if current_time + dt > self.T_final:
                dt = self.T_final - current_time
                r = self.alpha * dt / (self.dx ** 2)

                # 重新构建矩阵
                main_diag = (1 + 2 * r) * np.ones(n)
                off_diag = -r * np.ones(n - 1)
                A = np.zeros((3, n))
                A[0, 1:] = off_diag
                A[1, :] = main_diag
                A[2, :-1] = off_diag

            # 构建右端项（内部节点）
            b = u[1:-1].copy()

            # 求解三对角系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, b)

            # 更新解
            u[1:-1] = u_internal
            # 边界条件已隐含保持

            # 更新时间
            current_time += dt

            # 在指定时间点存储解
            if any(abs(current_time - pt) < dt / 2 for pt in plot_times):
                results['times'].append(current_time)
                results['solutions'].append(u.copy())

        # 记录执行时间
        results['execution_time'] = time.time() - start_time

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

        # 计算扩散数 r
        r = self.alpha * dt / (2 * self.dx ** 2)

        # 构建左端矩阵 A（内部节点）
        n = self.nx - 2  # 内部节点数
        main_diag = (1 + 2 * r) * np.ones(n)
        off_diag = -r * np.ones(n - 1)

        # 创建带状矩阵
        A = np.zeros((3, n))
        A[0, 1:] = off_diag  # 上对角线
        A[1, :] = main_diag  # 主对角线
        A[2, :-1] = off_diag  # 下对角线

        # 初始化解数组
        u = self.u_initial.copy()
        current_time = 0.0

        # 创建结果存储
        results = {
            'method': 'Crank-Nicolson',
            'times': [],
            'solutions': [],
            'r': r,
            'dt': dt,
            'execution_time': 0.0
        }

        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(current_time)
            results['solutions'].append(u.copy())

        # 记录开始时间
        start_time = time.time()

        # 时间步进循环
        while current_time < self.T_final:
            # 确保不会超过最终时间
            if current_time + dt > self.T_final:
                dt = self.T_final - current_time
                r = self.alpha * dt / (2 * self.dx ** 2)

                # 重新构建矩阵
                main_diag = (1 + 2 * r) * np.ones(n)
                off_diag = -r * np.ones(n - 1)
                A = np.zeros((3, n))
                A[0, 1:] = off_diag
                A[1, :] = main_diag
                A[2, :-1] = off_diag

            # 构建右端向量
            rhs = np.zeros(n)
            rhs[1:-1] = r * u[0:-4] + (1 - 2 * r) * u[1:-3] + r * u[2:-2]

            # 第一个内部节点
            rhs[0] = r * u[0] + (1 - 2 * r) * u[1] + r * u[2]
            # 最后一个内部节点
            rhs[-1] = r * u[-3] + (1 - 2 * r) * u[-2] + r * u[-1]

            # 求解线性系统
            u_internal = scipy.linalg.solve_banded((1, 1), A, rhs)

            # 更新解
            u[1:-1] = u_internal
            # 边界条件已隐含保持

            # 更新时间
            current_time += dt

            # 在指定时间点存储解
            if any(abs(current_time - pt) < dt / 2 for pt in plot_times):
                results['times'].append(current_time)
                results['solutions'].append(u.copy())

        # 记录执行时间
        results['execution_time'] = time.time() - start_time

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
        # 重构包含边界条件的完整解
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal

        # 计算二阶导数 (使用laplace算子)
        d2u_dx2 = laplace(u_full, mode='constant', cval=0.0) / (self.dx ** 2)

        # 返回内部节点的时间导数
        return self.alpha * d2u_dx2[1:-1]

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

        # 确保最终时间在绘图时间点中
        if self.T_final not in plot_times:
            plot_times.append(self.T_final)

        # 提取内部节点初始条件
        u0_internal = self.u_initial[1:-1]

        # 记录开始时间
        start_time = time.time()

        # 调用 solve_ivp 求解
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times
        )

        # 记录执行时间
        exec_time = time.time() - start_time

        # 重构包含边界条件的完整解
        solutions = []
        for i in range(sol.y.shape[1]):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            solutions.append(u_full)

        # 返回结果字典
        return {
            'method': f'solve_ivp ({method})',
            'times': sol.t,
            'solutions': solutions,
            'r': None,
            'dt': 'adaptive',
            'execution_time': exec_time
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

        # 打印求解信息
        print(f"\n{'=' * 50}")
        print(f"求解热传导方程: L={self.L}, alpha={self.alpha}")
        print(f"空间网格点数: {self.nx}, 最终时间: {self.T_final}")
        print(f"时间步长设置: 显式={dt_explicit}, 隐式={dt_implicit}, C-N={dt_cn}")
        print(f"ODE求解器方法: {ivp_method}")
        print(f"比较时间点: {plot_times}")
        print(f"{'=' * 50}\n")

        # 调用四种求解方法
        results = {}

        print("使用显式方法求解...")
        results['explicit'] = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)

        print("使用隐式方法求解...")
        results['implicit'] = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)

        print("使用Crank-Nicolson方法求解...")
        results['crank_nicolson'] = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)

        print(f"使用solve_ivp ({ivp_method})求解...")
        results['solve_ivp'] = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)

        # 打印每种方法的计算时间和稳定性参数
        # 在 compare_methods 方法中
        print("\n方法比较结果:")
        print("-" * 85)
        print(f"{'方法':<25} | {'时间步长':<10} | {'r参数':<10} | {'执行时间(s)':<12} | {'状态':<10}")
        print("-" * 85)

        for key, res in results.items():
            method_name = res['method']
            dt_val = res['dt']
            r_val = res['r'] if res['r'] is not None else 'N/A'
            exec_time = res['execution_time']

            # 格式化 dt_val 和 r_val
            dt_str = f"{dt_val:.4f}" if isinstance(dt_val, float) else str(dt_val)
            r_str = f"{r_val:.4f}" if isinstance(r_val, float) else str(r_val)

            # 检查显式方法的稳定性
            status = "稳定"
            if 'Explicit' in method_name and isinstance(r_val, float) and r_val > 0.5:
                status = "可能不稳定!"

            print(f"{method_name:<25} | {dt_str:<10} | {r_str:<10} | {exec_time:12.4f} | {status:<10}")

        print("-" * 85)
        return results

    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。

        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
        """
        # 获取时间点（使用第一个方法的时间点）
        times = methods_results[list(methods_results.keys())[0]]['times']

        # 创建2x2子图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs = axs.flatten()

        # 为每种方法绘制解曲线
        for i, (method, result) in enumerate(methods_results.items()):
            ax = axs[i]

            # 对于每个时间点，绘制温度分布
            for j, t in enumerate(result['times']):
                # 只绘制指定的时间点
                if t in times:
                    ax.plot(self.x, result['solutions'][j], label=f't={t}s')

            ax.set_title(result['method'])
            ax.set_xlabel('位置 (x)')
            ax.set_ylabel('温度 (u)')
            ax.grid(True)
            ax.legend()
            ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()

        # 可选保存图像
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
            print(f"错误: 参考方法 '{reference_method}' 不存在!")
            return None

        # 获取参考解
        ref_results = methods_results[reference_method]
        ref_solutions = {t: sol for t, sol in zip(ref_results['times'], ref_results['solutions'])}

        # 存储分析结果
        accuracy_results = {}

        print("\n精度分析 (参考方法: {})".format(ref_results['method']))
        print("-" * 70)
        print(f"{'方法':<25} | {'时间点':<8} | {'最大误差':<12} | {'平均误差':<12}")
        print("-" * 70)

        # 计算各方法与参考解的误差
        for method, results in methods_results.items():
            if method == reference_method:
                continue  # 跳过参考方法本身

            method_name = results['method']
            errors = []
            max_errors = []
            avg_errors = []

            # 对于每个时间点
            for t, sol in zip(results['times'], results['solutions']):
                # 找到最接近的参考时间点
                ref_t = min(ref_solutions.keys(), key=lambda x: abs(x - t))
                ref_sol = ref_solutions[ref_t]

                # 计算误差
                error = np.abs(sol - ref_sol)
                max_error = np.max(error)
                avg_error = np.mean(error)

                errors.append(error)
                max_errors.append(max_error)
                avg_errors.append(avg_error)

                # 打印每个时间点的误差
                print(f"{method_name:<25} | {t:<8.2f} | {max_error:<12.6f} | {avg_error:<12.6f}")

            # 存储结果
            accuracy_results[method] = {
                'method': method_name,
                'max_errors': max_errors,
                'avg_errors': avg_errors,
                'overall_max_error': np.max(max_errors),
                'overall_avg_error': np.mean(avg_errors)
            }

        print("-" * 70)

        return accuracy_results


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=51, T_final=25.0)

    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.001,  # 显式方法需要小时间步
        dt_implicit=0.1,  # 隐式方法可以使用较大时间步
        dt_cn=0.5,  # C-N方法可以使用更大的时间步
        ivp_method='BDF',  # 使用BDF方法
        plot_times=[0, 1, 5, 15, 25]
    )

    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)

    # 分析精度 (以solve_ivp为参考)
    accuracy = solver.analyze_accuracy(results, reference_method='solve_ivp')

    # 返回结果
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()


def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5,
                    ivp_method='BDF', plot_times=None):
    """
    比较所有四种数值方法。
    """
    if plot_times is None:
        plot_times = [0, 1, 5, 15, 25]

    # 打印求解信息
    print(f"\n{'=' * 50}")
    print(f"求解热传导方程: L={self.L}, alpha={self.alpha}")
    print(f"空间网格点数: {self.nx}, 最终时间: {self.T_final}")
    print(f"时间步长设置: 显式={dt_explicit}, 隐式={dt_implicit}, C-N={dt_cn}")
    print(f"ODE求解器方法: {ivp_method}")
    print(f"比较时间点: {plot_times}")
    print(f"{'=' * 50}\n")

    # 调用四种求解方法
    results = {}

    print("使用显式方法求解...")
    results['explicit'] = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)

    print("使用隐式方法求解...")
    results['implicit'] = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)

    print("使用Crank-Nicolson方法求解...")
    results['crank_nicolson'] = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)

    print(f"使用solve_ivp ({ivp_method})求解...")
    results['solve_ivp'] = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)

    # 打印每种方法的计算时间和稳定性参数
    print("\n方法比较结果:")
    print("-" * 85)
    print(f"{'方法':<25} | {'时间步长':<10} | {'r参数':<10} | {'执行时间(s)':<12} | {'状态':<10}")
    print("-" * 85)

    for key, res in results.items():
        method_name = res['method']
        dt_val = res['dt']
        r_val = res['r'] if res['r'] is not None else 'N/A'
        exec_time = res['execution_time']

        # 格式化 dt_val 和 r_val
        dt_str = f"{dt_val:.4f}" if isinstance(dt_val, float) else str(dt_val)
        r_str = f"{r_val:.4f}" if isinstance(r_val, float) else str(r_val)

        # 检查显式方法的稳定性
        status = "稳定"
        if 'Explicit' in method_name and isinstance(r_val, float) and r_val > 0.5:
            status = "可能不稳定!"

        print(f"{method_name:<25} | {dt_str:<10} | {r_str:<10} | {exec_time:12.4f} | {status:<10}")

    print("-" * 85)

    return results
