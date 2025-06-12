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
        # 创建零数组
        u = np.zeros(self.nx)
        
        # 设置初始条件（10 <= x <= 11 区域为1）
        mask = (self.x >= 10.0) & (self.x <= 11.0)
        u[mask] = 1.0
        
        # 应用边界条件（已经是0，无需额外操作）
        return u
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算稳定性参数 r = alpha * dt / dx²
        r = self.alpha * dt / (self.dx ** 2)
        
        # 检查稳定性条件 r <= 0.5
        if r > 0.5:
            print(f"警告: 显式方法不稳定 (r = {r:.4f} > 0.5)")
        
        # 初始化解数组和时间变量
        u = self.u_initial.copy()
        t = 0.0
        
        # 创建结果存储字典
        results = {'times': [], 'solutions': []}
        
        # 存储初始条件
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 时间步进循环
        while t < self.T_final:
            # 使用 laplace(u) 计算空间二阶导数
            u_laplacian = laplace(u, mode='constant', cval=0.0)
            
            # 更新解：u += r * laplace(u)
            u += r * u_laplacian
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            # 更新时间
            t += dt
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        
        # 构建三对角矩阵（内部节点）
        n = self.nx - 2  # 内部节点数
        diagonal = np.ones(n) * (1 + 2 * r)
        sub_diagonal = np.ones(n-1) * (-r)
        super_diagonal = np.ones(n-1) * (-r)
        
        # 组合为带状矩阵格式 [上对角线, 主对角线, 下对角线]
        ab = np.zeros((3, n))
        ab[0, 1:] = super_diagonal
        ab[1, :] = diagonal
        ab[2, :-1] = sub_diagonal
        
        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': []}
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 时间步进循环
        while t < self.T_final:
            # 构建右端项（内部节点）
            rhs = u[1:-1].copy()
            
            # 使用 scipy.linalg.solve_banded 求解
            u_internal = scipy.linalg.solve_banded((1, 1), ab, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            # 更新时间
            t += dt
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        return results
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        
        # 构建左端矩阵 A（内部节点）
        n = self.nx - 2  # 内部节点数
        diagonal_A = np.ones(n) * (1 + r)
        sub_diagonal_A = np.ones(n-1) * (-r/2)
        super_diagonal_A = np.ones(n-1) * (-r/2)
        
        # 组合为带状矩阵格式
        ab_A = np.zeros((3, n))
        ab_A[0, 1:] = super_diagonal_A
        ab_A[1, :] = diagonal_A
        ab_A[2, :-1] = sub_diagonal_A
        
        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': []}
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 时间步进循环
        while t < self.T_final:
            # 构建右端向量：(r/2)*u[:-2] + (1-r)*u[1:-1] + (r/2)*u[2:]
            rhs = (r/2) * u[:-2] + (1-r) * u[1:-1] + (r/2) * u[2:]
            
            # 求解线性系统
            u_internal = scipy.linalg.solve_banded((1, 1), ab_A, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            # 更新时间
            t += dt
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        return results
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        """
        # 重构完整解向量（包含边界条件）
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        
        # 使用 laplace(u_full) / dx² 计算二阶导数
        u_laplacian = laplace(u_full, mode='constant', cval=0.0) / (self.dx ** 2)
        
        # 返回内部节点的时间导数：alpha * d²u/dx²
        return self.alpha * u_laplacian[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 提取内部节点初始条件
        u0 = self.u_initial[1:-1]
        
        # 调用 solve_ivp 求解
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0,
            method=method,
            t_eval=plot_times
        )
        
        # 重构包含边界条件的完整解
        results = {'times': sol.t, 'solutions': []}
        for i in range(len(sol.t)):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            results['solutions'].append(u_full)
        
        return results
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 打印求解信息
        print("="*50)
        print(f"热传导方程求解器配置")
        print(f"空间域长度 L = {self.L}")
        print(f"热扩散系数 alpha = {self.alpha}")
        print(f"空间网格点数 nx = {self.nx}")
        print(f"最终时间 T_final = {self.T_final}")
        print("="*50)
        
        # 计算稳定性参数
        r_explicit = self.alpha * dt_explicit / (self.dx ** 2)
        r_implicit = self.alpha * dt_implicit / (self.dx ** 2)
        r_cn = self.alpha * dt_cn / (self.dx ** 2)
        
        # 调用四种求解方法
        print("开始求解...")
        
        # 显式方法
        start_time = time.time()
        explicit_results = self.solve_explicit(dt_explicit, plot_times)
        explicit_time = time.time() - start_time
        
        # 隐式方法
        start_time = time.time()
        implicit_results = self.solve_implicit(dt_implicit, plot_times)
        implicit_time = time.time() - start_time
        
        # Crank-Nicolson方法
        start_time = time.time()
        cn_results = self.solve_crank_nicolson(dt_cn, plot_times)
        cn_time = time.time() - start_time
        
        # solve_ivp方法
        start_time = time.time()
        ivp_results = self.solve_with_solve_ivp(ivp_method, plot_times)
        ivp_time = time.time() - start_time
        
        # 打印每种方法的计算时间和稳定性参数
        print("\n计算结果比较:")
        print(f"显式方法 (r = {r_explicit:.4f}): {explicit_time:.4f} 秒")
        print(f"隐式方法 (r = {r_implicit:.4f}): {implicit_time:.4f} 秒")
        print(f"Crank-Nicolson方法 (r = {r_cn:.4f}): {cn_time:.4f} 秒")
        print(f"solve_ivp方法 ({ivp_method}): {ivp_time:.4f} 秒")
        
        # 返回所有结果的字典
        return {
            'explicit': {'results': explicit_results, 'time': explicit_time, 'r': r_explicit},
            'implicit': {'results': implicit_results, 'time': implicit_time, 'r': r_implicit},
            'crank_nicolson': {'results': cn_results, 'time': cn_time, 'r': r_cn},
            'solve_ivp': {'results': ivp_results, 'time': ivp_time, 'method': ivp_method}
        }
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        """
        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # 方法名称和标签
        method_names = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        method_labels = ['显式方法', '隐式方法', 'Crank-Nicolson方法', 'solve_ivp方法']
        
        # 为每种方法绘制解曲线
        for i, method in enumerate(method_names):
            ax = axes[i]
            results = methods_results[method]['results']
            
            for j, t in enumerate(results['times']):
                ax.plot(self.x, results['solutions'][j], label=f't = {t:.1f}')
            
            ax.set_title(method_labels[i])
            ax.set_xlabel('x')
            ax.set_ylabel('温度 u(x,t)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        
        # 可选保存图像
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        """
        # 验证参考方法存在
        if reference_method not in methods_results:
            print(f"错误: 参考方法 '{reference_method}' 不存在")
            return None
        
        # 获取参考解
        ref_results = methods_results[reference_method]['results']
        ref_times = ref_results['times']
        ref_solutions = ref_results['solutions']
        
        # 初始化误差分析结果
        accuracy_results = {}
        
        # 对每种方法计算误差
        for method, data in methods_results.items():
            if method == reference_method:
                continue
                
            results = data['results']
            errors = []
            
            # 对每个时间点计算误差
            for i, t in enumerate(results['times']):
                # 找到最接近的参考时间点
                idx = np.argmin(np.abs(np.array(ref_times) - t))
                ref_sol = ref_solutions[idx]
                curr_sol = results['solutions'][i]
                
                # 计算误差
                error = np.abs(curr_sol - ref_sol)
                errors.append(error)
            
            # 统计误差指标
            max_errors = [np.max(e) for e in errors]
            avg_errors = [np.mean(e) for e in errors]
            
            # 存储结果
            accuracy_results[method] = {
                'max_errors': max_errors,
                'avg_errors': avg_errors,
                'times': results['times']
            }
        
        # 打印精度分析结果
        print("\n精度分析:")
        print(f"参考方法: {reference_method}")
        
        for method, data in accuracy_results.items():
            max_max_error = np.max(data['max_errors'])
            max_avg_error = np.max(data['avg_errors'])
            print(f"{method}: 最大误差 = {max_max_error:.6e}, 平均误差 = {max_avg_error:.6e}")
        
        return accuracy_results
    
    def plot_accuracy_comparison(self, accuracy_results, reference_method='solve_ivp'):
        """
        绘制不同方法的精度比较图。
        """
        # 创建2x1子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 方法名称和标签
        method_names = list(accuracy_results.keys())
        method_labels = ['显式方法', '隐式方法', 'Crank-Nicolson方法']
        
        # 绘制最大误差
        for i, method in enumerate(method_names):
            data = accuracy_results[method]
            ax1.plot(data['times'], data['max_errors'], 'o-', label=method_labels[i])
        
        ax1.set_title('最大误差随时间变化')
        ax1.set_xlabel('时间 t')
        ax1.set_ylabel('最大误差')
        ax1.set_yscale('log')  # 使用对数刻度以便更好地比较
        ax1.grid(True)
        ax1.legend()
        
        # 绘制平均误差
        for i, method in enumerate(method_names):
            data = accuracy_results[method]
            ax2.plot(data['times'], data['avg_errors'], 's-', label=method_labels[i])
        
        ax2.set_title('平均误差随时间变化')
        ax2.set_xlabel('时间 t')
        ax2.set_ylabel('平均误差')
        ax2.set_yscale('log')  # 使用对数刻度以便更好地比较
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=201, T_final=25.0)
    
    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.0001,  # 较小的时间步长以确保稳定性
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF'
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    
    # 绘制精度比较图
    if accuracy:
        solver.plot_accuracy_comparison(accuracy)
    
    # 返回结果
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
