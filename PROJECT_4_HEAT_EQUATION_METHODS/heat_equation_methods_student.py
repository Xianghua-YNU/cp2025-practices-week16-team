import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    A comprehensive solver for the 1D heat equation using multiple numerical methods.
    
    The heat equation: du/dt = alpha * d²u/dx²
    Boundary conditions: u(0,t) = 0, u(L,t) = 0
    Initial condition: u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        Initialize the heat equation solver.
        
        Args:
            L (float): Domain length [0, L]
            alpha (float): Thermal diffusivity coefficient
            nx (int): Number of spatial grid points
            T_final (float): Final simulation time
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # Spatial grid
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # Initialize solution array
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        Set the initial condition: u(x,0) = 1 for 10 <= x <= 11, 0 otherwise.
        
        Returns:
            np.ndarray: Initial temperature distribution
        """
        u0 = np.zeros(self.nx)
        mask = (self.x >= 10) & (self.x <= 11)
        u0[mask] = 1.0
        # Apply boundary conditions
        u0[0] = 0.0
        u0[-1] = 0.0
        return u0
        
def solve_explicit(self, dt=0.01, plot_times=None):
        """
        Solve using explicit finite difference method (FTCS).
        
        Args:
            dt (float): Time step size
            plot_times (list): Time points for plotting
            
        Returns:
            dict: Solution data including time points and temperature arrays
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # Stability check
        r = self.alpha * dt / (self.dx**2)
        if r > 0.5:
            print(f"Warning: Stability condition violated! r = {r:.4f} > 0.5")
            print(f"Consider reducing dt to < {0.5 * self.dx**2 / self.alpha:.6f}")
        
        # Initialize
        u = self.u_initial.copy()
        t = 0.0
        nt = int(self.T_final / dt) + 1
        
        # Storage for results
        results = {'times': [], 'solutions': [], 'method': 'Explicit FTCS'}
        
        # Store initial condition
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # Time stepping
        for n in range(1, nt):
            # Apply Laplacian using scipy.ndimage.laplace
            du_dt = r * laplace(u)
            u += du_dt
            
            # Apply boundary conditions
            u[0] = 0.0
            u[-1] = 0.0
            
            t = n * dt
            
            # Store solution at specified times
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        Solve using implicit finite difference method (BTCS).
        
        Args:
            dt (float): Time step size
            plot_times (list): Time points for plotting
            
        Returns:
            dict: Solution data including time points and temperature arrays
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # Parameters
        r = self.alpha * dt / (self.dx**2)
        nt = int(self.T_final / dt) + 1
        
        # Initialize
        u = self.u_initial.copy()
        
        # Build tridiagonal matrix for internal nodes
        num_internal = self.nx - 2
        banded_matrix = np.zeros((3, num_internal))
        banded_matrix[0, 1:] = -r  # Upper diagonal
        banded_matrix[1, :] = 1 + 2*r  # Main diagonal
        banded_matrix[2, :-1] = -r  # Lower diagonal
        
        # Storage for results
        results = {'times': [], 'solutions': [], 'method': 'Implicit BTCS'}
        
        # Store initial condition
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # Time stepping
        for n in range(1, nt):
            # Right-hand side (internal nodes only)
            rhs = u[1:-1].copy()
            
            # Solve tridiagonal system
            u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix, rhs)
            
            # Update solution
            u[1:-1] = u_internal_new
            u[0] = 0.0  # Boundary conditions
            u[-1] = 0.0
            
            t = n * dt
            
            # Store solution at specified times
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results

def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        Solve using Crank-Nicolson method.
        
        Args:
            dt (float): Time step size
            plot_times (list): Time points for plotting
            
        Returns:
            dict: Solution data including time points and temperature arrays
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # Parameters
        r = self.alpha * dt / (self.dx**2)
        nt = int(self.T_final / dt) + 1
        
        # Initialize
        u = self.u_initial.copy()
        
        # Build coefficient matrices for internal nodes
        num_internal = self.nx - 2
        
        # Left-hand side matrix A
        banded_matrix_A = np.zeros((3, num_internal))
        banded_matrix_A[0, 1:] = -r/2  # Upper diagonal
        banded_matrix_A[1, :] = 1 + r  # Main diagonal
        banded_matrix_A[2, :-1] = -r/2  # Lower diagonal
        
        # Storage for results
        results = {'times': [], 'solutions': [], 'method': 'Crank-Nicolson'}
        
        # Store initial condition
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # Time stepping
        for n in range(1, nt):
            # Right-hand side vector
            u_internal = u[1:-1]
            rhs = (r/2) * u[:-2] + (1 - r) * u_internal + (r/2) * u[2:]
            
            # Solve tridiagonal system A * u^{n+1} = rhs
            u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix_A, rhs)
            
            # Update solution
            u[1:-1] = u_internal_new
            u[0] = 0.0  # Boundary conditions
            u[-1] = 0.0
            
            t = n * dt
            
            # Store solution at specified times
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results
    
    def _heat_equation_ode(self, t, u_internal):
        """
        ODE system for solve_ivp method.
        
        Args:
            t (float): Current time
            u_internal (np.ndarray): Internal node temperatures
            
        Returns:
            np.ndarray: Time derivatives for internal nodes
        """
        # Reconstruct full solution with boundary conditions
        u_full = np.concatenate(([0.0], u_internal, [0.0]))
        
        # Compute second derivative using Laplacian
        d2u_dx2 = laplace(u_full) / (self.dx**2)
        
        # Return derivatives for internal nodes only
        return self.alpha * d2u_dx2[1:-1]
def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        Solve using scipy.integrate.solve_ivp.
        
        Args:
            method (str): Integration method ('RK45', 'BDF', 'Radau', etc.)
            plot_times (list): Time points for plotting
            
        Returns:
            dict: Solution data including time points and temperature arrays
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # Initial condition for internal nodes only
        u0_internal = self.u_initial[1:-1]
        
        start_time = time.time()
        
        # Solve ODE system
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times,
            rtol=1e-8,
            atol=1e-10
        )
        
        computation_time = time.time() - start_time
        
        # Reconstruct full solutions with boundary conditions
        results = {
            'times': sol.t.tolist(),
            'solutions': [],
            'method': f'solve_ivp ({method})',
            'computation_time': computation_time
        }
        
        for i in range(len(sol.t)):
            u_full = np.concatenate(([0.0], sol.y[:, i], [0.0]))
            results['solutions'].append(u_full)
        
        return results
def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        Compare all four numerical methods.
        
        Args:
            dt_explicit (float): Time step for explicit method
            dt_implicit (float): Time step for implicit method
            dt_cn (float): Time step for Crank-Nicolson method
            ivp_method (str): Integration method for solve_ivp
            plot_times (list): Time points for comparison
            
        Returns:
            dict: Results from all methods
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        print("Solving heat equation using four different methods...")
        print(f"Domain: [0, {self.L}], Grid points: {self.nx}, Final time: {self.T_final}")
        print(f"Thermal diffusivity: {self.alpha}")
        print("-" * 60)
        
        # Solve with all methods
        methods_results = {}
        
        # Explicit method
        print("1. Explicit finite difference (FTCS)...")
        methods_results['explicit'] = self.solve_explicit(dt_explicit, plot_times)
        print(f"   Computation time: {methods_results['explicit']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['explicit']['stability_parameter']:.4f}")
        
        # Implicit method
        print("2. Implicit finite difference (BTCS)...")
        methods_results['implicit'] = self.solve_implicit(dt_implicit, plot_times)
        print(f"   Computation time: {methods_results['implicit']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['implicit']['stability_parameter']:.4f}")
        
        # Crank-Nicolson method
        print("3. Crank-Nicolson method...")
        methods_results['crank_nicolson'] = self.solve_crank_nicolson(dt_cn, plot_times)
        print(f"   Computation time: {methods_results['crank_nicolson']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['crank_nicolson']['stability_parameter']:.4f}")
        
        # solve_ivp method
        print(f"4. solve_ivp method ({ivp_method})...")
        methods_results['solve_ivp'] = self.solve_with_solve_ivp(ivp_method, plot_times)
        print(f"   Computation time: {methods_results['solve_ivp']['computation_time']:.4f} s")
        
        print("-" * 60)
        print("All methods completed successfully!")
        
        return methods_results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        Plot comparison of all methods.
        
        Args:
            methods_results (dict): Results from compare_methods
            save_figure (bool): Whether to save the figure
            filename (str): Filename for saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        method_names = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, method_name in enumerate(method_names):
            ax = axes[idx]
            results = methods_results[method_name]
            
            # Plot solutions at different times
            for i, (t, u) in enumerate(zip(results['times'], results['solutions'])):
                ax.plot(self.x, u, color=colors[i], label=f't = {t:.1f}', linewidth=2)
            
            ax.set_title(f"{results['method']}\n(Time: {results['computation_time']:.4f} s)")
            ax.set_xlabel('Position x')
            ax.set_ylabel('Temperature u(x,t)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, self.L)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {filename}")
        
        plt.show()

 def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        Analyze the accuracy of different methods.
        
        Args:
            methods_results (dict): Results from compare_methods
            reference_method (str): Method to use as reference
            
        Returns:
            dict: Accuracy analysis results
        """
        if reference_method not in methods_results:
            raise ValueError(f"Reference method '{reference_method}' not found in results")
        
        reference = methods_results[reference_method]
        accuracy_results = {}
        
        print(f"\nAccuracy Analysis (Reference: {reference['method']})")
        print("-" * 50)
        
        for method_name, results in methods_results.items():
            if method_name == reference_method:
                continue
                
            errors = []
            # Align solutions by time points (find closest match)
            for ref_t, ref_u in zip(reference['times'], reference['solutions']):
                # Find closest solution in test method
                idx = np.argmin(np.abs(np.array(results['times']) - ref_t))
                test_u = results['solutions'][idx]
                error = np.linalg.norm(ref_u - test_u, ord=2)
                errors.append(error)
            
            max_error = max(errors) if errors else 0
            avg_error = np.mean(errors) if errors else 0
            
            accuracy_results[method_name] = {
                'max_error': max_error,
                'avg_error': avg_error,
                'errors': errors
            }
            
            print(f"{results['method']:25} - Max Error: {max_error:.2e}, Avg Error: {avg_error:.2e}")
        
        return accuracy_results


def main():
    """
    Demonstration of the HeatEquationSolver class.
    """
    # Create solver instance
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=21, T_final=25.0)
    
    # Compare all methods
    plot_times = [0, 1, 5, 15, 25]
    results = solver.compare_methods(
        dt_explicit=0.01,
        dt_implicit=0.1, 
        dt_cn=0.5,
        ivp_method='BDF',
        plot_times=plot_times
    )
    
    # Plot comparison
    solver.plot_comparison(results, save_figure=True)
    
    # Analyze accuracy
    accuracy = solver.analyze_accuracy(results, reference_method='solve_ivp')
    
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()


