#!/usr/bin/env python3
"""
Heat Equation Solver with Multiple Numerical Methods
Alternative Implementation with Same Functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    Solver for the 1D heat equation using various numerical techniques
    """
    
    def __init__(self, domain_length=20.0, diffusivity=10.0, grid_points=21, end_time=25.0):
        """
        Initialize the heat equation solver
        
        Parameters:
            domain_length (float): Length of the spatial domain [0, L]
            diffusivity (float): Thermal diffusion coefficient
            grid_points (int): Number of spatial discretization points
            end_time (float): Simulation end time
        """
        self.domain_length = domain_length
        self.diffusivity = diffusivity
        self.grid_points = grid_points
        self.end_time = end_time
        
        # Create spatial grid
        self.spatial_grid = np.linspace(0, domain_length, grid_points)
        self.dx = domain_length / (grid_points - 1)
        
        # Set initial temperature distribution
        self.initial_temp = self._initialize_temp_profile()
    
    def _initialize_temp_profile(self):
        """
        Create the initial temperature distribution:
        u(x,0) = 1 for 10 <= x <= 11, 0 otherwise
        """
        temp = np.zeros(self.grid_points)
        # Set initial pulse between x=10 and x=11
        temp[(self.spatial_grid >= 10) & (self.spatial_grid <= 11)] = 1.0
        # Apply fixed boundary conditions
        temp[0] = 0.0
        temp[-1] = 0.0
        return temp
    
    def explicit_method(self, time_step=0.01, output_times=None):
        """
        Solve using explicit finite difference (FTCS) scheme
        
        Parameters:
            time_step (float): Time increment
            output_times (list): Times to record solution
            
        Returns:
            dict: Solution data with times and temperature profiles
        """
        if output_times is None:
            output_times = [0, 1, 5, 15, 25]
            
        # Calculate stability parameter
        r_val = self.diffusivity * time_step / (self.dx**2)
        if r_val > 0.5:
            print(f"Stability warning: r = {r_val:.4f} > 0.5")
        
        # Initialize temperature array
        temp = self.initial_temp.copy()
        current_time = 0.0
        total_steps = int(self.end_time / time_step) + 1
        
        # Setup results container
        solution_data = {
            'times': [], 
            'profiles': [], 
            'method': 'Explicit FTCS',
            'execution_time': 0.0,
            'r_value': r_val
        }
        
        # Record initial condition if requested
        if 0 in output_times:
            solution_data['times'].append(0.0)
            solution_data['profiles'].append(temp.copy())
        
        start = time.perf_counter()
        
        # Time iteration loop
        for step in range(1, total_steps):
            # Compute spatial derivative using Laplacian
            d2u = laplace(temp)
            temp += r_val * d2u
            
            # Maintain boundary conditions
            temp[0] = 0.0
            temp[-1] = 0.0
            
            current_time = step * time_step
            
            # Save solution at requested times
            for t in output_times:
                if abs(current_time - t) < time_step/2 and t not in solution_data['times']:
                    solution_data['times'].append(current_time)
                    solution_data['profiles'].append(temp.copy())
        
        solution_data['execution_time'] = time.perf_counter() - start
        
        return solution_data
    
    def implicit_method(self, time_step=0.1, output_times=None):
        """
        Solve using implicit finite difference (BTCS) scheme
        
        Parameters:
            time_step (float): Time increment
            output_times (list): Times to record solution
            
        Returns:
            dict: Solution data with times and temperature profiles
        """
        if output_times is None:
            output_times = [0, 1, 5, 15, 25]
            
        # Calculate parameter
        r_val = self.diffusivity * time_step / (self.dx**2)
        total_steps = int(self.end_time / time_step) + 1
        
        # Initialize temperature array
        temp = self.initial_temp.copy()
        
        # Construct tridiagonal system for internal points
        internal_nodes = self.grid_points - 2
        matrix = np.zeros((3, internal_nodes))
        matrix[0, 1:] = -r_val  # Upper diagonal
        matrix[1, :] = 1 + 2*r_val  # Main diagonal
        matrix[2, :-1] = -r_val  # Lower diagonal
        
        # Setup results container
        solution_data = {
            'times': [], 
            'profiles': [], 
            'method': 'Implicit BTCS',
            'execution_time': 0.0,
            'r_value': r_val
        }
        
        # Record initial condition if requested
        if 0 in output_times:
            solution_data['times'].append(0.0)
            solution_data['profiles'].append(temp.copy())
        
        start = time.perf_counter()
        
        # Time iteration loop
        for step in range(1, total_steps):
            # Prepare right-hand side vector
            rhs_vector = temp[1:-1].copy()
            
            # Solve linear system
            internal_temp = scipy.linalg.solve_banded((1, 1), matrix, rhs_vector)
            
            # Update solution with boundary conditions
            temp[1:-1] = internal_temp
            temp[0] = 0.0
            temp[-1] = 0.0
            
            current_time = step * time_step
            
            # Save solution at requested times
            for t in output_times:
                if abs(current_time - t) < time_step/2 and t not in solution_data['times']:
                    solution_data['times'].append(current_time)
                    solution_data['profiles'].append(temp.copy())
        
        solution_data['execution_time'] = time.perf_counter() - start
        
        return solution_data
    
    def crank_nicolson_method(self, time_step=0.5, output_times=None):
        """
        Solve using Crank-Nicolson scheme
        
        Parameters:
            time_step (float): Time increment
            output_times (list): Times to record solution
            
        Returns:
            dict: Solution data with times and temperature profiles
        """
        if output_times is None:
            output_times = [0, 1, 5, 15, 25]
            
        # Calculate parameter
        r_val = self.diffusivity * time_step / (self.dx**2)
        total_steps = int(self.end_time / time_step) + 1
        
        # Initialize temperature array
        temp = self.initial_temp.copy()
        
        # Construct tridiagonal system for Crank-Nicolson
        internal_nodes = self.grid_points - 2
        matrix = np.zeros((3, internal_nodes))
        matrix[0, 1:] = -r_val/2  # Upper diagonal
        matrix[1, :] = 1 + r_val  # Main diagonal
        matrix[2, :-1] = -r_val/2  # Lower diagonal
        
        # Setup results container
        solution_data = {
            'times': [], 
            'profiles': [], 
            'method': 'Crank-Nicolson',
            'execution_time': 0.0,
            'r_value': r_val
        }
        
        # Record initial condition if requested
        if 0 in output_times:
            solution_data['times'].append(0.0)
            solution_data['profiles'].append(temp.copy())
        
        start = time.perf_counter()
        
        # Time iteration loop
        for step in range(1, total_steps):
            # Prepare right-hand side vector
            internal_temp = temp[1:-1]
            rhs_vector = (r_val/2) * temp[:-2] + (1 - r_val) * internal_temp + (r_val/2) * temp[2:]
            
            # Solve linear system
            internal_temp = scipy.linalg.solve_banded((1, 1), matrix, rhs_vector)
            
            # Update solution with boundary conditions
            temp[1:-1] = internal_temp
            temp[0] = 0.0
            temp[-1] = 0.0
            
            current_time = step * time_step
            
            # Save solution at requested times
            for t in output_times:
                if abs(current_time - t) < time_step/2 and t not in solution_data['times']:
                    solution_data['times'].append(current_time)
                    solution_data['profiles'].append(temp.copy())
        
        solution_data['execution_time'] = time.perf_counter() - start
        
        return solution_data
    
    def _system_derivatives(self, time_point, internal_temp):
        """
        Define ODE system for solve_ivp method
        
        Parameters:
            time_point (float): Current time
            internal_temp (np.array): Temperatures at internal nodes
            
        Returns:
            np.array: Time derivatives for internal nodes
        """
        # Reconstruct full solution with boundaries
        full_temp = np.concatenate(([0.0], internal_temp, [0.0]))
        
        # Compute second spatial derivative
        spatial_deriv = laplace(full_temp) / (self.dx**2)
        
        # Return time derivatives for internal nodes
        return self.diffusivity * spatial_deriv[1:-1]
    
    def solve_with_ode_integrator(self, solver_method='BDF', output_times=None):
        """
        Solve using scipy's ODE integrator
        
        Parameters:
            solver_method (str): Integration algorithm
            output_times (list): Times to record solution
            
        Returns:
            dict: Solution data with times and temperature profiles
        """
        if output_times is None:
            output_times = [0, 1, 5, 15, 25]
            
        # Initial condition for internal nodes
        initial_internal = self.initial_temp[1:-1]
        
        start = time.perf_counter()
        
        # Solve the ODE system
        solution = solve_ivp(
            fun=self._system_derivatives,
            t_span=(0, self.end_time),
            y0=initial_internal,
            method=solver_method,
            t_eval=output_times,
            rtol=1e-8,
            atol=1e-10
        )
        
        elapsed = time.perf_counter() - start
        
        # Reconstruct full solutions
        solution_data = {
            'times': solution.t.tolist(),
            'profiles': [],
            'method': f'ODE Integrator ({solver_method})',
            'execution_time': elapsed,
            'r_value': None
        }
        
        # Add boundary conditions to each solution
        for i in range(len(solution.t)):
            full_profile = np.concatenate(([0.0], solution.y[:, i], [0.0]))
            solution_data['profiles'].append(full_profile)
        
        return solution_data
    
    def compare_solution_methods(self, explicit_dt=0.01, implicit_dt=0.1, cn_dt=0.5, 
                                ode_method='BDF', output_times=None):
        """
        Execute and compare all solution methods
        
        Parameters:
            explicit_dt (float): Time step for explicit method
            implicit_dt (float): Time step for implicit method
            cn_dt (float): Time step for Crank-Nicolson method
            ode_method (str): Method for ODE solver
            output_times (list): Times to compare solutions
            
        Returns:
            dict: Results from all methods
        """
        if output_times is None:
            output_times = [0, 1, 5, 15, 25]
            
        print("Solving heat equation with multiple methods...")
        print(f"Domain: [0, {self.domain_length}], Points: {self.grid_points}")
        print(f"Diffusivity: {self.diffusivity}, End time: {self.end_time}")
        print("-" * 60)
        
        # Execute all solution methods
        results = {}
        
        print("1. Explicit finite difference method...")
        results['explicit'] = self.explicit_method(explicit_dt, output_times)
        print(f"   Time: {results['explicit']['execution_time']:.4f}s, r: {results['explicit']['r_value']:.4f}")
        
        print("2. Implicit finite difference method...")
        results['implicit'] = self.implicit_method(implicit_dt, output_times)
        print(f"   Time: {results['implicit']['execution_time']:.4f}s, r: {results['implicit']['r_value']:.4f}")
        
        print("3. Crank-Nicolson method...")
        results['crank_nicolson'] = self.crank_nicolson_method(cn_dt, output_times)
        print(f"   Time: {results['crank_nicolson']['execution_time']:.4f}s, r: {results['crank_nicolson']['r_value']:.4f}")
        
        print(f"4. ODE solver ({ode_method})...")
        results['ode_solver'] = self.solve_with_ode_integrator(ode_method, output_times)
        print(f"   Time: {results['ode_solver']['execution_time']:.4f}s")
        
        print("-" * 60)
        print("All methods completed successfully!")
        
        return results
    
    def visualize_comparison(self, solution_data, save_plot=False, filename='heat_equation_comparison.png'):
        """
        Create visual comparison of solution methods
        
        Parameters:
            solution_data (dict): Results from comparison
            save_plot (bool): Save figure to file
            filename (str): Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        method_keys = ['explicit', 'implicit', 'crank_nicolson', 'ode_solver']
        color_palette = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, method_key in enumerate(method_keys):
            ax = axes[idx]
            data = solution_data[method_key]
            
            # Plot each time snapshot
            for i, (t, profile) in enumerate(zip(data['times'], data['profiles'])):
                ax.plot(self.spatial_grid, profile, color=color_palette[i], 
                        label=f't = {t:.1f}', linewidth=2)
            
            ax.set_title(f"{data['method']}\n(Time: {data['execution_time']:.4f}s)")
            ax.set_xlabel('Position (x)')
            ax.set_ylabel('Temperature u(x,t)')
            ax.grid(alpha=0.3)
            ax.legend()
            ax.set_xlim(0, self.domain_length)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {filename}")
        
        plt.show()
    
    def evaluate_accuracy(self, solution_data, reference_method='ode_solver'):
        """
        Evaluate accuracy of methods relative to reference
        
        Parameters:
            solution_data (dict): Results from comparison
            reference_method (str): Method to use as reference
            
        Returns:
            dict: Accuracy metrics
        """
        if reference_method not in solution_data:
            raise ValueError(f"Reference method '{reference_method}' not found")
        
        reference = solution_data[reference_method]
        accuracy_metrics = {}
        
        print(f"\nAccuracy Evaluation (Reference: {reference['method']})")
        print("-" * 60)
        
        for method_key, data in solution_data.items():
            if method_key == reference_method:
                continue
                
            errors = []
            # Compare each time point
            for i, (ref_profile, test_profile) in enumerate(zip(reference['profiles'], data['profiles'])):
                if i < len(data['profiles']):
                    # Compute L2 norm of difference
                    error = np.linalg.norm(ref_profile - test_profile)
                    errors.append(error)
            
            max_err = max(errors) if errors else 0
            avg_err = np.mean(errors) if errors else 0
            
            accuracy_metrics[method_key] = {
                'max_error': max_err,
                'average_error': avg_err,
                'all_errors': errors
            }
            
            print(f"{data['method']:30} - Max Error: {max_err:.2e}, Avg Error: {avg_err:.2e}")
        
        return accuracy_metrics


def execute_demonstration():
    """
    Execute the heat equation solver demonstration
    """
    # Create solver with default parameters
    solver = HeatEquationSolver(domain_length=20.0, diffusivity=10.0, 
                               grid_points=21, end_time=25.0)
    
    # Define output times
    time_points = [0, 1, 5, 15, 25]
    
    # Compare all solution methods
    comparison_results = solver.compare_solution_methods(
        explicit_dt=0.01,
        implicit_dt=0.1,
        cn_dt=0.5,
        ode_method='BDF',
        output_times=time_points
    )
    
    # Visualize results
    solver.visualize_comparison(comparison_results, save_plot=True)
    
    # Evaluate accuracy
    accuracy_report = solver.evaluate_accuracy(comparison_results, reference_method='ode_solver')
    
    return solver, comparison_results, accuracy_report


if __name__ == "__main__":
    solver_instance, results_data, accuracy_info = execute_demonstration()
