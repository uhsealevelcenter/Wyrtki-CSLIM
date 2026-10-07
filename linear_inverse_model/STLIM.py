
############################################################################
#
#  STLIM: A *ST*ationary *L*inear *I*nverse *M*odel
#  Author: Yuxin Wang <yuxinw@hawaii.edu>

############################################################################

"""
Linear Inverse Model (LIM) is a stochastic climate model rooted in the Hasselmann (1976) framework
and extended to high-dimensional climate fields (Hasselmann, 1988; Penland and Sardeshmukh 1995).
The LIM assumes that the key dynamics can be represented as a linear system forced by stochastic noise.

Specifically, the LIM can be categorized to two types: the stationary LIM (ST-LIM) and the cyclostaionary LIM (CS-LIM).
This script will focus on the ST-LIM. But you can also find the CS-LIM.py for the CS-LIM in the same folder.

If you encounter problems in running `STLIM` or have questions, please feel free to contact Yuxin Wang (yuxinw@hawaii.edu).
 
"""

import numpy as np
import xarray as xr
from numpy.linalg import pinv, eigvals, eig, eigh, inv
from scipy.linalg import logm, expm
import math
import warnings

def is_hermitian(Q, tol=1e-10):
    return np.allclose(Q, Q.H, atol=tol)     


class STLIM(object):
    
    """
    The STLIM class
    
    """    
    def __init__(self, x_LIM_training, tau_for_L=1, tolerance_for_Q_negative_eigenvalue_number=0):
        
        """        
        Initialize a STLIM.
        
        Parameters:
        -----------                
        x_LIM_training: numpy.ndarray or xarray.DataArray
            Two-dimensional (2-D) state vector for STLIM training.
            - Shape: (spatial_length, time_length)
            - Typical content: Principal Components (PCs) derived from input fields
              to reduce the system’s degrees of freedom.
            - Notes: Rows = spatial dimensions/variables; columns = time.
            If an xarray.DataArray is provided, it is converted to NumPy.
            
        tau_for_L: int, optional
            Lag time for constructing the LIM dynamical operator 'L'.
            - Must be a positive integer.
            - Default: 1
            - Recommendation:
                 * Use `1` if the original data is not temporally smoothed.
                 * Use `3` if the original data is temporally smoothed 
                   (e.g., with a 3-month running mean).    
        
        tolerance_for_Q_negative_eigenvalue_number: float, optional
            Tolerance for the fraction of negative eigenvalues in the noise covariance
            matrix 'Q' (used in `self._noise_covariance_Q`).
            - Must be in the range [0, 1].
            - Default: 0 (no negative eigenvalues allowed).
            - Interpretation: Specifies the allowable fraction of eigenvalues that
              can be negative before raising an error or triggering correction.              
                        
        Raises
        ------
        TypeError
            If parameters are not of the expected type.
        ValueError
            If parameters are outside the valid range.
        
        """        
        
        # ---- Validate input ----

        # validate x_LIM_training
        # Allow xarray.DataArray, convert to numpy
        if isinstance(x_LIM_training, xr.DataArray):
            x_LIM_training = x_LIM_training.values    
        if not isinstance(x_LIM_training, np.ndarray):
            raise TypeError("'x_LIM_training' must be a NumPy array or xarray.DataArray.")    
        if x_LIM_training.ndim != 2:
            raise ValueError("'x_LIM_training' must be 2-D with shape (spatial_length, time_length).")    
        spatial_length, time_length = x_LIM_training.shape
        if time_length <= tau_for_L:
            raise ValueError(f"time length of the x_LIM_training ({time_length}) must be > tau_for_L ({tau_for_L}).")    
        if not np.isfinite(x_LIM_training).any():
            raise ValueError("'x_LIM_training' contains NaN or Inf! Please remove or fill them before proceeding.")
            
        # validate tau_for_L
        if not isinstance(tau_for_L, int):
            raise TypeError("'tau_for_L' must be an integer.")
        if tau_for_L < 1:
            raise ValueError("'tau_for_L' must be a positive integer.")
            
        # Validate tolerance_for_Q_negative_eigenvalue_number
        if not isinstance(tolerance_for_Q_negative_eigenvalue_number, (float, int)):
            raise TypeError("'tolerance_for_Q_negative_eigenvalue_number' must be a float or int.")
        if not (0 <= tolerance_for_Q_negative_eigenvalue_number <= 1):
            raise ValueError("'tolerance_for_Q_negative_eigenvalue_number' must be in the range [0, 1].")
            
        self.x_LIM_training = x_LIM_training
        self.spatial_length = spatial_length
        self.x_training_time_length = time_length
        self.tau_for_L = tau_for_L
        self.tolerance_for_Q_negative_eigenvalue_number = tolerance_for_Q_negative_eigenvalue_number
                        
    # -------------------------------------------------------------------------   
             
    def dynamical_operator_L(self):   
                 
        """        
        Calculate the dynamical operator L for a STLIM.
             
        Returns
        -------
        L: numpy.ndarray
            The dynamical operator (shape: spatial_length x spatial_length).
        
        Notes
        -----
        L is computed via:
            L = 1/τ * C(τ) @ C(0)^{-1}
        where
            C(0)  = cov(x(t), x(t))
            C(τ)  = cov(x(t+τ), x(t))            
        LIM stability requires that all eigenvalues of L have non-positive real parts,
        this function checks the condition and raises an error if not met.
                
        """
        
        # ---- Computation ----
        # Calculate 0-lag covariance od x
        x_0 = self.x_LIM_training[:,:-self.tau_for_L]
        C_0 = x_0 @ x_0.T / (x_0.shape[-1] - 1)
        
        # Calculate tau-lag covariance of x     
        x_tau = self.x_LIM_training[:,self.tau_for_L:]
        C_tau = x_tau @ x_0.T / (x_tau.shape[-1] - 1)   

        G_tau = np.dot(C_tau, pinv(C_0))
        G_tau_eigenvalues, G_tau_eigenvectors = eig(G_tau)
        L_eigenvalues = np.log(G_tau_eigenvalues)/self.tau_for_L
                      
        if (L_eigenvalues.real > 0).any():
            raise ValueError(
                "Positive eigenvalues detected in L. "
                "The LIM would be unstable. Consider using a longer time series "
                "or reducing the spatial dimension."
            )
        
        L = G_tau_eigenvectors @ np.diag(L_eigenvalues) @ pinv(G_tau_eigenvectors)
        L = np.real(L)
        
        return L

    # -------------------------------------------------------------------------   
         
    def noise_covariance_Q(self):   
                 
        """
        Calculate the noise covariance Q for a STLIM.

        Returns
        -------
        Q: numpy.ndarray
            Final noise covariance, shape (spatial_length, spatial_length), Hermitian (complex).
            - If Q_original is Hermitian and no negative eigenvalues: Q == Q_original.
            - If Q_original is Hermitian with some negatives but the fraction ≤ self.tolerance_for_Q_negative_eigenvalue_number:
              Q is an adjusted version of Q_original obtained by clipping negatives to 0 and rescaling positives
              to preserve tr(Q).
            
        Q_eigenvalues: numpy.ndarray
            Final eigenvalues of Q, shape (spatial_length,).
            - If Q_original is Hermitian and no negative eigenvalues: Q_eigenvalues == Q_original_eigenvalues.
            - If Q_original is Hermitian with some negatives but the fraction ≤ self.tolerance_for_Q_negative_eigenvalue_number:
              Q_eigenvalues is obtained by clipping negatives to 0 and rescaling positives to preserve tr(Q).
            
        Q_eigenvectors: numpy.ndarray
            Corresponding eigenvectors (columns), shape (spatial_length, spatial_length).
            
        Q_original_negative_eigenvalue_number: int
            Number of negative eigenvalues found in the original (pre-adjustment) Q.
            
        Q_original: numpy.ndarray
            Unadjusted noise covariance (same shape and dtype as Q). Returned for inspection
            regardless of whether an adjustment was applied.
            
        Q_original_eigenvalues: numpy.ndarray
            Eigenvalues of Q_original (pre-adjustment; shape: (spatial_length,)).
            
        Notes
        -----
        - Definition (fluctuation–dissipation relation):
            Q = -(L @ C0 + C0 @ L^H),
          where C0 is the 0-lag covariance of x and L is the STLIM dynamical operator.
          
        - What this function checks for STLIM stability (and how it reacts):
          1) Hermitian check for Q:
             - If Q is not symmetric/Hermitian, it raises an error.
          2) Positive-semidefinite check for Q via eigenvalues:
             - If some eigenvalues of Q are negative, but the fraction of negatives is
               ≤ `self.tolerance_for_Q_negative_eigenvalue_number`:
                 clip negatives to 0 and rescale positives to preserve `trace(Q)`,
                 and raise a RuntimeWarning.
             - If the fraction of negatives exceeds the tolerance, it raises an error.

        """  
        
        # ---- Computation ----
        # Calculate 0-lag covariance of x
        x_0 = self.x_LIM_training[:,:-self.tau_for_L]
        C_0 = x_0 @ x_0.T / (x_0.shape[-1] - 1)
        
        # Calculate L       
        L = self.dynamical_operator_L()        
        L_matrix = np.matrix(L)
        
        # Calculate Q        
        Q_original = -(L_matrix @ C_0 + C_0 @ L_matrix.H)
        # Calculate Q's eigenvalues and eigenvectors
        Q_original_eigenvalues, Q_eigenvectors = eigh(Q_original)
        
        # Q must be positive definite for LIM's stability:
        # (1) check if Q is symmetric or Hermitian
        # (2) check if Q's eigenvalues are all positive
        
        # check if Q is Hermitian
        if is_hermitian(Q_original):
            # check if it has any negative eigenvalues
            if (Q_original_eigenvalues < 0).any():
                Q_original_negative_eigenvalue_number = (Q_original_eigenvalues < 0).sum()
                # if Q has negative eigenvalues, we pre-set a tolerance for the proportion of negative eigenvalues in Q
                # check if the number exceeds the tolerance
                if Q_original_negative_eigenvalue_number <= self.spatial_length * self.tolerance_for_Q_negative_eigenvalue_number:
                    # if not exceeding the tolerance                    
                    
                    # Calculate original Q's trace
                    original_Qtrace = np.trace(Q_original)
                    Q_eigenvalues_clipped = np.copy(Q_original_eigenvalues)
                    
                    # Set negative eigenvalues to zero
                    Q_eigenvalues_clipped[Q_eigenvalues_clipped < 0] = 0
                    
                    # Calculate original Q's trace after removing the negative values                    
                    clipped_Qtrace = np.sum(Q_eigenvalues_clipped)
                    
                    # Rescale positive eigenvalues to preserve trace
                    Q_adjusted_eigenvalues = Q_eigenvalues_clipped * (original_Qtrace / clipped_Qtrace)
                    
                    # Calculate the adjusted Q
                    Q_adjusted = Q_eigenvectors @ np.diag(Q_adjusted_eigenvalues) @ Q_eigenvectors.T

                    Q = Q_adjusted
                    Q_eigenvalues = Q_adjusted_eigenvalues
                   
                    msg = (
                        f"Adjusted Q: negative eigenvalues were clipped to 0 and "
                        "positive eigenvalues were rescaled to preserve tr(Q)."
                    )
                    warnings.warn(msg, category=RuntimeWarning, stacklevel=2)
                    
                else:
                    raise ValueError('Q is not positive semidefinite within tolerance: too many negative eigenvalues.')
               
            else: # if Q does not have any negative eigenvalues     
                Q = Q_original
                Q_eigenvalues = Q_original_eigenvalues
                Q_original_negative_eigenvalue_number = 0
                
        else:
           raise ValueError('Q is invalid: Q is not symmetric or Hermitian')
        
        return Q, Q_eigenvalues, Q_eigenvectors, Q_original_negative_eigenvalue_number, Q_original, Q_original_eigenvalues

    # -------------------------------------------------------------------------   
        
    def propagation_operator_G(self, tau_for_G):           
          
        """
        Calculate the propagation operator G(τ) for a STLIM.
        
        Parameters:
        -------                 
        tau_for_G: int or 1D array-like or list-like of int
            Lead time(s) τ for the propagation operator 'G(τ)'. Each τ must be a positive integer.
            - If an int is given, returns a single τ result.
            - If an array or list is given, returns a stack over all τ values.
            
        Returns
        -------
        G_tau: numpy.ndarray
            Propagation operator at lag τ
            - If `tau_for_G` is an int: shape (spatial_length, spatial_length).
            - If `tau_for_G` is a 1D array of length K: shape (K, spatial_length, spatial_length).
              The first axis preserves the order of `tau_for_G` (no sorting).  
          
        Notes
        -----
        G(τ) is derived from the dynamical operator L via matrix exponentiation:
        G(τ) = exp(L * τ), where L is the LIM dynamical operator.          
          
        """    
        
        # ---- Validate input ----         
        tau_arr = np.asarray(tau_for_G)        
        # Shape: allow scalar or 1D only
        if tau_arr.ndim == 0:
            tau_arr = tau_arr.reshape(1)
        if tau_arr.ndim == 1:
            tau_arr = tau_arr
        else:
            raise TypeError("'tau_for_G' must be a scalar or 1D array-like.")        
        # Numeric + NaN check
        try:
            _tauf = tau_arr.astype(np.float64)
        except Exception:
            raise TypeError("All τ values must be numeric (int-like).")
        if np.isnan(_tauf).any():
            raise ValueError("τ must not contain NaN.")        
        # Integer-valued and > 0
        if not np.allclose(_tauf, np.round(_tauf)):
            raise TypeError("Each τ must be an integer.")
        tau_arr = np.round(_tauf).astype(np.int64)
        if (tau_arr <= 0).any():
            raise ValueError("Each τ must be > 0.")
            
        # ---- Compute L, then G(τ) ----   

        how_many_taus = tau_arr.shape[0]
        G_tau = np.full((how_many_taus, self.spatial_length, self.spatial_length), np.nan, dtype=complex)
        L = self.dynamical_operator_L()
        
        for itau_idx, itau in enumerate(tau_arr):
            # Build τ-step propagators via modular products       
            G_tau[itau_idx] = expm(L * itau)
            
        if tau_arr.shape[0] == 1:
            G_tau = np.squeeze(G_tau, axis=0)
        
        return G_tau     
    
    # -------------------------------------------------------------------------   
                             
    def deterministic_forecast(self, lead_tau_forecast, x_forecast_from=None):   
                 
        """
        Make the ST-LIM deterministic forecast.        
        
        Parameters:
        -----------                             
        lead_tau_forecast: int or 1D array-like or list-like of int
            Lead time(s) τ for the forecast. Each τ must be a positive integer.
            - If an int is given, returns a single τ result.
            - If an array or list is given, returns a stack over all τ values.

        x_forecast_from: array-like (numpy.ndarray, xarray.DataArray or list), optional
            Initial state(s) in the PC space to forecast from.
            - Default: self.x_LIM_training
            - Accepted shapes:
                • (spatial_length, time_length) for time_length ≥ 1  → forecasts for each column (time sample)
                • (spatial_length,)              → treated as a single state (time_length = 1)
            - Requirements: no NaNs; first dimension must equal self.spatial_length.
              If an xarray.DataArray is provided, it will be converted to NumPy.
            
        Returns
        -------
        x_tau_forecast: numpy.ndarray
            Forecasted state vector from 'x_forecast_from' at lead τ
            - If `lead_tau_forecast` is an int: shape (spatial_length, time_length_x_forecast_from).
            - If `lead_tau_forecast` is a 1D array of length K: shape (K, spatial_length, time_length_x_forecast_from).
              The first axis preserves the order of `lead_tau_forecast` (no sorting).  
            
        """
        
        # ---- Validate input ----                
        # Validate lead_tau_forecast
        tau_arr = np.asarray(lead_tau_forecast)        
        # Shape: allow scalar or 1D only
        if tau_arr.ndim == 0:
            tau_arr = tau_arr.reshape(1)
        if tau_arr.ndim == 1:
            tau_arr = tau_arr
        else:
            raise TypeError("'lead_tau_forecast' must be a scalar or 1D array-like.")        
        # Numeric + NaN check
        try:
            _tauf = tau_arr.astype(np.float64)
        except Exception:
            raise TypeError("All τ values must be numeric (int-like).")
        if np.isnan(_tauf).any():
            raise ValueError("τ must not contain NaN.")        
        # Integer-valued and > 0
        if not np.allclose(_tauf, np.round(_tauf)):
            raise TypeError("Each τ must be an integer.")
        tau_arr = np.round(_tauf).astype(np.int64)
        if (tau_arr <= 0).any():
            raise ValueError("Each τ must be > 0.")

        # Validate x_forecast_from
        if x_forecast_from is None:
            x_forecast_from = self.x_LIM_training       
        # Ensure NumPy array
        x_forecast_from = np.asarray(x_forecast_from)        
        # Shape: allow (spatial_length,) or (spatial_length, T≥1)
        if x_forecast_from.ndim == 1:
            if x_forecast_from.size != self.spatial_length:
                raise ValueError(
                    f"'x_forecast_from' 1-D length must equal spatial_length "
                    f"({x_forecast_from.size} != {self.spatial_length})."
                )
            x_forecast_from = x_forecast_from.reshape(self.spatial_length, 1)
        elif x_forecast_from.ndim == 2:
            if x_forecast_from.shape[0] != self.spatial_length:
                raise ValueError(
                    f"Mismatch in spatial_length: input has {x_forecast_from.shape[0]}, "
                    f"expected {self.spatial_length}."
                )
            if x_forecast_from.shape[1] < 1:
                raise ValueError("'x_forecast_from' must have at least one column (time_length ≥ 1).")
        else:
            raise ValueError(
                "'x_forecast_from' must be 1-D (spatial_length,) or 2-D (spatial_length, time_length)."
            )        
        # Numeric dtype and NaN/Inf checks
        if not np.issubdtype(x_forecast_from.dtype, np.number):
            try:
                x_forecast_from = x_forecast_from.astype(np.float64)
            except Exception as e:
                raise TypeError("'x_forecast_from' must be numeric.") from e        
        if not np.isfinite(x_forecast_from).any():
            raise ValueError("'x_forecast_from' contains NaN or Inf! Please remove or fill them before proceeding.")                    
        # Sizes for downstream use
        spatial_length_from_data, time_length_from_data = x_forecast_from.shape

        # ---- Computation ----
        how_many_taus = tau_arr.shape[0]
        x_tau_forecast = np.full((how_many_taus, self.spatial_length, time_length_from_data), np.nan, dtype=complex)
        G_tau = self.propagation_operator_G(tau_arr)
        
        if tau_arr.shape[0] == 1:
            x_tau_forecast = G_tau @ x_forecast_from
        else:  
            for itau_idx, itau in enumerate(tau_arr):
                x_tau_forecast[itau_idx] = G_tau[itau_idx] @ x_forecast_from

        return x_tau_forecast       

    # -------------------------------------------------------------------------   
                   
    def simulations_with_noise(self, time_steps, simulation_length, initial_state=None, seed=None):
                 
        """
        Generate stochastic simulations using a STLIM.
                 
        Parameters:
        -----------
        time_steps: int
            Number of sub-steps per model time unit (Δt = 1 / time_steps).
            - Must be > 1.
            - Example: For monthly data, `time_steps=60` corresponds to a time step of ~12 hours.
        
        simulation_length: int
            Length of model time units to simulate (samples stored once per unit).
            - Must be a positive integer.
            - Example: For monthly data, `simulation_length=2400` generates 200 years of monthly data.
            
        initial_state: numpy.ndarray or xarray.DataArray, optional
            Initial state (i.e., starting state) for the long simulation.
            - Shape: (spatial_length,) or (spatial_length, 1)
            - Default: np.zeros((spatial_length,1))
            - Order along the spatial_length axis must match the variable order in x_LIM_training used for STLIM training.   
            
        seed: int or None, optional
            Seed for the random number generator used in the stochastic forcing.
            - If None (default), each run will produce different noise.
        
        Returns
        -------
        simulations: numpy.ndarray
            Simulated series sampled every model time unit.
            - Shape: (spatial_length, simulation_length)
            
        """
                
        # ---- Validate input ----            
        # validate time_steps
        if not isinstance(time_steps, int):
            raise TypeError(f"'time_steps' must be an integer, got {type(time_steps).__name__}.")
        if time_steps <= 1:
            raise ValueError(f"'time_steps' must be > 1, got {time_steps}.")
        
        # validate simulation_length
        if not isinstance(simulation_length, int):
            raise TypeError(f"'simulation_length' must be an integer, got {type(simulation_length).__name__}.")
        if simulation_length <= 0:
            raise ValueError(f"'simulation_length' must be positive, got {simulation_length}.")   
            
        # validate initial_state            
        if initial_state is None:
            initial_state_new=np.zeros((self.spatial_length,1))   
        else:
            initial_state = np.asarray(initial_state)
            if initial_state.shape[0] != self.spatial_length:
                raise ValueError("'initial_state' length must match spatial_length.")                    
            if initial_state.ndim == 2 and initial_state.shape[1] == 1:
                initial_state_new = initial_state
            if initial_state.ndim == 1:
                initial_state_new=np.zeros((self.spatial_length,1))
                initial_state_new[:,0] = initial_state                
            if not np.isfinite(initial_state).all():
                raise ValueError("'initial_state' contains NaN or Inf! Please remove or fill them before proceeding.")
        
        # validate seed            
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("'seed' must be an integer or None.")
            # Setting the seed here ensures reproducibility of np.random.normal
            rng = np.random.default_rng(seed)

        # ---- Computation ----        
        # Get the dynamical operator L and noise covariance Q
        L = self.dynamical_operator_L()
        Q, Q_eigenvalues, Q_eigenvectors, _, _, _ = self.noise_covariance_Q()        
           
        y_t = initial_state_new
                 
        delta_t = 1/time_steps            
        total_steps = time_steps * simulation_length
        
        # Generate an empty array for simulations afterwards
        simulations = np.full((self.spatial_length,simulation_length), np.nan, dtype=float)
        
        # Calculate stochastic forcing amplitude matrix S
        S =  Q_eigenvectors @ np.sqrt(np.diag(Q_eigenvalues))
            
        for istep in range(total_steps):
            r_t = np.zeros((self.spatial_length,1))
            if seed is not None:
                r_t[:,0] = rng.standard_normal(size=(self.spatial_length))    
            else:
                r_t[:,0] = np.random.normal(size=(self.spatial_length))
            y_t_plus_delta_t = y_t + delta_t * L @ y_t + np.sqrt(delta_t) * S @ r_t
            
            y_t_plus_half_delta_t = (y_t_plus_delta_t + y_t)/2  
            
            # Store once per model time unit
            if (istep+1) % time_steps == 0:
                time_index = istep//time_steps
                simulations[:,time_index] = np.real(y_t_plus_half_delta_t.ravel())
            
            y_t = y_t_plus_delta_t        
                           
        return simulations                  
        
    # -------------------------------------------------------------------------   
                       
    def optimal_initial_condition(self, optimization_time, initial_norm_vector=None, final_norm_vector=None):           
                 
        """
        Compute the optimal initial condition that maximizes amplification over a given optimization time in the STLIM.       
         
        Parameters:
        -----------        
        optimization_time : int
            Positive lead time τ at which the amplification is evaluated.
            - Must be a positive integer.
            
        initial_norm_vector: numpy.ndarray or xarray.DataArray, optional
            Vector (d) defining the initial-state norm (directional weighting).
            - Shape: (spatial_length,) or (spatial_length, 1)
            - Must not be the all-zero vector.
            - Normalization is handled internally (input need not be unit length)
            - If not None, the initial norm is D = d @ d^T
            - If None, the initial norm D is the Euclidean norm (i.e., I).

        final_norm_vector: numpy.ndarray or xarray.DataArray, optional
            Vector (n) defining the final-state norm (directional weighting).
            - Shape: (spatial_length,) or (spatial_length, 1)
            - Must not be the all-zero vector.
            - Normalization is handled internally (input need not be unit length)
            - If not None, the final norm is N = n @ n^T
            - If None, the final norm N is the Euclidean norm (i.e., I).

        Returns
        -------
        maximum_amplification_factor : float
            Leading eigenvalue (real part) of inv(D) @ G(τ)^T @ N @ G(τ), i.e., the maximum
            amplification achievable at time τ under the specified norms.
    
        optimal_initial_condition: numpy.ndarray
            Eigenvector associated with the leading eigenvalue of inv(D) @ G(τ)^T @ N @ G(τ),
            giving the optimal initial condition. Shape: (spatial_length,)

        evoloved_targeted_final_condition: numpy.ndarray
            The evolved state at lead τ for that optimal initial condition:
            G(τ) @ optimal_initial_condition. Shape: (spatial_length,)
       
        """      
        
        # ---- Validate input ----        
        # validate optimization_time
        if not isinstance(optimization_time, int):
            raise TypeError(f"'optimization_time' must be an integer, got {type(optimization_time).__name__}.")
        if optimization_time < 1:
            raise ValueError(f"'optimization_time' must be > 0, got {optimization_time}.")

        # ---- Computation ----
        L = self.dynamical_operator_L()
        G_tau = self.propagation_operator_G(tau_for_G=optimization_time)
                        
        epsilon = 10**(-9)

        # validate initial_norm_vector        
        if initial_norm_vector is None:
            D = np.identity(self.spatial_length) + epsilon * np.identity(self.spatial_length)
        else:
            initial_norm_vector = np.asarray(initial_norm_vector)
            if initial_norm_vector.shape[0] != self.spatial_length:
                raise ValueError("'initial_norm_vector' length must match spatial_length.")                
            if initial_norm_vector.ndim == 2 and initial_norm_vector.shape[1] == 1:
                initial_norm_vector_new = initial_norm_vector
            if initial_norm_vector.ndim == 1:
                initial_norm_vector_new = np.zeros((self.spatial_length,1))
                initial_norm_vector_new[:,0] = initial_norm_vector
            if not np.isfinite(initial_norm_vector).any():
                raise ValueError("'initial_norm_vector' contains NaN or Inf! Please remove or fill them before proceeding.")                
            # normalize the initial norm vector
            d_norm = np.linalg.norm(initial_norm_vector_new)
            if d_norm != 0:
                d_normalized = initial_norm_vector_new / d_norm
            else:
                raise ValueError('Initial kernal vector elements are all 0')                
            # Compute the initial norm D
            D = d_normalized @ d_normalized.T + epsilon * np.identity(self.spatial_length)
         
        # validate final_norm_vector        
        if final_norm_vector is None:
            N = np.identity(self.spatial_length) + epsilon * np.identity(self.spatial_length)
        else:
            final_norm_vector = np.asarray(final_norm_vector)
            if final_norm_vector.shape[0] != self.spatial_length:
                raise ValueError("'final_norm_vector' length must match spatial_length.")                
            if final_norm_vector.ndim == 2 and final_norm_vector.shape[1] == 1:
                final_norm_vector_new = final_norm_vector
            if final_norm_vector.ndim == 1:
                final_norm_vector_new = np.zeros((self.spatial_length,1))
                final_norm_vector_new[:,0] = final_norm_vector
            if not np.isfinite(final_norm_vector_new).all():
                raise ValueError("'final_norm_vector_new' contains NaN or Inf! Please remove or fill them before proceeding.")                
            # normalize the final norm vector
            n_norm = np.linalg.norm(final_norm_vector_new)
            if n_norm != 0:
                n_normalized = final_norm_vector_new / n_norm                
            else:
                raise ValueError('Final kernal vector elements are all 0')                
            # Compute the final norm N              
            N = n_normalized @ n_normalized.T + epsilon * np.identity(self.spatial_length)
        
        G_tau_transpose = G_tau.transpose()
        invDGT = inv(D) @ G_tau_transpose
        invDGTN = invDGT @ N
        invDGTNG = invDGTN @ G_tau        
        # calculate its eigenvalues and eigenvectors
        w, v = eig(invDGTNG)
        
        # Sort indices by descending eigenvalue
        idx = np.argsort(w)[::-1]        
        # Reorder both eigenvalues and eigenvectors
        w = w[idx]
        v = v[:, idx]
        
        # the maximum_amplification_factor is the the leading eigenvalue
        maximum_amplification_factor = w[0]
        # the optimal_inital_condition is eigenvector corresponding to the leading eigenvalue
        optimal_inital_condition = v[:,0]

        evoloved_targeted_final_condition = G_tau @ optimal_inital_condition
        
        return maximum_amplification_factor, optimal_inital_condition, evoloved_targeted_final_condition     
        
    # -------------------------------------------------------------------------       
            
    def POPs(self):   
                
        """
        Calculate the principal_oscillation_patterns (POPs) of STLIM, by the eigenanalysis of the dynmical operator L.

        Returns
        -------
        POP_period: numpy.ndarray, shape (m,)
            Oscillation periods (positive; same units as lag time;
            extremely large values or np.inf for statonary or non-oscillatory modes).
        POP_decay: numpy.ndarray, shape (m,)
            E-folding decay times (positive; same units as lag time)
        POP_patterna: numpy.ndarray, shape (m, n_modes)
            Real parts of eigenvectors of L. Rows correspond to spatial dimension (m), 
            and columns correspond to POP modes (n_modes).
            Note: since L is an m × m operator, technically m = n_modes.
        POP_patternb: numpy.ndarray, shape (m, n_modes)
            Imaginary parts of eigenvectors of L. Rows correspond to spatial dimension (m), 
            and columns correspond to POP modes (n_modes).
            Note: since L is an m × m operator, technically m = n_modes.

        Notes
        -----
        For an eigenvalue of L, λ = σ + iω:
          - Decay time τ = -1/σ
          - Period  T = 2π/|ω|
        Results are sorted by descending decay time (least damped first).
        Oscillatory modes appear in complex-conjugate pairs with identical decay times
        and periods; stationary modes have T = ∞ or huge number.

        """      
    
        # ---- Computation ----

        L = self.dynamical_operator_L()
        L_eigenvalues, L_eigenvectors = eig(L)
        
        # Sort indices by descending eigenvalue of L
        sorted_indices = np.argsort(-1/L_eigenvalues.real)[::-1]
        sorted_L_eigenvalues = L_eigenvalues[sorted_indices]
        sorted_L_eigenvectors = L_eigenvectors[:, sorted_indices]
        
        POP_period = np.abs(2*math.pi/sorted_L_eigenvalues.imag)
        POP_decay = -1/sorted_L_eigenvalues.real
        
        POP_patterna = np.real(sorted_L_eigenvectors)
        POP_patternb = np.imag(sorted_L_eigenvectors)            
        
        return POP_period, POP_decay, POP_patterna, POP_patternb
    

    # -------------------------------------------------------------------------   
            
    def POPs_reconstruction(self, modes_want_for_reconstruction=None):
        
        """
        Reconstruct the state vector used for the STLIM training, using selected Principal Oscillation Patterns (POPs)
        derived from the STLIM.
    
        Parameters
        ----------
        modes_want_for_reconstruction: list or numpy.ndarray, optional
            1-based indices of POPs to use for reconstruction.
            - Default: None (use all modes)
            - Modes are ordered by **descending decay time** (least damped first), consistent with `_POPs`.
            - Elements must be positive integers, and the maximum index must be <= spatial dimension.
    
        Returns
        -------
        x_recons_by_POP: numpy.ndarray
            Reconstructed state vector , shape (spatial_length, time_length).
            
        """
        # ---- eigen-decomposition and sorting by decay time ----
        L = self.dynamical_operator_L()
        L_eigenvalues, L_eigenvectors = eig(L)
        # left eigenvectors (Hermitian of inverse of right-eigenvector matrix)
        L_eigenvectors_H = np.array(np.matrix(inv(L_eigenvectors)).getH())
    
        # Sort by descending decay time τ = -1/Re(λ)  (least damped first)
        # (Re(λ) typically < 0; larger Re(λ) => longer decay time)
        sort_order = np.argsort(-L_eigenvalues.real)  # larger real part first
        U_sorted = L_eigenvectors[:, sort_order]
        V_sorted = L_eigenvectors_H[:, sort_order]
    
        # ---- handle modes_want_for_reconstruction ----
        if modes_want_for_reconstruction is None:
            modes = np.arange(1, self.spatial_length + 1, dtype=int)  # 1..spatial_length
        else:
            arr = np.asarray(modes_want_for_reconstruction)
            if arr.size == 0:
                raise ValueError("modes_want_for_reconstruction cannot be empty.")   
            # ensure 1-D
            arr = arr.reshape(-1)
    
            # ensure numeric and integer-valued
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError("modes_want_for_reconstruction must be numeric (int).")
            if not np.isfinite(arr).any():
                raise ValueError("modes_want_for_reconstruction contains NaN/Inf.")
            if not np.all(np.equal(arr, np.floor(arr))):
                raise ValueError("All mode indices must be integers (1-based).")   
            modes = arr.astype(int)    
            # positivity and bounds (1-based indexing)
            if np.any(modes <= 0):
                raise ValueError("All mode indices must be positive (1-based): min index is 1.")
            if np.any(modes > self.spatial_length):
                raise ValueError(
                    f"Mode indices must be ≤ spatial_length={self.spatial_length}."
                )    
            # no duplicates (avoids double-counting)
            if np.unique(modes).size != modes.size:
                raise ValueError("Duplicate mode indices are not allowed.")
    
        # convert to 0-based indices for numpy indexing
        idx0 = modes - 1
    
        # ---- reconstruction ----
        ui = U_sorted[:, idx0]              # (spatial_length, k)
        vi = V_sorted[:, idx0]              # (spatial_length, k)
        viH = np.matrix(vi).getH()
        
        alpha = viH @ self.x_LIM_training
        x_recons_by_POP = ui @ alpha  
    
        return np.asarray(x_recons_by_POP.real)
    
