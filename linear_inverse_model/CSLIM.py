############################################################################
#
#  CSLIM: A *C*yclo*S*tationary *L*inear *I*nverse *M*odel
#  Author: Yuxin Wang <yuxinw@hawaii.edu>

############################################################################


"""
Linear Inverse Model (LIM) is a stochastic climate model rooted in the Hasselmann (1976) framework
and extended to high-dimensional climate fields (Hasselmann, 1988; Penland and Sardeshmukh 1995).
The LIM assumes that the key dynamics can be represented as a linear system forced by stochastic noise.

Specifically, the LIM can be categorized to two types: the stationary LIM (ST-LIM) and the cyclostaionary LIM (CS-LIM).
This script will focus on the CS-LIM. But you can also find the ST-LIM.py for the CS-LIM in the same folder.

If you encounter problems in running `CSLIM` or have questions, please feel free to contact Yuxin Wang (yuxinw@hawaii.edu).
 
"""

import numpy as np
import xarray as xr
from numpy.linalg import pinv, eigvals, eig, eigh, inv
from scipy.linalg import logm, expm
import math
import warnings

def is_hermitian(Q, tol=1e-10):
    return np.allclose(Q, Q.H, atol=tol)     
             
def reshape_state_vector(x_LIM_training, period_T = 12):   
        
    """
    Reshape the training state vector for a CSLIM.

    Purpose
    -------
    Convert (spatial_length, time_length) into (spatial_length, n_cycles, period_T),
    where n_cycles = time_length / period_T. This makes periodic structure
    explicit along the second axis (length = period_T).
    
    Parameters:
    -----------
    x_LIM_training: numpy.ndarray or xarray.DataArray
        Two-dimensional (2-D) state vector for CSLIM training.
        - Shape: (spatial_length, time_length)
        - Typical content: Principal Components (PCs) derived from input fields
          to reduce the system’s degrees of freedom.
        - Notes: The first axis (rows) represents spatial dimension or variables; 
          the second axis (columns) represents time. If provided as an 
          xarray.DataArray, it will be automatically converted to a NumPy array.
          
    period_T: int
        Period of the data (in time steps).
        - Must be an integer ≥ 2, but it must be < x_LIM_training
        - Example: For monthly data with an annual cycle, use period_T = 12. 
        
     Returns
     -------
     x_LIM_training_reshaped : numpy.ndarray
         The reshaped state vector for the afterwards CSLIM training.
         - Shape: (spatial_length, n_cycles, period_T)         
     
     """  
    
    # ---- Validate input ----
    # Accept xarray.DataArray and convert to NumPy
    if isinstance(x_LIM_training, xr.DataArray):
        x_LIM_training = x_LIM_training.values
    if not isinstance(x_LIM_training, np.ndarray):
        raise TypeError("'x_LIM_training' must be a NumPy array or xarray.DataArray.")    
    if x_LIM_training.ndim != 2:
        raise ValueError("'x_LIM_training' must be 2-D with shape (spatial_length, time_length).")           
    spatial_length, time_length = x_LIM_training.shape
    if time_length <= period_T:
        raise ValueError("'time_length' must be greater than 'period_T'.")
    if time_length % period_T != 0:
        raise ValueError(f"'time_length' ({time_length}) must be a multiple of 'period_T' ({period_T}).")            
    if not np.isfinite(x_LIM_training).any():
        raise ValueError("'x_LIM_training' contains NaN or Inf!! Please remove or fill them before proceeding.")

    n_cycles = time_length//period_T
    if n_cycles < 2:
        raise ValueError("Not enough cycles (need at least 2)")
    
    x_CSLIM_training = np.full((spatial_length, n_cycles, period_T), np.nan, dtype=float)
    for it in range(time_length):
        # Map time index 'it' to (phase, cycle)
        cycle = it // period_T         # 0..n_cycles-1
        phase = it % period_T          # 0..period_T-1
        # Fill slice: source is (spatial_length,), dest is (spatial_length,)
        x_CSLIM_training[:, cycle, phase] = x_LIM_training[:, it]
                   
    return x_CSLIM_training


# ---------------------------------   

class CSLIM(object):
    
    """
    The CSLIM class
    
    """    
    
    def __init__(self, x_CSLIM_training, period_T, tolerance_for_Q_negative_eigenvalue_number=0):
        
        """
        Initialize a CSLIM .
            
        Parameters:
        -----------  
        x_CSLIM_training : numpy.ndarray or xarray.DataArray
           State vector for CSLIM training.
           - Shape: (spatial_length, n_cycles, period_T)
           - Typical content: Principal Components (PCs) derived from input fields
             to reduce the system’s degrees of freedom.
           - Axis meaning:
               * Axis 0 (rows): spatial dimension or variable index
               * Axis 1 (columns): cycles
               * Axis 2: phases within each cycle (length = period_T)
           - Notes: If provided as an xarray.DataArray, it will be automatically
             converted to a NumPy array.
             
        period_T : int
            Period of the data (in time steps).
            - Must be an integer ≥ 2.
            - Example: For monthly data with an annual cycle, use period_T = 12. 
        
        tolerance_for_Q_negative_eigenvalue_number : float, optional
            Tolerance for the proportion of negative eigenvalues in matrix 'Q'.
            - Must be in the range [0, 1].
            - Default: 0 (no negative eigenvalues allowed).
            - Interpretation: Specifies the allowable fraction of eigenvalues that
              can be negative before raising an error or triggering correction.              
            - Notes: For the CS-LIM, negative eigenvalues are more likely due to smaller
              sample sizes compared to the stationary LIM (ST-LIM). Therefore, a higher
              tolerance may be appropriate. Wang et al. (2023, 2025) used 40%.
            - Recommendation: Plot the variance maps (diagonal elements of Q transformed
              to geographical space) and check that the adjusted and original versions
              differ only slightly, ensuring the variance structure remains consistent.
                                      
        Raises
        ------
        TypeError
            If parameters are not of the expected type.
        ValueError
            If parameters are outside the valid range.
        
        """        
         
        # ---- Validate input ----
        # validate x_CSLIM_training        
        # Ensure NumPy array
        if isinstance(x_CSLIM_training, xr.DataArray):
            x_CSLIM_training = x_CSLIM_training.values
        if not isinstance(x_CSLIM_training, np.ndarray):
            raise TypeError("'x_CSLIM_training' must be a NumPy array or xarray.DataArray.")    
        if x_CSLIM_training.ndim != 3:
            raise ValueError("x_CSLIM_training must be 3-D with shape (spatial_length, n_cycles, period_T).")       
        spatial_length, n_cycles, period_T_from_data = x_CSLIM_training.shape
        if period_T_from_data != period_T:
            raise ValueError(f"Mismatch in period_T: input has period_T={period_T}, "
                f"but expected period_T={period_T_from_data} (from initialization).")
        if n_cycles < 2:
            raise ValueError("Not enough cycles (need at least 2)")    
        if not np.isfinite(x_CSLIM_training).any():
            raise ValueError("'x_CSLIM_training' contains NaN values! Please remove or fill them before proceeding.")
                
        # validate period_T
        if not isinstance(period_T, int):
            raise TypeError("'period_T' must be an integer.")
        if period_T < 2:
            raise ValueError("'period_T' must be a integer >=2.")

        # validate tolerance_for_Q_negative_eigenvalue_number
        if not isinstance(tolerance_for_Q_negative_eigenvalue_number, (float, int)):
            raise TypeError("'tolerance_for_Q_negative_eigenvalue_number' must be a float or int.")
        if not (0 <= tolerance_for_Q_negative_eigenvalue_number <= 1):
            raise ValueError("'tolerance_for_Q_negative_eigenvalue_number' must be in the range [0, 1].")
            
        self.x_CSLIM_training = x_CSLIM_training
        self.x_training_n_cycles = n_cycles
        self.period_T = period_T
        self.spatial_length = spatial_length
        self.tolerance_for_Q_negative_eigenvalue_number = tolerance_for_Q_negative_eigenvalue_number
                            
     # ---------------------------------   
             
    def propogation_operator_G1(self):   
                 
        """
        Compute the periodic one-step propagation operators G_{j→j+1} for a CSLIM.

        Definitions
        -----------
        - j ∈ {0, …, T−1} indexes the phase within a period of length T = self.period_T.
        - Training data are arranged as x_CSLIM_training with shape
          (spatial_length, n_cycles, period_T). For a fixed phase j, the slice
          x_CSLIM_training[:, :, j] stacks consecutive cycles at that phase.

        Returns
        -------
        Gj_1 : np.ndarray
            Periodic propagation operator, shape (period_T, spatial_length, spatial_length),
            where Gj_1[j] maps phase j to j+1.
            
        Notes
        -----
        Gj_1 is computed via:
            Gj_1 = Cj(1) @ Cj(0)^{-1}
        where
            Cj(0)  = cov(xj(t), xj(t))
            Cj(1)  = cov(xj(t+1), xj(t))   

        """ 
           
        # Allocate: G_{j→j+1} for all phases j
        Gj_1 = np.full((self.period_T,self.spatial_length,self.spatial_length), np.nan, dtype=complex)

        # Regular phases: j = 0 .. T-2 (use consecutive phase slices)
        for j in range(self.period_T-1):
            xj = self.x_CSLIM_training[:,:,j]
            C0_j = xj @ xj.T / (xj.shape[-1] - 1)
            
            xjplus1 = self.x_CSLIM_training[:,:,j+1]
            C1_j = xjplus1 @ xj.T / (xjplus1.shape[-1] - 1)
            
            Gj_1[j] = np.dot(C1_j, pinv(C0_j))  
            
        # Wrap-around phase: j = T-1 → j+1 ≡ 0
        # Align cycles so the “next” phase samples match the same cycle index.
        for j in range(self.period_T-1,self.period_T):
            xj = self.x_CSLIM_training[:,:self.x_training_n_cycles-1,j]  # drop last to align
            C0_j = xj @ xj.T / (xj.shape[-1] - 1)
            
            xjplus1 = self.x_CSLIM_training[:,1:,0]   # drop first to align
            C1_j = xjplus1 @ xj.T / (xjplus1.shape[-1] - 1)
            
            Gj_1[j] = np.dot(C1_j, pinv(C0_j))
            Gj_1[j] = np.real(Gj_1[j])
            
        return Gj_1
             
     # ---------------------------------   
             
    def dynamical_operator_Lj(self):   
                
        """
        Compute the periodic dynamical operator Lj for a CSLIM.
        j is the j_th phase.
        
        Definition
        -------
        For each phase j, Lj satisfies:
            exp(L_j * tau) = G_{j→j+1}
        with tau = 1 model time unit (e.g., one month) in this implementation.
    
        Returns
        -------
        Lj : np.ndarray
            Periodic dynamical operators, shape (period_T, spatial_length, spatial_length),
            Lj[j] = logm(Gj_1[j]).

        Notes
        -----
        Lj is obtained via:
            Gj_1 = V @ Λ @ V^{-1},  Lj = V @ log(Λ) @ V^{-1}

        """ 

        Gj_1 = self.propogation_operator_G1()
        
        # Allocate
        Lj = np.full((self.period_T,self.spatial_length,self.spatial_length), np.nan, dtype=complex)
                
        for j in range(self.period_T):
            Gj_1_eigenvalues, Gj_1_eigenvectors = eig(Gj_1[j])
            Lj_eigenvalues = np.log(Gj_1_eigenvalues)/1   # tau = 1 model unit
            Lj[j] = Gj_1_eigenvectors @ np.diag(Lj_eigenvalues) @ pinv(Gj_1_eigenvectors)
            Lj[j] = np.real(Lj[j])
            
        return Lj

    # ---------------------------------   
             
    def monodromy_matrix_Mj(self):   
                
        """
        Compute the periodic monodromy matrix Mj for a CSLIM.

        Definition
        ----------
        M_j is the product of one full period of one-step propagators starting
        at phase j:
            M_j = G_{j+T-1→j} ··· G_{j+1→j+2} · G_{j→j+1}
        (indices wrap modulo T). Thus, M_j maps phase j back to itself
        after one full cycle.
    
        Returns
        -------
        Mj : np.ndarray
            Periodic monodromy matrices, shape (period_T, spatial_length, spatial_length),
            product of one full period starting at each phase j.

        Stability Check
        ---------------
        The Floquet exponents μ_i defined by
            μ_i = (1/T) * log(λ_i(M_j))
        should have strictly negative real parts for a stable (dissipative)
        cyclostationary system. If any Re(μ_i) ≥ 0, we raise a ValueError.
        
        """ 
           
        # Allocate
        # Monodromy matrices M_j: product over a full season starting at j
        Mj = np.full((self.period_T,self.spatial_length,self.spatial_length), np.nan, dtype=complex)

        Gj_1 = self.propogation_operator_G1()
         
        for istart in range(self.period_T):
            # Build order: j-1, j-2, ..., 0, T-1, ..., j  (G[j] is multiplied last)
            order = np.roll(np.arange(self.period_T), -istart)
            # Left-multiply to maintain forward-in-time order
            Mj_inter = Gj_1[order[0]]
            for idx in order[1:]:
                Mj_inter = Gj_1[idx] @ Mj_inter
            Mj[istart] = np.real(Mj_inter)
        
        # Stability check:
        # # While Lj can be temporally unstable, the monodromy matrix Mj must be dissipative, hence ensuring the stationarity over a cycle
        # That is, the real part of Floquet exponents, Re(μ_i), must be < 0
        
        M1_eigenvalues, M1_eigenvectors = eig(Mj[0])
   
        miu1_Floquet_exponents = np.log(M1_eigenvalues)/self.period_T
        if (np.real(miu1_Floquet_exponents)>=0).any():
            raise ValueError('Floquet exponents have positive real parts, the CS-LIM would not be stable')      
         
        return Mj
    
    # ---------------------------------   
    
    def propogate_operator_Gtau(self, tau_for_G):   
                 
        """
        Compute τ-step propagators G_j(τ) for each phase j in a CSLIM.
        
        Definition
        -------
        For each phase j (0..period_T-1), compute the τ-step propagator
            G_j(τ) = G_{j+τ-1} ··· G_{j+1} G_j
        with phase indices taken modulo T = self.period_T.        
         
        Parameters:
        -----------
        tau_for_G: int or 1D array-like of int either in list or np.array
            Lead time(s) τ in phase steps. Each τ must be a positive integer.
            - If an int is given, returns a single τ result.
            - If an array is given, returns a stack over all τ values.
        
        Returns
        -------
        Gj_tau : np.ndarray
            Propagation operators at lag τ for each phase j.
            - If `tau_for_G` is an int: shape (period_T, spatial_length, spatial_length).
            - If `tau_for_G` is a 1D array of length K: shape (K, period_T, spatial_length, spatial_length).
              The first axis preserves the order of `tau_for_G` (no sorting).

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
               
        Gj_1 = self.propogation_operator_G1()
        how_many_taus = tau_arr.shape[0]
        Gj_tau = np.full((how_many_taus, self.period_T, self.spatial_length, self.spatial_length), np.nan, dtype=complex)

        for itau_idx, itau in enumerate(tau_arr):
            # Build τ-step propagators via modular products
            for j in range(self.period_T):
                inter_matrix = np.eye(self.spatial_length, dtype=complex)
                # Multiply G_{j}, G_{j+1}, ..., G_{j+τ-1}  (indices mod period_T)
                for k in range(itau):
                    idx = (j + k) % self.period_T
                    inter_matrix = Gj_1[idx] @ inter_matrix
                Gj_tau[itau_idx,j] = np.real(inter_matrix)
                
        if tau_arr.shape[0] == 1:
            Gj_tau = np.squeeze(Gj_tau, axis=0)
            Gj_tau = np.real(Gj_tau)      
            
        return Gj_tau
    
    # ---------------------------------   
    
    def noise_covariance_Qj(self):   
                 
        """
        Compute the phase-dependent noise covariance matrices Q_j for a CSLIM.
                
        Returns
        -------
        Qj : numpy.ndarray
            Final phase-dependent noise covariances, shape (period_T, spatial_length, spatial_length), complex Hermitian.
            - If Qj_original is Hermitian and no negative eigenvalues: Qj == Qj_original.
            - If Qj_original is Hermitian with some negatives but the fraction ≤ self.tolerance_for_Q_negative_eigenvalue_number:
              Qj is an adjusted version of Qj_original obtained by clipping negatives to 0 and rescaling positives
              to preserve tr(Qj).
        
        Qj_eigenvalues : numpy.ndarray
            Final eigenvalues of Qj, shape (period_T, spatial_length).
            - If Qj_original is Hermitian and no negative eigenvalues: Qj_eigenvalues == Qj_original_eigenvalues.
            - If Qj_original is Hermitian with some negatives but the fraction ≤ self.tolerance_for_Q_negative_eigenvalue_number:
              Qj_eigenvalues is obtained by clipping negatives to 0 and rescaling positives to preserve tr(Qj).
              
        Qj_eigenvectors : numpy.ndarray
            Corresponding eigenvectors, shape (period_T, spatial_length, spatial_length) (columns = eigenvectors).
            
        Qj_original_negtive_eigenvalue_number : numpy.ndarray
            Count of negative eigenvalues found in the original (pre-adjustment) Q_j, shape (period_T,).
            
        Qj_original: numpy.ndarray
            Unadjusted noise covariance (same shape and dtype as Qj). Returned for inspection
            regardless of whether an adjustment was applied.
            
        Qj_original_eigenvalues: numpy.ndarray
            Eigenvalues of Qj_original (pre-adjustment; shape: (period_T, spatial_length)).
            
        Stability Check
        ---------------               
        We then enforce Hermitian by clipping small negative eigenvalues up to zero
        if the fraction of negative eigenvalues is within the configured tolerance,
        and rescaling positive eigenvalues to preserve trace of Q_j.
        
        """
        
        # Allocate outputs for all phases
        Qj = np.full((self.period_T,self.spatial_length,self.spatial_length), np.nan, dtype=complex)
        Qj_eigenvalues = np.full((self.period_T,self.spatial_length), np.nan, dtype=complex)
        Qj_eigenvectors = np.full((self.period_T,self.spatial_length,self.spatial_length), np.nan, dtype=complex)
        Qj_original_negtive_eigenvalue_number = np.zeros(self.period_T)    
        Qj_original = np.full((self.period_T,self.spatial_length,self.spatial_length), np.nan, dtype=complex)
        Qj_original_eigenvalues = np.full((self.period_T,self.spatial_length), np.nan, dtype=complex)

        Lj = self.dynamical_operator_Lj()
        
        for j in range(self.period_T):
            # Build phase-j covariances: C0_j, C_{j-1}, C_{j+1}
            # Edge phases (j=0 and j=T−1) use shifted slices to align cycles;
            # interior phases use the full slice at phase j.
            
            if j==0:
                xj = self.x_CSLIM_training[:,1:,j]
                C0_j = xj @ xj.T / (xj.shape[-1] - 1)
                
                xjminus1 = self.x_CSLIM_training[:,:-1,(j-1)%self.period_T]
                Cminus1_j = xjminus1 @ xjminus1.T / (xjminus1.shape[-1] - 1)
         
                xjplus1 = self.x_CSLIM_training[:,1:,(j+1)%self.period_T]
                Cplus1_j = xjplus1 @ xjplus1.T / (xjplus1.shape[-1] - 1)   
                                           
            elif (j>0) & (j<self.period_T-1):                           
                xj = self.x_CSLIM_training[:,:,j]
                C0_j = xj @ xj.T / (xj.shape[-1] - 1)
                
                xjminus1 = self.x_CSLIM_training[:,:,(j-1)%self.period_T]
                Cminus1_j = xjminus1 @ xjminus1.T / (xjminus1.shape[-1] - 1)
         
                xjplus1 = self.x_CSLIM_training[:,:,(j+1)%self.period_T]
                Cplus1_j = xjplus1 @ xjplus1.T / (xjplus1.shape[-1] - 1)   
         
            elif j==self.period_T-1:
                xj = self.x_CSLIM_training[:,:-1,j]
                C0_j = xj @ xj.T / (xj.shape[-1] - 1)
            
                xjminus1 = self.x_CSLIM_training[:,:-1,(j-1)%self.period_T]
                Cminus1_j = xjminus1 @ xjminus1.T / (xjminus1.shape[-1] - 1)
         
                xjplus1 = self.x_CSLIM_training[:,1:,(j+1)%self.period_T]
                Cplus1_j = xjplus1 @ xjplus1.T / (xjplus1.shape[-1] - 1)         
            
            Lj_matrix = np.matrix(Lj[j])
         
            # check if Qj is positive definite:
            # (1) check if Qj is symmetric or Hermitian
            # (2) check if Qj's eigenvalues are all positive
            
            Qj_original[j] = (Cplus1_j-Cminus1_j)/2-(Lj_matrix @ C0_j + C0_j @ Lj_matrix.H)
            Qj_original_eigenvalues[j], Qj_eigenvectors[j] = eigh(Qj_original[j])
            # 1) Hermitian requirement
            if is_hermitian(np.matrix(Qj_original[j])):
                if (Qj_original_eigenvalues[j] < 0).any():
                    # 2) If any negatives, decide whether to repair or raise
                    Qj_original_negtive_eigenvalue_number[j] = (Qj_original_eigenvalues[j] < 0).sum()
                    # Tolerance on the *count* of negative eigenvalues
                    if Qj_original_negtive_eigenvalue_number[j] <= self.spatial_length * self.tolerance_for_Q_negative_eigenvalue_number:
                        
                        # Calculate original Q's trace
                        # Preserve total variance: rescale after clipping
                        original_Qjtrace = np.trace(Qj_original[j])
                        Qj_eigenvalues_clipped = np.copy(Qj_original_eigenvalues[j])
                        
                        # Set negative eigenvalues to zero
                        Qj_eigenvalues_clipped[Qj_eigenvalues_clipped < 0] = 0
                        
                        # Calculate original Q's trace after removing the negative values                    
                        clipped_Qjtrace = np.sum(Qj_eigenvalues_clipped)
                        
                        # Rescale positive eigenvalues to preserve trace
                        Qj_adjusted_eigenvalues = Qj_eigenvalues_clipped * (original_Qjtrace / clipped_Qjtrace)
                        
                        # Calculate the adjusted Q
                        Qj_adjusted = Qj_eigenvectors[j] @ np.diag(Qj_adjusted_eigenvalues) @ Qj_eigenvectors[j].T
                        
                        # Save adjusted outputs for phase j
                        Qj[j] = np.real(Qj_adjusted)
                        Qj_eigenvalues[j] = np.real(Qj_adjusted_eigenvalues)
                        
                        msg = (
                            f"Adjusted Q{j+1}: negative eigenvalues were clipped to 0 and "
                            "positive eigenvalues were rescaled to preserve tr(Qj)."
                        )
                        warnings.warn(msg, category=RuntimeWarning, stacklevel=2)

                    else:
                        raise ValueError(f'Q{j+1} is not positive semidefinite within tolerance: too many negative eigenvalues.')
         
                else:
                    Qj[j] = np.real(Qj_original[j])
                    Qj_eigenvalues[j] = np.real(Qj_original_eigenvalues[j])
         
            else: # if Q does not have any negative eigenvalues     
                raise ValueError(f'Q{j+1} is invalid: Q{j+1} is not symmetric or Hermitian')
                
            Qj_eigenvectors[j] = np.real(Qj_eigenvectors[j])     
            Qj_original[j] = np.real(Qj_original[j])
            Qj_original_eigenvalues[j] = np.real(Qj_original_eigenvalues[j])
            
        return Qj, Qj_eigenvalues, Qj_eigenvectors, Qj_original_negtive_eigenvalue_number, Qj_original, Qj_original_eigenvalues
    
    # ---------------------------------   
               
    def deterministic_forecast(self, x_forecast_from, x_forecast_from_j_index, lead_tau_forecast):           
         
        """
        Make the CS-LIM deterministic forecast. 
                 
        Parameters:
        -----------                                   
        x_forecast_from: numpy.ndarray or xarray.DataArray
            The state vector you  want to forecast from in the PCs space
            - Size: [self.spatial_length, 1] or [self.spatial_length,]
            - Notes: If provided as an xarray.DataArray, it will be automatically
            converted to a NumPy array.
                                    
        x_forecast_from_j_index: int
            The phase index j that the first of the x_forecast_from is. Must be a positive integer >0.
            - Note: the index starts from 1
            
        lead_tau_forecast: int or 1D array-like or list-like of int (1-based indices)
            Lead time(s) τ for the forecast. Each τ must be a positive integer.
            - If an int is given, returns a single τ result.
            - If an array or list is given, returns a stack over all τ values.    
            
        Returns
        -------
        x_tau_forecast: numpy.ndarray
            Forecasted state vector from 'x_forecast_from' at lead τ
            - If `lead_tau_forecast` is an int: shape (spatial_length,).
            - If `lead_tau_forecast` is a 1D array of length K: shape (K, spatial_length).
              The first axis preserves the order of `lead_tau_forecast` (no sorting).  
              
        """      

        # ---- Validate input ----
        # Validate or set lead_tau_forecast
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

        # Validate or set x_forecast_from
        # Ensure NumPy array
        if isinstance(x_forecast_from, xr.DataArray):
            x_forecast_from = x_forecast_from.values
        if not isinstance(x_forecast_from, np.ndarray):
            raise TypeError("'x_forecast_from' must be a NumPy array or xarray.DataArray.")                
        x_forecast_from = np.asarray(x_forecast_from)
        if x_forecast_from.shape[0] != self.spatial_length:
            raise ValueError("'x_forecast_from' length must match spatial_length.")                    
        if x_forecast_from.ndim == 2 and x_forecast_from.shape[1] == 1:
            x_forecast_from_new = x_forecast_from
        if x_forecast_from.ndim == 1:
            x_forecast_from_new=np.zeros((self.spatial_length,1))
            x_forecast_from_new[:,0] = x_forecast_from   
        if not np.isfinite(x_forecast_from_new).any():
            raise ValueError("'x_forecast_from' contains NaN or Inf! Please remove or fill them before proceeding.")

        # validate x_forecast_from_j_index
        if not isinstance(x_forecast_from_j_index, int):
            raise TypeError("'x_forecast_from_j_index' must be an integer.")
        if x_forecast_from_j_index < 1:
            raise ValueError("'x_forecast_from_j_index' must be a positive integer.")
        if x_forecast_from_j_index > self.period_T:
            raise ValueError("'x_forecast_from_j_index' must < self.period_T.")
        
        xj_tau_forecast = np.full((tau_arr.shape[0], self.spatial_length), np.nan, dtype=complex)
        Gj_tau = self.propogate_operator_Gtau(tau_for_G=tau_arr)

        for itau_idx, itau in enumerate(tau_arr):
            Gj_tau_for_forecast = Gj_tau[itau_idx,x_forecast_from_j_index-1]
            xj_tau_forecast[itau_idx] = (Gj_tau_for_forecast @ x_forecast_from_new)[:,0]
            xj_tau_forecast[itau_idx] = np.real(xj_tau_forecast[itau_idx])
                   
        return xj_tau_forecast 
               
    # ---------------------------------   
    
    def optimal_initial_condition(self, optimization_time, target_final_condtion_j_index, initial_norm_vector=None, final_norm_vector=None):   
                  
        """
        Compute the optimal initial condition that maximizes amplification at lead τ for a specified phase j in the CSLIM.
                 
        Parameters:
        -----------             
        optimization_time : int
            Positive lead time τ at which the amplification is evaluated.
            - Must be a positive integer.
            
        target_final_condtion_j_index : int
            1-based phase index of the target final state (1 ≤ index ≤ self.period_T).
            Specifies the phase j at which the evolved state is evaluated at lead τ.
            Example (monthly data): target_final_condtion_j_index = 1 → January (2 → February, …, 12 → December).
             
        initial_norm_vector : numpy.ndarray or xarray.DataArray, optional
            Vector (d) defining the initial-state norm (directional weighting).
            - Shape: (spatial_length,) or (spatial_length, 1)
            - Must not be the all-zero vector.
            - Normalization is handled internally (input need not be unit length)
            - If not None, the initial norm is D = d @ d^T
            - If None, the initial norm D is the Euclidean norm (i.e., I).

        final_norm_vector : numpy.ndarray or xarray.DataArray, optional
            Vector (n) defining the final-state norm (directional weighting).
            - Shape: (spatial_length,) or (spatial_length, 1)
            - Must not be the all-zero vector.
            - Normalization is handled internally (input need not be unit length)
            - If not None, the final norm is N = n @ n^T
            - If None, the final norm N is the Euclidean norm (i.e., I).

        Returns
        -------
        maximum_amplification_factor : float
            Leading eigenvalue (real part) of inv(D) @ Gj(τ)^T @ N @ Gj(τ), i.e., the maximum
            amplification achievable at time τ for target final condtion with phase j, under the specified norms.
    
        optimal_initial_condition : numpy.ndarray
            Eigenvector associated with the leading eigenvalue of inv(D) @ Gj(τ)^T @ N @ Gj(τ),
            giving the optimal initial condition at time τ for target final condtion with phase j. Shape: (spatial_length,)

        evoloved_targeted_final_condition: numpy.ndarray
            The evolved state at lead τ from the optimal initial condition:
            G{j_for_initial}(τ) @ optimal_initial_condition, where j_for_initial = target_final_condtion_j_index - optimization_time.
            Shape: (spatial_length,)
       
        """     
        
        # ---- Validate input ----
        # validate optimization_time
        if not isinstance(optimization_time, int):
            raise TypeError(f"'optimization_time' must be an integer, got {type(optimization_time).__name__}.")
        if optimization_time < 1:
            raise ValueError(f"'optimization_time' must be > 0, got {optimization_time}.")
            
        # validate x_forecast_from_j_index
        if not isinstance(target_final_condtion_j_index, int):
            raise TypeError("'target_final_condtion_j_index' must be an integer.")
        if target_final_condtion_j_index < 1:
            raise ValueError("'target_final_condtion_j_index' must be a positive integer.")
        if target_final_condtion_j_index > self.period_T:
            raise ValueError("'target_final_condtion_j_index' must < self.period_T.")

        # ---- Computation ----
        # Propagate to obtain G_j(τ) for all phases
        Gj_tau = self.propogate_operator_Gtau(tau_for_G=optimization_time)
        # Select the target phase’s propagator G_j(τ) (target_final_condition_index is 1-based)
        Gj_tau_target_final_condtion_j_index = Gj_tau[target_final_condtion_j_index-1]
                
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
            if not np.isfinite(final_norm_vector_new).any():
                raise ValueError("'final_norm_vector_new' contains NaN or Inf! Please remove or fill them before proceeding.")                
            # normalize the final norm vector
            n_norm = np.linalg.norm(final_norm_vector_new)
            if n_norm != 0:
                n_normalized = final_norm_vector_new / n_norm                
            else:
                raise ValueError('Final kernal vector elements are all 0')                
            # Compute the final norm N              
            N = n_normalized @ n_normalized.T + epsilon * np.identity(self.spatial_length)
            
        
        Gj_tau_target_final_condtion_j_index_transpose = Gj_tau_target_final_condtion_j_index.transpose()
        invDGT = inv(D) @ Gj_tau_target_final_condtion_j_index_transpose
        invDGTN = invDGT @ N
        invDGTNG = invDGTN @ Gj_tau_target_final_condtion_j_index
        
        # calculate its eigenvalues and eigenvectors
        w, v = eig(invDGTNG)

        # Sort indices by descending eigenvalue
        idx = np.argsort(w)[::-1]        
        # Reorder both eigenvalues and eigenvectors
        w = w[idx]
        v = v[:, idx]
        
        # Leading eigenvalue/eigenvector → maximum amplification and optimal initial condition
        maximum_amplification_factor = w[0]
        optimal_inital_condition = v[:,0]
        
        evoloved_targeted_final_condition = Gj_tau[(target_final_condtion_j_index-optimization_time-1)%self.period_T] @ optimal_inital_condition
        
        return maximum_amplification_factor, optimal_inital_condition, evoloved_targeted_final_condition     
        
    # ---------------------------------   
    
    def simulations_with_noise(self, time_steps, simulation_length, initial_condition=None, initial_condition_j_index=None, seed=None):   
                   
        """
        Generate stochastic simulations using a CSLIM.
                 
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

        initial_state_j_index : int
            1-based phase index of the initial state (1 ≤ index ≤ self.period_T).
            Specifies the phase j at which the initial state is.
            - Default: 1
            - Example (monthly data): target_final_condtion_j_index = 1 → January (2 → February, …, 12 → December).  
            
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

        # validate initial_state            
        if initial_condition_j_index is None:
            initial_condition_j_index=1   
        else:
            if not isinstance(initial_condition_j_index, int):
                raise TypeError("'initial_condition_j_index' must be an integer.")
            if initial_condition_j_index < 1:
                raise ValueError("'initial_condition_j_index' must be a positive integer.")
            if initial_condition_j_index > self.period_T:
                raise ValueError("'initial_condition_j_index' must < self.period_T.")
            
        # validate seed            
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("'seed' must be an integer or None.")
            # Setting the seed here ensures reproducibility of np.random.normal
            rng = np.random.default_rng(seed)    
        
        # ---- Computation ----        
        # Get the dynamical operator L and noise covariance Q
        Lj = self.dynamical_operator_Lj()
        Qj, Qj_eigenvalues, Qj_eigenvectors, _, _, _ = self.noise_covariance_Qj()        
           
        y_t = initial_state_new
                 
        delta_t = 1/time_steps            
        total_steps = time_steps * simulation_length
        
        # Generate an empty array for simulations afterwards
        simulations = np.full((self.spatial_length,simulation_length), np.nan, dtype=float)        
        
        for istep in range(total_steps):      
            
            r_t = np.zeros((self.spatial_length,1))
            if seed is not None:
                r_t[:,0] = rng.standard_normal(size=(self.spatial_length))    
            else:
                r_t[:,0] = np.random.normal(size=(self.spatial_length))
                
            j_index = (initial_condition_j_index-1 + (istep//time_steps)% self.period_T) % self.period_T
            
            Sj =  Qj_eigenvectors[j_index] @ np.sqrt(Qj_eigenvalues[j_index]) 
            
            y_t_plus_delta_t = y_t + delta_t * Lj[j_index] @ y_t + np.sqrt(delta_t) * Sj @ r_t
            
            y_t_plus_half_delta_t = (y_t_plus_delta_t + y_t)/2  
                    
            # Store once per model time unit
            if (istep+1) % time_steps == 0:
                time_index = int(istep//time_steps)-1
                simulations[:,time_index] = np.real(y_t_plus_half_delta_t.ravel())
            
            y_t = y_t_plus_delta_t             
            
        return simulations                  

    # ---------------------------------   
    
    def POPs(self):   
         
        """
        Calculate the principal_oscillation_patterns (POPs) of CSLIM , by the eigenanalysis of the Floquet exponent ρ.
        
        Returns
        -------
        POP_period: numpy.ndarray, shape (m,)
            Oscillation periods (positive; same units as lag time;
            extremely large values or np.inf for statonary or non-oscillatory modes).
        POP_decay: numpy.ndarray, shape (m,)
            E-folding decay times (positive; same units as lag time)
        POP_patterna: numpy.ndarray, shape (self.period_T, m, n_modes)
            Real parts of eigenvectors of Mj. The second axis correspond to spatial dimension (m), 
            and the third axis correspond to POP modes (n_modes).
            Note: since L is an m × m operator, technically m = n_modes.
        POP_patternb: numpy.ndarray, shape (self.period_T, m, n_modes)
            Imaginary parts of eigenvectors of Mj. The second axis correspond to spatial dimension (m), 
            and the third axis correspond to POP modes (n_modes).
            Note: since Mj is an m × m operator, technically m = n_modes.
        
        Notes
        -----
        For the kth Floquet exponent ρ_k = σ_k + iω_k, it is derived by λ_k = ln(ρ_k)/self.period_T
        where λ_k is the kth eigenvalue of Mj:
          - Decay time τ = -1/σ_k
          - Period  T = 2π/|ω_k|
        Eigenvalues of every Mj would be the same.
        Results are sorted by descending decay time (least damped first).
        Oscillatory modes appear in complex-conjugate pairs with identical decay times
        and periods; stationary modes have T = ∞ or huge number.
        
        """
        
        # ---- Computation ----    
        POP_patterna = np.zeros((self.period_T,self.spatial_length))*np.nan
        POP_patternb = np.zeros((self.period_T,self.spatial_length))*np.nan
        
        Mj = self.monodromy_matrix_Mj()        
        for j in range(self.period_T):
            Mj_eigenvalues, Mj_eigenvectors = eig(Mj[j])      

            rho_j = Mj_eigenvalues
            miu_j = np.log(rho_j)/self.period_T
            
            sigma_real_j = miu_j.real
            omega_imag_j = miu_j.imag
            
            POP_period_j = 2*math.pi/np.abs(omega_imag)
            POP_decay_j = -1/sigma_real
            
            # Sort eigenvalues and eigenvectors by the decay time
            sorted_indices = np.argsort(POP_decay_j)[::-1]
            sorted_Mj_eigenvectors = Mj_eigenvectors[:, sorted_indices]            
            POP_patterna[j] = np.real(sorted_Mj_eigenvectors)
            POP_patternb[j] = np.imag(sorted_Mj_eigenvectors)
            
        # eigenvalues of every Mj would be the same, and thus the POP period and decay time,
        # here we use the last Mj to calculate the the POP period and decay time       
        POP_period = POP_period_j[sorted_indices]
        POP_decay = POP_decay_j[sorted_indices]
        
        return POP_period, POP_decay, POP_patterna, POP_patternb
        
    # -------------------------------------------------------------------------   
            
    def POPs_reconstruction(self, modes_want_for_reconstruction=None):
        
        """
        Reconstruct the state vector used for the CSLIM training, using selected Principal Oscillation Patterns (POPs)
        derived from the CSLIM.
    
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
           Reconstructed state vector. Shape: (spatial_length, n_cycles, period_T), same shape as self.x_CSLIM_training
           - Axis meaning:
               * Axis 0 (rows): spatial dimension or variable index
               * Axis 1 (columns): cycles
               * Axis 2: phases within each cycle (length = period_T)
            
        """
        
        x_recons_by_POP = np.full_like(self.x_CSLIM_training)
        
        Mj = self.monodromy_matrix_Mj()
        
        Mj_eigenvectors_sorted = np.zeros((self.period_T,self.spatial_length,self.spatial_length))*np.nan
        Mj_eigenvectors_sorted_H = np.zeros((self.period_T,self.spatial_length,self.spatial_length))*np.nan
        for j in range(self.period_T):
            Mj_eigenvalues, Mj_eigenvectors = eig(Mj[j])   
           
            # calculate the Floquet exponent
            rho_j = Mj_eigenvalues
            miu_j = np.log(rho_j)/self.period_T
            
            sigma_real_j = miu_j.real
            omega_imag_j = miu_j.imag
            
            POP_period_j = 2*math.pi/np.abs(omega_imag)
            POP_decay_j = -1/sigma_real
            
            # Sort eigenvalues and eigenvectors by the decay time
            sorted_indices = np.argsort(POP_decay_j)[::-1]
            Mj_eigenvectors_sorted = Mj_eigenvectors[:, sorted_indices]            
            Mj_eigenvectors_sorted[j] = sorted_Mj_eigenvectors
            Mj_eigenvectors_sorted_H[i] = np.array(np.matrix(inv(Mj_eigenvectors_sorted[i])).getH())

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
        for j in range(self.period_T):
            ui_j = Mj_eigenvectors_sorted[j :, idx0]              # (spatial_length, k)
            vi_j = Mj_eigenvectors_sorted_H[j :, idx0]              # (spatial_length, k)
            viH_j = np.matrix(vi_j).getH()
            
            alpha = viH_j @ self.x_CSLIM_training[:,:,j]
            x_recons_by_POP[:,:,j] = ui_j @ alpha  
            
        return np.asarray(x_recons_by_POP.real)





