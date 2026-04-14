import numpy as np
from scipy.interpolate import interp1d
from Utils.RotateCoord import TO_EVEN_HDG

def resample(T,X,T_new,  method='linear' , unwrapDEG=False):
    if (unwrapDEG):
        X=TO_EVEN_HDG(X)
    if method == 'linear':
        pp_obj = pp_linear(T, X)
    elif method == 'const':
        pp_obj = pp_const(T, X)
    elif method == 'cubic':
        pp_obj = pp_cubic(T, X)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'const', or 'cubic'")     

    return ppval(pp_obj, T_new)

def pp_linear(T, X):
    T = np.array(T)
    X = np.array(X)
    
    return interp1d(T, X, kind='linear', bounds_error=False, fill_value=(X[0], X[-1]))  #fill_value='extrapolate'


def pp_const(T, X):
    T = np.array(T)
    X = np.array(X)
    
    return interp1d(T, X, kind='previous', bounds_error=False, fill_value=(X[0], X[-1]))


def pp_cubic(T, X):
    T = np.array(T)
    X = np.array(X)
    
    if len(T) < 4:
        # Fall back to linear for insufficient points
        print("Warning: Not enough points for cubic interpolation, using linear")
        return pp_linear(T, X)
    
    return interp1d(T, X, kind='cubic', bounds_error=False, fill_value='extrapolate')


def ppval(pp_obj, T_new):
    T_new = np.array(T_new)
    return pp_obj(T_new)


def pp_interp(T, X, T_new, method='linear'):
    if method == 'linear':
        pp_obj = pp_linear(T, X)
    elif method == 'const':
        pp_obj = pp_const(T, X)
    elif method == 'cubic':
        pp_obj = pp_cubic(T, X)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'const', or 'cubic'")
    
    return ppval(pp_obj, T_new)


# Test functions
def test_ppform():
    """Test the ppform functions"""
    print("Testing ppform module...")
    
    # Test data
    T = np.array([0, 1, 2, 3, 4, 5])
    X = np.array([10, 15, 12, 18, 20, 16])
    T_new = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    
    # Test linear interpolation
    print("\n1. Testing linear interpolation:")
    pp_lin = pp_linear(T, X)
    X_lin = ppval(pp_lin, T_new)
    print(f"T_new: {T_new}")
    print(f"X_linear: {X_lin}")
    
    # Test constant interpolation
    print("\n2. Testing constant interpolation:")
    pp_const_obj = pp_const(T, X)
    X_const = ppval(pp_const_obj, T_new)
    print(f"X_const: {X_const}")
    
    # Test cubic interpolation
    print("\n3. Testing cubic interpolation:")
    pp_cub = pp_cubic(T, X)
    X_cubic = ppval(pp_cub, T_new)
    print(f"X_cubic: {X_cubic}")
    
    # Test convenience function
    print("\n4. Testing convenience function:")
    X_conv = pp_interp(T, X, T_new, 'linear')
    print(f"X_convenience: {X_conv}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_ppform()


    """
ppform.py - MATLAB-like piecewise polynomial interpolation

Provides MATLAB-compatible functions for interpolation:
- pp_linear(T, X): Create linear interpolation object
- pp_const(T, X): Create step-constant interpolation object  
- pp_cubic(T, X): Create cubic interpolation object
- ppval(pp_obj, T_new): Evaluate interpolation at new points

Usage example:
    import numpy as np
    from ppform import pp_linear, ppval
    
    T = np.array([0, 1, 2, 3, 4, 5])
    X = np.array([10, 15, 12, 18, 20, 16])
    
    # Create interpolation object
    PPX_T = pp_linear(T, X)
    
    # Use it
    T_new = np.arange(0, 5, 0.1)
    X_new = ppval(PPX_T, T_new)
"""

