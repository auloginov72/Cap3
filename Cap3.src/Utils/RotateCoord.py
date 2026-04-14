import numpy as np

def TO_180_180(fi):
    fi = np.asarray(fi)           # convert to array (safe for scalars)
    fi = fi % 360                 # reduce to 0–360
    fi = np.where(fi > 180, fi - 360, fi)
    return fi

def TO_EVEN_HDG(fi):
    """
    Unwrap heading angles (in degrees) assuming each consecutive
    measurement cannot change by more than ±180 degrees.
    
    fi : array of angles in degrees (any wrap)
    return : continuous angle array
    """
    fi = np.asarray(fi)
    fi_out = np.zeros_like(fi, dtype=float)
    
    # first value unchanged
    fi_out[0] = fi[0]
    
    for i in range(1, len(fi)):
        # raw difference
        diff = fi[i] - fi[i-1]
        
        # normalize difference to -180..180
        diff = TO_180_180(diff)
        
        # build continuous result
        fi_out[i] = fi_out[i-1] + diff
    
    return fi_out

def RotateCoord(X, Y, K, Coord):

    # Convert to numpy arrays and ensure they are column vectors
    X = np.array(X).flatten()
    Y = np.array(Y).flatten()
    K = np.array(K).flatten()
    
    # Check dimensions - return empty arrays if they don't match
    if len(X) != len(Y) or len(X) != len(K):
        return np.array([]), np.array([])
    
    if Coord == "loc2glob":
        return loc2glob(X, Y, K)
    elif Coord == "glob2loc":
        return glob2loc(X, Y, K)
    else:
        # Invalid Coord parameter - return empty arrays
        return np.array([]), np.array([])

def loc2glob(X, Y, K):
    """Transform from local to global coordinates"""
    a1 = np.cos(K)
    a2 = -np.sin(K)
    a3 = -a2  # sin(K)
    a4 = a1   # cos(K)
    
    X_ = a1 * X + a2 * Y
    Y_ = a3 * X + a4 * Y
    
    return X_, Y_

def glob2loc(X, Y, K):
    """Transform from global to local coordinates"""
    a1 = np.cos(K)
    a2 = np.sin(K)
    a3 = -a2  # -sin(K)
    a4 = a1   # cos(K)
    
    X_ = a1 * X + a2 * Y
    Y_ = a3 * X + a4 * Y
    
    return X_, Y_


    """
    Rotate coordinates between local and global coordinate systems
    
    Parameters:
    X, Y : array-like, coordinates
    K : array-like, rotation angles
    Coord : str, either "loc2glob" or "glob2loc"
    
    Returns:
    X_, Y_ : numpy arrays, transformed coordinates (empty if input sizes don't match)
    """