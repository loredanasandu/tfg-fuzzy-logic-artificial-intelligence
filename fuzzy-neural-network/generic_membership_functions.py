# --- Generic membership functions for input variables ---

def triangular_membership_function(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

    Args:
        x (float): input value
        a (float): leftmost point of the triangle
        b (float): middle point of the triangle
        c (float): rightmost point of the triangle

    Returns:
        float: membership value between 0 and 1
    """

    try:
        if x < a:
            return 0
        elif a <= x <= b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return (c - x) / (c - b)
        else:
            return 0
    except ZeroDivisionError:
        return 0
    

def left_trapezoidal_membership_function(x: float, a: float, b: float) -> float:
    """
    Left trapezoidal membership function.

    Args:
        x (float): input value
        a (float): leftmost point of the trapezoid
        b (float): rightmost point of the trapezoid

    Returns:
        float: membership value between 0 and 1
    """
    try:
        if x <= a:
            return 1
        elif a <= x <= b:
            return (b - x) / (b - a)
        else:
            return 0
    except ZeroDivisionError:     
        return 0
    
def right_trapezoidal_membership_function(x: float, a: float, b: float) -> float:
    """
    Right trapezoidal membership function.

    Args:
        x (float): input value
        a (float): leftmost point of the trapezoid
        b (float): rightmost point of the trapezoid

    Returns:
        float: membership value between 0 and 1
    """
    try:
        if x <= a:
            return 0
        elif a <= x <= b:
            return (x - a) / (b - a)
        else:
            return 1
    except ZeroDivisionError:
        return 0