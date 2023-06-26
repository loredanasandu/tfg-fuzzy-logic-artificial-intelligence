import numpy as np

# --- T-norms ---

def t_norm_min(x: float, y: float) -> float:
    """
    Minimum t-norm.
    Defined as the minimum of x and y.

    Args:
        x (float): first value, membership value between 0 and 1
        y (float): second value, membership value between 0 and 1

    Returns:
        float: minimum of x and y, membership value between 0 and 1
    """
    return np.min([x, y], axis=0)


def t_norm_product(x: float, y: float) -> float:
    """
    Product t-norm.
    Defined as the product of x and y.

    Args:
        x (float): first value, membership value between 0 and 1
        y (float): second value, membership value between 0 and 1

    Returns:
        float: product of x and y, membership value between 0 and 1
    """
    return x * y


def t_norm_lukasiewicz(x: float, y: float) -> float:
    """
    Lukasiewicz t-norm.
    Defined as the maximum of 0 and x + y - 1.

    Args:
        x (float): first value, membership value between 0 and 1
        y (float): second value, membership value between 0 and 1

    Returns:
        float: maximum of 1 and x + y, membership value between 0 and 1
    """
    return np.max([np.zeros(x.shape), x + y - 1], axis=0)


# --- T-conorms ---

def t_conorm_max(x: float, y: float) -> float:
    """
    Maximum t-conorm.
    Defined as the maximum of x and y.

    Args:
        x (float): first value, membership value between 0 and 1
        y (float): second value, membership value between 0 and 1

    Returns:
        float: maximum of x and y, membership value between 0 and 1
    """
    return np.max([x, y], axis=0)

def t_conorm_sum(x: float, y: float) -> float:
    """
    Sum t-conorm.
    Defined as x + y - x * y.

    Args:
        x (float): first value, membership value between 0 and 1
        y (float): second value, membership value between 0 and 1

    Returns:
        float: sum of x and y, membership value between 0 and 1
    """
    return x + y - x * y

def t_conorm_lukasiewicz( x: float, y: float) -> float:
    """
    Lukasiewicz t-conorm.
    Defined as the minimum of 1 and x + y.

    Args:
        x (float): first value, membership value between 0 and 1
        y (float): second value, membership value between 0 and 1

    Returns:
        float: minimum of 1 and x + y, membership value between 0 and 1
    """
    return np.min([np.ones(x.shape), x + y], axis=0)

