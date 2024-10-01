# Import the C extension using a relative import
import cfor


def c4(
        data,
        threshold=None,
        condition=">",
        use_mp=None,
        use_simd=True
        ):
    """
    A user-friendly wrapper around the C extension `c4`.

    Parameters:
    - data (list or pandas Series): The data to process.
    - threshold (float, optional): The threshold for the condition.
    - condition (str, optional): Condition to apply. Options: ">", "<", "==", <=, >= or None for no condition.
    - use_mp (bool, optional): Whether to use multi-processing. Defaults to True for large datasets.
    - use_simd (bool, optional): Whether to use simd. Defaults to True
    
    Returns:
    - list: Transformed data based on the condition.
    """
    
    # Validate input data
    if not isinstance(data, list):
        raise ValueError("Data must be provided as a Python list")
    
    # Default threshold to 0 if None provided
    if threshold is None:
        threshold = 0.0

    # Map condition strings to condition types used in the C extension
    condition_mapping = {
        ">": 0,
        "<": 1,
        "==": 2,
        None: -1,  # No condition
        ">=": 3,
        "<=": 4
    }

    # Set condition_type based on user input
    if condition not in condition_mapping:
        raise ValueError("Invalid condition. Choose from '>', '<', '==', '>=', '<=', or None")

    condition_type = condition_mapping[condition]

    # Default use_mp based on the size of the data (use multi-processing for large datasets)
    if use_mp is None:
        use_mp = len(data) > 10000  # Enable MP for datasets larger than 10,000 elements

    # Convert `use_mp` from bool to the expected int for the C extension
    use_mp_flag = 1 if use_mp else 0

    # Convert `use_simd` from bool to the expected int for the C extension
    use_simd_flag = 1 if use_simd else 0

    # Call the C extension and return the result
    return cfor.c4(data, threshold, condition_type, use_mp_flag, use_simd_flag)
