
def calculate_guessing_parameter(spline_curve):
    """
    Calculate the average of the first values from all non-empty curves in a dictionary.
    Args:
        spline_curve (dict): Dictionary containing curves as numpy arrays.
    Returns:
        float: The average of the first values from all non-empty curves.
    """
    # Initialize variables to store the sum of first values and the count of non-empty curves
    first_values_sum = 0
    count = 0

    # Iterate through all curves in the dictionary
    for curve in spline_curve.values():
        # Check if the curve has at least one element
        if curve.size > 0:
            # Accumulate the first value
            first_values_sum += curve[0]
            # Increment the count
            count += 1

    # Calculate and return the average
    if count > 0:
        guessing_parameter = first_values_sum / count
        print(f"Task 1 -- Guessing parameter: {guessing_parameter}")
        return guessing_parameter
    else:
        return 0
