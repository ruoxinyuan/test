import numpy as np
import pandas as pd

def process_action_diffs(X_test_loaded, spline_curve):
    """
    Process action differences and generate a summary dataframe.
    Args:
        X_test_loaded (torch.Tensor): Tensor containing sequences of (action, time) pairs.
        spline_curve (dict): Dictionary containing fitted B-spline curves for each sample ID.
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame summarizing median, mean, and count of action differences.
            - pd.DataFrame: Top 5 actions with highest median differences.
            - pd.DataFrame: Bottom 5 actions with lowest median differences.
    """
    
    def create_test_df(X_test_loaded):
        """
        Create a DataFrame from X_test_loaded containing 'id', 'action', and 'time' columns.
        """
        # Initialize lists to hold data for DataFrame
        ids = []
        actions = []
        times = []

        # Iterate over each sample in X_test_loaded
        for sample_id, sample in enumerate(X_test_loaded):
            # sample is a 2D tensor of shape (length, 2), where 2 represents (action, time)
            for step in sample:
                ids.append(sample_id)  # Assign unique ID for each sample
                actions.append(step[0].item())  # Extract action
                times.append(step[1].item())  # Extract time

        # Create a DataFrame
        test_df = pd.DataFrame({
            'id': ids,
            'action': actions,
            'time': times
        })
        test_df = test_df[test_df['action'] != -1].reset_index(drop=True)

        return test_df

    test_df = create_test_df(X_test_loaded)

    # Filter based on spline_curve keys
    test_slope1 = test_df[test_df['id'].isin(spline_curve.keys())].copy()

    # Initialize a list to store action differences
    action_diffs = []

    # Iterate through each valid sample ID
    for id_ in spline_curve.keys():
        x = np.arange(1, len(spline_curve[id_]) + 1)
        fitted_curve = spline_curve[id_]

        # Compute differences between adjacent values
        diffs = fitted_curve[1:] - fitted_curve[:-1]
        action_diffs.extend(diffs)

        # Insert NaN at the last position for alignment
        diffs_with_nan = np.insert(diffs, x[-1] - 1, np.nan)
        test_slope1.loc[test_slope1['id'] == id_, 'action_diffs'] = diffs_with_nan

    # Group by 'action' and calculate statistics
    grouped_median = test_slope1.groupby('action')['action_diffs'].median().reset_index()
    grouped_mean = test_slope1.groupby('action')['action_diffs'].mean().reset_index(drop=True)
    grouped_count = test_slope1.groupby('action')['action_diffs'].count().reset_index(drop=True)

    # Combine results into a single DataFrame
    grouped = pd.concat([grouped_median, grouped_mean, grouped_count], axis=1)
    grouped.columns = ['action', 'median', 'mean', 'count']
    
    # Sort by 'median' column
    grouped_sorted = grouped.sort_values(by='median', ascending=False).dropna(subset=['median'])

    # Select top 5 and bottom 5 actions by 'median'
    good_1grams = grouped_sorted.head(5).reset_index(drop=True)
    bad_1grams = grouped_sorted.tail(5).reset_index(drop=True)

    print("Task 2 -- ")
    print("\nTop 5 1-grams with highest median differences:")
    print(good_1grams)
    print("\nBottom 5 1-grams with lowest median differences:")
    print(bad_1grams)

    return grouped, good_1grams, bad_1grams


def process_2gram_diffs(X_test_loaded, spline_curve):
    """
    Process 2-gram action differences and generate a summary dataframe.
    """
    def create_test_df(X_test_loaded):
        """
        Create a DataFrame from X_test_loaded containing 'id', 'action', and 'time' columns.
        """
        ids = []
        actions = []
        times = []
        for sample_id, sample in enumerate(X_test_loaded):
            for step in sample:
                ids.append(sample_id)
                actions.append(step[0].item())
                times.append(step[1].item())
        test_df = pd.DataFrame({
            'id': ids,
            'action': actions,
            'time': times
        })
        test_df = test_df[test_df['action'] != -1].reset_index(drop=True)
        return test_df

    # Create DataFrame
    test_df = create_test_df(X_test_loaded)

    # Filter based on spline_curve keys
    test_slope2 = test_df[test_df['id'].isin(spline_curve.keys())].copy()

    # Add columns for pre_action and 2gram
    test_slope2['pre_action'] = test_slope2['action'].shift(1)
    test_slope2['2gram'] = test_slope2.apply(lambda row: (row['pre_action'], row['action']), axis=1)

    # Initialize a list to store 2-gram differences
    gram_diffs = []

    # Iterate through each valid sample ID
    for id_ in spline_curve.keys():
        fitted_curve = spline_curve[id_]

        # Compute differences for 2-gram
        diffs = fitted_curve[2:] - fitted_curve[:-2]
        gram_diffs.extend(diffs)

        # Insert NaN for alignment
        diffs_with_nan = np.insert(diffs, 0, np.nan)  # Add NaN at the beginning
        diffs_with_nan = np.insert(diffs_with_nan, len(diffs_with_nan), np.nan)  # Add NaN at the end
        test_slope2.loc[test_slope2['id'] == id_, 'gram_diffs'] = diffs_with_nan

    # Group by '2gram' and calculate statistics
    grouped_median = test_slope2.groupby('2gram')['gram_diffs'].median().reset_index()
    grouped_mean = test_slope2.groupby('2gram')['gram_diffs'].mean().reset_index(drop=True)
    grouped_count = test_slope2.groupby('2gram')['gram_diffs'].count().reset_index(drop=True)

    # Combine results into a single DataFrame
    grouped = pd.concat([grouped_median, grouped_mean, grouped_count], axis=1)
    grouped.columns = ['2gram', 'median', 'mean', 'count']
    
    # Sort by 'median' column
    grouped_sorted = grouped.sort_values(by='median', ascending=False).dropna(subset=['median'])

    # Select top 5 and bottom 5 2-grams by 'median'
    good_2grams = grouped_sorted.head(5).reset_index(drop=True)
    bad_2grams = grouped_sorted.tail(5).reset_index(drop=True)

    print("\nTop 5 2-grams with highest median differences:")
    print(good_2grams)
    print("\nBottom 5 2-grams with lowest median differences:")
    print(bad_2grams)

    return grouped, good_2grams, bad_2grams
