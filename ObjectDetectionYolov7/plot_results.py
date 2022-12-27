import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(sns.color_palette("husl", 3))


def get_training_results(path):
    """"
        Read a csv file with training results and return it as a Pandas DataFrame.

    Parameters:
        path (str): Path to the csv file.

    Returns:
        Pandas DataFrame: DataFrame with training results.
    """""
    return pd.read_csv(path,
                       header=None,  # No header row
                       names=['Epoch', 'gpu_mem', 'train/box_loss', 'train/obj_loss', 'train/cls_loss',
                              'train/total_loss', 'train/labels', 'train/img_size',
                              'metrics/precision', 'metrics/recall', 'metrics/mAP_0.25', 'metrics/mAP_0.50',
                              'metrics/mAP_0.75', 'val/box_loss', 'val/obj_loss', 'val/cls_loss'],  # Column names
                       delim_whitespace=True)


def get_test_results(path):
    """"
        This function reads in a test results file and returns a pandas DataFrame.
        
        Parameters:
            path (str): path to the test results file
            
        Returns:
            pandas.DataFrame: test results data
    """""
    return pd.read_csv(path, delim_whitespace=True)


def plot_metrics(training_results):
    """"
        Plots the validation mAP for different IoU thresholds (0.25, 0.5, and 0.75).
        
    Parameters:
        training_results (Pandas DataFrame): DataFrame containing the training results. 
                                             The DataFrame should have three columns:
                                             'metrics/mAP_0.25', 'metrics/mAP_0.50', and 'metrics/mAP_0.75'.
        
    Returns:
        None
    """""

    # Initialize figure with 1 row and 3 columns, with specified size
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # Flatten the axes array so it can be iterated over
    axs = axs.flatten()

    # Set up the different metric types that will be plotted
    metric_types = ['mAP_0.25', 'mAP_0.50', 'mAP_0.75']
    # Iterate over each metric type
    for i, metric_type in enumerate(metric_types):
        # Plot the metric values for each epoch on the corresponding axis
        axs[i].plot(range(len(training_results)), training_results[f'metrics/{metric_type}'])
        # Set the title for the plot
        axs[i].set_title(f'Validation {metric_type}')
        # Set the y-axis limits
        axs[i].set_ylim((0, 1.1))
        # Set the x-axis label
        axs[i].set_xlabel('Epoch')
    # Show the plot
    plt.show()


def plot_losses(training_results):
    """"
    Plots training and validation loss for the objective, box, and cls losses.

    Parameters:
        training_results (DataFrame): A DataFrame containing the training and validation loss values.
            The dictionary should have the following keys: 'train/obj_loss', 'val/obj_loss',
            'train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss'.

    Returns:
        None
    """""
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    axs = axs.flatten()

    # Set the loss types to plot
    loss_types = ['obj', 'box', 'cls']

    # Iterate through loss types
    for i, loss_type in enumerate(loss_types):
        # Plot training and validation loss for the current loss type
        axs[i].plot(range(len(training_results)), training_results[f'train/{loss_type}_loss'], label='Training loss')
        axs[i].plot(range(len(training_results)), training_results[f'val/{loss_type}_loss'], label='Validation loss')
        # Set the title and x-axis label for the current loss type
        axs[i].set_title(f'Training and Validation {loss_type.upper()} Loss')
        axs[i].set_xlabel('Epoch')
    # Show the legend
    plt.legend()
    # Show the plot
    plt.show()


def plot_test_results(test_results):
    """"
    Plot the results from a test.

    Parameters:
    test_results (DataFrame): The test results to plot. Must include the following columns:
        'Class': The class label for each row.
        'AP@.25': The AP@.25 value for each row.
        'AP@.5': The AP@.5 value for each row.
        'AP@.75': The AP@.75 value for each row.

    Returns:
    None
    """""

    # Set the figure size
    plt.figure(figsize=(20, 6))

    # Define the width of the bars
    bar_width = 0.25

    # Define the labels for the bar plot
    labels = test_results['Class'][1:]

    # Define the values for the bar plot
    ap_25 = test_results['AP@.25'][1:]
    ap_5 = test_results['AP@.5'][1:]
    ap_75 = test_results['AP@.75'][1:]

    # Create the bar plot
    plt.bar([float(x) for x in range(len(labels))], ap_25, label='AP@.25', width=bar_width)
    plt.bar([float(x) + bar_width for x in range(len(labels))], ap_5, label='AP@.5', width=bar_width)
    plt.bar([float(x) + 2 * bar_width for x in range(len(labels))], ap_75, label='AP@.75', width=bar_width)
    plt.ylim((0, 1.1))

    # Set the tick labels
    plt.xticks([r + bar_width for r in range(len(labels))], labels)

    # Add a legend to the plot
    plt.legend()

    # Add the value of each bar to the plot
    for i, v in enumerate(ap_25):
        plt.text(i - bar_width / 2, v + 0.01, str(v))
    for i, v in enumerate(ap_5):
        plt.text(i + bar_width / 2, v + 0.01, str(v))
    for i, v in enumerate(ap_75):
        plt.text(i + 3 * bar_width / 2, v + 0.01, str(v))

    plt.title('Test Results')

    # Show the plot
    plt.show()


def plot_all_experiments_test_results(all_test_results):
    """"
        Plots test results for multiple experiments on the same plot.

    Parameters:
        all_test_results (list of DataFrames): A list of DataFrames containing the test results for each experiment.
            Each DataFrame should have the following structure:
                - 'Class': Class labels for each object.
                - 'AP@.25', 'AP@.5', 'AP@.75': Average precision values for each object at different IoU thresholds.

    Returns:
        None
    """""
    metric_types = ['AP@.25', 'AP@.5', 'AP@.75']
    bar_width = 0.25
    for i, metric_type in enumerate(metric_types):
        plt.figure(figsize=(18, 6))

        for j, exp in enumerate(all_test_results):
            labels = exp['Class'].loc[1:]
            # Create the bar plot
            plt.bar([float(x) + j * bar_width for x in range(len(labels))], exp[metric_type].loc[1:],
                    label=f'Experiment {j+1}', width=bar_width)
        # Set the tick labels
        plt.xticks([r + bar_width for r in range(len(labels))], labels)

        # Add a legend to the plot
        plt.legend()
        plt.ylim((0, 1.1))
        plt.title(f'Test {metric_type}')

        # Add the value of each bar to the plot
        for k, exp in enumerate(all_test_results):
            for t, v in enumerate(exp[metric_type].loc[1:]):
                plt.text(t + k * bar_width - bar_width / 2, v + 0.01, str(round(v, 2)))

        # Show the plot
        plt.show()


def plot_all_experiments_losses(all_train_results):
    """"
        Plots training and validation loss for the objective, box, and cls losses for all training experiments.
        
    Parameters:
    all_train_results (list): A list of DataFrames containing the training and validation loss values for each 
        experiment. Each DataFrame should have the following keys: 'train/obj_loss', 'val/obj_loss',
        'train/box_loss', 'val/box_loss', 'train/cls_loss', 'val/cls_loss'.

    Returns:
        None
    """""
    loss_types = ['obj', 'box', 'cls']

    for set_type in ['train', 'val']:
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))
        axs = axs.flatten()
        for i, loss_type in enumerate(loss_types):
            for j, exp in enumerate(all_train_results):
                axs[i].plot(range(len(exp)), exp[f'{set_type}/{loss_type}_loss'], label=f'Experiment {j+1}')
            axs[i].set_title(f'{set_type.upper()} {loss_type.upper()} Loss')
            axs[i].set_xlabel('Epoch')
        plt.legend()
        plt.show()


def total_metrics_ovel_all_the_experiments(all_test_results):
    """"
        Plots the average precision over all experiments for AP@.25, AP@.50, and AP@.75.

    Parameters:
        all_test_results (List[DataFrame]): A list of DataFrames containing the test results for each experiment.
            Each DataFrame should have the following columns: 'Class', 'AP@.25', 'AP@.50', and 'AP@.75'.

    Returns:
        None
    """""
    fig, axs = plt.subplots(1, 3, figsize=(16, 7))
    axs = axs.flatten()
    bar_width = 0.25

    metric_types = ['AP@.25', 'AP@.5', 'AP@.75']
    for i, metric_type in enumerate(metric_types):
        values = [exp[metric_type][0] for j, exp in enumerate(all_test_results)]
        axs[i].bar([float(x) + bar_width for x in range(len(all_test_results))], values, width=bar_width)
        axs[i].set_ylim((0, 1.1))
        axs[i].set_title(f'Test m{metric_type}')

        # Add the value of each bar to the plot
        for j, v in enumerate(values):
            axs[i].text(j + bar_width / 2, v + 0.01, str(v))

        # Set the tick labels
        axs[i].set_xticks([r + bar_width for r in range(len(all_test_results))],
                          [f'Experiment {r+1}' for r in range(len(all_test_results))])

    plt.show()


def plot_all_experiments_training_metrics(all_train_results):
    """"
        Plots the validation mAP metric for each experiment in the given list of training results.

    Parameters:
        all_train_results (List[DataFrame]): A list of training results DataFrames. Each DataFrame should contain
            the 'metrics/mAP_0.25', 'metrics/mAP_0.50', and 'metrics/mAP_0.75' columns.

    Returns:
        None
    """""
    metric_types = ['mAP_0.25', 'mAP_0.50', 'mAP_0.75']
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs = axs.flatten()
    for i, metric_type in enumerate(metric_types):
        for j, exp in enumerate(all_train_results):
            axs[i].plot(range(len(exp)), exp[f'metrics/{metric_type}'], label=f'Experiment {j+1}')
        axs[i].set_title(f'Validation {metric_type}')
        axs[i].set_xlabel('Epoch')
    plt.legend()
    plt.show()
