from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import os
import csv
from collections import Counter

def find_best_l1(losses):
    # Initialize variables to store the best L1 factor and its corresponding loss
    best_loss = float('inf')
    best_l1_factor = None
    # Iterate through the losses dictionary to find the factor with the lowest loss
    for l1_factor, val_losses in losses.items():
        # You can choose to consider the final loss, or the average/minimum loss
        final_loss = val_losses[-1]  # Considering the final loss of the last epoch
        # final_loss = np.mean(val_losses)  # Alternatively, consider the average loss
    #    final_loss = min(val_losses)  # Or, consider the minimum loss

        if final_loss < best_loss:
            best_loss = final_loss
            best_l1_factor = l1_factor
    return best_l1_factor


def plot_losses(losses):
    # Plot the reconstruction losses for each L1 factor
    plt.figure(figsize=(10, 6))
    for l1_factor, loss in losses.items():
        plt.plot(range(1, 51), loss, label=f'L1={l1_factor}')

    plt.title('Reconstruction Loss vs. L1 Regularization Factor')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.show()

def get_scaled_data(file_path):

    data = pd.read_csv(file_path)
    y = data['target']
    data = data.drop('target', axis=1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, y

def get_losses(file_path, encoding_dim):
    scaled_data, y = get_scaled_data(file_path)
    losses = {}
     # Define a list of L1 regularization factors to experiment with
    l1_factors = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # If your goal is feature learning or data denoising, 
    # where maintaining the data structure is important, you might choose an encoding_dim close to 10. 
    # This would not compress your data much but can help the model learn a useful representation of your data.#   
    input_dim = scaled_data .shape[1]

    # this is our input placeholder
    input_img = Input(shape=(input_dim))

    # Assuming 'scaled_data' is your dataset and 'y' is your labels
    # Split the dataset into training and testing sets using scikit-learn
    x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=42)

    # Loop through the L1 factors and train autoencoders
    for l1_factor in l1_factors:
        # Encoder and Decoder layers
        encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(input_img)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)  # Adjusted to 'input_dim'

        # Define and compile the autoencoder model
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Train the autoencoder
        history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
        
        # Store the reconstruction loss for this L1 factor
        losses[l1_factor] = history.history['val_loss']
        return losses

def write_to_file(path_dir, csv_file_name, factors):
    # Ensure the directory exists or create it if necessary
    os.makedirs(path_dir, exist_ok=True)

    # Create a list with the data you want to write to the CSV file
    # Here, assuming each sublist in 'factors' should be a separate row
    data_to_write = [["Autoencoder_Factors"]] + factors
    
    file_path = os.path.join(path_dir, csv_file_name)
    
    try:
        # Open the CSV file for writing
        with open(file_path, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)
            
            # Write the data to the CSV file
            csv_writer.writerows(data_to_write)

        print(f"Data has been written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def find_top_dim_overlapped_lists(factors, dim):
    # Count the frequency of each element across all lists
    element_frequency = Counter(element for sublist in factors for element in sublist)

    # Calculate the overlap score for each sublist (sum of frequencies of its elements)
    overlap_scores = [(sum(element_frequency[element] for element in sublist), index) for index, sublist in enumerate(factors)]

    # Sort the sublists by their overlap scores in descending order and get the top 'dim' sublists
    top_dim_overlapped_indices = sorted(overlap_scores, reverse=True)[:(dim-1)]

    # Get the most overlapped sublists using the found indices
    top_dim_overlapped_sublists = [factors[index] for _, index in top_dim_overlapped_indices]
    return top_dim_overlapped_sublists

    print("The top", dim, "overlapped sublists are:", top_dim_overlapped_sublists)
