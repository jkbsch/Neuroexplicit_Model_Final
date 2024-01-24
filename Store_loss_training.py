# copied from chatgpt
import os
import re
import csv

def extract_information(file_path):
    # Initialize variables to store information
    data = []
    current_epoch = None
    current_data = {}

    # Open and read the .out file
    with open(file_path, 'r') as file:
        for line in file:
            # Extract information using regular expressions
            epoch_match = re.match(r'Epoch: (\d+), i = (\d+), Alpha = ([\d.]+)', line)
            accuracy_loss_match = re.match(r'Train Accuracy \(Average\):[^\d]+([\d.]+)%[^\d]+([\d.]+)', line)

            if epoch_match:
                # Update current_epoch and initialize a new dictionary for the data
                current_epoch = int(epoch_match.group(1))
                current_data = {'epoch': current_epoch}
            elif accuracy_loss_match:
                # Update current_data with accuracy and loss information
                current_data.update({
                    'i': int(epoch_match.group(2)),
                    'alpha': float(epoch_match.group(3)),
                    'accuracy': float(accuracy_loss_match.group(1)),
                    'loss': float(accuracy_loss_match.group(2))
                })
                data.append(current_data)

    return data

def save_to_txt(data, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Create the output file path
    output_file = os.path.join(output_folder, 'output.txt')

    # Write data to the output file in CSV format
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'i', 'alpha', 'accuracy', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write data
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    # Input file path
    input_file = input("Kopie_HPC/scripts/slurm-23153864.out")

    # Extract information from the .out file
    extracted_data = extract_information(input_file)

    # Output folder path
    output_folder = 'results'

    # Save extracted data to a .txt file in the 'results' folder
    save_to_txt(extracted_data, output_folder)

    print("Extraction and saving complete. Results saved in 'results/output.txt'.")
