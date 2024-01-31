# adapted, original: chat gpt
import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

def extract_information(file_path):
    # Initialize variables to store information
    data = []
    current_epoch = None
    current_i = None
    current_data = {}


    # Open and read the .out file
    with open(file_path, 'r') as file:
        for line in file:
            # Extract information using regular expressions
            epoch_match = re.match(r'Epoch: (\d+), i = (\d+), Alpha = ([\d.]+)', line)
            accuracy_loss_match = re.match(r'Train Accuracy \(Average\):[^\d]+([\d.]+)%[^\d]+([\d.]+)', line)
            # accuracy_loss_match = re.match(r'Train Accuracy \(Average\):[^\d]+([\d.]+)%[^\d]*(-?[\d.]+)', line)
            if epoch_match:
                # Update current_epoch and initialize a new dictionary for the data
                current_epoch = int(epoch_match.group(1))
                current_i = int(epoch_match.group(2))
                current_alpha = float(epoch_match.group(3))
                current_data = {'epoch': current_epoch}
            elif accuracy_loss_match:
                # Update current_data with accuracy and loss information
                beg_acc = accuracy_loss_match.regs[2][0]-1
                loss = float(accuracy_loss_match.group(2))
                if line[beg_acc] == '-':
                    loss *= -1

                current_data.update({
                    'i': current_i,
                    'alpha': current_alpha,
                    'accuracy': float(accuracy_loss_match.group(1)),
                    'loss': loss
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

def plot_loss_acc(extracted_data, show_alpha = False, fail=False):


    if show_alpha:
        fig, ax = plt.subplots(3, 1)
    else:
        fig, ax = plt.subplots(2, 1)

    # Plot loss
    epoch = []
    cnt = 0
    loss = []
    accuracy = []
    alpha = []

    while extracted_data[cnt]['epoch'] <= extracted_data[cnt+1]['epoch']:
        epoch.append(int(extracted_data[cnt]['epoch']))
        loss.append(extracted_data[cnt]['loss'])
        accuracy.append(extracted_data[cnt]['accuracy'])
        alpha.append(extracted_data[cnt]['alpha'])
        cnt += 1

    avg_loss = []
    avg_acc = []
    avg_alpha = []
    avg_epochs = []
    if fail:
        av = 1
        av2 = av-1
    else:
        av = 1
        av2 = av-1
    for i in range(av2, len(loss), av):
        avg_loss.append(sum(loss[i-av2:i+1])/av)
        avg_acc.append(sum(accuracy[i-av2:i+1])/av)
        avg_epochs.append(np.min(epoch[i-av2:i+av]))
        avg_alpha.append(sum(alpha[i-av2:i+1])/av)
    if fail:
        color = 'slategrey'
    else:
        color = 'navy'

    fig.suptitle('Average Results during Training')
    ax[0].sharex(ax[1])
    ax[0].set_ylabel('Loss')
    ax[0].plot(avg_epochs, avg_loss, label='Loss', color=color)
    ax[0].set_ylim([0, 0.35])

    if show_alpha:
        ax[1].sharex(ax[2])
        ax[2].set_xlabel('Epochs')
        ax[2].set_ylabel('Alpha')
        ax[2].plot(avg_epochs, avg_alpha, label='Alpha', color=color)
        ax[2].set_ylim([0.05, 0.2])
    else:
        ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy [%]')
    ax[1].plot(avg_epochs, avg_acc, label='Accuracy', color=color)
    ax[1].set_ylim([80, 100])





    """ax2 = ax.twinx()
    ax2.plot(accuracy, label='Accuracy', color='cornflowerblue')"""

    plt.show()
    fig.tight_layout()

    if show_alpha:
        fig.savefig(f'results_n/loss_acc_alpha.png', dpi=1200)
    else:
        fig.savefig(f'results_n/loss_acc.png', dpi=1200)



if __name__ == "__main__":
    # Input file path
    input_file = "../Kopie_HPC/scripts/slurm-23153868.out"
    # Extract information from the .out file
    extracted_data = extract_information(input_file)

    # Output folder path
    output_folder = 'results'

    # Save extracted data to a .txt file in the 'results' folder
    # save_to_txt(extracted_data, output_folder)
    for show_alpha in [True, False]:
        plot_loss_acc(extracted_data, show_alpha, fail=False)


