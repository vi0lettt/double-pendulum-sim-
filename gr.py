import matplotlib.pyplot as plt
import pandas as pd
import glob

plt.style.use('Solarize_Light2')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

files = glob.glob("pendulum_data_*.csv")


for i, filename in enumerate(files):
    data = pd.read_csv(filename)
    method_name = filename.split('_')[-1].split('.')[0]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Метод интегрирования: {method_name}', fontsize=16)
    
    axs[0,0].plot(data['Time'], data['Theta1'], color=colors[i])
    axs[0,0].set_title('Угол theta1', fontsize=12)
    axs[0,0].set_xlabel('Время, с', fontsize=10)
    axs[0,0].set_ylabel('Радианы', fontsize=10)
    
    axs[0,1].plot(data['Time'], data['Theta2'], color=colors[i])
    axs[0,1].set_title('Угол theta2', fontsize=12)

    axs[1,0].plot(data['Time'], data['Omega1'], color=colors[i])
    axs[1,0].set_title('Скорость omega1', fontsize=12)

    axs[1,1].plot(data['Time'], data['Omega2'], color=colors[i])
    axs[1,1].set_title('Скорость omega2', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'plots_{method_name}.png', dpi=300)
    plt.close()