import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  

# Загрузка данных
data_rk4 = pd.read_csv("pendulum_data_RK4.csv")
data_dp8 = pd.read_csv("pendulum_data_DP8.csv")

abs_diff_theta1 = np.abs(data_rk4['Theta1'] - data_dp8['Theta1'])
abs_diff_theta2 = np.abs(data_rk4['Theta2'] - data_dp8['Theta2'])
abs_diff_omega1 = np.abs(data_rk4['Omega1'] - data_dp8['Omega1'])
abs_diff_omega2 = np.abs(data_rk4['Omega2'] - data_dp8['Omega2'])

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(data_rk4['Time'], abs_diff_theta1, color='#1f77b4')
plt.yscale('log')  # Логарифмическая шкала по Y
plt.title('Абсолютная разница theta1 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Абс. разница, радианы (log)', fontsize=10)

plt.subplot(2, 2, 2)
plt.plot(data_rk4['Time'], abs_diff_theta2, color='#ff7f0e')
plt.yscale('log')
plt.title('Абсолютная разница theta2 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Абс. разница, радианы (log)', fontsize=10)

plt.subplot(2, 2, 3)
plt.plot(data_rk4['Time'], abs_diff_omega1, color='#2ca02c')
plt.yscale('log')
plt.title('Абсолютная разница omega1 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Абс. разница, радианы/с (log)', fontsize=10)

plt.subplot(2, 2, 4)
plt.plot(data_rk4['Time'], abs_diff_omega2, color='#d62728')
plt.yscale('log')
plt.title('Абсолютная разница omega2 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Абс. разница, радианы/с (log)', fontsize=10)

plt.tight_layout()
plt.savefig('difference_RK4_DP8_log.png', dpi=300)
plt.close()

diff_theta1 = data_rk4['Theta1'] - data_dp8['Theta1']
diff_theta2 = data_rk4['Theta2'] - data_dp8['Theta2']
diff_omega1 = data_rk4['Omega1'] - data_dp8['Omega1']
diff_omega2 = data_rk4['Omega2'] - data_dp8['Omega2']

print("Среднее значение разницы theta1: ", np.mean(diff_theta1))
print("Среднее значение разницы theta2: ", np.mean(diff_theta2))
print("Среднее значение разницы omega1: ", np.mean(diff_omega1))
print("Среднее значение разницы omega2: ", np.mean(diff_omega2))