import matplotlib.pyplot as plt
import pandas as pd


data_rk4 = pd.read_csv("pendulum_data_RK4.csv")
data_dp8 = pd.read_csv("pendulum_data_DP8.csv")

diff_theta1 = data_rk4['Theta1'] - data_dp8['Theta1']
diff_theta2 = data_rk4['Theta2'] - data_dp8['Theta2']
diff_omega1 = data_rk4['Omega1'] - data_dp8['Omega1']
diff_omega2 = data_rk4['Omega2'] - data_dp8['Omega2']

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(data_rk4['Time'], diff_theta1, color='#1f77b4')
plt.title('Разница theta1 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Разница, радианы', fontsize=10)

plt.subplot(2, 2, 2)
plt.plot(data_rk4['Time'], diff_theta2, color='#ff7f0e')
plt.title('Разница theta2 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Разница, радианы', fontsize=10)

plt.subplot(2, 2, 3)
plt.plot(data_rk4['Time'], diff_omega1, color='#2ca02c')
plt.title('Разница omega1 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Разница, радианы/с', fontsize=10)

plt.subplot(2, 2, 4)
plt.plot(data_rk4['Time'], diff_omega2, color='#d62728')
plt.title('Разница omega2 (RK4 - DP8)', fontsize=12)
plt.xlabel('Время, с', fontsize=10)
plt.ylabel('Разница, радианы/с', fontsize=10)

plt.tight_layout()
plt.savefig('difference_RK4_DP8.png', dpi=300)
plt.close()


print("Среднее значение разницы theta1: ", sum(diff_theta1)/len(diff_theta1))
print("Среднее значение theta2: ", sum(diff_theta2)/len(diff_theta1))
print("Среднее значение omega1: ", sum(diff_omega1)/len(diff_omega1))
print("Среднее значение omega2: ", sum(diff_omega2)/len(diff_omega2))