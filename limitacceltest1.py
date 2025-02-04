from jormungandr.optimization import OptimizationProblem
import matplotlib.pyplot as plt
from jormungandr.autodiff import sin, cos, abs, pow, sqrt
import math

N = 500
dt_guess = 0.05
T_max = 20
elevator_min_len = 36 * 0.0254 # 36 inches -> m (0.9144)
elevator_max_len = 80 * 0.0254 # 60 inches -> m (2.032)

pivot_min_theta = math.radians(30)
pivot_max_theta = math.radians(90)

pivot_max_accel = math.radians(100) # deg/s^2
# pivot_accel_reduction_per_meter = 0.3 # deg/s^2/m
elevator_max_accel = 2 # m/s^2

endeff_x_max = (28.25 + 18) * 0.0254 # max amount to the left -> m
endeff_x_min = 0 # max amount to the right -> m

def get_end_eff_pos(ext_len, theta):
  x = ext_len * cos(theta)
  y = ext_len * sin(theta)
  return x, y

def get_elevator_len_arm_angle(x, y):
  return math.sqrt(x ** 2 + y ** 2), math.atan2(y, x)

def visualize_results(pivot_X, elevator_X, acc_input, dt_s, N):
  import numpy as np
    
  # Extract values from decision variables
  pivot_angles = [pivot_X[0, k].value() for k in range(N + 1)]
  pivot_velocities = [pivot_X[1, k].value() for k in range(N + 1)]
  elevator_lengths = [elevator_X[0, k].value() for k in range(N + 1)]
  elevator_velocities = [elevator_X[1, k].value() for k in range(N + 1)]
  pivot_accelerations = [acc_input[0, k].value() for k in range(N)]
  elevator_accelerations = [acc_input[1, k].value() for k in range(N)]
  dt_values = [dt_s[0, k].value() for k in range(N + 1)]

  # Compute end-effector positions
  endeff_x = [elevator_lengths[k] * math.cos(pivot_angles[k]) for k in range(N + 1)]
  endeff_y = [elevator_lengths[k] * math.sin(pivot_angles[k]) for k in range(N + 1)]

  time_values = [sum(dt_values[:k+1]) for k in range(N + 1)]

  # Normalize time for colormap
  normalized_time = np.linspace(0, 1, len(endeff_x))

  # Plot pivot state
  plt.figure(figsize=(16, 12))
  plt.subplot(3, 2, 1)
  plt.plot(time_values, [math.degrees(angle) for angle in pivot_angles], label="Pivot Angle (deg)")
  plt.xlabel("Time (s)")
  plt.ylabel("Angle (deg)")
  plt.legend()
  plt.grid()

  plt.subplot(3, 2, 2)
  plt.plot(time_values, pivot_velocities, label="Pivot Velocity (rad/s)", color="orange")
  plt.xlabel("Time (s)")
  plt.ylabel("Velocity (rad/s)")
  plt.legend()
  plt.grid()

  # Plot elevator state
  plt.subplot(3, 2, 3)
  plt.plot(time_values, elevator_lengths, label="Elevator Length (m)", color="green")
  plt.xlabel("Time (s)")
  plt.ylabel("Length (m)")
  plt.legend()
  plt.grid()

  plt.subplot(3, 2, 4)
  plt.plot(time_values, elevator_velocities, label="Elevator Velocity (m/s)", color="red")
  plt.xlabel("Time (s)")
  plt.ylabel("Velocity (m/s)")
  plt.legend()
  plt.grid()

  # Plot end-effector position
  plt.subplot(3, 2, 5)
  cmap = plt.cm.viridis
  for i in range(len(endeff_x) - 1):
    plt.plot(endeff_x[i:i+2], endeff_y[i:i+2], color=cmap(normalized_time[i]), lw=2)
  plt.scatter(0, 0, color="black", label="Origin", zorder=5)
  plt.scatter(endeff_x[0], endeff_y[0], color="blue", label="Start Point", zorder=5)
  plt.scatter(endeff_x[-1], endeff_y[-1], color="red", label="End Point", zorder=5)
  plt.xlabel("X Position (m)")
  plt.ylabel("Y Position (m)")
  plt.legend()
  plt.grid()
  plt.title("End-Effector Path (Color indicates progression)")

  # Plot accelerations
  plt.subplot(3, 2, 6)
  plt.plot(time_values[:-1], pivot_accelerations, label="Pivot Acceleration (rad/s^2)", color="brown")
  plt.plot(time_values[:-1], elevator_accelerations, label="Elevator Acceleration (m/s^2)", color="cyan")
  plt.xlabel("Time (s)")
  plt.ylabel("Acceleration")
  plt.legend()
  plt.grid()
    
  plt.tight_layout()
  plt.show()


def main():
  problem = OptimizationProblem()
  pivot_X = problem.decision_variable(2, N + 1) # angle(rad), velocity
  elevator_X = problem.decision_variable(2, N + 1) # extension length, velocity
  acc_input = problem.decision_variable(2, N + 1) # acc_pivot, acc_elevator
  dt_s = problem.decision_variable(1, N + 1) # seconds

  for k in range(N + 1):
    problem.subject_to(pivot_X[0, k] >= pivot_min_theta)
    problem.subject_to(pivot_X[0, k] <= pivot_max_theta)

    problem.subject_to(elevator_X[0, k] >= elevator_min_len)
    problem.subject_to(elevator_X[0, k] <= elevator_max_len)

    pos = get_end_eff_pos(elevator_X[0, k], pivot_X[0, k])
    problem.subject_to(endeff_x_min <= pos[0])
    problem.subject_to(pos[0] <= endeff_x_max)
    problem.subject_to(pos[1] > 0)


  for k in range(N):
    pivot_state_k = pivot_X[:, k]
    pivot_state_k1 = pivot_X[:, k + 1]
    elevator_state_k = elevator_X[:, k]
    elevator_state_k1 = elevator_X[:, k + 1]
    acc_k = acc_input[:, k]
    
    # Constraining all dt to be equal to each other. better for solver perf
    dt_s[0, k].set_value(dt_guess)
    dt_s[0, k + 1].set_value(dt_guess)
    problem.subject_to(dt_s[k + 1] == dt_s[k])
    problem.subject_to(dt_s[k + 1] > 0)
    # problem.subject_to(dt_s[k] <= T_max / N)


    # Dynamics
    problem.subject_to(pivot_state_k1[0] == pivot_state_k[0] + dt_s[k] * pivot_state_k[1] + 0.5 * dt_s[k] ** 2 * acc_k[0])
    problem.subject_to(pivot_state_k1[1] == pivot_state_k[1] + dt_s[k] * acc_k[0])

    problem.subject_to(elevator_state_k1[0] == elevator_state_k[0] + dt_s[k] * elevator_state_k[1] + 0.5 * dt_s[k] ** 2 * acc_k[1])
    problem.subject_to(elevator_state_k1[1] == elevator_state_k[1] + dt_s[k] * acc_k[1])

    elevator_state_k[0].set_value(elevator_min_len) # avoid singularities at the initial 0 length state
    pivot_accel_limit = pivot_max_accel * pow(elevator_min_len / elevator_state_k[0], 2)
    # pivot_accel_limit = max(pivot_accel_limit, 1e-4)
    # pivot_accel_limit = pivot_max_accel
    # problem.subject_to(pivot_accel_limit > 0)
    # problem.subject_to(elevator_state_k1[0] < pivot_max_accel / pivot_accel_reduction_per_meter)
    problem.subject_to(acc_k[0] >= -pivot_accel_limit)
    problem.subject_to(acc_k[0] <= pivot_accel_limit)
    problem.subject_to(acc_k[1] >= -elevator_max_accel)
    problem.subject_to(acc_k[1] <= elevator_max_accel)

  # problem.subject_to(pivot_X[0, 0] == math.radians(45))
  problem.subject_to(pivot_X[1, 0] == 0)

  # problem.subject_to(elevator_X[0, 0] == elevator_min_len)
  problem.subject_to(elevator_X[1, 0] == 0)

  # problem.subject_to(pivot_X[0, N] == math.radians(60))
  problem.subject_to(pivot_X[1, N] == 0)

  # problem.subject_to(elevator_X[0, N] == 60 * 0.0254)
  problem.subject_to(elevator_X[1, N] == 0)

  start_pos = get_elevator_len_arm_angle(0.2, 1.5)
  end_pos = get_elevator_len_arm_angle(0.4, 1)

  assert start_pos[0] > elevator_min_len
  assert start_pos[0] < elevator_max_len

  assert end_pos[0] > elevator_min_len
  assert end_pos[0] < elevator_max_len

  assert start_pos[1] > pivot_min_theta
  assert start_pos[1] < pivot_max_theta

  assert end_pos[1] > pivot_min_theta
  assert end_pos[1] < pivot_max_theta

  problem.subject_to(start_pos[0] == elevator_X[0, 0])
  problem.subject_to(start_pos[1] == pivot_X[0, 0])
  problem.subject_to(end_pos[0] == elevator_X[0, N])
  problem.subject_to(end_pos[1] == pivot_X[0, N])

  total_time = sum(dt_s[0, k] for k in range(N+1))
  problem.subject_to(total_time <= T_max)
  J = total_time
  
  problem.minimize(J)

  problem.solve(diagnostics = True)
  # print(f"With pivot accel reduction max extension is {pivot_max_accel / pivot_accel_reduction_per_meter}m")
  print(f"Start pos: ({start_pos[0]} meters, {math.degrees(start_pos[1])}°)")
  print(f"End pos: ({end_pos[0]} meters, {math.degrees(end_pos[1])}°)")
  print(f"Total time: {total_time.value()}s")
  print(elevator_X[0, N].value())
  visualize_results(pivot_X, elevator_X, acc_input, dt_s, N) 
 
if __name__ == "__main__":
  main()


