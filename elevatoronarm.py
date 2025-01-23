from jormungandr.optimization import OptimizationProblem
from jormungandr.autodiff import sin, cos, abs
import math

N = 100
dt_guess = 0.05
elevator_min_len = 36 * 0.0254 # 36 inches -> m
elevator_max_len = 80 * 0.0254 # 60 inches -> m

pivot_min_theta = math.radians(10)
pivot_max_theta = math.radians(100)

endeff_x_min = -(28.25 + 18) * 0.0254 # max amount to the left -> m
endeff_x_max = (28.25 + 18) * 0.0254 # max amount to the right -> m

def get_end_eff_pos(ext_len, theta):
  x = ext_len * cos(theta)
  y = ext_len * sin(theta)
  return x, y

def main():
  problem = OptimizationProblem()
  pivot_X = problem.decision_variable(2, N + 1) # angle(rad), velocity
  elevator_X = problem.decision_variable(2, N + 1) # extension length, velocity
  acc_input = problem.decision_variable(2, N + 1) # acc_pivot, acc_elevator
  dt_s = problem.decision_variable(N + 1) # seconds

  for k in range(N):
    pivot_state_k = pivot_X[:, k]
    pivot_state_k1 = pivot_X[:, k + 1]
    elevator_state_k = elevator_X[:, k]
    elevator_state_k1 = elevator_X[:, k + 1]
    acc_k = acc_input[:, k]
    
    dt_s[k].set_value(dt_guess)
    problem.subject_to(dt_s[k + 1] == dt_s[k])

    # Dynamics
    problem.subject_to(pivot_state_k1[0] == pivot_state_k[0] + dt_s[k] * pivot_state_k[1] + 0.5 * dt_s[k] ** 2 * acc_k[0])
    problem.subject_to(pivot_state_k1[1] == pivot_state_k[1] + dt_s[k] * acc_k[0])

    problem.subject_to(elevator_state_k1[0] == elevator_state_k[0] + dt_s[k] * elevator_state_k[1] + 0.5 * dt_s[k] ** 2 * acc_k[1])
    problem.subject_to(elevator_state_k1[1] == elevator_state_k[1] + dt_s[k] * acc_k[1])

    problem.subject_to(acc_k[0] >= -10)
    problem.subject_to(acc_k[0] <= 10)
    problem.subject_to(acc_k[1] >= -10)
    problem.subject_to(acc_k[1] <= 10)


  problem.subject_to(pivot_X[0, 0] == math.radians(45))
  problem.subject_to(pivot_X[1, 0] == 0)

  problem.subject_to(elevator_X[0, 0] == elevator_min_len)
  problem.subject_to(elevator_X[1, 0] == 0)

  problem.subject_to(pivot_X[0, N] == math.radians(60))
  problem.subject_to(pivot_X[1, N] == 0)

  problem.subject_to(elevator_X[0, N] == 60 * 0.0254)
  problem.subject_to(elevator_X[1, N] == 0)

  problem.solve(diagnostics = True)
 
  # x = 18.0, y = 6.0
  # print(f"x = {x.value()}, y = {y.value()}")
 
 
if __name__ == "__main__":
  main()