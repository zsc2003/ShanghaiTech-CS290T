# Project1
# Deadline: 2025/03/21  23:59

Name: Zhou Shouchen

Student ID: 2021533042

The Project1 consists of two parts.

## PartA (30 pts)
There are 2 problems.
Please print it out and complete the question on it, and submit the paper version of the question to TA.

Every problem is 15 pts.

## PartB (70 pts)
**Create a virtual environment using [Anaconda](https://www.anaconda.com/download), with Python 3.6.13 and gym 0.9.4:**
```bash
conda create -n gym_094 python==3.6.13
conda activate gym_094
pip install gym==0.9.4
```

Please carefully read Part_B.pdf and complete the 5 questions Q1-5.

For Q3 and Q4, please write your answer below:

#### Q3: Compare your Policy iteration with the Value iteration to obtain the final policy. Are there any differences between them? Please explain the reason.

Steps of Policy Iteration:
- Start with an initial policy.
- Evaluate the value function for the current policy.
- Improve the policy based on the value function.
- Repeat until the policy stops changing.

Steps of Value Iteration:
- Directly update the value function for each state.
- Once the value function converges, derive the optimal policy.

Both methods involve the steps of strategy enhancement and strategy evaluation. However, policy iteration enhances the strategy based on the pre update value function to obtain a greedy policy, and then selects the greedy action in state s according to the greedy policy to update the value function corresponding to s. Value iteration combines two steps into a one-step update method through Bellman Optimization Equation。

Both algorithms will eventually give the same optimal policy, but with different speeds and methods. The policy iteration requires more iterations to converge, but each iteration is faster, while the value iteration requires fewer iterations to converge, but each iteration is slower.


#### Q4: Compare the convergence speed of Policy iteration and Value iteration. Which one can converge faster? Please explain the reason.

In our cliff walking experiment(a timer is added in the code), the policy iteration totolly requires: ``60+72+44+12+1=189`` iterations to converge, with total time ``0.023s``. But the value iteration requires only ``14`` iterations to converge, with total time ``0.006s``. Which means the value iteration converges faster.

But for each iteration, the policy iteration requires ``0.023 / 189 = 1.22e-4s`` per iteration, while the value iteration requires ``0.006 / 14 = 4.29e-4s`` per iteration. Which means that the policy iteration less time per iteration.

Value Iteration optimizes the value function directly through the Bellman optimal equation, which is an implicit strategy update. Each iteration simultaneously optimizes the value function and improves the strategy. The Cliff Walking environment is relatively simple and deterministic. Due to the clear direction of policy improvement, sparse rewards, and small state space, updating the value function can find the optimal policy with fewer iterations. Although it requires more time per iteration, it converges faster.

Policy Iteration is explicit policy update, which requires policy evaluation until the value function converges, and then policy improvement. In sparse reward environments, the policy evaluation phase of Policy Iteration may waste a significant amount of time on unimportant states, resulting in high computational costs. Although it requires less time per iteration, it requires much more iterations and converges slower.


(Q1)20 + (Q2)20 + (Q3)10 + (Q4)10 + (Q5)10 = 70 pts


Finally, compress the entire folder into a zip file (e.g. 张三_2025233111.zip) and send it to wangyc2023@shanghaitech.edu.cn