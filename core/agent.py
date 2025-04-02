import numpy as np
from mesa import Agent

class CulturalAgent(Agent):
    """
    An agent playing the spatial Prisoner's Dilemma with cultural utility
    and endogenous cultural evolution via success-biased imitation.
    """
    def __init__(self, unique_id, model, initial_strategy, C):
        super().__init__(unique_id, model)
        self.strategy = initial_strategy  # 1 for C, 0 for D
        self.C = C                        # Cultural parameter [0, 1]
        self.next_strategy = self.strategy # Initialize next strategy
        self.next_C = self.C              # Initialize next C value
        self.current_utility = 0.0

    def calculate_utility(self):
        """Calculate the agent's utility based on interactions with neighbors."""
        self.current_utility = 0.0
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        if not neighbors:
            return
        
        for neighbor in neighbors:
            # Ensure neighbor has calculated its strategy for the *current* step if needed
            # In our BaseScheduler setup, all strategies are from the *previous* completed step, which is correct.
            my_payoff, neighbor_payoff = self.model.payoff_matrix[self.strategy][neighbor.strategy]
            self.current_utility +=  self.C*my_payoff + (1-self.C) * neighbor_payoff
#            self.current_utility +=  (1-self.C)*my_payoff + self.C * neighbor_payoff
#            self.current_utility +=  my_payoff + self.C * neighbor_payoff

#        print(f"Agent {self.unique_id} utility: {self.current_utility}")

# agent.py


    def calculate_utility1(self):
        """Calculate the agent's utility based on interactions with neighbors."""
        # print(f"--- Step {self.model.schedule.steps}, Agent {self.unique_id} Calculating Utility ---") # 可选：取消注释以跟踪智能体
        self.current_utility = 0.0 # 确保每次计算前都重置为0

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        if not neighbors:
            return # 没有邻居，效用就是0

        for i, neighbor in enumerate(neighbors):
            try:
                # --- 1. 获取收益 ---
                payoffs = self.model.payoff_matrix[self.strategy][neighbor.strategy]
                my_payoff, neighbor_payoff = payoffs

                # --- 2. !!! 添加收益检查代码 !!! ---
                # 在这里检查获取到的收益值是否为零
                payoff_is_zero = (my_payoff == 0 and neighbor_payoff == 0) # 检查是否 *两者都* 为零
                payoff_one_is_zero = (my_payoff == 0 or neighbor_payoff == 0) # 检查是否 *至少一个* 为零

                # 你可以选择打印所有情况，或者只打印你关心的零收益情况
                # 选项 A: 打印每一次交互的收益，无论是否为零
                # print(f"  DEBUG PAYOFF: A{self.unique_id}(S={self.strategy}) vs N{neighbor.unique_id}(S={neighbor.strategy}) -> Payoffs: My={my_payoff}, Neighbor={neighbor_payoff}")

                # 选项 B: 只在检测到零收益时打印（更常用）
#                if payoff_one_is_zero: # 如果你更关心是否 *至少一个* 是零
#                     print(f"  DEBUG **ZERO PAYOFF DETECTED**: A{self.unique_id}(S={self.strategy}, C={self.C:.2f}) interacts with N{neighbor.unique_id}(S={neighbor.strategy}) -> Payoffs: My={my_payoff}, Neighbor={neighbor_payoff}")
                # else:
                     # 如果需要，也可以打印非零的情况
                     # print(f"  DEBUG NON-ZERO PAYOFF: A{self.unique_id}(S={self.strategy}) vs N{neighbor.unique_id}(S={neighbor.strategy}) -> Payoffs: My={my_payoff}, Neighbor={neighbor_payoff}")

                # --- 3. 继续计算效用 (使用获取到的 my_payoff 和 neighbor_payoff) ---
                if self.C is None or np.isnan(self.C):
                     print(f"    ERROR: Agent {self.unique_id}'s Culture (C) is invalid: {self.C}")
                     interaction_utility_contribution = my_payoff
                else:
                     interaction_utility_contribution = my_payoff + self.C * neighbor_payoff

                self.current_utility += interaction_utility_contribution


            except KeyError:
                # 如果策略值不是预期的 0 或 1，访问收益矩阵会出错
                print(f"    ERROR: Invalid strategy key accessing payoff matrix! My strategy: {self.strategy}, Neighbor strategy: {neighbor.strategy}")
                # 可以在这里决定如何处理错误，例如跳过这个邻居或将收益设为0
                my_payoff, neighbor_payoff = 0, 0 # 假设错误时收益为0
                # 也可以在这里打印零收益的调试信息
                print(f"  DEBUG **ERROR ZERO PAYOFF**: A{self.unique_id}(S={self.strategy}) vs N{neighbor.unique_id}(S={neighbor.strategy}) due to KeyError -> Setting Payoffs to 0")


        # print(f"  Agent {self.unique_id}: FINAL Calculated Utility: {self.current_utility:.4f}") # 可选：取消注释看最终效用





    def decide_strategy_update(self): # Renamed from select_next_strategy
        """Determine the agent's strategy for the next step using Fermi rule."""
        self.next_strategy = self.strategy # Default: keep current strategy

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if not neighbors:
            return

        neighbor_to_compare = self.random.choice(neighbors)
        delta_utility = neighbor_to_compare.current_utility - self.current_utility

        
        # 可以先处理 K 接近 0 的确定性情况（避免除以零或极小数）
        if self.model.K < 1e-9: # 使用一个很小的阈值代替 == 0，更安全
            # 如果邻居效用更高，则采纳概率为1，否则为0
            prob_adopt = 1.0 if delta_utility > 0 else 0.0
        else:
            # 计算指数项
            argument = -delta_utility / self.model.K
            # 检查指数是否过大，可能导致 exp() 溢出
            # 对于标准的 64 位浮点数，np.exp 大约在参数 > 709 时溢出
            # 我们可以用一个稍微保守的值，比如 700
            if argument > 700:
                # 如果指数过大，我们知道 exp 会溢出，最终概率应为 0
                prob_adopt = 0.0
            else:
                # 只有在参数安全时才计算 exp
                # 注意：np.exp 对于非常小的负数（导致下溢）处理得很好，会得到 0.0
                prob_adopt = 1 / (1 + np.exp(argument))
        # 用计算出的概率与随机数比较
#        print(prob_adopt)
#        if self.random.random() < 0.3:
        if self.random.random() < prob_adopt:
            self.next_strategy = neighbor_to_compare.strategy
        else:
            self.next_strategy = self.strategy




    def decide_culture_update(self):
        """
        Potentially determine the agent's cultural value C for the next step
        using Fermi rule based on utility, with probability p_update_C.
        """
        self.next_C = self.C # Default: keep current culture

        # Attempt cultural update only with probability p_update_C
        if self.random.random() < self.model.p_update_C:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            if not neighbors:
                return

            neighbor_to_compare = self.random.choice(neighbors)
            delta_utility = neighbor_to_compare.current_utility - self.current_utility

            # Use the cultural noise parameter K_C
            if self.model.K_C <= 1e-9:
                 prob_adopt_culture = 1.0 if delta_utility > 0 else 0.0
            else:
                 prob_adopt_culture = 1 / (1 + np.exp(-delta_utility / self.model.K_C))

            if self.random.random() < prob_adopt_culture:
                self.next_C = neighbor_to_compare.C # Adopt neighbor's C

    def mutate_culture(self):
        """Apply random mutation to the cultural value C with probability p_mut."""
        if self.random.random() < self.model.p_mut:
            # Generate a new C value uniformly between 0 and 1
            self.C = self.random.uniform(0, 1)
            # Ensure next_C is also updated if mutation happens after advance sets C = next_C
            # If mutate happens *after* advance, we directly change self.C
            self.next_C = self.C # Keep next_C consistent if mutated after advance

    def advance(self):
        """
        Apply the determined next strategy and next cultural value.
        This is called by the scheduler's step() function.
        """
        self.strategy = self.next_strategy
        self.C = self.next_C
        # Mutation is handled separately in the model step loop AFTER advance


    # Remove the old step() method as its logic is now split
    # def step(self):
    #     pass
