import numpy as np
from mesa import Agent

class CulturalAgent(Agent):
    """
    一个在空间囚徒困境中具有文化效用并通过成功偏向模仿进行内生文化演化的智能体
    An agent in spatial prisoner's dilemma with cultural utility that evolves endogenously through success-biased imitation
    """
    def __init__(self, unique_id, model, initial_strategy, C):
        """
        初始化文化智能体
        Args:
            unique_id: 智能体唯一标识符
            model: 所属模型实例
            initial_strategy: 初始策略 (1表示合作C, 0表示背叛D)
            C: 文化参数 [0,1]区间
        """
        super().__init__(unique_id, model)
        self.strategy = initial_strategy  # 1 for C, 0 for D
        self.C = C                        # Cultural parameter [0, 1]
        self.next_strategy = self.strategy # Initialize next strategy
        self.next_C = self.C              # Initialize next C value
        self.current_utility = 0.0        # 当前效用值

    def calculate_utility(self):
        """
        基于与邻居的交互计算智能体的效用
        计算方法: 效用 = Σ(自身收益 * C + 邻居收益 * (1-C))
        """
        self.current_utility = 0.0  # 重置当前效用值
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        if not neighbors:
            return  # 如果没有邻居，效用保持为0
        
        for neighbor in neighbors:
            # 确保邻居已计算其当前步骤的策略(如果需要)
            # 在我们的BaseScheduler设置中，所有策略都来自上一步骤，这是正确的
            my_payoff, neighbor_payoff = self.model.payoff_matrix[self.strategy][neighbor.strategy]
            # 计算文化效用: 结合自身收益和邻居收益，权重由文化参数C决定
            self.current_utility += self.C * my_payoff + (1 - self.C) * neighbor_payoff
#            self.current_utility +=  (1-self.C)*my_payoff + self.C * neighbor_payoff
#            self.current_utility +=  my_payoff + self.C * neighbor_payoff

#        print(f"Agent {self.unique_id} utility: {self.current_utility}")

# agent.py


    def calculate_utility1(self):
        """基于与邻居的交互计算智能体的效用(方法1)"""
        # print(f"--- Step {self.model.schedule.steps}, Agent {self.unique_id} Calculating Utility ---") # 调试用：取消注释可跟踪智能体
        self.current_utility = 0.0 # 确保每次计算前都重置为0

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        if not neighbors:
            return # 没有邻居，效用就是0

        for i, neighbor in enumerate(neighbors):
            try:
                # --- 1. 获取收益矩阵中的收益值 ---
                payoffs = self.model.payoff_matrix[self.strategy][neighbor.strategy]
                my_payoff, neighbor_payoff = payoffs

                # --- 2. 收益值检查 ---
                # 检查收益值是否为零
                payoff_is_zero = (my_payoff == 0 and neighbor_payoff == 0) # 检查双方收益是否都为零
                payoff_one_is_zero = (my_payoff == 0 or neighbor_payoff == 0) # 检查是否至少一方收益为零

                # 调试选项：
                # 选项 A: 打印所有交互的收益值
                # print(f"  DEBUG PAYOFF: A{self.unique_id}(S={self.strategy}) vs N{neighbor.unique_id}(S={neighbor.strategy}) -> Payoffs: My={my_payoff}, Neighbor={neighbor_payoff}")

                # 选项 B: 只打印零收益情况(推荐)
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
                # 当策略值不是预期的0或1时，访问收益矩阵会抛出KeyError
                print(f"    ERROR: Invalid strategy key accessing payoff matrix! My strategy: {self.strategy}, Neighbor strategy: {neighbor.strategy}")
                # 错误处理：将当前交互的收益设为0
                my_payoff, neighbor_payoff = 0, 0 # 错误情况下默认收益为0
                # 打印调试信息
                print(f"  DEBUG **ERROR ZERO PAYOFF**: A{self.unique_id}(S={self.strategy}) vs N{neighbor.unique_id}(S={neighbor.strategy}) due to KeyError -> Setting Payoffs to 0")


        # print(f"  Agent {self.unique_id}: FINAL Calculated Utility: {self.current_utility:.4f}") # 调试用：取消注释可查看最终计算出的效用





    def decide_strategy_update(self): # 从 select_next_strategy 重命名而来
        """
        使用费米规则确定智能体下一步的策略
        
        机制详解:
        1. 随机选择一个邻居进行比较
        2. 计算效用差: ΔU = 邻居效用 - 自身效用
        3. 使用费米函数计算采纳概率: P = 1/(1+exp(-ΔU/K))
           - K: 选择强度参数，控制随机性程度
           - 当K→0时变为确定性选择(优胜劣汰)
           - 当K较大时选择更随机
        4. 根据概率决定是否采纳邻居策略
        
        边界情况处理:
        1. 当K接近0时，直接根据效用差决定(确定性选择)
        2. 当ΔU/K过大时，进行数值溢出保护
        """
        self.next_strategy = self.strategy # 默认：保持当前策略

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if not neighbors:
            return

        neighbor_to_compare = self.random.choice(neighbors)
        delta_utility = neighbor_to_compare.current_utility - self.current_utility

        # 处理K接近0的边界情况（避免除以极小值）
        if self.model.K < 1e-9: # 使用极小阈值代替等于0的判断
            # 确定性选择：邻居效用更高则必然采纳
            prob_adopt = 1.0 if delta_utility > 0 else 0.0
        else:
            # 计算费米函数参数
            argument = -delta_utility / self.model.K
            # 溢出保护：当指数参数过大时限制最大值
            # 注：64位浮点数在参数>709时exp会溢出
            if argument > 700:  # 设置稍低于709的安全阈值
                prob_adopt = 0.0  # 极大负收益差导致采纳概率趋近0
            else:
                # 安全计算指数函数
                prob_adopt = 1 / (1 + np.exp(argument))
        
        # 根据概率决定是否采纳邻居策略
        if self.random.random() < prob_adopt:
            self.next_strategy = neighbor_to_compare.strategy  # 采纳邻居策略
        else:
            self.next_strategy = self.strategy  # 保持当前策略




    def decide_culture_update(self):
        """
        基于效用，以p_update_C的概率使用费米规则确定智能体下一步的文化值C
        
        文化更新机制详解:
        1. 更新触发: 以概率p_update_C决定是否尝试更新(控制更新频率)
        2. 邻居选择: 随机选择一个邻居进行比较
        3. 效用比较: 计算ΔU = 邻居效用 - 自身效用
        4. 费米决策: 使用P = 1/(1+exp(-ΔU/K_C))计算采纳概率
           - K_C: 文化更新的噪声参数
           - 当K_C→0时变为确定性选择(优胜劣汰)
           - 当K_C较大时选择更随机
        5. 文化采纳: 根据概率决定是否采纳邻居的C值
        
        边界情况处理:
        1. 当K_C接近0时，直接根据效用差决定(确定性选择)
        2. 当没有邻居时跳过更新
        """
        self.next_C = self.C # 默认保持当前文化值

        # 以p_update_C的概率尝试文化更新
        if self.random.random() < self.model.p_update_C:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            if not neighbors:
                return  # 没有邻居则跳过更新

            # 随机选择一个邻居进行比较
            neighbor_to_compare = self.random.choice(neighbors)
            delta_utility = neighbor_to_compare.current_utility - self.current_utility

            # 使用文化噪声参数K_C计算采纳概率
            if self.model.K_C <= 1e-9:  # 处理K_C接近0的边界情况
                prob_adopt_culture = 1.0 if delta_utility > 0 else 0.0  # 确定性选择
            else:
                # 使用费米函数计算概率
                prob_adopt_culture = 1 / (1 + np.exp(-delta_utility / self.model.K_C))

            # 根据概率决定是否采纳邻居的文化参数
            if self.random.random() < prob_adopt_culture:
                self.next_C = neighbor_to_compare.C  # 采纳邻居的文化参数

    def mutate_culture(self):
        """
        以p_mut的概率对文化值C进行随机突变
        
        突变机制详解:
        1. 突变触发: 以概率p_mut决定是否发生突变(控制突变频率)
        2. 突变生成: 在[0,1]区间内均匀随机生成新的C值
           - 完全随机探索文化参数空间
           - 不依赖于邻居或效用比较
        3. 状态同步: 确保next_C与突变后的C值保持一致
        
        边界情况处理:
        1. 突变概率p_mut=0时完全禁用突变
        2. 突变概率p_mut=1时每个时间步都强制突变
        3. 突变后的C值自动保持在[0,1]有效范围内
        """
        if self.random.random() < self.model.p_mut:
            # 在[0,1]区间内均匀随机生成新的C值
            self.C = self.random.uniform(0, 1)
            # 确保next_C与突变后的C值保持一致
            # 如果突变发生在advance()之后，我们直接修改self.C
            self.next_C = self.C  # 保持next_C与突变后的C值一致

    def advance(self):
        """
        应用确定的下一个策略和文化值
        
        状态更新机制详解:
        1. 策略更新: 将next_strategy赋值给当前strategy
           - next_strategy由decide_strategy_update()方法确定
        2. 文化更新: 将next_C赋值给当前C
           - next_C由decide_culture_update()方法确定
        3. 状态同步: 确保所有智能体同步更新状态
        
        调用关系说明:
        1. 由调度器的step()函数调用
        2. 在decide_strategy_update()和decide_culture_update()之后执行
        3. 在mutate_culture()之前执行
        
        重要注意事项:
        1. 该方法不包含任何决策逻辑，仅执行状态更新
        2. 突变(mutation)在模型的主循环中单独处理(在advance之后)
        3. 所有智能体的advance()调用是同步的，确保状态更新的一致性
        """
        self.strategy = self.next_strategy  # 更新策略
        self.C = self.next_C                # 更新文化参数
        # 突变(mutation)在模型的主循环中单独处理(在advance之后)


    # Remove the old step() method as its logic is now split
    # def step(self):
    #     pass
