# 引入日期类型，用于计算目标截止时间
from datetime import date

# 引入项目中定义的 FinancialGoal 数据模型
from app.state import FinancialGoal


def calculate_months_remaining(target_date: date) -> int:
    """
    计算从今天到目标截止日期还剩多少个月。

    参数：
    - target_date: 目标完成日期

    返回：
    - 剩余月份（至少为 1，避免后面除以 0）
    """
    # 获取今天日期
    today = date.today()

    # 计算剩余天数
    delta_days = (target_date - today).days

    # 如果目标日期已经过去（或今天），直接返回 1（避免除以0）
    if delta_days <= 0:
        return 1

    # 粗略将天数转换为月份（按30天一个月）
    months = delta_days // 30

    # 至少返回1个月
    return max(months, 1)


def calculate_required_monthly_saving(goal: FinancialGoal) -> float:
    """
    计算为了达成某个目标，每个月需要存多少钱。

    参数：
    - goal: 一个 FinancialGoal 对象

    返回：
    - 每月需要储蓄的金额
    """

    # 计算还差多少钱才能完成目标
    remaining_amount = goal.target_amount - goal.current_amount

    # 如果已经超过目标（负数），就当作 0
    remaining_amount = max(remaining_amount, 0.0)

    # 计算剩余月份
    months_remaining = calculate_months_remaining(goal.target_date)

    # 每月需要储蓄 = 剩余金额 / 剩余月份
    required_per_month = remaining_amount / months_remaining

    return required_per_month