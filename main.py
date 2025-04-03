import pandas as pd
import statsmodels.api as sm
from itertools import combinations, islice
from tqdm import tqdm
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

# 读取 dta 文件
files = ["1-de.dta"]
dfs = [pd.read_stata("C://Users//up2236384//Desktop//regression results//"+file) for file in files]

# 合并数据（假设有相同的 key 进行合并，否则需要调整）
df = pd.concat(dfs, axis=0, ignore_index=True)


# df = df.sample(n=5000, random_state=42)  # 设定 random_state 以保证可复现


required_vars = '''decoupling identifiedmw leverage roa_netincome mtb sox change_sales opercycle logassets zscore vcbacked1 auditorbig4big41 underwriter lnfirmage i59 i60 i70 i72 i73 i75 i79 i80 i82 i87 i42 i45 i48 i49 i50 i51 i52 i53 i54 i55 i58 i07 i10 i12 i13 i16 i20 i23 i24 i26 i28 i29 i30 i35 i36 i37 i38 i39 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22'''
df.columns = df.columns.str.strip()  # 去除前后空格和换行符
df.columns = df.columns.str.replace("\n", "", regex=True)

required_vars = required_vars.split(' ')
# 设定变量分组
#required_vars = ["Infoage", "Infirmage"]  # 必须包含的变量
delet_vars = ['firmage', '']
y_var = ['TEM'] #  # 更改y  2

target_vars = [ "decoupidentifiedmw"]  # 替换成你要关注的变量名
target_vars = [var for var in target_vars if var in df.columns]

# optional_vars = []
optional_vars = [col for col in df.columns if col not in required_vars + y_var  + delet_vars]
#print('\n' in df.columns.tolist())

missing_vars = [col for col in y_var  + required_vars + optional_vars if col not in df.columns]
if missing_vars:
    print("以下变量在数据中找不到:", missing_vars)

def remove_high_vif(X, threshold=10):
    """计算 VIF 并移除 VIF > threshold 的变量"""
    dropped = True  # 标记是否有变量被删除
    dropped_vars = []
    while dropped:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        high_vif_vars = vif_data[vif_data["VIF"] > threshold]["Variable"].tolist()
        if not high_vif_vars:  # 没有高 VIF 变量，停止
            dropped = False
        else:
            dropped_vars.extend(high_vif_vars)  # 将移除的变量加入列表
            X = X.drop(columns=high_vif_vars[:1])  # 仅移除 VIF 最高的变量
    print(f'过滤掉了{dropped_vars}')
    return X

# 处理缺失值
df = df.dropna(subset= y_var  + required_vars + optional_vars)
print("正在预筛选可选变量的共线性...")
X_initial = df[optional_vars]
X_filtered = remove_high_vif(X_initial, threshold=10)
filtered_optional_vars = X_filtered.columns.tolist()
print(f"共线性筛选后剩余 {len(filtered_optional_vars)} 个可选变量")

# 添加常数项
def fit_model(X, y):
    X = sm.add_constant(X)  # 添加截距项

    model = sm.OLS(y, X).fit()
    # 计算 R² 是否有效
    if not (model.rsquared >= 0 and model.rsquared <= 1):  # 处理 NaN 或 inf
        return None, []
    return model, X.columns

# 计算所有可能的变量组合的 p 值
best_model = None
best_p_value = float("inf")
best_vars = None

y = df["TEM"] # 更改y  1

max_combinations = 3  # 每个 r 只选 5 组
best_models = []  # 存储 (p值, 变量组合, 模型)

for r in tqdm(range(1, len(optional_vars) + 1)):
    shuffled_vars = random.sample(optional_vars, len(optional_vars))  # 打乱顺序
    comb_iter = combinations(shuffled_vars, r)  

    for opt_comb in tqdm(islice(comb_iter, max_combinations), leave=False):
        X = df[list(required_vars) + list(opt_comb)]
        model, final_vars = fit_model(X, y)
        if model is None:
            continue  # 跳过无效模型
        target_p_values = [model.pvalues[var] for var in target_vars if var in model.pvalues]


        if target_p_values:
            min_p_value = min(target_p_values)  # 取关键变量的最小 p 值
            best_models.append((min_p_value, final_vars.tolist(), model))
            #best_models.append((min_p_value, list(required_vars) + list(opt_comb), model))

            # 只保留最优的 3 个模型
            best_models = sorted(best_models, key=lambda x: x[0])[:3]  # 按 p 值排序
        #if target_p_values and max(target_p_values) < best_p_value:
        #    best_p_value = max(target_p_values)  
        #    best_model = model
        #    best_vars = list(required_vars) + list(opt_comb)

# 输出最佳模型结果
#print("最佳变量组合:", best_vars)
#print(best_model.summary())

for i, (p_value, variables, model) in enumerate(best_models, 1):
    print(f"\n===== Top {i} 模型 =====")
    print(f"关键变量的最小 p 值: {p_value:.5f}")
    print("变量组合:", variables)
    print(model.summary())
