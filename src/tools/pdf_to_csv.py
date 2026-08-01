import pandas as pd
import warnings

# 忽略 openpyxl 的样式警告
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ================= 配置参数 =================
excel_file = "./data/tables/GB12268/GB+12268-2025.xlsx"  # 你的原始大 Excel 文件
output_csv = "./data/tables/GB12268/GB+12268_tab.csv"

start_sheet = 12
end_sheet = 387

# 统一的标准图谱属性（列名）
standard_columns = [
    "un_number", "name_zh", "name_en", "class_or_division", 
    "subsidiary_hazard", "packing_group", "special_provisions", 
    "limited_quantities", "excepted_quantities", "packing_instruction", 
    "special_packing_provisions", "portable_tank_instruction", "portable_tank_special_provisions"
]
# ============================================

print("正在初始化 Excel 文件...")
excel_reader = pd.ExcelFile(excel_file)
sheet_names = excel_reader.sheet_names
sheets_to_combine = sheet_names[start_sheet - 1 : end_sheet]

all_dfs = []

print(f"开始面向知识图谱提取并对齐数据（共 {len(sheets_to_combine)} 个 Sheet）...")

for sheet in sheets_to_combine:
    # 1. 读取原始数据
    df_raw = pd.read_excel(excel_reader, sheet_name=sheet, header=None)
    if df_raw.empty:
        continue

    # 2. 动态清洗并定位表头
    header_idx = None
    scan_range = df_raw.iloc[:min(6, len(df_raw))].fillna("").astype(str)
    for row_idx in range(len(scan_range)):
        row_text = "".join(scan_range.iloc[row_idx].tolist())
        if "联合国" in row_text or "编号" in row_text:
            header_idx = row_idx
            break
            
    if header_idx is None:
        continue
        
    # 3. 截取数据行并规范列数
    data_start_idx = header_idx + 2
    df_data = df_raw.iloc[data_start_idx:].copy()
    df_data = df_data.iloc[:, :13]
    df_data.columns = standard_columns
    
    # 4. 彻底剔除跨页打印重复出现的表头和序号干扰行
    df_data = df_data[~df_data["un_number"].astype(str).str.contains(r"联合国|编号|确定|\(1\)|\(2\)", na=False)]
    df_data = df_data[~df_data["name_zh"].astype(str).str.contains(r"名称|说明|\(2\)", na=False)]
    
    # 5. 【知识图谱核心修复：合并单元格穿透填充】
    # 先清除全空行
    df_data.dropna(how='all', inplace=True)
    
    # 核心：对所有列进行前向填充（ffill），确保被合并单元格拆开后，每一行都拥有完整的属性值
    df_data = df_data.ffill()
    
    # 追加到列表中
    all_dfs.append(df_data)

# 合并所有
print("正在执行全局合并与图谱格式清洗...")
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 清洗掉联合国编号依旧为空的残余行
    combined_df = combined_df[combined_df["un_number"].notna()]
    
    # 6. 【知识图谱核心修复：清除单元格内的换行符】
    # 单元格内换行会彻底破坏 CSV 的行结构，将其替换为逗号或空格（这里用空格或分号隔离多个包装指南）
    for col in combined_df.columns:
        combined_df[col] = combined_df[col].astype(str).str.replace(r'\r+|\n+', ' ', regex=True).str.strip()
        # 清理 pandas 转换遗留的 'nan' 字符串为纯空字符串
        combined_df[col] = combined_df[col].replace('nan', '')

    # 7. 导出为图谱专用的 CSV 文件（使用 utf-8-sig 编码防止中文在某些系统乱码）
    print(f"正在写入图谱 CSV 文件...")
    combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"🚀 【图谱数据准备就绪】合并单元格已穿透填充，换行符已清洗！文件已保存至：{output_csv}")
else:
    print("❌ 未能成功提取数据。")