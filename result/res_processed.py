import pandas as pd

if __name__ == '__main__':

    # 读取原始Excel文件
    input_file = '论文04-实验数据.xlsx'  # 替换为你的输入文件名
    output_file = 'res_processed_output_beta_first.xlsx'  # 替换为你想要的输出文件名

    # 创建一个Excel writer对象
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 获取所有sheet名称
        xls = pd.ExcelFile(input_file)
        sheet_names = xls.sheet_names

        # 只处理前四个sheet
        for sheet_name in sheet_names[:4]:
            lambda1_s = [0.1, 0.5, 1, 2, 5, 10]
            beta1_s = [0.1, 0.5, 1, 2, 5, 10]
            combined_df = pd.DataFrame()  # 用于存储所有组合的结果

            for beta_ in beta1_s:
                for lambda_ in lambda1_s:
                    # 读取每个sheet的内容
                    df = pd.read_excel(xls, sheet_name)

                    # 筛选出lambda1=a且beta=b的行
                    filtered_df = df[(df['lambda1'] == lambda_) & (df['beta'] == beta_) & (df['Type'] == 'max_epoch')]

                    # 按ACC列降序排序，并提取前5行
                    top_5_df = filtered_df.sort_values(by='ACC', ascending=False).head(1)

                    # 将结果追加到combined_df
                    combined_df = pd.concat([combined_df, top_5_df], ignore_index=True)

            # 将所有组合的结果写入新的Excel文件的对应sheet
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"处理完成，结果已保存到 {output_file}")
