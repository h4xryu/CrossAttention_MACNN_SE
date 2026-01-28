#!/usr/bin/env python3
"""
논문 표 데이터를 CSV 및 XLSX로 내보내기
"""

import pandas as pd
import os

output_dir = '/home/work/Ryuha/ECG_CrossAttention-stored/paper/tables'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# 표 1. RR 간격 기반 시간적 특징 정의
# =============================================================================
table1_data = {
    'Feature': ['Pre-RR', 'Post-RR', 'Local-RR', 'Global-RR',
                'Pre-Global diff', 'Post-Global ratio', 'Pre-Post ratio'],
    'Formula': ['RR_{i-1}', 'RR_i', 'μ_l (최근 10개 평균)', 'μ_g (전역 평균)',
                'RR_{i-1} - μ_g', 'RR_{i-1} / μ_g', 'RR_{i-1} / RR_i'],
    '설명': ['직전 RR 간격', '직후 RR 간격', '최근 10개 RR 평균', '전역 RR 평균',
            '전역 평균 대비 편차', '전역 평균 대비 비율', 'Pre/Post 비율'],
    'opt1_index': ['-', '-', '-', '-', '-', '-', '-'],  # opt1은 ratio만 사용
    'opt2_index': ['0 (ratio)', '1 (ratio)', '-', '-', '-', '-', '-']
}

# opt1 (7-Dim) 상세
table1_opt1 = {
    'Index': [0, 1, 2, 3, 4, 5, 6],
    'Feature_Name': ['Pre/Local', 'Pre/Post', 'Pre/Prev(t-1)', 'Post/Local',
                     'Local/Global', 'Pre/Pre(t-2)', 'Post/Next_Post'],
    'Formula': ['Pre_RR / Local_RR', 'Pre_RR / Post_RR', 'Pre_RR[i] / Pre_RR[i-1]',
                'Post_RR / Local_RR', 'Local_RR / Global_RR',
                'Pre_RR[i] / Pre_RR[i-2]', 'Post_RR[i] / Post_RR[i+1]'],
    '역할': ['조기성 감지 (SVEB 핵심)', '보상 휴지기 확인', '급격한 변화 감지 (t-1)',
            '회복 후 리듬 확인', '환자 평소 대비 현재 상태',
            '급격한 변화 감지 (t-2)', '회복 안정성'],
    '중요도': ['핵심', '핵심', '핵심', '핵심', '보조', '보조', '보조']
}

df_table1 = pd.DataFrame(table1_data)
df_table1_opt1 = pd.DataFrame(table1_opt1)

# =============================================================================
# 표 2. 클래스별 샘플 분포
# =============================================================================
table2_data = {
    'Class': ['N', 'S', 'V', 'F', 'Q', 'Total'],
    'DS1_Train': [45842, 944, 3788, 415, 8, 50997],
    'DS2_Test': [44238, 1837, 3221, 388, 7, 49691],
    'Total': [90080, 2781, 7009, 803, 15, 100688],
    'Ratio_%': [89.46, 2.76, 6.96, 0.80, 0.01, 100.0]
}
df_table2 = pd.DataFrame(table2_data)

# =============================================================================
# 표 3. ResUNet 백본 기반 성능 비교
# =============================================================================
table3_data = {
    'Method': ['Baseline', 'A1 (Naive Concat)', 'A2 (Cross-Attention)',
               'B1 (Naive Concat TF)', 'B2 (Cross-Attention TF)'],
    'Fusion_Type': ['None', 'Naive Concat', 'Cross-Attention',
                    'Naive Concat (Transformer)', 'Cross-Attention (Transformer)'],
    'Macro_Acc_%': [96.48, 95.70, 96.44, 97.49, 97.72],
    'Macro_Se_%': [45.33, 47.58, 51.18, 59.71, 62.90],
    'Macro_Sp_%': [87.43, 89.93, 90.41, 93.99, 94.86],
    'Macro_Pr_%': [50.16, 38.02, 58.40, 62.78, 73.62],
    'Macro_F1_%': [46.88, 41.60, 52.36, 60.56, 65.69],
    'Weighted_Acc_%': [93.80, 92.74, 94.19, 96.25, 96.98],
    'Weighted_Se_%': [92.95, 91.40, 92.89, 94.98, 95.45],
    'Weighted_Sp_%': [56.76, 68.32, 68.75, 80.96, 83.97],
    'Weighted_Pr_%': [90.44, 88.84, 92.17, 94.49, 94.96],
    'Weighted_F1_%': [91.51, 89.81, 92.16, 94.63, 95.01]
}
df_table3 = pd.DataFrame(table3_data)

# =============================================================================
# 표 4. MACNN-SE 백본 기반 성능 비교
# =============================================================================
table4_data = {
    'Method': ['Baseline (opt2)', 'Naive Concat (opt1)', 'Proj Concat (opt1)',
               'MHCA h=1 (opt1)', 'Hybrid MHCA (opt3)'],
    'Fusion_Type': ['Early (DAEAC)', 'Late Concat', 'Late + Gate',
                    'Cross-Attention', 'Early + Cross-Attention'],
    'Input_Option': ['opt2', 'opt1', 'opt1', 'opt1', 'opt3'],
    'Macro_Acc_%': [96.18, 93.22, 96.74, 92.48, 93.74],
    'Macro_Se_%': [40.71, 47.56, 42.37, 43.28, 70.83],
    'Macro_Sp_%': [86.43, 86.13, 85.42, 86.16, 94.33],
    'Macro_Pr_%': [52.13, 51.16, 55.61, 49.95, 56.41],
    'Macro_F1_%': [43.19, 43.84, 44.53, 45.41, 58.54]
}
df_table4 = pd.DataFrame(table4_data)

# =============================================================================
# 표 5. Hybrid MHCA Ablation
# =============================================================================
table5_data = {
    'Configuration': ['Baseline', 'MHCA only', 'Early only', 'Hybrid (Early + MHCA)'],
    'Early_Fusion': ['Yes', 'No', 'Yes', 'Yes'],
    'Late_MHCA': ['No', 'Yes', 'No', 'Yes'],
    'Macro_Se_%': [40.71, 43.28, 40.71, 70.83],
    'Macro_F1_%': [43.19, 45.41, 43.19, 58.54],
    'Delta_Se_%p': ['-', '+2.57', '+0.00', '+30.12'],
    'Delta_F1_%p': ['-', '+2.22', '+0.00', '+15.35']
}
df_table5 = pd.DataFrame(table5_data)

# =============================================================================
# 표 6. S 클래스 (SVEB) 성능 비교 - ResUNet
# =============================================================================
table6_data = {
    'Method': ['Baseline', 'A1 (Concat)', 'A2 (Cross-Attn)',
               'B1 (Concat-TF)', 'B2 (MHCA-TF)', 'Delta (B2 vs Baseline)'],
    'S_Precision_%': [21.22, 36.49, 70.45, 70.45, 70.45, '+49.23'],
    'S_Sensitivity_%': [11.21, 35.44, 66.58, 66.58, 66.58, '+55.37'],
    'S_F1_%': [14.67, 35.96, 68.46, 68.46, 68.46, '+53.79']
}
df_table6 = pd.DataFrame(table6_data)

# =============================================================================
# 표 7. 백본별 크로스 어텐션 융합 효과 요약
# =============================================================================
table7_data = {
    'Backbone': ['ResUNet', 'MACNN-SE'],
    'Best_Method': ['B2 (TF-MHCA)', 'Hybrid MHCA'],
    'Macro_F1_%': [65.69, 58.54],
    'Baseline_F1_%': [46.88, 43.19],
    'Delta_F1_%p': ['+18.81', '+15.35'],
    'Macro_Se_%': [62.90, 70.83],
    'Baseline_Se_%': [45.33, 40.71],
    'Delta_Se_%p': ['+17.57', '+30.12']
}
df_table7 = pd.DataFrame(table7_data)

# =============================================================================
# 표 8. 기존 연구 대비 비교
# =============================================================================
table8_data = {
    'Method': ['de Chazal et al. (2004)', 'Sellami & Hwang (2019)',
               'Zhou et al. FCBA (2023)', 'Proposed (B2)'],
    'Protocol': ['DS1/DS2', 'DS1/DS2', 'DS1/DS2', 'DS1/DS2'],
    'Class_Set': ['N,S,V,F,Q', 'N,S,V,F,Q', 'N,S,V*', 'N,S,V,F,Q'],
    'Overall_Acc_%': [83.19, 88.34, 91.22, 97.72],
    'S_Precision_%': ['-', 30.44, 36.48, 70.45],
    'S_Sensitivity_%': ['-', 82.04, 82.88, 66.58],
    'S_F1_%': [37.60, 44.41, 50.66, 68.46],
    'Note': ['Classic baseline', 'CNN + RR', '3-class only*', 'Proposed']
}
df_table8 = pd.DataFrame(table8_data)

# =============================================================================
# 전체 실험 결과 (사용자 제공 데이터)
# =============================================================================
full_results_data = {
    'Method': [
        'opt3_mhca_h1_auprc', 'opt3_concat_auprc', 'baseline_opt2_recall',
        'baseline_opt2_f1', 'opt3_concat_proj_recall', 'opt3_concat_proj_f1',
        'mhca_h1_recall', 'mhca_h1_f1', 'opt3_mhca_h1_recall', 'opt3_mhca_h1_f1',
        'mhca_h1_auroc', 'fusion_concat_auroc', 'fusion_concat_proj_auprc',
        'baseline_opt2_auroc', 'fusion_concat_auprc', 'opt3_concat_proj_auprc',
        'opt3_concat_proj_auroc', 'opt3_concat_recall', 'opt3_concat_f1',
        'opt3_concat_auroc', 'fusion_concat_proj_recall', 'fusion_concat_proj_f1',
        'fusion_concat_recall', 'fusion_concat_f1', 'mhca_h1_auprc',
        'fusion_concat_proj_auroc', 'opt3_mhca_h1_auroc', 'baseline_opt2_auprc'
    ],
    'Macro_Acc_%': [
        93.74, 91.35, 93.45, 93.45, 85.29, 85.29,
        86.49, 86.49, 84.26, 84.26, 92.48, 93.42, 96.74,
        96.63, 93.22, 95.73, 95.73, 88.71, 88.71,
        83.64, 81.82, 81.82, 78.64, 78.64, 85.62,
        94.55, 94.52, 96.18
    ],
    'Macro_Se_%': [
        70.83, 64.57, 52.76, 52.76, 82.63, 82.63,
        53.48, 53.48, 66.98, 66.98, 43.28, 47.84, 42.37,
        41.46, 47.56, 35.27, 35.27, 41.99, 41.99,
        42.23, 58.78, 58.78, 54.71, 54.71, 49.25,
        25.23, 25.00, 40.71
    ],
    'Macro_Sp_%': [
        94.33, 92.05, 89.42, 89.42, 92.16, 92.16,
        87.85, 87.85, 90.93, 90.93, 86.16, 88.43, 85.42,
        85.33, 86.13, 80.84, 80.84, 91.75, 91.75,
        89.48, 87.46, 87.46, 87.72, 87.72, 90.36,
        75.14, 75.00, 86.43
    ],
    'Macro_F1_%': [
        58.54, 55.91, 48.79, 48.79, 46.60, 46.60,
        45.61, 45.61, 45.42, 45.42, 45.41, 45.36, 44.53,
        43.96, 43.84, 39.28, 39.28, 35.72, 35.72,
        35.31, 34.96, 34.96, 34.95, 34.95, 33.43,
        24.02, 23.55, 43.19
    ]
}
df_full_results = pd.DataFrame(full_results_data)

# =============================================================================
# Save all tables
# =============================================================================
tables = {
    'Table1_RR_Features_Basic': df_table1,
    'Table1_RR_Features_opt1_Detail': df_table1_opt1,
    'Table2_Class_Distribution': df_table2,
    'Table3_ResUNet_Results': df_table3,
    'Table4_MACNN_SE_Results': df_table4,
    'Table5_Hybrid_Ablation': df_table5,
    'Table6_SVEB_Performance': df_table6,
    'Table7_Backbone_Summary': df_table7,
    'Table8_SOTA_Comparison': df_table8,
    'Full_Experiment_Results': df_full_results
}

# Save as CSV
for name, df in tables.items():
    csv_path = os.path.join(output_dir, f'{name}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {csv_path}")

# Save all tables to single XLSX
xlsx_path = os.path.join(output_dir, 'All_Tables.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    for name, df in tables.items():
        # Sheet name max 31 chars
        sheet_name = name[:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Added sheet: {sheet_name}")

print(f"\nAll tables saved to: {xlsx_path}")
print(f"Individual CSVs saved to: {output_dir}/")
