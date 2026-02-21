# ====== SHAP分析 - 专注于目标基因 ======
# 在Jupyter Notebook中运行

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, accuracy_score)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置图形参数
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# 设置配色（与R中一致）
colors = ["#d25756", "#f1cb6e", "#7eb4db", "#b889b9"]

print("="*60)
print(" SHAP Analysis for Target Genes - GSE174367 ")
print("="*60)

# %% [markdown]
# ## 1. 数据加载 - 专注于目标基因

# %%
# 定义23个目标基因
target_genes = ["DRD2", "LYN", "NFE2L2", "CDK2", "NR3C1", "PIK3CA", 
                "CTSB", "LCK", "BRAF", "MAPK8", "MMP9", "GSK3B", 
                "PIK3CD", "CSF1R", "FYN", "PTGS2", "SMAD3", "NFKB1", 
                "MAP2K1", "HDAC1", "KIT", "ABL1", "MAPK14"]

print(f"\nTarget genes to analyze: {len(target_genes)} genes")

# 尝试先使用目标基因文件
try:
    X = pd.read_csv("SHAP_features_target_genes.csv", index_col=0)
    print("✓ Using target genes feature file")
    using_target_only = True
except:
    # 如果没有，使用top500基因文件并筛选
    X_full = pd.read_csv("SHAP_features_top500_genes.csv", index_col=0)
    target_genes_present = [g for g in target_genes if g in X_full.columns]
    
    if len(target_genes_present) > 0:
        X = X_full[target_genes_present]
        print(f"✓ Found {len(target_genes_present)} target genes in top500 file")
        print(f"  Genes found: {target_genes_present}")
        target_genes = target_genes_present
        using_target_only = False
    else:
        # 使用所有top基因
        X = X_full
        print("⚠ No target genes found, using all top genes for analysis")
        using_target_only = False

# 加载标签
y_df = pd.read_csv("SHAP_labels.csv")
y = y_df["y"].values
metadata = pd.read_csv("SHAP_sample_metadata.csv")

print(f"\nData shape: {X.shape}")
print(f"Number of samples: {len(y)}")
print(f"Number of features: {X.shape[1]}")
print(f"\nClass distribution:")
print(f"  AD (1): {sum(y==1)} samples")
print(f"  Control (0): {sum(y==0)} samples")

# %% [markdown]
# ## 2. 目标基因表达模式分析

# %%
if using_target_only or len([g for g in target_genes if g in X.columns]) > 0:
    
    # 筛选存在的目标基因
    target_genes_in_data = [g for g in target_genes if g in X.columns]
    print(f"\nAnalyzing {len(target_genes_in_data)} target genes")
    
    # 创建表达数据框
    target_expr = X[target_genes_in_data].copy()
    target_expr['Group'] = ['AD' if y[i] == 1 else 'Control' for i in range(len(y))]
    
    # 计算每个基因在两组的统计
    gene_stats = []
    for gene in target_genes_in_data:
        ad_values = target_expr[target_expr['Group'] == 'AD'][gene]
        control_values = target_expr[target_expr['Group'] == 'Control'][gene]
        
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(ad_values, control_values)
        
        gene_stats.append({
            'Gene': gene,
            'Mean_AD': ad_values.mean(),
            'Mean_Control': control_values.mean(),
            'Fold_Change': ad_values.mean() / control_values.mean() if control_values.mean() != 0 else 0,
            'P_value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    gene_stats_df = pd.DataFrame(gene_stats).sort_values('P_value')
    print("\nTarget genes expression statistics:")
    print(gene_stats_df.to_string())
    
    # 保存统计结果
    gene_stats_df.to_csv('target_genes_statistics.csv', index=False)

# %% [markdown]
# ## 3. 机器学习模型训练（使用可用的特征）

# %%
print("\n3. Model training...")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")

# 使用XGBoost（通常表现最好）
model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=3, 
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 训练模型
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  AUC: {auc_score:.3f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# %% [markdown]
# ## 4. SHAP分析 - 专注于目标基因

# %%
print("\n4. SHAP Analysis for target genes...")

# 创建SHAP解释器
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"SHAP values shape: {shap_values.shape}")

# %% [markdown]
# ## 5. 目标基因SHAP可视化

# %%
# 5.1 目标基因的SHAP重要性分析
target_genes_present = [g for g in target_genes if g in X.columns]

if len(target_genes_present) > 0:
    print(f"\n5. Target Genes SHAP Analysis")
    print(f"Analyzing {len(target_genes_present)} target genes:")
    
    # 获取目标基因的索引
    target_indices = [X_test.columns.get_loc(g) for g in target_genes_present]
    
    # 提取目标基因的SHAP值
    target_shap_values = shap_values[:, target_indices]
    
    # 计算重要性
    target_importance = pd.DataFrame({
        'Gene': target_genes_present,
        'SHAP_Importance': np.abs(target_shap_values).mean(axis=0),
        'Mean_SHAP': target_shap_values.mean(axis=0),
        'Std_SHAP': target_shap_values.std(axis=0),
        'Max_SHAP': np.abs(target_shap_values).max(axis=0),
        'Positive_Impact': (target_shap_values > 0).sum(axis=0),
        'Negative_Impact': (target_shap_values < 0).sum(axis=0)
    }).sort_values('SHAP_Importance', ascending=False)
    
    print("\nTarget Genes SHAP Importance Ranking:")
    print(target_importance.to_string())
    
    # 保存结果
    target_importance.to_csv('target_genes_SHAP_importance.csv', index=False)

# %%
# 5.2 目标基因SHAP Summary Plot
if len(target_genes_present) > 0:
    plt.figure(figsize=(10, max(6, len(target_genes_present) * 0.3)))
    
    # 创建目标基因的数据
    X_test_targets = X_test[target_genes_present]
    
    shap.summary_plot(
        target_shap_values, 
        X_test_targets,
        show=False
    )
    plt.title('SHAP Summary Plot - Target Genes Only', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('SHAP_target_genes_summary.png', bbox_inches='tight', dpi=300)
    plt.show()

# %%
# 5.3 目标基因重要性条形图
if len(target_genes_present) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：SHAP重要性
    ax = axes[0]
    y_pos = np.arange(len(target_importance))
    bars = ax.barh(y_pos, target_importance['SHAP_Importance'].values)
    
    # 根据重要性设置颜色
    max_importance = target_importance['SHAP_Importance'].max()
    bar_colors = [colors[0] if imp > max_importance * 0.5 else colors[2] 
                  for imp in target_importance['SHAP_Importance'].values]
    for bar, color in zip(bars, bar_colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(target_importance['Gene'].values)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_title('Target Genes SHAP Importance', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(target_importance['SHAP_Importance'].values):
        ax.text(v + max_importance * 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # 右图：正负影响比例
    ax = axes[1]
    positive = target_importance['Positive_Impact'].values
    negative = target_importance['Negative_Impact'].values
    
    y_pos = np.arange(len(target_importance))
    ax.barh(y_pos, positive, label='Positive Impact (→AD)', color=colors[0], alpha=0.7)
    ax.barh(y_pos, -negative, label='Negative Impact (→Control)', color=colors[2], alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(target_importance['Gene'].values)
    ax.set_xlabel('Number of Samples', fontsize=11)
    ax.set_title('Direction of SHAP Impact', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Target Genes SHAP Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('target_genes_SHAP_analysis.png', bbox_inches='tight', dpi=300)
    plt.show()

# %%
# 5.4 目标基因SHAP值热图
if len(target_genes_present) > 0:
    plt.figure(figsize=(12, 8))
    
    # 创建热图数据
    heatmap_data = pd.DataFrame(
        target_shap_values.T,
        index=target_genes_present,
        columns=X_test.index
    )
    
    # 按基因重要性排序
    gene_order = target_importance['Gene'].tolist()
    heatmap_data = heatmap_data.loc[gene_order]
    
    # 按样本分组排序
    sample_order = []
    for group in [0, 1]:
        group_samples = X_test.index[y_test == group].tolist()
        sample_order.extend(group_samples)
    
    heatmap_data = heatmap_data[sample_order]
    
    # 创建样本标签
    sample_labels = ['AD' if y_test[X_test.index.get_loc(s)] == 1 else 'Control' 
                     for s in sample_order]
    
    # 绘制热图
    sns.heatmap(
        heatmap_data,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'SHAP value'},
        xticklabels=False,
        yticklabels=True,
        vmin=-0.3,
        vmax=0.3
    )
    
    # 添加样本组标签
    ax = plt.gca()
    for i, label in enumerate(sample_labels):
        color = colors[0] if label == 'AD' else colors[2]
        ax.add_patch(plt.Rectangle((i, -0.5), 1, 0.3, 
                                   facecolor=color, alpha=0.5, 
                                   transform=ax.get_xaxis_transform()))
    
    plt.title('SHAP Values Heatmap - Target Genes', fontsize=14)
    plt.xlabel('Samples (AD vs Control)', fontsize=11)
    plt.ylabel('Genes', fontsize=11)
    plt.tight_layout()
    plt.savefig('SHAP_target_genes_heatmap.png', bbox_inches='tight', dpi=300)
    plt.show()

# %%
# 5.5 单个目标基因的详细分析（Top 6）
if len(target_genes_present) > 0:
    top_target_genes = target_importance.head(6)['Gene'].values
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, gene in enumerate(top_target_genes):
        ax = axes[idx]
        
        # 获取该基因的SHAP值和表达值
        gene_idx = target_genes_present.index(gene)
        gene_shap = target_shap_values[:, gene_idx]
        gene_expr = X_test[gene].values
        
        # 绘制SHAP依赖图
        scatter = ax.scatter(gene_expr, gene_shap, 
                           c=y_test, cmap='RdBu_r', 
                           alpha=0.6, s=50)
        
        # 添加趋势线
        z = np.polyfit(gene_expr, gene_shap, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(gene_expr), p(np.sort(gene_expr)), 
                "k--", alpha=0.5, linewidth=2)
        
        ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
        ax.set_xlabel(f'{gene} Expression', fontsize=10)
        ax.set_ylabel('SHAP value', fontsize=10)
        ax.set_title(f'{gene} (Rank: {idx+1})', fontsize=11)
        ax.grid(alpha=0.3)
        
        # 添加colorbar到最后一个子图
        if idx == 2:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Group (0=Control, 1=AD)', fontsize=9)
    
    plt.suptitle('SHAP Dependence Analysis - Top 6 Target Genes', fontsize=14)
    plt.tight_layout()
    plt.savefig('SHAP_target_genes_dependence.png', bbox_inches='tight', dpi=300)
    plt.show()

# %%
# 5.6 Waterfall plot for target genes (单个样本示例)
if len(target_genes_present) > 0:
    # 选择一个AD样本和一个Control样本
    ad_idx = np.where(y_test == 1)[0][0]
    control_idx = np.where(y_test == 0)[0][0]
    
    for sample_idx, sample_type in [(ad_idx, 'AD'), (control_idx, 'Control')]:
        plt.figure(figsize=(10, 6))
        
        # 只使用目标基因创建waterfall
        shap.waterfall_plot(
            shap.Explanation(
                values=target_shap_values[sample_idx],
                base_values=explainer.expected_value,
                data=X_test[target_genes_present].iloc[sample_idx],
                feature_names=target_genes_present
            ),
            max_display=len(target_genes_present),
            show=False
        )
        
        sample_name = X_test.index[sample_idx]
        plt.title(f'SHAP Waterfall - Target Genes\nSample: {sample_name} ({sample_type})', 
                 fontsize=12)
        plt.tight_layout()
        plt.savefig(f'SHAP_waterfall_target_genes_{sample_type}.png', 
                   bbox_inches='tight', dpi=300)
        plt.show()

# %%
# 6. 生成最终报告
print("\n" + "="*60)
print(" Analysis Summary ")
print("="*60)

if len(target_genes_present) > 0:
    print(f"\nTarget genes analyzed: {len(target_genes_present)}")
    print(f"Model Accuracy: {accuracy:.3f}")
    print(f"Model AUC: {auc_score:.3f}")
    
    print("\nTop 5 Most Important Target Genes (by SHAP):")
    for i, row in target_importance.head(5).iterrows():
        print(f"  {i+1}. {row['Gene']}: {row['SHAP_Importance']:.4f}")
    
    print("\nTarget Genes with Significant Expression Difference (p<0.05):")
    if 'gene_stats_df' in locals():
        sig_genes = gene_stats_df[gene_stats_df['Significant'] == 'Yes']
        for _, row in sig_genes.iterrows():
            print(f"  - {row['Gene']}: FC={row['Fold_Change']:.2f}, p={row['P_value']:.4f}")
    
    # 保存完整报告
    with open('target_genes_SHAP_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("Target Genes SHAP Analysis Report - GSE174367\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Number of target genes analyzed: {len(target_genes_present)}\n")
        f.write(f"Total samples: {len(y)}\n")
        f.write(f"Model Performance: Accuracy={accuracy:.3f}, AUC={auc_score:.3f}\n\n")
        
        f.write("Target Genes SHAP Importance:\n")
        f.write("-"*40 + "\n")
        f.write(target_importance.to_string() + "\n\n")
        
        if 'gene_stats_df' in locals():
            f.write("Target Genes Expression Statistics:\n")
            f.write("-"*40 + "\n")
            f.write(gene_stats_df.to_string() + "\n")
    
    print("\nFiles saved:")
    print("  - target_genes_SHAP_importance.csv")
    print("  - target_genes_statistics.csv")
    print("  - target_genes_SHAP_report.txt")
    print("  - Multiple visualization plots")

print("\n" + "="*60)
print(" Analysis Complete! ")
print("="*60)# 6PPD-Q-AD
