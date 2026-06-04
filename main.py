"""
主入口文件
运行三阶段渐进式迁移学习
"""
import torch
import numpy as np
from pathlib import Path
import json

from data_process.dataloader import EnterpriseRiskDataLoader
from data_process.konwledge_graph import EnterpriseKnowledgeGraph
from model.transfer_model import TransferLearningModel
from trainer.progressive_trainer import ProgressiveTransferTrainer


def set_random_seed(seed=42):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_domain_data(split_base_path, region_name):
    """
    加载单个域的时间划分数据

    Args:
        split_base_path: 时间划分数据基础路径
        region_name: 地区名称 ('aba' 或 'chengdu')

    Returns:
        hetero_data: 异构图数据
    """
    print(f"\n加载 {region_name.upper()} 域数据...")

    loader = EnterpriseRiskDataLoader("", "")
    region_data, split_info = loader.load_single_region_temporal_data(
        split_base_path, region_name
    )

    print(f"{region_name.upper()} 数据统计:")
    for table_name, df in region_data.items():
        if df is not None:
            print(f"  {table_name}: {df.shape}")

    # 构建知识图谱
    print(f"构建 {region_name.upper()} 知识图谱...")
    kg = EnterpriseKnowledgeGraph(region_data, split_info=split_info)
    kg.preprocess_data()
    hetero_data = kg.build_heterogeneous_graph()

    print(f"  企业节点: {hetero_data['company'].x.shape[0]:,}")
    print(f"  节点类型: {list(hetero_data.node_types)}")
    print(f"  边类型: {len(hetero_data.edge_types)}")

    if hasattr(hetero_data['company'], 'train_mask'):
        print(f"  训练集: {hetero_data['company'].train_mask.sum().item()}")
        print(f"  验证集: {hetero_data['company'].val_mask.sum().item()}")
        print(f"  测试集: {hetero_data['company'].test_mask.sum().item()}")

    return hetero_data


def main():
    """主函数"""
    print("=" * 80)
    print("三阶段渐进式迁移学习: 成都(源域) -> 阿坝(目标域)")
    print("=" * 80)

    # # 设置随机种子
    # set_random_seed(42)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # ✅ 修改为你的实际数据路径
    split_base_path = r"C:\Users\Claire\study\毕设\code\4\random_data_splits"

    # ========== Step 1: 加载数据 ==========
    print("\n" + "=" * 80)
    print("Step 1: 加载数据")
    print("=" * 80)

    # 加载源域(成都)
    source_data = load_domain_data(split_base_path, 'chengdu')
    source_data = source_data.to(device)

    # 加载目标域(阿坝)
    target_data = load_domain_data(split_base_path, 'aba')
    target_data = target_data.to(device)

    # ========== Step 2: 构建模型 ==========
    print("\n" + "=" * 80)
    print("Step 2: 构建迁移学习模型")
    print("=" * 80)

    # 获取元数据
    node_types = source_data.node_types
    edge_types = source_data.edge_types
    node_features = {}
    for node_type in node_types:
        if hasattr(source_data[node_type], 'x'):
            node_features[node_type] = source_data[node_type].x.shape[1]
        else:
            node_features[node_type] = 1
    metadata = (node_features, edge_types)

    # 创建模型
    # 注意：company 节点特征维度已从 12 扩展为 12 + TOPO_FEAT_DIM (=8) = 20
    # （在 knowledge_graph.py 的 build_heterogeneous_graph 中自动拼接）
    model = TransferLearningModel(
        metadata=metadata,
        hidden_dim=64,
        num_layers=2,
        shared_dim=48,
        private_dim=16,
        num_classes=2,
        use_mmd=True,
        topo_dim=8   # 与 TOPO_FEAT_DIM 一致
    )

    # ✅ 初始化延迟参数（用源域数据做一次dummy forward）
    print("初始化模型参数...")
    model.to(device)
    model.eval()
    with torch.no_grad():
        try:
            _ = model(
                source_data.x_dict,
                source_data.edge_index_dict,
                return_features=False
            )
            print("✅ 模型参数初始化成功")
        except Exception as e:
            print(f"⚠️ 初始化警告: {e}")

    print(f"✅ 模型构建完成")
    print(f"  模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

    # ========== Step 3: 创建训练器 ==========
    print("\n" + "=" * 80)
    print("Step 3: 创建渐进式训练器")
    print("=" * 80)

    trainer = ProgressiveTransferTrainer(
        model=model,
        source_data=source_data,
        target_data=target_data,
        device=device
    )

    print("✅ 训练器创建完成")

    # ========== Step 4: 训练模型 ==========
    print("\n" + "=" * 80)
    print("Step 4: 开始三阶段训练")
    print("=" * 80)

    results = trainer.run_full_pipeline()

    # ========== Step 20: 保存结果 ==========
    print("\n" + "=" * 80)
    print("Step 20: 保存结果")
    print("=" * 80)

    with open('result/transfer_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("✅ 结果已保存到 transfer_results.json")

    print("\n" + "=" * 80)
    print("迁移学习完成!")
    print("=" * 80)

    return model, results


if __name__ == "__main__":
    # 运行7次实验
    all_results = []

    for run_id in range(7):
        seed = 42 + run_id
        print(f"\n{'=' * 80}")
        print(f"🔄 开始第 {run_id + 1}/7 次实验 (seed={seed})")
        print(f"{'=' * 80}")

        # 设置随机种子
        set_random_seed(seed)

        # 运行实验
        model, results = main()
        results['seed'] = seed
        results['run_id'] = run_id + 1
        all_results.append(results)

    # ========== 新增：生成表格形式的总结果 ==========
    import pandas as pd

    # 1. 保存JSON格式（保留原有功能）
    with open('result/all_runs_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print("\n✅ 原始结果已保存到 result/run_baselines_ablation.json")

    # 2. 生成详细结果表格
    detailed_rows = []
    for r in all_results:
        for stage in ['stage1', 'stage2']:
            row = {
                'Run': r['run_id'],
                'Seed': r['seed'],
                'Stage': stage,
                'AUC': r[stage]['auc'],
                'AUPRC': r[stage]['auprc'],
                'F1': r[stage]['f1'],
                'F1_Pos': r[stage]['f1_pos'],
                'Precision_Pos': r[stage]['precision_pos'],
                'Recall_Pos': r[stage]['recall_pos'],
                'H_measure': r[stage]['h_measure'],
                'Specificity': r[stage]['specificity'],  # ✅ 新增
                'Sensitivity': r[stage]['sensitivity']  # ✅ 新增
            }
            detailed_rows.append(row)

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv('result/detailed_results.csv', index=False, encoding='utf-8-sig')
    print("✅ 详细结果已保存到 result/detailed_results.csv")

    # 3. 生成统计摘要表格
    summary_rows = []
    for stage in ['stage1', 'stage2']:
        metrics = ['auc', 'auprc', 'f1', 'f1_pos', 'precision_pos', 'recall_pos', 'h_measure', 'specificity',
                   'sensitivity']  # ✅ 添加新指标
        row = {'Stage': stage}

        for metric in metrics:
            values = [r[stage][metric] for r in all_results]
            row[f'{metric.upper()}_mean'] = np.mean(values)
            row[f'{metric.upper()}_std'] = np.std(values)

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('result/summary_results.csv', index=False, encoding='utf-8-sig')
    print("✅ 统计摘要已保存到 result/summary_results.csv")

    # 4. 打印表格到控制台
    print("\n" + "=" * 80)
    print("📊 7次实验统计摘要")
    print("=" * 80)
    print("\n" + df_summary.to_string(index=False))

    # 20. 生成改进幅度表格
    improvement_rows = []
    for r in all_results:
        improvement = {
            'Run': r['run_id'],
            'Seed': r['seed'],
            'Stage1_AUC': r['stage1']['auc'],
            'Stage2_AUC': r['stage2']['auc'],
            'Improvement': r['stage2']['auc'] - r['stage1']['auc'],
            'Improvement_%': (r['stage2']['auc'] - r['stage1']['auc']) / r['stage1']['auc'] * 100
        }
        improvement_rows.append(improvement)

    df_improvement = pd.DataFrame(improvement_rows)

    # 添加平均行
    avg_row = {
        'Run': 'Average',
        'Seed': '-',
        'Stage1_AUC': df_improvement['Stage1_AUC'].mean(),
        'Stage2_AUC': df_improvement['Stage2_AUC'].mean(),
        'Improvement': df_improvement['Improvement'].mean(),
        'Improvement_%': df_improvement['Improvement_%'].mean()
    }
    df_improvement = pd.concat([df_improvement, pd.DataFrame([avg_row])], ignore_index=True)

    df_improvement.to_csv('result/improvement_analysis.csv', index=False, encoding='utf-8-sig')
    print("✅ 改进分析已保存到 result/improvement_analysis.csv")

    # 6. 生成 Specificity 和 Sensitivity 变化表格
    print("\n生成 Type I/II Error 分析表格...")
    type_error_rows = []
    for r in all_results:
        row = {
            'Run': r['run_id'],
            'Seed': r['seed'],
            # Stage 1
            'Stage1_Specificity': r['stage1']['specificity'],
            'Stage1_Sensitivity': r['stage1']['sensitivity'],
            'Stage1_TypeI_Error': 1 - r['stage1']['specificity'],
            'Stage1_TypeII_Error': 1 - r['stage1']['sensitivity'],
            # Stage 2
            'Stage2_Specificity': r['stage2']['specificity'],
            'Stage2_Sensitivity': r['stage2']['sensitivity'],
            'Stage2_TypeI_Error': 1 - r['stage2']['specificity'],
            'Stage2_TypeII_Error': 1 - r['stage2']['sensitivity'],
            # 改进幅度
            'Specificity_Improvement': r['stage2']['specificity'] - r['stage1']['specificity'],
            'Sensitivity_Improvement': r['stage2']['sensitivity'] - r['stage1']['sensitivity'],
            'TypeI_Error_Reduction': (r['stage1']['specificity'] - r['stage2']['specificity']),
            'TypeII_Error_Reduction': (r['stage1']['sensitivity'] - r['stage2']['sensitivity'])
        }
        type_error_rows.append(row)

    df_type_error = pd.DataFrame(type_error_rows)

    # 添加平均行
    avg_type_error = {
        'Run': 'Average',
        'Seed': '-',
        'Stage1_Specificity': df_type_error['Stage1_Specificity'].mean(),
        'Stage1_Sensitivity': df_type_error['Stage1_Sensitivity'].mean(),
        'Stage1_TypeII_Error': df_type_error['Stage1_TypeII_Error'].mean(),
        'Stage1_TypeI_Error': df_type_error['Stage1_TypeI_Error'].mean(),
        'Stage2_Specificity': df_type_error['Stage2_Specificity'].mean(),
        'Stage2_Sensitivity': df_type_error['Stage2_Sensitivity'].mean(),
        'Stage2_TypeII_Error': df_type_error['Stage2_TypeII_Error'].mean(),
        'Stage2_TypeI_Error': df_type_error['Stage2_TypeI_Error'].mean(),
        'Specificity_Improvement': df_type_error['Specificity_Improvement'].mean(),
        'Sensitivity_Improvement': df_type_error['Sensitivity_Improvement'].mean(),
        'TypeII_Error_Reduction': df_type_error['TypeII_Error_Reduction'].mean(),
        'TypeI_Error_Reduction': df_type_error['TypeI_Error_Reduction'].mean()
    }
    df_type_error = pd.concat([df_type_error, pd.DataFrame([avg_type_error])], ignore_index=True)

    df_type_error.to_csv('result/type_error_analysis.csv', index=False, encoding='utf-8-sig')
    print("✅ Type I/II Error 分析已保存到 result/type_error_analysis.csv")

    print("\n" + "=" * 80)
    print("📊 Type I/II Error 分析（Specificity & Sensitivity）")
    print("=" * 80)
    print("\n" + df_type_error.to_string(index=False))

    # ========== 👇 修改最后的总结输出 ==========

    print("\n" + "=" * 80)
    print("🎉 所有实验完成！共生成5个结果文件：")
    print("  1. result/run_baselines_ablation.json - 完整JSON数据")
    print("  2. result/detailed_results.csv - 每次运行的详细指标")
    print("  3. result/summary_results.csv - 统计摘要(均值±标准差)")
    print("  4. result/improvement_analysis.csv - AUC改进幅度分析")
    print("  20. result/type_error_analysis.csv - Type I/II Error分析 ⭐NEW")
    print("=" * 80)

