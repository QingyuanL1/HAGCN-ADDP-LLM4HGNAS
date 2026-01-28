#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 eval.py 的架构评估功能
"""

import os
import sys
from eval import load_architectures, validate_architecture

def test_load_architectures():
    """测试架构加载功能"""
    print("="*60)
    print("测试 1: 加载最佳架构")
    print("="*60)

    model_dir = "outputs/pdp_20/test_agg_search_20260126T142547"

    # 测试加载最佳架构
    archs = load_architectures(model_dir, arch_source='best', top_k=1)
    print(f"[OK] 加载了 {len(archs)} 个架构")

    if archs:
        arch_info = archs[0]
        print(f"[OK] 架构来源: {arch_info['source']}")
        print(f"[OK] 训练验证成本: {arch_info.get('val_cost', 'N/A')}")
        print(f"[OK] Epoch: {arch_info.get('epoch', 'N/A')}")

        arch = arch_info['arch']
        if isinstance(arch, dict) and 'layers' in arch:
            print(f"[OK] 架构格式: 新格式（带 aggregation）")
            print(f"  Layers: {arch['layers']}")
            print(f"  Aggregation: {arch.get('aggregation', 'default')}")
        else:
            print(f"[OK] 架构格式: 旧格式（简单列表）")
            print(f"  Layers: {arch}")

        # 验证架构
        is_valid = validate_architecture(arch)
        print(f"[OK] 架构验证: {'通过' if is_valid else '失败'}")

    print()

def test_load_top_k():
    """测试加载 Top-K 架构"""
    print("="*60)
    print("测试 2: 加载 Top-5 架构")
    print("="*60)

    model_dir = "outputs/pdp_20/run_20260124T135054"

    # 测试加载 Top-5
    archs = load_architectures(model_dir, arch_source='history', top_k=5)
    print(f"[OK] 加载了 {len(archs)} 个架构")

    for i, arch_info in enumerate(archs, 1):
        print(f"\n架构 {i}:")
        print(f"  来源: {arch_info['source']}")
        print(f"  训练验证成本: {arch_info.get('val_cost', 'N/A'):.4f}")
        print(f"  Epoch: {arch_info.get('epoch', 'N/A')}")

        # 验证架构
        is_valid = validate_architecture(arch_info['arch'])
        print(f"  验证: {'[OK] 通过' if is_valid else '[FAIL] 失败'}")

    print()

def test_backward_compatibility():
    """测试向后兼容性"""
    print("="*60)
    print("测试 3: 向后兼容性（无架构文件）")
    print("="*60)

    # 测试不存在的目录
    model_dir = "outputs/pdp_20/nonexistent"

    archs = load_architectures(model_dir, arch_source='best', top_k=1)
    print(f"[OK] 加载了 {len(archs)} 个架构")

    if archs:
        arch_info = archs[0]
        print(f"[OK] 架构来源: {arch_info['source']}")
        print(f"[OK] 架构: {arch_info['arch']}")

        # 验证 None 架构
        is_valid = validate_architecture(arch_info['arch'])
        print(f"[OK] 架构验证: {'通过' if is_valid else '失败'}")

    print()

def test_architecture_formats():
    """测试不同架构格式的验证"""
    print("="*60)
    print("测试 4: 架构格式验证")
    print("="*60)

    # 测试新格式
    new_format = {
        'layers': [[1,0,1,0,1,0,0], [1,1,0,0,1,1,0], [1,1,1,1,1,0,0]],
        'aggregation': ['mean', 'sum', 'max']
    }
    print("新格式架构:")
    print(f"  {new_format}")
    is_valid = validate_architecture(new_format)
    print(f"  验证: {'[OK] 通过' if is_valid else '[FAIL] 失败'}")

    # 测试旧格式
    old_format = [[1,0,1,0,1,0,0], [1,1,0,0,1,1,0], [1,1,1,1,1,0,0]]
    print("\n旧格式架构:")
    print(f"  {old_format}")
    is_valid = validate_architecture(old_format)
    print(f"  验证: {'[OK] 通过' if is_valid else '[FAIL] 失败'}")

    # 测试 None
    print("\nNone 架构:")
    is_valid = validate_architecture(None)
    print(f"  验证: {'[OK] 通过' if is_valid else '[FAIL] 失败'}")

    # 测试无效格式
    invalid_format = [[1,0,1], [1,1,0]]  # 错误的关系数
    print("\n无效格式架构（错误的关系数）:")
    print(f"  {invalid_format}")
    is_valid = validate_architecture(invalid_format)
    print(f"  验证: {'[FAIL] 失败' if not is_valid else '[OK] 通过（不应该）'}")

    print()

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("eval.py 架构评估功能测试")
    print("="*60 + "\n")

    try:
        test_load_architectures()
        test_load_top_k()
        test_backward_compatibility()
        test_architecture_formats()

        print("="*60)
        print("[SUCCESS] 所有测试完成！")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
