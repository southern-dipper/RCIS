#!/usr/bin/env python3
"""
项目环境检查脚本
检查项目依赖和基本功能是否正常
"""

import sys
import importlib

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ❌ 需要Python 3.7或更高版本")
        return False
    else:
        print("  ✅ Python版本符合要求")
        return True

def check_dependencies():
    """检查必要的依赖包"""
    print("\n检查依赖包...")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
    ]
    
    missing_packages = []
    
    for package_name, display_name in required_packages:
        try:
            importlib.import_module(package_name)
            print(f"  ✅ {display_name} 已安装")
        except ImportError:
            print(f"  ❌ {display_name} 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_project_files():
    """检查项目文件"""
    print("\n检查项目文件...")
    
    required_files = [
        'A_star_rpis.py',
        'RRT_rpis.py',
        'quick_test.py',
        'requirements.txt',
        'README.md'
    ]
    
    import os
    missing_files = []
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"  ✅ {file_name} 存在")
        else:
            print(f"  ❌ {file_name} 不存在")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n缺少以下文件: {', '.join(missing_files)}")
        return False
    
    return True

def check_imports():
    """检查模块导入"""
    print("\n检查模块导入...")
    
    try:
        import numpy as np
        print("  ✅ numpy 导入成功")
    except Exception as e:
        print(f"  ❌ numpy 导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✅ matplotlib 导入成功")
    except Exception as e:
        print(f"  ❌ matplotlib 导入失败: {e}")
        return False
    
    try:
        from RRT_rpis import compute_robust_safe_set_optimized
        print("  ✅ RRT_rpis 模块导入成功")
    except Exception as e:
        print(f"  ❌ RRT_rpis 模块导入失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("RPIS Safe Planning - 项目环境检查")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_project_files(),
        check_imports()
    ]
    
    print("\n" + "=" * 60)
    
    if all(checks):
        print("✅ 所有检查通过！项目环境配置正确。")
        print("\n您可以运行以下命令开始使用：")
        print("  python quick_test.py    # 运行性能测试")
    else:
        print("❌ 项目环境检查失败！请解决上述问题。")
        print("\n建议的解决步骤：")
        print("1. 确保Python版本 >= 3.7")
        print("2. 运行: pip install -r requirements.txt")
        print("3. 确认所有源代码文件都在当前目录")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
