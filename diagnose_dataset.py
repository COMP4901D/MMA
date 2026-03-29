"""
diagnose_dataset.py  —  诊断 UTD-MHAD 目录结构
"""
import os, sys, glob

root = sys.argv[1] if len(sys.argv) > 1 else "./UTD-MHAD"
root = os.path.abspath(root)

print(f"\n{'='*60}")
print(f"  Diagnosing: {root}")
print(f"{'='*60}")

if not os.path.exists(root):
    print(f"\n  ✖ Path does NOT exist: {root}")
    sys.exit(1)

# 1) 列出顶层目录
print(f"\n  Top-level contents of {root}:")
for item in sorted(os.listdir(root)):
    full = os.path.join(root, item)
    if os.path.isdir(full):
        n = len(os.listdir(full))
        print(f"    📁 {item}/   ({n} items)")
    else:
        print(f"    📄 {item}   ({os.path.getsize(full)} bytes)")

# 2) 递归查找所有 .mat 文件
print(f"\n  Searching for *.mat files recursively…")
mats = glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
print(f"    Found {len(mats)} .mat files total")

if mats:
    # 按文件夹分类
    by_dir = {}
    for p in mats:
        d = os.path.dirname(p)
        by_dir.setdefault(d, []).append(os.path.basename(p))

    for d, files in sorted(by_dir.items()):
        print(f"\n    📁 {os.path.relpath(d, root)}/   ({len(files)} .mat files)")
        for f in sorted(files)[:5]:
            print(f"       {f}")
        if len(files) > 5:
            print(f"       … and {len(files)-5} more")

# 3) 特别检查 Inertial / Depth
for name in ["Inertial", "inertial", "Depth", "depth",
             "RGB", "Skeleton", "skeleton"]:
    path = os.path.join(root, name)
    if os.path.exists(path):
        files = os.listdir(path)
        print(f"\n  ✔ {name}/ exists with {len(files)} files")
        if files:
            print(f"    First 3: {sorted(files)[:3]}")
    else:
        print(f"\n  ✖ {name}/ does NOT exist")

print(f"\n{'='*60}")
print("  Done. Use the information above to verify directory structure.")
print(f"{'='*60}\n")