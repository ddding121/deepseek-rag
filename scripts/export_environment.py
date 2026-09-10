"""在已验收的 Windows Python 3.11 虚拟环境导出版本，不读取密钥或网络地址。"""
import importlib.metadata as metadata
import platform
import sys
from pathlib import Path

if sys.prefix == sys.base_prefix:
    raise SystemExit('请先激活本项目 .venv，再运行本脚本。')
if platform.system() != 'Windows' or sys.version_info[:2] != (3, 11):
    raise SystemExit('此导出用于已跑通的 Windows Python 3.11 环境，请在对应环境运行。')
packages = {}
for dist in metadata.distributions():
    name = dist.metadata.get('Name')
    if name and name.lower() not in ('pip','setuptools','wheel'):
        packages[name.lower()] = f'{name}=={dist.version}'
path = Path(__file__).resolve().parents[1] / 'requirements-windows-lock.txt'
if path.exists():
    raise SystemExit('requirements-windows-lock.txt 已存在，请先备份或重命名后再导出。')
path.write_text('# Windows Python 3.11 installed package snapshot; not a cross-platform lock.\n'
                + '\n'.join(packages[k] for k in sorted(packages)) + '\n', encoding='utf-8')
print('已导出 requirements-windows-lock.txt；该文件只包含包名和版本。')
print('请运行 python -m pip check，并在新建虚拟环境安装和验收后再声明可复现。')
