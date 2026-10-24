# 自动测试

截至 2026-10-24，已执行原有能带 16 项和新增拓扑/开放链 40 项，共 **56 项通过，0 失败、0 错误、0 跳过**。历史记录见[能带测试文本](../results/logs/ssh_bands_pytest_run01.txt)和[全套测试文本](../results/logs/ssh_topology_pytest_run02.txt)。XML 含机器信息，仅保留本地。

能带测试覆盖厄米性、固定动量的复跃迁相位、数值与解析能谱、带隙、等跃迁闭隙及动量周期性。

新增测试覆盖绕数符号、几何、网格与能量缩放、闭隙和未分辨保护；开链键、终止方式、傅里叶约定、厄米与手征对称、本征方程、归一化、谱配对；近零与局域性联合判据、概率衰减、完全二聚化和均匀链解析基准、六种链长劈裂；输入校验、保存数据重构和输出目录拒绝覆盖。

在项目根目录用现有 `.venv` 执行：

```powershell
& '.\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider tests
```

只验证能带时，将末尾 `tests` 替换为 `tests/test_ssh_bands.py`。通常无需指定 `--basetemp`；如指定，必须使用新的专用临时目录，因为 pytest 会清理该目录。

测试不证明所有参数、不同终止方式、无序或一般扰动都成立，也不覆盖尚未实现的 Zak 相位。详细参数及验证边界见[拓扑报告](../results/ssh_topology_run02/report.md)。
