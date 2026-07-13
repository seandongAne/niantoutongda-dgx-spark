# Spark 安全清扫记录（2026-07-13）

## 范围与结论

- 目标主机：spark-72（实际 hostname：`spark-48f0`）。
- 经队员授权，使用登录密码完成 root 级只读审计、精确清扫和复验；密码仅在本地进程内存中读取并通过 `sudo -S` 输入，未写入仓库、远端文件、环境变量或命令行。
- 结论：已清除发现的 Koske 残件和休眠的不安全 Jupyter unit；清扫后 root 级复验与普通用户增强健康检查均通过。

## 清扫前发现

- `/home/Developer/koske`：空目录，时间戳与此前入侵时段一致。
- `/dev/shm/hideproc.so`：AArch64 ELF 共享对象，导出 `readdir`，字符串含 `hideproc.c`、`koske` 与 `/proc`；SHA-256 为 `3aa0af2d6d110b23e0c9f2564a746ea9496bfbacc4f75ecf4585c4fc9f8a3c43`。
- `/dev/shm/.hiddenpid`：记录了一个已不存在的 PID，属于残留状态文件。
- `~/.config/systemd/user/jupyter-workshop.service`：已禁用且未运行，但仍配置为在 `0.0.0.0:8888` 上启动 Jupyter，属于可再次暴露公网映射的危险休眠配置。
- root 范围未发现正在运行的 Koske/矿工进程、已加载的 `hideproc.so`、`/etc/ld.so.preload` 注入、相关 cron/systemd 持久化或 Docker 容器。

## 执行的操作

- 删除 `/home/Developer/koske`、`/dev/shm/hideproc.so` 和 `/dev/shm/.hiddenpid`。
- 停用并删除 `jupyter-workshop.service`，刷新 user systemd 配置。
- 保留队员确认归属正常的两把 `authorized_keys` 公钥。
- 未改动无确证关联的 `/tmp/setup_stepfun_remote.sh`，未启动模型、训练或其他 GPU 工作负载。

## 复验结果

- root 级复验：`✅ ROOT KOSKE SWEEP CLEAN`。
- 独立增强健康检查：`✅ SPARK CLEAN (未发现已知 IOC; load=0.01; 8888→8072,9000→9072 未暴露)`。
- 健康检查现已额外覆盖 `hideproc.so`、`.hiddenpid`、进程映射、系统级持久化和休眠的不安全 Jupyter unit；已知残件再次出现时会 fail closed。

## 边界

本次结论表示在已知 IOC、持久化点、进程映射、容器和敏感监听范围内未再发现异常，不等同于完整磁盘取证或可信重装。若后续再次出现 IOC，应立即停止使用并优先重装/重置主机。
