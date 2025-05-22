
# 项目介绍

本项目是基于 [verl](https://github.com/volcengine/verl/tree/82cbc43dc7c42484cc0626247c9b44fd307c0513) 的二次开发，旨在增强原有框架的灵活性和性能，主要针对 reward 模型的调用和管理方式进行了优化。

## 核心功能(更新到0415)

### 1. 远程 Reward 调用机制

重构了 reward 模型的调用方式，实现了通过远程接口访问 reward 模型的功能。这使得用户可以：
- 将 reward 模型部署在独立服务器上
- 通过 API 接口调用外部 reward 服务
- 降低主训练流程的资源占用

### 2. 多 Reward 加权融合

增加了对多个 reward 模型同时调用并进行加权融合的支持：
- 可配置多个 reward 模型的权重
- 灵活组合不同评估维度的 reward 信号
- 支持自定义融合策略

### 3. 协程多线程优化

引入基于协程的多线程调用机制，显著提高了模型训练效率：
- 并行处理 reward 计算请求
- 降低 I/O 等待时间
- 优化资源利用率

### 4. 完善二次开发教程

提供了全面的二次开发教程：
- reward 模型远程部署指南
- 多 reward 配置教程
- 性能调优建议
- 二次开发指南
- 常见问题排查与调试方法

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuanzhoulvpi2017/nano_rl&type=Date)](https://www.star-history.com/#yuanzhoulvpi2017/nano_rl&Date)
