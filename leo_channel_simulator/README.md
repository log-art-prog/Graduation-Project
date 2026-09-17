# 低轨卫星星地信道仿真器 (LEO Satellite Channel Simulator)

基于真实TLE轨道驱动的低轨卫星星地信道仿真工具，支持多环境对比分析。

## 功能特性

### 1. 真实轨道驱动
- 支持读取标准TLE (Two-Line Element) 数据
- 使用Skyfield进行高精度轨道外推
- 实时计算卫星仰角、方位角、斜距、径向速度
- 自动计算过顶点、过境窗口

### 2. 多环境支持
| 环境类型 | 阴影衰落σ | 最大时延扩展 | 莱斯K因子范围 | 适用场景 |
|----------|-----------|--------------|---------------|----------|
| Open (开阔地) | 1.0 dB | 10 ns | 10~25 dB | 农村、沙漠、海洋 |
| Suburban (城郊) | 3.5 dB | 100 ns | 3~15 dB | 郊区、乡镇 |
| Urban (城市) | 6.0 dB | 1000 ns | -10~5 dB | 城市中心、高楼密集区 |

### 3. 完整信道模型
- 自由空间路径损耗
- 大气吸收损耗 (ITU-R P.676)
- 降雨衰减 (ITU-R P.838)
- 阴影衰落 (Loo模型)
- 小尺度莱斯/瑞利衰落
- 多普勒频移与扩展
- 多径时延扩展

### 4. 可视化输出
- 轨道轨迹与仰角变化
- 多普勒频移曲线
- 路径损耗分解
- 接收功率与SNR
- 衰落包络统计
- 多环境对比图

## 文件结构

```
leo_channel_simulator/
├── README.md                   # 说明文档
├── requirements.txt            # 依赖包
├── main.py                     # 主程序入口
├── src/
│   ├── tle_orbit.py            # TLE轨道计算模块
│   ├── channel_model.py        # 信道模型核心
│   ├── propagation.py          # 传播损耗模型
│   ├── fading.py               # 衰落模型
│   ├── environment.py          # 环境参数定义
│   └── visualization.py        # 可视化模块
├── data/
│   └── starlink_tle.txt        # 星链卫星TLE数据示例
└── output/                     # 结果输出目录
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行仿真
```bash
python main.py
```

### 3. 自定义参数
修改 `main.py` 中的配置：
- 替换TLE数据文件路径
- 设置地面站经纬度
- 选择仿真频段（S/C/Ku/Ka/Q/V）
- 切换环境类型或同时对比多种环境
- 设置雨率（mm/h）

## TLE数据获取
可从以下网站获取最新TLE数据：
- CelesTrak: https://celestrak.org/
- Space-Track: https://www.space-track.org/

## 参考标准
- ITU-R P.618: 地空传播预测
- ITU-R P.676: 大气气体衰减
- ITU-R P.838: 降雨衰减模型
- ITU-R P.840: 云雾衰减
- 3GPP TR 38.811: 5G卫星通信信道模型
- 3GPP TR 38.901: 地面移动通信信道模型
