# 推理功能（实验性质）

本教程用于介绍如何使用BasicTS内置功能实现模型的推理能力

## ⏬ 推理脚本

使用推理脚本，可以从指定文件中读取输入数据，使用指定模型推理，并将输出

```bash
cd /path/to/your/project
git clone https://github.com/zezhishao/BasicTS.git
```

## 💿 Web页面

### 操作系统

我们建议在 Linux 系统（如 Ubuntu 或 CentOS）上使用 BasicTS。

### Python

需要 Python 3.6 或更高版本（建议使用 3.8 或更高版本）。

我们推荐使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/) 来创建虚拟 Python 环境。

### PyTorch

BasicTS 对 PyTorch 版本非常灵活。您可以根据 Python 版本[安装 PyTorch](https://pytorch.org/get-started/previous-versions/)。我们建议使用 `pip` 进行安装。

### 其他依赖项

确保 PyTorch 正确安装后，您可以安装其他依赖项：

```bash
pip install -r requirements.txt
```

### 示例设置

#### 示例 1：Python 3.11 + PyTorch 2.3.1 + CUDA 12.1 (推荐)

```bash
# 安装 Python
conda create -n BasicTS python=3.11
conda activate BasicTS
# 安装 PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# 安装其他依赖项
pip install -r requirements.txt
```

#### 示例 2：Python 3.9 + PyTorch 1.10.0 + CUDA 11.1

```bash
# 安装 Python
conda create -n BasicTS python=3.9
conda activate BasicTS
# 安装 PyTorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# 安装其他依赖项
pip install -r requirements.txt
```

## 📦 API服务

您可以从 [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp?usp=sharing) 或 [百度网盘](https://pan.baidu.com/s/1shA2scuMdZHlx6pj35Dl7A?pwd=s2xe) 下载 `all_data.zip` 文件。将文件解压到 `datasets/` 目录：

```bash
cd /path/to/BasicTS # not BasicTS/basicts
unzip /path/to/all_data.zip -d datasets/
```

这些数据集已预处理完毕，可以直接使用。

> [!NOTE]
> `data.dat` 文件是以 `numpy.memmap` 格式存储的数组，包含原始时间序列数据，形状为 [L, N, C]，其中 L 是时间步数，N 是时间序列数，C 是特征数。
>
> `desc.json` 文件是一个字典，存储了数据集的元数据，包括数据集名称、领域、频率、特征描述、常规设置和缺失值。
>
> 其他文件是可选的，可能包含附加信息，如表示时间序列间预定义图结构的 `adj_mx.pkl`。

> [!NOTE]
> 如果您对预处理步骤感兴趣，可以参考[预处理脚本](../scripts/data_preparation) 和 `raw_data.zip`。

## 🧑‍💻 进一步探索

本教程为您提供了 BasicTS 的基础知识，但还有更多内容等待您探索。在深入其他主题之前，我们先更详细地了解 BasicTS 的结构：

<div align="center">
  <img src="figures/DesignConvention.jpeg" height=350>
</div>

BasicTS 的核心组件包括 `Dataset`、`Scaler`、`Model`、`Metrics`、`Runner` 和 `Config`。为简化调试过程，BasicTS 作为一个本地化框架运行，所有代码都直接在您的机器上运行。无需 `pip install basicts`，只需克隆仓库，即可本地运行代码。

以下是一些高级主题和附加功能，帮助您充分利用 BasicTS：

- **🎉 [快速上手](./getting_started_cn.md)**
- **💡 [了解 BasicTS 的设计理念](./overall_design_cn.md)**
- **📦 [探索数据集设计并自定义数据集](./dataset_design_cn.md)**
- **🛠️ [了解数据缩放器设计并创建自定义缩放器](./scaler_design_cn.md)**
- **🧠 [深入了解模型设计并构建自定义模型](./model_design_cn.md)**
- **📉 [了解评估指标设计并开发自定义损失函数与评估指标](./metrics_design_cn.md)**
- **🏃‍♂️ [掌握执行器设计并创建自定义执行器](./runner_design_cn.md)**
- **📜 [解析配置文件设计并自定义配置](./config_design_cn.md)**
- **🔍 [探索多种基线模型](../baselines/)**
