使用conda创建一个新的python环境，推荐3.11

```bash
conda create -n verl_custom python=3.11
```

进入项目目录并安装

```bash
conda activate verl_custom
cd verl_proj
pip install -e . -i https://mirrors.aliyun.com/pypi/simple
```

如果没有vllm，再安装一下vllm（当前verl支持的vllm版本为0.8.x）

```bash
pip install vllm -i https://mirrors.aliyun.com/pypi/simple
```