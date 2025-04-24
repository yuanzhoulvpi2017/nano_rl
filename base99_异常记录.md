
# error1
```bash
from torch.multiprocessing.reductions import ForkingPickler

或者是

ForkingPickler error when installing verl with torch v2.6.0(+) on AMD GPU #700
```

解决办法，参考链接： https://github.com/volcengine/verl/issues/700
```
pip install "tensordict==0.6.2"
```