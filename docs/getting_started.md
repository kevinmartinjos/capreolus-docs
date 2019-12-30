# Basic usage

```python
from capreolus import train

pipeline_config = {
    "reranker": "KNRM",
    "benchmark": "dummy",
    "niters": 40,
}

train.main(pipeline_config)
```