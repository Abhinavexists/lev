# lev

An open System One decision model. Give it a state and typed questions (yes/no, choice, score), and it answers all of them in one forward pass with calibrated probabilities over exactly the options you supplied. A LoRA adapter on Qwen3.5-4B, served over TypeSafe's `/v1/systemone` protocol.

```bash
pip install "lev[serve] @ git+https://github.com/Abhinavexists/lev#subdirectory=packages/lev"
```

```python
import lev

model = lev.load("interfaze-ai/lev")
```

The core install is pure Python (schema, prompt layouts, router, calibration); `[train]` adds the model stack, `[serve]` the HTTP server. Benchmarks, quickstart and design: [github.com/Abhinavexists/lev](https://github.com/Abhinavexists/lev). Weights: [interfaze-ai/lev](https://huggingface.co/interfaze-ai/lev).
