# levbench

Benchmark harness for System One decision models: TypeSafe's Jev and any `/v1 systemone`-compatible server, through the same code path. Reports accuracy, log loss, Brier, ECE with reliability bins, selective accuracy, latency, tokens, cost and schema-retry counts.

```bash
levbench eval --backend jev --tasks data/s1bench
levbench eval --backend lev --tasks data/s1bench --base-url http://localhost:8000
```

levbench does not depend on `lev`. Docs and results: [github.com/Abhinavexists/lev](https://github.com/Abhinavexists/lev).
