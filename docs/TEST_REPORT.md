# Test & benchmark report

```
........................................................................ [ 63%]
..........................................                               [100%]
114 passed in 1.17s
```

## Eval (mock)
```
PASS  route-simple                0.0 ms  tier=0 want=0
PASS  route-code                  0.0 ms  tier=1 want=1
PASS  route-hard                  0.3 ms  tier=2 want=2
PASS  route-classify              0.1 ms  tier=0 want=0
PASS  mem-stem                    4.8 ms  top3=['sister_name']
PASS  mem-value                   0.6 ms  top3=['arcade']
PASS  ans-ready                   0.3 ms  mock/mock-small 0.00s 'ready'
PASS  ans-json                    0.2 ms  mock/mock-small 0.00s '{"ok": true}'
PASS  ans-math                    0.1 ms  mock/mock-small 0.00s '42'
PASS  tool-multi-status           0.5 ms  mock: ready
PASS  tool-review-empty           0.1 ms  Nothing is waiting for your approval.

11/11 passed (mock)
```

## Benchmarks
```

MEMORY RECALL  (1930 stored facts, 30 natural-language questions)
                           before           after
  precision@1               0.533           0.833
  recall@3                  0.667           0.967
  median ms/query            2.94            0.38
  p95 ms/query               3.74            4.18

STARTUP  (median of 5 cold imports)
                                                                before           after
  import gemini+memory ms                                         45.3            48.1
  first use of AI hub ms                                     n/a (new)            57.2
  (before/after import gap is within run-to-run noise)                                

TOKENS  (our estimator; 'before' = what an unmanaged request would carry — the original had no equivalent)
                                       before           after
  60-turn history                        6581             437
  tool defs (19 tools)                   4879            2082
  needed tool still offered                 -             5/5
  provider calls, 100 repeats             100               1

RELIABILITY  (SIMULATED providers; measures our logic, not anyone's servers)
                                       before           after
  answered, primary down       0/200 (by design)         200/200
  calls spent on dead primary    not measured               2
  calls for 50 failing reqs      not measured               2

ROUTING COST  (ARITHMETIC on the default price table, not a real bill)
                           always strongest          routed
  100 mixed requests, USD             0.8          0.1696
  saved                                             78.8%

Saved /home/claude/My-Project/bench/results.json
'by design': the original code has no second provider, so when every Gemini model fails the feature fails. The original ladder does keep its own per-model cooldowns; that was not benchmarked.
```

## Avatar frame cost (520px canvas, r=220)
```
real            median 16.4 ms   p95 22.8 ms
holo            median 8.3 ms   p95 11.5 ms
```
