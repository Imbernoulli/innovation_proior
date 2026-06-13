Measured results — `baseline:awq` (`is_final,true`), seed 42. FP16 reference perplexity
4.9071. Lower is better.

## ptq-7b-int4 (INT4, group 128)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 5.0647 | 4.9071 | 0.1577 | 79.3 |

## ptq-7b-int3 (INT3, group 128)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 5.7776 | 4.9071 | 0.8706 | 79.0 |

## ptq-7b-int4-g64 (INT4, group 64)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 5.0454 | 4.9071 | 0.1383 | 80.2 |
