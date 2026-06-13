Measured results — `baseline:gptq` (`is_final,true`), seed 42. FP16 reference perplexity
4.9071. Lower is better.

## ptq-7b-int4 (INT4, group 128)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 5.0711 | 4.9071 | 0.1640 | 219.9 |

## ptq-7b-int3 (INT3, group 128)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 6.1011 | 4.9071 | 1.1940 | 220.6 |

## ptq-7b-int4-g64 (INT4, group 64)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 5.0435 | 4.9071 | 0.1363 | 219.1 |
