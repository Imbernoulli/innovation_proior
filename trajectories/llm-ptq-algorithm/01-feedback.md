Measured results — `baseline:round_to_nearest` (`is_final,true`), seed 42. FP16 reference perplexity
4.9071. Lower is better.

## ptq-7b-int4 (INT4, group 128)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 5.1343 | 4.9071 | 0.2271 | 22.5 |

## ptq-7b-int3 (INT3, group 128)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 6.7341 | 4.9071 | 1.8270 | 21.8 |

## ptq-7b-int4-g64 (INT4, group 64)
| seed | wikitext2_ppl | fp16_ppl | degradation | quant_time (s) |
|---|---|---|---|---|
| 42 | 5.0890 | 4.9071 | 0.1819 | 22.8 |
