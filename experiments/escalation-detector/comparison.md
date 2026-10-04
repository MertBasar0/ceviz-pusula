
## history 1
| model | AUC | @0.5 P/R/FP/FN | best thr | P | R | FP | FN | best with FP<=5: thr P R FN | p50/p95 ms |
| Kev 0.8B | 0.845 | 0.926/0.581/4/36 | 0.41 | 0.638 | 0.965 | 47 | 3 | 0.50 0.926 0.581 36 | 27.1/30.2 |
| Decider 2B | 0.912 | 0.863/0.802/11/17 | 0.43 | 0.782 | 0.919 | 22 | 7 | 0.56 0.924 0.709 25 | 30.2/31.5 |
| this-that 1.2 | 0.947 | 0.982/0.628/1/32 | 0.05 | 0.949 | 0.86 | 4 | 12 | 0.05 0.949 0.86 12 | 26.9/28.9 |
| qwen3:14b | 0.955 | 0.927/0.884/6/10 | 0.87 | 0.974 | 0.884 | 2 | 10 | 0.87 0.974 0.884 10 | 233.2/253.6 |
| regex | - | 0.533/0.372/28/54 | - | | | | | | ~0 |

## history 3
| model | AUC | @0.5 P/R/FP/FN | best thr | P | R | FP | FN | best with FP<=5: thr P R FN | p50/p95 ms |
| Kev 0.8B | 0.882 | 0.976/0.465/1/46 | 0.41 | 0.698 | 0.942 | 35 | 5 | 0.46 0.928 0.744 22 | 25.2/34.2 |
     categories at best thr: {'explicit_correction': '20/20', 'paraphrased_correction': '28/30', 'stt_correction': '9/12', 'trigger_word_new_task': '11/25', 'non_correction_followup': '14/25', 'repeat_unsatisfied': '15/15', 'repeat_satisfied': '4/8', 'negation_trap': '6/10', 'history_dependent': '11/13'}
| Decider 2B | 0.9 | 0.844/0.756/12/21 | 0.36 | 0.741 | 0.965 | 29 | 3 | 0.57 0.93 0.616 33 | 32.3/34.4 |
     categories at best thr: {'explicit_correction': '20/20', 'paraphrased_correction': '28/30', 'stt_correction': '11/12', 'trigger_word_new_task': '19/25', 'non_correction_followup': '16/25', 'repeat_unsatisfied': '15/15', 'repeat_satisfied': '0/8', 'negation_trap': '8/10', 'history_dependent': '9/13'}
| this-that 1.2 | 0.955 | 1.0/0.372/0/54 | 0.05 | 0.931 | 0.779 | 5 | 19 | 0.05 0.931 0.779 19 | 37.5/39.4 |
     categories at best thr: {'explicit_correction': '17/20', 'paraphrased_correction': '22/30', 'stt_correction': '5/12', 'trigger_word_new_task': '24/25', 'non_correction_followup': '24/25', 'repeat_unsatisfied': '14/15', 'repeat_satisfied': '8/8', 'negation_trap': '9/10', 'history_dependent': '11/13'}
| qwen3:14b | 0.988 | 0.965/0.953/3/4 | 0.13 | 0.965 | 0.965 | 3 | 3 | 0.13 0.965 0.965 3 | 236.5/261.4 |
     categories at best thr: {'explicit_correction': '20/20', 'paraphrased_correction': '29/30', 'stt_correction': '10/12', 'trigger_word_new_task': '24/25', 'non_correction_followup': '25/25', 'repeat_unsatisfied': '15/15', 'repeat_satisfied': '8/8', 'negation_trap': '10/10', 'history_dependent': '11/13'}
| regex | - | 0.533/0.372/28/54 | - | | | | | | ~0 |

## history 5
| model | AUC | @0.5 P/R/FP/FN | best thr | P | R | FP | FN | best with FP<=5: thr P R FN | p50/p95 ms |
| Kev 0.8B | 0.865 | 0.939/0.36/2/55 | 0.42 | 0.696 | 0.907 | 34 | 8 | 0.46 0.923 0.698 26 | 33.0/44.1 |
| Decider 2B | 0.918 | 0.901/0.744/7/22 | 0.39 | 0.769 | 0.965 | 25 | 3 | 0.51 0.925 0.721 24 | 45.2/47.5 |
| this-that 1.2 | 0.934 | 1.0/0.221/0/67 | 0.05 | 0.964 | 0.616 | 2 | 33 | 0.05 0.964 0.616 33 | 36.1/37.8 |
| qwen3:14b | 0.985 | 0.975/0.895/2/9 | 0.05 | 0.975 | 0.907 | 2 | 8 | 0.05 0.975 0.907 8 | 228.8/273.4 |
| regex | - | 0.533/0.372/28/54 | - | | | | | | ~0 |
