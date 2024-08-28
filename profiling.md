# Without Changes

```
I0828 21:41:49.546723 1 model.py:284]          8629 function calls in 0.019 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.004    0.004    0.019    0.019 /models/postprocess/1/model.py:113(handle_request)
        7    0.005    0.001    0.005    0.001 {built-in method numpy.array}
        1    0.005    0.005    0.005    0.005 {transpose}
     8400    0.004    0.000    0.004    0.000 {minMaxLoc}
        1    0.001    0.001    0.001    0.001 {NMSBoxes}
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 /opt/tritonserver/backends/python/triton_python_backend_utils.py:123(get_input_tensor_by_name)
      207    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        2    0.000    0.000    0.000    0.000 {built-in method time.perf_counter}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.000    0.000    0.000    0.000 {built-in method builtins.len}



I0828 21:41:49.546957 1 model.py:285] Total Time: 0.018770832999507547
```

# With Changes

```
I0828 21:45:18.024403 1 model.py:284]          8645 function calls in 0.032 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.004    0.004    0.032    0.032 /models/postprocess/1/model.py:113(handle_request)
        7    0.017    0.002    0.017    0.002 {built-in method numpy.array}
        1    0.008    0.008    0.008    0.008 {transpose}
     8400    0.003    0.000    0.003    0.000 {minMaxLoc}
        1    0.000    0.000    0.000    0.000 {NMSBoxes}
        1    0.000    0.000    0.000    0.000 /opt/tritonserver/backends/python/triton_python_backend_utils.py:123(get_input_tensor_by_name)
        6    0.000    0.000    0.000    0.000 {method 'astype' of 'numpy.ndarray' objects}
        1    0.000    0.000    0.000    0.000 /models/postprocess/1/model.py:155(<listcomp>)
        2    0.000    0.000    0.000    0.000 /models/postprocess/1/model.py:77(calculate_score)
      209    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
        2    0.000    0.000    0.000    0.000 {built-in method time.perf_counter}
        4    0.000    0.000    0.000    0.000 {built-in method builtins.max}
        1    0.000    0.000    0.000    0.000 /models/postprocess/1/model.py:156(<listcomp>)
        4    0.000    0.000    0.000    0.000 {built-in method builtins.min}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.000    0.000    0.000    0.000 {method 'add' of 'set' objects}
        2    0.000    0.000    0.000    0.000 {built-in method builtins.len}



I0828 21:45:18.024780 1 model.py:285] Total Time: 0.03214170799947169
```
