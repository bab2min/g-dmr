# Incorporating Citation Impact into Analysis of Research Trends

This introduces new generalized DMR topic model, Citation Impact DMR(CI-DMR) topic model. 
It can capture more than one metadata into topic using Legendre polynomial as the basis of topic hyperparameter.

## Installation
Requirement:

* C++11-compatible compiler (gcc >= 4.8 or msvc >= 12)
* Eigen 3.3.7
* OpenCL 1.2 (with graphic card supporting OpenCL, optional)

You can compile using .sln(Visual Studio) or Makefile(linux or macOS).

## Experiments

```
$> gdmr.exe gdmr data/dataset1.txt -K 40 -F 2 -D 4,3 -I 800 --oi 20 --bi 200 -a 1e-2 -s 0.25 --s0 3 -V 1 --alphaEps 1e-10 --mdm 1997,0 --mdM 2017,1
$> gdmr.exe gdmr data/dataset2.txt -K 30 -F 2 -D 4,3 -I 800 --oi 20 --bi 200 -a 1e-2 -s 0.25 --s0 3 -V 1 --alphaEps 1e-10 --mdm 2000,0 --mdM 2017,1
```

If you run the experiment at GPU, you check the OpenCL device first. The following command shows available OpenCL devices.
```
$> gdmr.exe --clList
```
To run training at GPU, you should pass --cl with the device number shown in the above command, like:
```
$> gdmr.exe gdmr data/dataset1.txt --cl 1 -K 40 -F 2 -D 4,3 -I 800 --oi 20 --bi 200 -a 1e-2 -s 0.25 --s0 3 -V 1 --alphaEps 1e-10 --mdm 1997,0 --mdM 2017,1
```

### Plotting topic distribution of g-DMR
We also provide Python code for plotting. You should install requirements first to run plot.py.
```
pip3 install -r plot_requirements.txt
```

Next, you should have the parameter result of g-DMR. You can find the result of this paper at `results/gdmr.lis.40.txt` and `results/gdmr.tm.30.txt`.
To draw plots, use following command.
```
mkdir output_dir
python3 results/gdmr.lis.40.txt output_dir --dim 4,3 --width 800 --height 600
```

## Result
The result of the paper were conducted on NVIDIA GTX960. You may get different results when running on a CPU or a different GPU.


## License
A part of this code is based on a repository `tomoto`(https://github.com/bab2min/tomotopy) which is licensed under the terms of MIT License. 
The rest of this code is also licensed under the terms of MIT License. Please feel free to use this code.
