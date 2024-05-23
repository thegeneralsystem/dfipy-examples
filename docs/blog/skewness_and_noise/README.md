# Code companion to "Limitations and challenges in mobility data: skewness and noise"

## Reproducibility

- This is the code companion to the blog post [Limitations and challenges in mobility data: skewness and noise](https://www.generalsystem.com/blog/skewness-and-noise). 
- It comes as part of the [dfipy-example](https://github.com/thegeneralsystem/dfipy-examples/tree/main) repository, open sourced by General System under [Apache License, Version 2.0](https://github.com/thegeneralsystem/dfipy-examples/blob/main/LICENCE).

### Data skewness

To reproduce the first part of the blog post about the data skewness you will need a mobility dataset, either stored in an S3 bucket or stored in a DFI instance.

The Australia dataset used for the post is not provided within the code, and you are invited to use your own, simply replacing the data source in the code below.

If you are interested in trying out DFI, please follow the [quick start guide](https://github.com/thegeneralsystem/quickstart-guide), and then add `dfipy==1.0.1` to the requirements.

Without DFI you can still re-create the graphs for your dataset following the first part of the notebook `data_skewness.ipynb`.

### Noisy data

The second part of the blog post is completely reproducible with the only library requirements provided in the `requirements.txt` and the code in the enclosed files.
All the code to create synthetic trajectories and to add noise is open sourced and available to the reader.
You are invited to play with it and create your own trajectories and use the noise model to make it realistic.

The starting point is the notebook `data_with_noise.ipynb` to add noise to a single device and see the effect of each individual noise pattern on the dataset.

To reproduce the results in the blog post you can run the code in `data_with_noise_multiple_devices.ipynb`, where noise is applied to a group of devices that is then grouped in H3 and timestamps.

### Python environment

To re-create both the pipeline for the first part, create a python virtualenvironment and activated with:

```bash
virtualenv venv -p python3.11
source venv/bin/activate
```
