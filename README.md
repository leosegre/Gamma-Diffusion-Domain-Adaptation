# Denoising Diffusion Gamma Models



## Running the Experiments
The code has been tested on PyTorch 1.7.

### Train a model
```
python main.py --config {DATASET_NOISE}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --ni
```
We provide 6 config files for Celeba and Church to train with gaussian noise and gamma noise.

### Sampling from the model

#### Sampling from the generalized model for FID evaluation
```
python main.py --config {DATASET_NOISE}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS} --eta {ETA} --ni
```
where
- `ETA` controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM).
- `STEPS` controls how many timesteps used in the process.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.


#### Sampling from the model for image inpainting
Use `--interpolation` option instead of `--fid`.

#### Sampling from the sequence of images that lead to the sample
Use `--sequence` option instead.

The above two cases contain some hard-coded lines specific to producing the image, so modify them according to your needs.


## References and Acknowledgements


This implementation is based on the code at [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim)
