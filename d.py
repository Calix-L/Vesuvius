import kagglehub

# Download latest version
path = kagglehub.model_download("wyz114514/nnunet-391epoch-3d-fullres/pyTorch/default")

print("Path to model files:", path)