import torch
import torch.nn.functional as F


def create_gaussian_kernel(kernel_size, sigma):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    # make a grid of (x, y) coordinates
    ax = torch.linspace(-(kernel_size // 2), kernel_size // 2, steps=kernel_size)
    xx, yy = torch.meshgrid([ax, ax], indexing='ij')
    xx = xx.float()
    yy = yy.float()
    # compute the 2D Gaussian function
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    # normalize the kernel so that sum is 1
    kernel = kernel / torch.sum(kernel)
    # check the shape before reshaping
    if kernel.numel() != kernel_size * kernel_size:
        raise RuntimeError(f"Kernel size mismatch: expected {kernel_size * kernel_size}, got {kernel.numel()}")
    # reshape to be compatible with conv2d (out_channels, in_channels, H, W)
    return kernel.view(1, 1, kernel_size, kernel_size)


def create_laplacian_kernel():
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    return laplacian_kernel.view(1, 1, 3, 3)


def abslog(image, iter_num=2, kernel_size=3, sigma=0.6):
    # create the Gaussian and Laplacian kernels
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(image.device)
    laplacian_kernel = create_laplacian_kernel().to(image.device)
    now_input = image
    # apply absLoG operation
    for _ in range(iter_num):
        # apply the Gaussian smoothing
        smoothed = F.conv2d(now_input, gaussian_kernel, padding=kernel_size // 2)
        # apply the Laplacian filter
        log_out = F.conv2d(smoothed, laplacian_kernel, padding=1)
        # abs and norm
        log_out = torch.abs(log_out)
        abslog_out = (log_out - log_out.min()) / (log_out.max() - log_out.min())
        # update now_input
        now_input = abslog_out
    return abslog_out


if __name__ == "__main__":
    img = torch.zeros(1, 256, 256)
    edge = abslog(img)
    print(edge.shape)