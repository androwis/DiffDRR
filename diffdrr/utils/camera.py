import torch

from .backend import get_device


class Detector:
    """
    Construct a 6 DoF X-ray detector system. This model is based on a C-Arm.

    Inputs
    ------
    height : int
        Height of the X-ray detector (ie, DRR height)
    width : int
        Width of the X-ray detector (ie, DRR width)
    delx : float
        Pixel spacing in the X-direction of the X-ray detector
    dely : float
        Pixel spacing in the Y-direction of the X-ray detector
    device : str or torch.device
        Compute device. If str, either "cpu", "cuda", or "mps".
    """

    def __init__(self, height, width, delx, dely, device):
        self.height = height
        self.width = width
        self.delx = delx
        self.dely = dely
        self.device = device if isinstance(device, torch.device) else get_device(device)

    def make_xrays(self, sdr, rotations, translations):
        """
        Inputs
        ------
            sdr : torch.Tensor
                Source-to-Detector radius (half of the Source-to-Detector distance)
            rotations : torch.Tensor
                Vector of C-Arm rotations (theta, phi, gamma) for azimuthal, polar, and roll
            translations : torch.Tensor
                Vector of volume translations (bx, by, bz)
        """

        # Get the detector plane normal vector
        source, center, u, v, n_batches = _get_basis(sdr, rotations, self.device)
        source += translations
        center += translations
        center = center.unsqueeze(1).unsqueeze(1)

        # Construt the detector plane
        t = torch.arange(-self.height // 2, self.height // 2, device=self.device) + 1
        s = torch.arange(-self.width // 2, self.width // 2, device=self.device) + 1
        t = t * self.delx
        s = s * self.dely
        coefs = torch.cartesian_prod(t, s).reshape(1, self.height, self.width, 2)
        coefs = coefs.expand(n_batches, -1, -1, -1)
        basis = torch.stack([u, v], dim=1).unsqueeze(1)
        rays = coefs @ basis
        rays += center
        return source, rays


def _get_basis(rho, rotations, device):
    # Get the rotation of 3D space
    n_batches = len(rotations)
    R = rho * Rxyz(rotations, n_batches, device)

    # Get the detector center and X-ray source
    source = R[:, :, 0]
    center = -source

    # Get the basis of the detector plane (before translation)
    # TODO: normalizing the vectors seems to break the gradient, fix in future
    u, v = R[:, :, 1], R[:, :, 2]
    # u_ = u / torch.norm(u)
    # v_ = v / torch.norm(v)

    return source, center, u, v, n_batches


# Define 3D rotation matrices
def Rxyz(rotations, n_batches, device):
    theta = rotations[:, 0]
    phi = rotations[:, 1]
    gamma = rotations[:, 2]
    t0 = torch.zeros(n_batches, device=device)
    t1 = torch.ones(n_batches, device=device)
    return (
        Rz(theta, t0, t1, n_batches)
        @ Ry(phi, t0, t1, n_batches)
        @ Rx(gamma, t0, t1, n_batches)
    )


def Rx(theta, t0, t1, n_batches):
    return (
        torch.stack(
            [
                t1,
                t0,
                t0,
                t0,
                torch.cos(theta),
                -torch.sin(theta),
                t0,
                torch.sin(theta),
                torch.cos(theta),
            ]
        )
        .reshape(3, 3, n_batches)
        .permute(2, 0, 1)
    )


def Ry(theta, t0, t1, n_batches):
    return (
        torch.stack(
            [
                torch.cos(theta),
                t0,
                torch.sin(theta),
                t0,
                t1,
                t0,
                -torch.sin(theta),
                t0,
                torch.cos(theta),
            ]
        )
        .reshape(3, 3, n_batches)
        .permute(2, 0, 1)
    )


def Rz(theta, t0, t1, n_batches):
    return (
        torch.stack(
            [
                torch.cos(theta),
                -torch.sin(theta),
                t0,
                torch.sin(theta),
                torch.cos(theta),
                t0,
                t0,
                t0,
                t1,
            ]
        )
        .reshape(3, 3, n_batches)
        .permute(2, 0, 1)
    )
