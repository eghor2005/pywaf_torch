import torch

class AtomicKernel:
    @classmethod
    def ft_up(cls, t: torch.Tensor, nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{up}{(x)}

        :param t: real scalar or array
        :param nprod: integer scalar, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if nprod < 1:
            raise Exception('nprod must be greater than 1')

        # Ensure t is at least 1D tensor
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Create powers of 2
        p = torch.pow(2, torch.linspace(1, nprod, nprod, device=t.device, dtype=t.dtype))

        # Compute sinc(t/(2^p * pi)) = sin(t/(2^p)) / (t/(2^p))
        # Using torch.sinc which computes sin(pi*x)/(pi*x)
        t_expanded = t.unsqueeze(1)  # Shape: (n, 1)
        p_expanded = p.unsqueeze(0)  # Shape: (1, nprod)

        # sinc(t/(2^p * pi)) = torch.sinc(t/(2^p * pi) / pi) = torch.sinc(t/(2^p * pi^2))
        # Actually, numpy's sinc is sin(pi*x)/(pi*x), so we need to adjust
        # numpy.sinc(x) = sin(pi*x)/(pi*x)
        # So numpy.sinc(x/pi) = sin(x)/(x)
        # Therefore: numpy.sinc(t/(p * pi)) = sin(t/p)/(t/p)

        # Create the argument for torch.sinc
        # torch.sinc(x) computes sin(pi*x)/(pi*x)
        # We want sin(t/p)/(t/p) = numpy.sinc(t/(p * pi))
        # Let x = t/(p * pi), then torch.sinc(x) = sin(pi*x)/(pi*x) = sin(t/p)/(t/p)
        x = t_expanded / (p_expanded * torch.pi)

        # Compute product along the last dimension
        out = torch.prod(torch.sinc(x), dim=1)
        return out

    @classmethod
    def up(cls, x: torch.Tensor, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{up}{(x)}

        :param x: real scalar or array
        :param nsum: integer scalar, nsum=100 by default
        :param nprod: integer scalar, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if nprod < 1:
            raise Exception('nprod must be greater than 0')
        if nsum < 1:
            raise Exception('nsum must be greater than 0')

        # Ensure x is at least 1D tensor
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_up(torch.pi * idx, nprod)

        # Compute cosine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(pi * x * idx)
        cos_terms = torch.cos(torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension
        out = 0.5 + torch.sum(coeff * cos_terms, dim=1)

        # Apply condition: 0 where |x| > 1
        return torch.where(torch.abs(x) <= 1., out, torch.tensor(0., device=device, dtype=dtype))

    @classmethod
    def up_deriv(cls, x: torch.Tensor, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{up}{(x)} 1st derivation

        :param x: real scalar or array
        :param nsum: integer scalar, nsum=100 by default
        :param nprod: integer scalar, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if nprod < 1:
            raise Exception('nprod must be greater than 0')
        if nsum < 1:
            raise Exception('nsum must be greater than 0')

        # Ensure x is at least 1D tensor
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_up(torch.pi * idx, nprod)

        # Compute sine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(pi * x * idx)
        sin_terms = torch.sin(torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition: 0 where |x| > 1
        return torch.where(torch.abs(x) <= 1., out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def ft_upm(cls, t: torch.Tensor, m: int = 1, nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{up}_m{(x)}

        :param t: real scalar or array
        :param m: integer scalar, m=1 by default
        m>=1 for appropriate computation
        :param nprod: integer scalar, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        dtype = t.dtype

        # Create powers of 2*m
        p = torch.pow(2 * m, torch.linspace(1, nprod, nprod, device=device, dtype=dtype))

        # Prepare tensors for broadcasting
        t_expanded = t.unsqueeze(1)  # Shape: (n, 1)
        p_expanded = p.unsqueeze(0)  # Shape: (1, nprod)

        # Compute numerator: sinc(m*t/(p * pi))^2
        # numpy.sinc(x/pi) = sin(x)/(x)
        numerator_arg = m * t_expanded / (p_expanded * torch.pi)
        numerator = torch.sinc(numerator_arg) ** 2

        # Compute denominator: sinc(t/(p * pi))
        denominator_arg = t_expanded / (p_expanded * torch.pi)
        denominator = torch.sinc(denominator_arg)

        # Compute product along the last dimension
        return torch.prod(numerator / denominator, dim=1)


    @classmethod
    def upm(cls, x: torch.Tensor, m: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{up}_m{(x)}

        :param x: real scalar or array
        :param m: integer scalar, m=1 by default,
        m>=1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default,
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_upm(torch.pi * idx, m, nprod)

        # Compute cosine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(pi * x * idx)
        cos_terms = torch.cos(torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension
        out = 0.5 + torch.sum(coeff * cos_terms, dim=1)

        # Apply condition: 0 where |x| > 1
        return torch.where(torch.abs(x) <= 1., out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def upm_deriv(cls, x: torch.Tensor, m: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{up}_m{(x)} 1st derivation

        :param x: real scalar or array
        :param m: integer scalar, m=1 by default,
        m>=1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_upm(torch.pi * idx, m, nprod)

        # Compute sine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(pi * x * idx)
        sin_terms = torch.sin(torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition: 0 where |x| > 1
        return torch.where(torch.abs(x) <= 1., out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def ft_ha(cls, t: torch.Tensor, a: float = 2., nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{h}_a{(x)}

        :param t: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param nprod: integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        dtype = t.dtype

        # Create powers of a
        p = torch.pow(a, torch.linspace(1, nprod, nprod, device=device, dtype=dtype))

        # Prepare tensors for broadcasting
        t_expanded = t.unsqueeze(1)  # Shape: (n, 1)
        p_expanded = p.unsqueeze(0)  # Shape: (1, nprod)

        # Compute sinc(t/(p * pi))
        arg = t_expanded / (p_expanded * torch.pi)

        # Compute product along the last dimension
        return torch.prod(torch.sinc(arg), dim=1)


    @classmethod
    def ha(cls, x: torch.Tensor, a: float = 2., nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{h}_a{(x)}

        :param x: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default,
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        mlt = a - 1

        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Apply condition on x
        x_clamped = torch.where(torch.abs(x) <= 1. / mlt, x, torch.tensor(1. / mlt, device=device, dtype=dtype))

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_ha(mlt * torch.pi * idx, a, nprod)

        # Compute cosine terms
        x_expanded = x_clamped.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(pi * mlt * x * idx)
        cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension
        return mlt * (0.5 + torch.sum(coeff * cos_terms, dim=1))


    @classmethod
    def ha_deriv(cls, x: torch.Tensor, a: float = 2., nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{h}_a{(x)} 1st derivation

        :param x: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        mlt = a - 1

        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_ha(mlt * torch.pi * idx, a, nprod)

        # Compute sine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(pi * mlt * x * idx)
        sin_terms = torch.sin(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def ft_xin(cls, t: torch.Tensor, n: int = 1, nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{xi}_n{(x)}

        :param t: real scalar or array
        :param n: integer scalar, n=1 by default,
        n>=1 for appropriate computation
        :param nprod: integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        dtype = t.dtype

        # Create powers of (n + 1)
        p = torch.pow(n + 1, torch.linspace(1, nprod, nprod, device=device, dtype=dtype))

        # Prepare tensors for broadcasting
        t_expanded = t.unsqueeze(1)  # Shape: (n, 1)
        p_expanded = p.unsqueeze(0)  # Shape: (1, nprod)

        # Compute sinc(t/(p * pi))^n
        arg = t_expanded / (p_expanded * torch.pi)
        tmp = torch.sinc(arg) ** n

        # Compute product along the last dimension
        return torch.prod(tmp, dim=1)


    @classmethod
    def xin(cls, x: torch.Tensor, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{xi}_n{(x)}

        :param x: real scalar or array
        :param n: integer scalar, n=1 by default,
        n>=1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default,
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_xin(torch.pi * idx, n, nprod)

        # Compute cosine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(pi * x * idx)
        cos_terms = torch.cos(torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension
        out = 0.5 + torch.sum(coeff * cos_terms, dim=1)

        # Apply condition: 0 where |x| > 1
        return torch.where(torch.abs(x) <= 1., out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def xin_deriv(cls, x: torch.Tensor, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{xi}_n{(x)} 1st derivation

        :param x: real scalar or array
        :param n: integer scalar, n=1 by default,
        n>=1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_xin(torch.pi * idx, n, nprod)

        # Compute sine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(pi * x * idx)
        sin_terms = torch.sin(torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition: 0 where |x| > 1
        return torch.where(torch.abs(x) <= 1., out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def ft_fupn(cls, t: torch.Tensor, n: int = 1, nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{fup}_n{(x)}

        :param t: real scalar or array
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nprod: integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        dtype = t.dtype

        # Create powers of 2
        p = torch.pow(2., torch.linspace(1, nprod, nprod, device=device, dtype=dtype))

        # Prepare tensors for broadcasting
        t_expanded = t.unsqueeze(1)  # Shape: (n, 1)
        p_expanded = p.unsqueeze(0)  # Shape: (1, nprod)

        # Compute first multiplier: sinc(0.5 * t / pi)^n
        arg1 = 0.5 * t / torch.pi
        mult01 = torch.sinc(arg1) ** n

        # Compute second multiplier: product of sinc(t/(p * pi))
        arg2 = t_expanded / (p_expanded * torch.pi)
        mult02 = torch.prod(torch.sinc(arg2), dim=1)

        return mult01 * mult02


    @classmethod
    def fupn(cls, x: torch.Tensor, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{fup}_n{(x)}

        :param x: real scalar or array
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default,
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        mlt = 2. / (n + 2)

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_fupn(mlt * torch.pi * idx, n, nprod)

        # Compute cosine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(mlt * pi * x * idx)
        cos_terms = torch.cos(mlt * torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension
        out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=1))

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def fupn_deriv(cls, x: torch.Tensor, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{fup}_n{(x)} 1st derivation

        :param x: real scalar or array
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        mlt = 2. / (n + 2)

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_fupn(mlt * torch.pi * idx, n, nprod)

        # Compute sine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(mlt * pi * x * idx)
        sin_terms = torch.sin(mlt * torch.pi * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def ft_chan(cls, t: torch.Tensor, a: float = 2., n: int = 1, nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{ch}_{a,n}{(x)}

        :param t: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=1 for appropriate computation
        :param nprod: integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        dtype = t.dtype

        # Create powers of a
        p = torch.pow(a, torch.linspace(1, nprod, nprod, device=device, dtype=dtype))

        # Prepare tensors for broadcasting
        t_expanded = t.unsqueeze(1)  # Shape: (n, 1)
        p_expanded = p.unsqueeze(0)  # Shape: (1, nprod)

        # Compute sinc(t/(p * pi))^n
        arg = t_expanded / (p_expanded * torch.pi)
        tmp = torch.sinc(arg) ** n

        # Compute product along the last dimension
        return torch.prod(tmp, dim=1)


    @classmethod
    def chan(cls, x: torch.Tensor, a: float = 2., n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{ch}_{a,n}{(x)}

        :param x: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default,
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        mlt = (a - 1) / n

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_chan(mlt * torch.pi * idx, a, n, nprod)

        # Compute cosine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(pi * mlt * x * idx)
        cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension
        out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=1))

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def chan_deriv(cls, x: torch.Tensor, a: float = 2., n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{ch}_{a,n}{(x)} 1st derivation

        :param x: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=1 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        mlt = (a - 1) / n

        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Apply condition on x
        x_clamped = torch.where(torch.abs(x) <= 1. / mlt, x, torch.tensor(1. / mlt, device=device, dtype=dtype))

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_chan(mlt * torch.pi * idx, a, n, nprod)

        # Compute sine terms
        x_expanded = x_clamped.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(pi * mlt * x * idx)
        sin_terms = torch.sin(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def ft_fipan(cls, t: torch.Tensor, a: float = 2., n: int = 1, nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{fip}_{a,n}{(x)}

        :param t: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nprod: integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        dtype = t.dtype

        # Create powers of a
        p = torch.pow(a, torch.linspace(1, nprod, nprod, device=device, dtype=dtype))

        # Prepare tensors for broadcasting
        t_expanded = t.unsqueeze(1)  # Shape: (n, 1)
        p_expanded = p.unsqueeze(0)  # Shape: (1, nprod)

        # Compute first multiplier: sinc(0.5 * t / pi)^n
        arg1 = 0.5 * t / torch.pi
        mult01 = torch.sinc(arg1) ** n

        # Compute second multiplier: product of sinc(t/(p * pi))
        arg2 = t_expanded / (p_expanded * torch.pi)
        mult02 = torch.prod(torch.sinc(arg2), dim=1)

        return mult01 * mult02


    @classmethod
    def fipan(cls, x: torch.Tensor, a: float = 2., n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{fip}_{a,n}{(x)}

        :param x: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default,
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        l = n + 2. / (a - 1.)
        mlt = 2. / l

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_fipan(mlt * torch.pi * idx, a, n, nprod)

        # Compute cosine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(pi * mlt * x * idx)
        cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension
        out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=1))

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def fipan_deriv(cls, x: torch.Tensor, a: float = 2., n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{fip}_{a,n}{(x)} 1st derivation

        :param x: real scalar or array
        :param a: real scalar, a=2 by default,
        a>1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        l = n + 2. / (a - 1.)
        mlt = 2. / l

        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Apply condition on x
        x_clamped = torch.where(torch.abs(x) <= 1. / mlt, x, torch.tensor(1. / mlt, device=device, dtype=dtype))

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_fipan(mlt * torch.pi * idx, a, n, nprod)

        # Compute sine terms
        x_expanded = x_clamped.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(pi * mlt * x * idx)
        sin_terms = torch.sin(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))


    @classmethod
    def ft_fpmn(cls, t: torch.Tensor, m: int = 2, n: int = 1, nprod: int = 10) -> torch.Tensor:
        r""" Fourier transform of atomic function \mathrm{fp}_{m,n}{(x)}

        :param t: real scalar or array
        :param m: integer scalar, m=2 by default,
        m>=1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nprod: integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        dtype = t.dtype

        # Compute first multiplier: sinc(0.5 * t / pi)^n
        arg1 = 0.5 * t / torch.pi
        mult01 = torch.sinc(arg1) ** n

        # Compute second multiplier: ft_upm(t, m, nprod)
        mult02 = cls.ft_upm(t, m, nprod)

        return mult01 * mult02


    @classmethod
    def fpmn(cls, x: torch.Tensor, m: int = 2, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{fp}_{m,n}{(x)}

        :param x: real scalar or array
        :param m: integer scalar, m=2 by default,
        m>=1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default,
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        mlt = 2. / (n + 2)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        # Apply condition on x
        x_clamped = torch.where(torch.abs(x) <= 1. / mlt, x, torch.tensor(1. / mlt, device=device, dtype=dtype))

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_fpmn(mlt * torch.pi * idx, m, n, nprod)

        # Compute cosine terms
        x_expanded = x_clamped.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute cos(pi * mlt * x * idx)
        cos_terms = torch.cos(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension
        out = mlt * (0.5 + torch.sum(coeff * cos_terms, dim=1))

        # Apply condition
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))

    @classmethod
    def fpmn_deriv(cls, x: torch.Tensor, m: int = 2, n: int = 1, nsum: int = 100, nprod: int = 10) -> torch.Tensor:
        r""" Fourier series of atomic function \mathrm{fp}_{m,n}{(x)} 1st derivation

        :param x: real scalar or array
        :param m: integer scalar, m=2 by default,
        m>=1 for appropriate computation
        :param n: integer scalar, n=1 by default,
        n>=0 for appropriate computation
        :param nsum:  nsum is an integer, nsum=100 by default
        :param nprod:  nprod is an integer, nprod=10 by default,
        nprod>=5 for appropriate computation
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        mlt = 2. / (n + 2)

        # Create indices
        idx = torch.linspace(1, nsum, nsum, device=device, dtype=dtype)

        # Compute coefficients
        coeff = cls.ft_fpmn(mlt * torch.pi * idx, m, n, nprod)

        # Compute sine terms
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1)
        idx_expanded = idx.unsqueeze(0)  # Shape: (1, nsum)

        # Compute sin(pi * mlt * x * idx)
        sin_terms = torch.sin(torch.pi * mlt * x_expanded * idx_expanded)

        # Sum over idx dimension with coefficients
        out = -torch.pi * torch.sum(idx * coeff * sin_terms, dim=1)

        # Apply condition: 0 where |x| > 1/mlt
        return torch.where(torch.abs(x) <= 1. / mlt, out, torch.tensor(0., device=device, dtype=dtype))

