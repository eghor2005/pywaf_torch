import torch
try:
    from .kernel import AtomicKernel as atom
except ImportError:
    # Если запускаем файл напрямую
    from kernel import AtomicKernel as atom

window_list = ['up', 'upm', 'ha',
               'xin', 'fupn', 'chan', 'fipan', 'fpmn']


def normalize(window: torch.Tensor, mode: str, npt: int) -> torch.Tensor:
    """
    Нормализация окна.

    Args:
        window: Тензор значений окна
        mode: Режим нормализации ('max' или 'area')
        npt: Количество точек

    Returns:
        Нормализованный тензор
    """
    if mode == 'max':
        return window / torch.max(window)
    elif mode == 'area':
        dx = 2. / (npt - 1)
        area = torch.trapezoid(window, dx=dx)
        return window / area
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def up(npt: int, mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция up.

    Args:
        npt: Количество точек
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    t = torch.linspace(-1, 1, npt, device=device, dtype=torch.float32)
    window = atom.up(t)
    return normalize(window, mode, npt)


def upm(npt: int, m: int = 2, mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция upm.

    Args:
        npt: Количество точек
        m: Параметр m
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    t = torch.linspace(-1, 1, npt, device=device, dtype=torch.float32)
    window = atom.upm(t, m)
    return normalize(window, mode, npt)


def ha(npt: int, a: float = 2., mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция ha.

    Args:
        npt: Количество точек
        a: Параметр a (a > 1)
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    mlt = a - 1.
    t = torch.linspace(-1. / mlt, 1. / mlt, npt, device=device, dtype=torch.float32)
    window = atom.ha(t, a)
    return normalize(window, mode, npt)


def xin(npt: int, n: int = 2, mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция xin.

    Args:
        npt: Количество точек
        n: Параметр n (n >= 1)
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    t = torch.linspace(-1, 1, npt, device=device, dtype=torch.float32)
    window = atom.xin(t, n)
    return normalize(window, mode, npt)


def fupn(npt: int, n: int = 0, mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция fupn.

    Args:
        npt: Количество точек
        n: Параметр n (n >= 0)
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    mlt = (n + 2) / 2
    t = torch.linspace(-mlt, mlt, npt, device=device, dtype=torch.float32)
    window = atom.fupn(t, n)
    return normalize(window, mode, npt)


def chan(npt: int, a: float = 2., n: int = 2, mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция chan.

    Args:
        npt: Количество точек
        a: Параметр a (a > 1)
        n: Параметр n (n >= 1)
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    mlt = n / (a - 1.)
    t = torch.linspace(-mlt, mlt, npt, device=device, dtype=torch.float32)
    window = atom.chan(t, a, n)
    return normalize(window, mode, npt)


def fipan(npt: int, a: float = 2., n: int = 2, mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция fipan.

    Args:
        npt: Количество точек
        a: Параметр a (a > 1)
        n: Параметр n (n >= 0)
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    l_val = n + 2. / (a - 1.)
    t = torch.linspace(-l_val / 2., l_val / 2., npt, device=device, dtype=torch.float32)
    window = atom.fipan(t, a, n)
    return normalize(window, mode, npt)


def fpmn(npt: int, m: int = 2, n: int = 2, mode: str = 'max', device: torch.device = None) -> torch.Tensor:
    """
    Оконная функция fpmn.

    Args:
        npt: Количество точек
        m: Параметр m (m >= 1)
        n: Параметр n (n >= 0)
        mode: Режим нормализации ('max' или 'area')
        device: Устройство для вычислений (CPU/GPU)

    Returns:
        Тензор значений окна
    """
    if device is None:
        device = torch.device('cpu')

    mlt = (n + 2) / 2
    t = torch.linspace(-mlt, mlt, npt, device=device, dtype=torch.float32)
    window = atom.fpmn(t, m, n)
    return normalize(window, mode, npt)

