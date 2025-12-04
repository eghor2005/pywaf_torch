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


# Функция для получения всех окон
def get_all_windows(npt: int, mode: str = 'max', device: torch.device = None) -> dict:
    """
    Получить все оконные функции.

    Args:
        npt: Количество точек
        mode: Режим нормализации
        device: Устройство для вычислений

    Returns:
        Словарь с оконными функциями
    """
    windows = {}

    # Базовые функции
    windows['up'] = up(npt, mode, device)
    windows['upm'] = upm(npt, mode=mode, device=device)
    windows['ha'] = ha(npt, mode=mode, device=device)
    windows['xin'] = xin(npt, mode=mode, device=device)
    windows['fupn'] = fupn(npt, mode=mode, device=device)
    windows['chan'] = chan(npt, mode=mode, device=device)
    windows['fipan'] = fipan(npt, mode=mode, device=device)
    windows['fpmn'] = fpmn(npt, mode=mode, device=device)

    return windows


# Пример использования
if __name__ == "__main__":
    # Тестирование на CPU
    print("Testing on CPU:")
    npt = 101

    # Получить все окна
    windows_cpu = get_all_windows(npt, mode='max')

    for name, window in windows_cpu.items():
        print(f"{name}: shape={window.shape}, dtype={window.dtype}, "
              f"max={window.max().item():.4f}, min={window.min().item():.4f}")

    # Тестирование отдельных функций
    print("\nTesting individual functions:")

    # up функция
    up_window = up(npt, mode='area')
    print(f"up window (area normalized): sum={up_window.sum().item():.4f}")

    # fupn функция с различными параметрами
    for n_val in [0, 1, 2, 3]:
        fupn_window = fupn(npt, n=n_val, mode='max')
        print(f"fupn(n={n_val}): max={fupn_window.max().item():.4f}")

    # Тестирование на GPU (если доступно)
    if torch.cuda.is_available():
        print("\nTesting on GPU:")
        device_gpu = torch.device('cuda')

        up_gpu = up(npt, mode='max', device=device_gpu)
        print(f"up on GPU: shape={up_gpu.shape}, device={up_gpu.device}")

        # Проверка согласованности CPU/GPU
        up_cpu = up(npt, mode='max')
        diff = torch.max(torch.abs(up_cpu - up_gpu.cpu())).item()
        print(f"CPU/GPU difference: {diff:.6e}")