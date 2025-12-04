import torch
import numpy as np
import math
try:
    from .kernel import AtomicKernel
except ImportError:
    # Если запускаем файл напрямую
    from kernel import AtomicKernel


class Wavelet(AtomicKernel):

    def __init__(self, waf: str, coef: int = 2):
        """
        Инициализация вейвлета.

        Args:
            waf: Тип вейвлета ('up', 'upm', 'meyer4', 'meyer6')
            coef: Коэффициент для upm вейвлета
        """
        waf_list = ['up', 'upm', 'meyer4', 'meyer6']
        if waf not in waf_list:
            raise ValueError('waf_torch must be "up", "upm", "meyer4" or "meyer6"')
        self.coef = coef
        self.waf = waf

    def chi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисление функции chi.

        Args:
            x: Входной тензор

        Returns:
            Тензор значений chi
        """
        m = self.coef

        def sum_func(f, x, a, b, m):
            """ Вспомогательная функция суммирования """
            return f(a * x - b, m) + f(a * x, m) + f(a * x + b, m)

        # Проверяем, является ли x скаляром
        if x.dim() == 0:
            x = x.unsqueeze(0)

        device = x.device
        dtype = x.dtype

        if self.waf == 'up':
            factor = 3 / (2 * math.pi)
            return (self.up(factor * x + 1) +
                    self.up(factor * x) +
                    self.up(factor * x - 1))

        elif self.waf == 'upm':
            return sum_func(self.upm, x, 3 / (2 * math.pi), 1, m)

        elif self.waf == 'meyer4':
            def nu(x: torch.Tensor) -> torch.Tensor:
                """ Функция nu для Meyer4 """
                return x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)

            # Векторизованное вычисление
            abs_x = torch.abs(x)
            factor = 3 / (2 * math.pi)

            # Создаем маску для условий
            condition1 = (abs_x >= 2 * math.pi / 3) & (abs_x <= 4 * math.pi / 3)
            condition2 = (abs_x <= 2 * math.pi / 3) & (abs_x >= -2 * math.pi / 3)

            # Вычисляем значения
            result = torch.zeros_like(x)

            # Условие 1
            result[condition1] = torch.cos(math.pi / 2 * nu(factor * abs_x[condition1] - 1))

            # Условие 2
            result[condition2] = 1.0

            # За пределами условий остается 0
            return result ** 2

        elif self.waf == 'meyer6':
            def nu(x: torch.Tensor) -> torch.Tensor:
                """ Функция nu для Meyer6 """
                return (x ** 6 * (462 - 1980 * x + 3465 * x ** 2 -
                                  3080 * x ** 3 + 1386 * x ** 4 - 252 * x ** 5))

            # Векторизованное вычисление
            abs_x = torch.abs(x)
            factor = 3 / (2 * math.pi)

            # Создаем маску для условий
            condition1 = (abs_x >= 2 * math.pi / 3) & (abs_x <= 4 * math.pi / 3)
            condition2 = (abs_x <= 2 * math.pi / 3) & (abs_x >= -2 * math.pi / 3)

            # Вычисляем значения
            result = torch.zeros_like(x)

            # Условие 1
            result[condition1] = torch.cos(math.pi / 2 * nu(factor * abs_x[condition1] - 1))

            # Условие 2
            result[condition2] = 1.0

            # За пределами условий остается 0
            return result ** 2

        else:
            raise ValueError('waf_torch must be "up", "upm", "meyer4" or "meyer6"')

    def phi_f(self, x: torch.Tensor) -> torch.Tensor:
        """
        Функция phi в частотной области.

        Args:
            x: Входной тензор

        Returns:
            Тензор значений phi_f
        """
        return torch.sqrt(torch.abs(self.chi(x)))

    def H(self, x: torch.Tensor) -> torch.Tensor:
        """
        Функция H.

        Args:
            x: Входной тензор

        Returns:
            Тензор значений H
        """
        n = torch.tensor([-1, 0, 1], device=x.device, dtype=x.dtype)
        sum_result = torch.zeros_like(x)

        for item in n:
            sum_result += self.phi_f(2 * (x - 2 * math.pi * item))

        return sum_result

    def dec_lo(self, n: torch.Tensor) -> torch.Tensor:
        """
        Декомпозиционный низкочастотный фильтр.

        Args:
            n: Тензор индексов

        Returns:
            Коэффициенты фильтра
        """
        device = n.device
        dtype = n.dtype

        # Создаем частотную ось
        w = torch.linspace(-math.pi, math.pi, 10000, device=device, dtype=dtype)

        # Вычисляем H(w)
        H_w = self.H(w)

        # Вычисляем коэффициенты Фурье
        h = []
        for i in n:
            integral = torch.trapezoid(H_w * torch.exp(1j * w * i), w)
            h.append(torch.real(integral) / (math.sqrt(2) * math.pi))

        return torch.stack(h)

    def rec_lo(self, dec_lo: torch.Tensor) -> torch.Tensor:
        """
        Реконструкционный низкочастотный фильтр.

        Args:
            dec_lo: Декомпозиционный низкочастотный фильтр

        Returns:
            Реконструкционный низкочастотный фильтр
        """
        # Комплексное сопряжение и обратный порядок
        return torch.conj(torch.flip(dec_lo, dims=[0]))

    def dec_hi(self, dec_lo: torch.Tensor, n: torch.Tensor, N: int) -> torch.Tensor:
        """
        Декомпозиционный высокочастотный фильтр.

        Args:
            dec_lo: Декомпозиционный низкочастотный фильтр
            n: Тензор индексов
            N: Длина фильтра

        Returns:
            Декомпозиционный высокочастотный фильтр
        """
        coef = torch.pow(-torch.ones(N, device=dec_lo.device, dtype=dec_lo.dtype), n)
        return coef * torch.flip(dec_lo, dims=[0])

    def rec_hi(self, dec_lo: torch.Tensor, n: torch.Tensor, N: int) -> torch.Tensor:
        """
        Реконструкционный высокочастотный фильтр.

        Args:
            dec_lo: Декомпозиционный низкочастотный фильтр
            n: Тензор индексов
            N: Длина фильтра

        Returns:
            Реконструкционный высокочастотный фильтр
        """
        coef = torch.pow(-torch.ones(N, device=dec_lo.device, dtype=dec_lo.dtype), n + 1)
        return coef * dec_lo

    def filter(self, N: int) -> dict:
        """
        Генерация фильтров для вейвлет-преобразования.

        Args:
            N: Длина фильтра

        Returns:
            Словарь с фильтрами
        """
        device = torch.device('cpu')  # Можно изменить на нужное устройство
        n = torch.arange(-N // 2, N // 2, device=device, dtype=torch.float32)

        dec_lo_mass = self.dec_lo(n)
        dec_hi_mass = self.dec_hi(dec_lo_mass, n, N)
        rec_lo_mass = self.rec_lo(dec_lo_mass)
        rec_hi_mass = self.rec_hi(dec_lo_mass, n, N)

        return {
            'dec_lo': dec_lo_mass,
            'dec_hi': dec_hi_mass,
            'rec_lo': rec_lo_mass,
            'rec_hi': rec_hi_mass
        }

    def psi_f(self, w: torch.Tensor) -> torch.Tensor:
        """
        Вейвлет в частотной области.

        Args:
            w: Частотная ось

        Returns:
            Тензор значений psi_f
        """
        return (torch.exp(1j * w / 2) *
                (self.phi_f(w - 2 * math.pi) + self.phi_f(w + 2 * math.pi)) *
                self.phi_f(w / 2))

    def psi(self, x: torch.Tensor, N: int = 10000) -> torch.Tensor:
        """
        Вейвлет во временной области.

        Args:
            x: Временная ось
            N: Количество точек для интегрирования

        Returns:
            Тензор значений psi
        """
        device = x.device
        dtype = x.dtype

        # Создаем частотную ось
        w = torch.linspace(2 * math.pi / 3, 8 * math.pi / 3, N, device=device, dtype=dtype)

        # Вычисляем C(w)
        C = self.phi_f(w / 2) * self.phi_f(w - 2 * math.pi)

        # Вычисляем psi(x) для каждого x
        psi_values = []
        for item in x:
            integral = torch.trapezoid(C * torch.cos(w * (item + 0.5)), w)
            psi_values.append(integral)

        psi_tensor = torch.stack(psi_values)
        return psi_tensor / math.pi

    def phi(self, x: torch.Tensor, N: int = 10000) -> torch.Tensor:
        """
        Масштабирующая функция во временной области.

        Args:
            x: Временная ось
            N: Количество точек для интегрирования

        Returns:
            Тензор значений phi
        """
        device = x.device
        dtype = x.dtype

        # Создаем частотную ось
        w = torch.linspace(0, 4 * math.pi / 3, N, device=device, dtype=dtype)

        # Вычисляем C(w)
        C = self.phi_f(w)

        # Вычисляем phi(x) для каждого x
        phi_values = []
        for item in x:
            integral = torch.trapezoid(C * torch.cos(w * item), w)
            phi_values.append(integral)

        phi_tensor = torch.stack(phi_values)
        return phi_tensor / math.pi


