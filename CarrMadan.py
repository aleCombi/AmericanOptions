import numpy as np

class CarrMadan:
    def __init__(self, boundary, alpha, step):
        self.boundary = boundary
        self.alpha = alpha
        self.step = step
    
    def CallTransform(self, r, T, phi, v):
        numerator = np.exp(- r * T) * phi(v - (self.alpha + 1) * 1j)
        denominator = self.alpha ** 2 + self.alpha - v ** 2 + 1j * (2 * self.alpha + 1) * v
        return numerator / denominator
    
    def CallPrice(self, option, model):
        damp = np.exp(- self.alpha * np.log(option.strike)) / (2 * np.pi)
        v = np.linspace(-self.boundary, self.boundary, int(self.boundary / self.step))
        phi = lambda u: model.CharacteristicFunction(u, option.maturity)
        integrand = self.CallTransform(model.r, option.maturity, phi, v) * np.exp(- 1j * v * np.log(option.strike))
        integral = np.trapz(integrand, v)
        return np.real(damp * integral)