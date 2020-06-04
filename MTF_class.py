import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.ndimage import sobel, median_filter
from numpy import pi, cos, arctan, sinc, exp
from scipy.fftpack import fft, fftshift


def linear(x, a, b):
    return a * x + b


def gauss(x, a, mu, s):
    return a * exp(-(x - mu) ** 2 / (2 * s ** 2))


class EdgeImageAnalysis:
    def __init__(self, img, p):
        super(EdgeImageAnalysis).__init__()
        self.img = img
        self.p = p

    def denoise(self):
        return median_filter(self.img, 25)

    def lsf(self):
        img_n = self.denoise()
        return sobel(img_n, 1)  # edge detector based on 1st derivative

    def getangle(self):
        # The center of the profile must be estimated in each row - lsf is fitted by gaussian function,
        # the position of the line is then mean value of ths function - parameter from curve fitting
        mu = []
        lsf = self.lsf()
        x = np.arange(0, lsf.shape[1])
        for i in range(lsf.shape[0]):
            y = lsf[i][:]
            best_val, cov = curve_fit(gauss, x, y, p0=[1, 300, 10])
            mu.append(int(best_val[1]))

        y = np.arange(0, lsf.shape[0])
        best_val, cov = curve_fit(linear, mu, y, p0=[0.5, 400])
        n = abs(int(best_val[0]))
        print("Num of lines:", n)
        alpha = abs(arctan(1 / n))
        print("Edge angle:", np.degrees(alpha))
        return n, alpha

    def oversample(self):
        lsf = self.lsf()
        n = self.getangle()[0]
        lsf_over = []
        for i in range(int(lsf.shape[0] / n)):
            lsf_over_part = lsf[i * n:(i + 1) * n, :]
            lsf_over_part = list(
                lsf_over_part.reshape((lsf_over_part.shape[0] * lsf_over_part.shape[1]), order='F'))  # 'F' - by columns
            # to get rid of the esf shift in each group:
            lsf_over_part = lsf_over_part[len(lsf_over_part) - i * n:] + lsf_over_part[:len(
                lsf_over_part) - i * n]
            lsf_over.append(lsf_over_part)

        # Averaging the ESF is preferred than averaging the MTFs  (individual MTFs may be subject to bias errors
        # caused by noise)
        lsf_avg = np.mean(lsf_over, axis=0)[i * n:]
        return lsf_avg

    def mtf(self, stop):
        lsf = self.oversample()
        m = len(lsf)  # total length of the LSF
        mtf = abs(fftshift(fft(lsf)))  # fftshift - zero-centered fft
        mtf = mtf / mtf[int(len(mtf) / 2)]

        # Correction for the transfer function of the derivation filter = sinc function
        n, alpha = self.getangle()
        delta_x = self.p / n
        u = np.asarray([k / (2 * delta_x * m) for k in range(int(m / 2))])  # frequencies
        # mtf_der = [sinc(pi * fr * p / n) for fr in u[1:]]
        # mtf_der = [1] + mtf_der
        # mtf = mtf[int(len(mtf) / 2):int(len(mtf) / 2) + stop] / mtf_der[:stop]
        mtf = mtf[int(len(mtf) / 2):int(len(mtf) / 2) + stop]
        return mtf, u, alpha


# MTF theoretic
class MTFMath:

    def __init__(self, u, p):
        super(MTFMath).__init__()
        self.u = u
        self.p = p

    def mtfBox(self):
        f = sinc(pi * u * p)
        return f

    def mtfPSF(self, sigma):
        f = exp(-2 * (pi * u * sigma) ** 2)
        return f

    def mtfBoxPSF(self, sigma):
        f = self.mtfBox() * self.mtfPSF(sigma)
        return f


# ================================================================================
# load file with edge
fName = 'Poisson_0_5_edge_1_5blurred35.txt'
img = np.loadtxt(fName)

p = 0.025  # mm  - bude se menit dle receptoru obrazu - vycist z dicom hlavicky
# Call class
eia = EdgeImageAnalysis(img, p)
# LSF
lsf = eia.oversample()
r = int(lsf.shape[0] / 2)
x = p * np.linspace(-r, r, 2 * r)
n, alpha = eia.getangle()
alpha = np.degrees(alpha)
plt.plot(x, lsf, 'k--', label=r'with noise,N={n:d},$\alpha$={a:.2f}'.format(n=n, a=alpha))

# ----second file

fName = 'edge_1_5blurred35_512.txt'
img = np.loadtxt(fName)
# Call class
eia = EdgeImageAnalysis(img, p)
# LSF
lsf = eia.oversample()
r = int(lsf.shape[0] / 2)
x = p * np.linspace(-r, r, 2 * r)
n, alpha = eia.getangle()
alpha = np.degrees(alpha)
plt.plot(x, lsf, 'b--', label=r'without noise,N={n:d},$\alpha$={a:.2f}'.format(n=n, a=alpha))

plt.title('Average LSF')
plt.legend()
plt.grid()
plt.savefig('lsf_comparison.png')
plt.show()

exit()
#
# ====MTF=====
stop = 20
mtf, u, alpha = eia.mtf(stop=stop)

fig, ax = plt.subplots()
ax.plot(u[:stop] / cos(alpha), mtf[:stop], 'k--', label='MTF calc')
# ax.plot(mtf[:stop], 'k--',label='MTF calc')
plt.xlim(0, 0.6)
# plt.ylim(10 ** (-6), 1)
ax.set_ylabel("$mtf$ [-]")
ax.set_xlabel(r"$u = 1/(2\cdot M\cdot\Delta x)\cdot 1/cos\alpha$ [lp/mm]")
ax.grid(True, which='major')
ax.set_yscale('log')

# ax.xaxis.set_minor_locator(MultipleLocator(0.02))
# plt.savefig('mtf_corr2.png')
# MTFteor = np.loadtxt('MTF_teor.txt')
# ax.plot(u[:stop], MTFteor[:stop], label='MTFteor')
# ax.plot(MTFteor[:stop], label='MTFteor')
# plt.savefig('mtf_shortRange.png')

u = np.asarray(u)
mtf = MTFMath(u, p)

sigma = 35 * p
mtf_t = mtf.mtfBoxPSF(sigma)
ax.plot(u[:stop], mtf_t[:stop], label='MTF_th')
plt.legend()
plt.title(
    r'Edge angle: $\alpha = 1,5^\circ$, Gaussian blur: $\sigma = 35$ px,' + '\n' + 'Poisson noise: $\lambda=0.5$, matrix: 512 x 512')
# plt.savefig('mtf_from_noised_blurred_Img_Sobel.png')
plt.show()
