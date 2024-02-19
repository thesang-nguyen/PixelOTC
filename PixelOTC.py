import numpy as np
import numba as nb
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import ot  # POT package


@nb.njit
def cost_matrix(m, n):
    '''
    Compute the squared distances cost matrix for (flattened) 2D OT problem on 
    m x n images.
    '''
    C = np.zeros((m*n, m*n))
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    C[i*n+j, k*n+l] = (i-k)**2 + (j-l)**2
    return C


@nb.njit
def image_geodesic(m, n, pi, t=0.5):
    '''
    Compute geodesic (image) between two images given OT plan `pi` at time `t`.
    
    Parameters
    ----------  
    m : int
        Number of rows in image.
    n : int
        Number of columns in image.
    pi : ndarray
        OT plan between two (flattened) images.
    t : float
        Time parameter between 0 and 1.
    '''
    assert 0 <= t <= 1, 't must be between 0 and 1'

    mut = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for k in range(m):
                for l in range(n):
                    x = int((1-t)*i + t*k + 0.5)  # round up .5
                    y = int((1-t)*j + t*l + 0.5)
                    mut[x, y] += pi[i*n+j, k*n+l]
    return mut


def otc_curve(pi, C, dists):
    '''
    Compute OTC curve for given OT plan `pi` and cost matrix `C` at distances
    `dists`.

    Parameters
    ----------
    pi : ndarray
        OT plan between two (flattened) images.
    C : ndarray
        Squared distances cost matrix for (flattened) 2D OT problem on 
        m x n images.
    dists : ndarray or list of floats
        Distances at which to compute OTC curve.
    '''
    ind = np.argsort(C.flatten())
    C = C.flatten()[ind]
    pi = pi.flatten()[ind]
    I = np.searchsorted(C, dists)
    curve = np.cumsum(pi)[I]
    return curve
    # curve = []
    # for dist in dists:
    #     I, J = np.where(C < dist*dist)  # C represents *squared* distances
    #     curve.append(pi[I, J].sum())
    # return curve


def scale256(im):
    '''Scale image to values between 0 and 255.'''
    return np.uint8(im/im.max() * 255)


def gray2rgb(im, coeff=(1, 0, 0), scale=True):
    '''Convert grayscale image to RGB image.'''
    r, g, b = coeff
    im = im[:, :, None]
    im = np.concatenate((r*im, g*im, b*im), axis=2)
    im = scale256(im) if scale else im
    return im


class PixelOTC:
    '''
    Class for computing OT between two images and plotting geodesics between
    them at different times.
    '''
    def __init__(self, imA, imB, unbalanced=False, lam=1.):
        '''
        Parameters
        ----------
        imA : ndarray
            First image.
        imB : ndarray
            Second image.
        unbalanced : bool
            Whether to solve unbalanced OT problem.
        lam : float
            Regularization parameter for unbalanced OT problem.
        '''
        self.imA = imA
        self.imB = imB
        self.unbalanced = unbalanced
        self.lam = lam

        self.m, self.n = imA.shape
        self.C = cost_matrix(self.m, self.n)
        self.mu = imA.flatten()
        self.nu = imB.flatten()

        self.pi = None  # optimal transport plan, calculated by solve()
        self.optimal_cost = None

    def solve(self):
        '''
        Solve OT problem between two images.

        Returns
        -------
        pi : ndarray
            Optimal transport plan between two (flattened) images.
        optimal_cost : float
            Optimal transport cost.
        '''
        mu = self.mu
        nu = self.nu
        C = self.C.astype(dtype=np.float64)

        if self.unbalanced:
            mu = np.append(mu, nu.sum())
            nu = np.append(nu, mu[:-1].sum())
            C = np.pad(
                C, ((0, 1), (0, 1)),
                mode='constant', constant_values=self.lam/2
            )
            C[-1, -1] = 0.
            np.testing.assert_approx_equal(mu.sum(), nu.sum(), err_msg='Something did not sum up, check code and inputs')
            mu = mu/mu.sum() * np.sum(nu)
        else:
            mu = mu/mu.sum()
            nu = nu/nu.sum()

        self.pi = ot.emd(mu, nu, C)
        self.optimal_cost = np.sum(self.pi * C)

        if self.unbalanced:
            self.pi = self.pi[:-1, :-1]
        return self.pi, self.optimal_cost
    
    def plot(self, ts=None, rgb=True):
        '''
        Plot images and geodesics between them at times `ts`.

        Parameters
        ----------
        ts : ndarray or list of floats
            Times at which to plot geodesics.
        rgb : bool
            Whether to plot images in RGB or grayscale.
        '''
        if ts is None:
            ts = np.linspace(0, 1, 5)[1:-1]

        if rgb:
            imA = gray2rgb(self.imA, coeff=(1, 0, 0), scale=True)
            imB = gray2rgb(self.imB, coeff=(0, 1, 0), scale=True)
        else:
            imA = self.imA
            imB = self.imB

        fig, axs = plt.subplots(ncols=len(ts)+2, figsize=((len(ts)+2)*3, 3))

        axs[0].set_title('imA')
        axs[0].imshow(imA, origin='lower')
        axs[0].axis('off')
        axs[-1].set_title('imB')
        axs[-1].imshow(imB, origin='lower')
        axs[-1].axis('off')

        for i, t in enumerate(ts):
            mut = image_geodesic(self.m, self.n, self.pi, t=t)
            if rgb:
                mut = gray2rgb(mut, coeff=(1-t, t, 0), scale=True)

            axs[i+1].set_title(f'$\mu_t$ for t={t}')
            axs[i+1].imshow(mut, origin='lower')
            axs[i+1].axis('off')

        fig.tight_layout()
        plt.show()

    def otc_curve(self, dists):
        '''
        OTC curve for distances `dists`.

        Parameters
        ----------
        dists : ndarray or list of floats
            Distances at which to compute OTC curve.
        '''
        curve = otc_curve(self.pi, self.C, dists)
        return curve

    def plot_otc_curve(self, dists):
        '''
        Plot OTC curve for distances `dists`.

        Parameters
        ----------
        dists : ndarray or list of floats
            Distances at which to compute OTC curve.
        '''
        curve = otc_curve(self.pi, self.C, dists)
        plt.figure(figsize=(6, 4))
        plt.plot(dists, curve)
        plt.xlabel('t')
        plt.ylabel('OTC(t)')
        plt.title('OTC curve')
        plt.grid()
        plt.show()

    def make_gif(
        self, ts=None, show_imA=False, show_imB=False, rgb=True,
        filename='geodesic.gif', duration=1,
    ):
        '''
        Make GIF of geodesics between two images at times `ts`.

        Parameters
        ----------
        ts : ndarray or list of floats
            Times at which to plot geodesics.
        show_imA : bool
            Whether to show first image at beginning of GIF.
        show_imB : bool
            Whether to show second image at end of GIF.
        rgb : bool
            Whether to plot images in RGB or grayscale.
        filename : str
            Filename of GIF.
        duration : float
        '''
        if ts is None:
            ts = np.linspace(0, 1, 101)

        frames = []
        for t in ts:
            mut = image_geodesic(self.m, self.n, self.pi, t=t)
            if rgb:
                mut = gray2rgb(mut, coeff=(1-t, t, 0), scale=False)
            mut = scale256(mut)
            mut = Image.fromarray(mut)  # optional: mode='RGB' if rgb else 'L'
            frames.append(mut)

        if show_imA:
            imA = self.imA
            if rgb:
                imA = gray2rgb(imA, coeff=(1, 0, 0), scale=False)
            imA = scale256(imA)
            imA = Image.fromarray(imA)  # optional: mode='RGB' if rgb else 'L'
            frames.insert(0, imA)
        if show_imB:
            imB = self.imB
            if rgb:
                imB = gray2rgb(imB, coeff=(0, 1, 0), scale=False)
            imB = scale256(imB)
            imB = Image.fromarray(imB)  # optional: mode='RGB' if rgb else 'L'
            frames.append(imB)

        frame_one = frames[0]
        frame_one.save(
            filename, format="GIF", append_images=frames,
            save_all=True, duration=duration, loop=0
        )


    def slider_plot(self, show_imA=True, show_imB=True, rgb=True):
        '''
        Plot images and geodesics between them at times `ts` with slider.

        Parameters
        ----------
        show_imA : bool
            Whether to show first image at beginning of GIF.
        show_imB : bool
            Whether to show second image at end of GIF.
        rgb : bool
            Whether to plot images in RGB or grayscale.

        Returns
        -------
        slider : matplotlib.widgets.Slider
            Slider to control time parameter `t`.
        button : matplotlib.widgets.Button
            Button to reset slider to initial value.

        Notes
        -----
        For the slider and button to remain responsive in a jupyter notebook 
        you must maintain a reference to them, i.e. store the variables 
        returned `matplotlib.widgets` objects `slider` and `button`.
        '''
        ncols = 1 + bool(show_imA) + bool(show_imB)
        fig, axs = plt.subplots(ncols=ncols, figsize=(3*ncols, 4))

        if show_imA:
            imA = self.imA
            if rgb:
                imA = gray2rgb(imA, coeff=(1, 0, 0), scale=True)
            axs[0].set_title('imA')
            axs[0].imshow(imA, origin='lower')
            axs[0].axis('off')
        if show_imB:
            imB = self.imB
            if rgb:
                imB = gray2rgb(imB, coeff=(0, 1, 0), scale=True)
            axs[-1].set_title('imB')
            axs[-1].imshow(imB, origin='lower')
            axs[-1].axis('off')

        t = 0.5  # initial value
        mut = image_geodesic(self.m, self.n, self.pi, t=t)
        if rgb:
            mut = gray2rgb(mut, coeff=(1-t, t, 0), scale=True)
        im = axs[1].imshow(mut, origin='lower')
        axs[1].set_title('$\mu_t$')
        axs[1].axis('off')

        # adjust main plot to make room for slider
        fig.subplots_adjust(bottom=0.25)

        # horizontal slider to control the frequency
        t_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(
            ax=t_ax,
            label='t',
            valmin=0,
            valmax=1,
            valinit=t,
        )

        # function to be called anytime slider's value changes
        def update(val):
            mut = image_geodesic(self.m, self.n, self.pi, t=val)
            if rgb:
                mut = gray2rgb(mut, coeff=(1-val, val, 0), scale=True)
            im.set_data(mut)
            fig.canvas.draw_idle()

        # register update function with each change of slider value
        slider.on_changed(update)

        # button to reset slider to initial value
        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            # NOTE: event parameter is required for button.on_clicked to work
            slider.reset()
        button.on_clicked(reset)

        plt.show()

        return slider, button
