import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML, display
import VBMicrolensing
import TripleLensing
import math
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from TestML import get_crit_caus, getphis_v3, get_allimgs_with_mu, testing

class ThreeLens1S:
    def __init__(self, t0, tE, rho, u0_list, q2, q3, s2, s3, alpha_deg, psi_deg,
                 rs, secnum, basenum, num_points):

        self.t0 = t0
        self.tE = tE
        self.rho = rho
        self.u0_list = u0_list
        self.q2 = q2
        self.q3 = q3
        self.s2 = s2
        self.s3 = s3
        self.alpha_deg = alpha_deg
        self.psi_deg = psi_deg
        self.rs = rs
        self.secnum = secnum
        self.basenum = basenum
        self.num_points = num_points

        self.alpha_rad = np.radians(alpha_deg)
        self.psi_rad = np.radians(psi_deg)
        self.tau = np.linspace(-2, 2, num_points)
        self.t = self.t0 + self.tau * self.tE

        self.TRIL = TripleLensing.TripleLensing()
        self.colors = [plt.colormaps['BuPu'](i) for i in np.linspace(1.0, 0.4, len(u0_list))]
        self.systems = self._prepare_systems()

        import VBMicrolensing
        self.VBM = VBMicrolensing.VBMicrolensing()
        self.VBM.RelTol = 1e-3
        self.VBM.Tol = 1e-3
        self.VBM.astrometry = True
        self.VBM.SetMethod(self.VBM.Method.Nopoly)

    def get_lens_geometry(self):
        m1 = 1 / (1 + self.q2 + self.q3)
        m2 = self.q2 * m1
        m3 = self.q3 * m1
        mlens = [m1, m2, m3]
        x1, y1 = 0.0, 0.0
        x2, y2 = self.s2, 0.0
        x3 = self.s3 * np.cos(self.psi_rad)
        y3 = self.s3 * np.sin(self.psi_rad)
        zlens = [x1, y1, x2, y2, x3, y3]
        return mlens, zlens

    def _prepare_systems(self):
        systems = []
        mlens, zlens = self.get_lens_geometry()
        z = [[zlens[0], zlens[1]], [zlens[2], zlens[3]], [zlens[4], zlens[5]]]
        critical, caustics = get_crit_caus(mlens, z, len(mlens))
        caus_x = np.array([pt[0] for pt in caustics])
        caus_y = np.array([pt[1] for pt in caustics])

        for idx, u0 in enumerate(self.u0_list):
            y1s = u0 * np.sin(self.alpha_rad) + self.tau * np.cos(self.alpha_rad)
            y2s = u0 * np.cos(self.alpha_rad) - self.tau * np.sin(self.alpha_rad)

            cent_x, cent_y = [], []
            for i in range(self.num_points):
                Phis = getphis_v3(mlens, z, y1s[i], y2s[i], self.rs, 2000, caus_x, caus_y,
                                  secnum=self.secnum, basenum=self.basenum, scale=10)[0]
                imgXS, imgYS, imgMUs, *_ = get_allimgs_with_mu(
                    mlens, z, y1s[i], y2s[i], self.rs, len(mlens), Phis)

                if len(imgMUs) == 0 or sum(imgMUs) == 0:
                    cent_x.append(np.nan)
                    cent_y.append(np.nan)
                else:
                    cx = np.sum(np.array(imgMUs) * np.array(imgXS)) / np.sum(imgMUs)
                    cy = np.sum(np.array(imgMUs) * np.array(imgYS)) / np.sum(imgMUs)
                    cent_x.append(cx)
                    cent_y.append(cy)

            systems.append({
                'u0': u0,
                'color': self.colors[idx],
                'y1s': y1s,
                'y2s': y2s,
                'cent_x': np.array(cent_x),
                'cent_y': np.array(cent_y),
                'mlens': mlens,
                'zlens': zlens
            })

        return systems
    
    def plot_caustics_and_critical(self):
        param = [
            np.log(self.s2), np.log(self.q2), self.u0_list[0], self.alpha_deg,
            np.log(self.rho), np.log(self.tE), self.t0,
            np.log(self.s3), np.log(self.q3), self.psi_rad
        ]
        _ = self.VBM.TripleLightCurve(param, self.t)  # sets internal lens geometry

        caustics = self.VBM.Multicaustics()
        criticalcurves = self.VBM.Multicriticalcurves()

        plt.figure(figsize=(6, 6))
        for c in caustics:
            plt.plot(c[0], c[1], 'r', lw=1.2)
        for crit in criticalcurves:
            plt.plot(crit[0], crit[1], 'k--', lw=0.8)

        lens_pos = self.get_lens_geometry()[1]
        for i in range(0, 6, 2):
            plt.plot(lens_pos[i], lens_pos[i+1], 'ko')

        plt.title("Caustics and Critical Curves (VBM)")
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.show()

    def plot_light_curve(self):
        plt.figure(figsize=(6, 4))
        for u0, color in zip(self.u0_list, self.colors):
            param = [
                np.log(self.s2), np.log(self.q2), u0, self.alpha_deg,
                np.log(self.rho), np.log(self.tE), self.t0,
                np.log(self.s3), np.log(self.q3), self.psi_rad
            ]
            mag, *_ = self.VBM.TripleLightCurve(param, self.t)
            plt.plot(self.tau, mag, color=color, label=fr"$u_0$ = {u0}")
        plt.xlabel(r"$\tau$")
        plt.ylabel("Magnification")
        plt.title("Triple Lens Light Curve (VBM)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()    

    def plot_centroid_trajectory(self):
        plt.figure(figsize=(6, 6))
        for system in self.systems:
            dx = system['cent_x'] - system['y1s']
            dy = system['cent_y'] - system['y2s']
            plt.plot(dx, dy, color=system['color'], label=fr"$u_0$ = {system['u0']}")
        plt.xlabel(r"$\theta_x$ ($\theta_E$)")
        plt.ylabel(r"$\theta_y$ ($\theta_E$)")
        plt.title("Centroid Shift Trajectories")
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_shift_vs_time(self):
        plt.figure(figsize=(8, 5))
        for system in self.systems:
            dx = system['cent_x'] - system['y1s']
            dy = system['cent_y'] - system['y2s']
            dtheta = np.sqrt(dx**2 + dy**2)
            plt.plot(self.tau, dtheta, label=fr"$u_0$ = {system['u0']}", color=system['color'])
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$|\delta \vec{\Theta}|$")
        plt.title("Centroid Shift vs Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def animate(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(i):
            ax.cla()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal")
            ax.set_title("Triple Lens Event Animation")
            for system in self.systems:
                testing(ax, system['mlens'], system['zlens'], system['y1s'][i], system['y2s'][i], self.rs,
                        secnum=self.secnum, basenum=self.basenum,
                        full_trajectory=(system['y1s'], system['y2s']), cl=system['color'])
            return ax,

        ani = FuncAnimation(fig, update, frames=self.num_points, blit=False)
        plt.close(fig)
        return HTML(ani.to_jshtml())
    
    def animate_combined(self):
        # First, prepare the caustics and critical curves once using VBM
        param = [
            np.log(self.s2), np.log(self.q2), self.u0_list[0], self.alpha_deg,
            np.log(self.rho), np.log(self.tE), self.t0,
            np.log(self.s3), np.log(self.q3), self.psi_rad
        ]
        _ = self.VBM.TripleLightCurve(param, self.t)  # set lens geometry
        caustics = self.VBM.Multicaustics()
        criticalcurves = self.VBM.Multicriticalcurves()

        fig, ax = plt.subplots(figsize=(6, 6))

        def update(i):
            ax.cla()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal")
            ax.set_title("Triple Lens Microlensing Event")

            # Plot VBM caustics and criticals
            for c in caustics:
                ax.plot(c[0], c[1], 'r', lw=1.2)
            for crit in criticalcurves:
                ax.plot(crit[0], crit[1], 'k--', lw=0.8)

            for system in self.systems:
                # Plot the full source trajectory
                ax.plot(system['y1s'], system['y2s'], '--', color=system['color'], alpha=0.5)

                # Plot source position at frame i
                ax.plot(system['y1s'][i], system['y2s'][i], 'o', color=system['color'])

                # Plot the lens positions
                zlens = system['zlens']
                ax.plot(zlens[0], zlens[1], 'ko')
                ax.plot(zlens[2], zlens[3], 'ko')
                ax.plot(zlens[4], zlens[5], 'ko')

                # Optional: Plot image positions (using TripleLensing)
                imgXS, imgYS, imgMUs, *_ = get_allimgs_with_mu(
                    system['mlens'], [[zlens[0], zlens[1]], [zlens[2], zlens[3]], [zlens[4], zlens[5]]],
                    system['y1s'][i], system['y2s'][i], self.rs, len(system['mlens']),
                    getphis_v3(system['mlens'], [[zlens[0], zlens[1]], [zlens[2], zlens[3]], [zlens[4], zlens[5]]],
                            system['y1s'][i], system['y2s'][i], self.rs, 2000,
                            np.array([pt[0] for pt in caustics[0]]),  # Just using 1st loop
                            np.array([pt[1] for pt in caustics[0]]),
                            secnum=self.secnum, basenum=self.basenum, scale=10)[0]
                )

                if len(imgXS) > 0:
                    ax.scatter(imgXS, imgYS, s=30, edgecolors='black', facecolors='none', label='Images')

            return ax,

        ani = FuncAnimation(fig, update, frames=self.num_points, blit=False)
        plt.close(fig)
        return HTML(ani.to_jshtml())