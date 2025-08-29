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
    

class TwoLens1S:
    def __init__(self, t0, tE, rho, u0_list, q, s, alpha, t_lc=None, a1=0.0):
        self.t0 = float(t0)
        self.tE = float(tE)
        self.rho = float(rho)
        self.u0_list = list(u0_list)
        self.q = float(q)
        self.s = float(s)
        self.alpha = float(alpha)
        self.theta = np.radians(self.alpha)

        self.tau = np.linspace(-4, 4, 1000)
        self.t    = self.t0 + self.tau * self.tE
        self.tau_hr = np.linspace(-4, 4, 1000)
        self.t_hr   = self.t0 + self.tau_hr * self.tE

        if t_lc is not None:
            self.t_lc = np.asarray(t_lc, dtype=float)
            self.tau_lc = (self.t_lc - self.t0) / self.tE
        else:
            self.tau_lc = np.linspace(-4, 4, 200)
            self.t_lc   = self.t0 + self.tau_lc * self.tE

        # Limb darkening
        self.a1 = float(a1) if a1 is not None else 0.0  
        self.n_rings = 24

        
        self.VBM = VBMicrolensing.VBMicrolensing()
        self.VBM.RelTol = 1e-3
        self.VBM.Tol    = 1e-3
        self.VBM.astrometry = True
        
        try:
            self.VBM.SetLDprofile(self.VBM.LDlinear)
        except AttributeError:
            self.VBM.SetLDprofile(self.VBM.LDprofiles.LDlinear)
        self.VBM.a1 = self.a1

        
        self.colors = [plt.colormaps['BuPu'](i) for i in np.linspace(1.0, 0.4, len(self.u0_list))]
        self.systems = self._prepare_systems()

    @staticmethod
    def _poly_area(x, y):
        x = np.asarray(x); y = np.asarray(y)
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _uniform_cumulants(self, y1, y2, r_eff):
        """Return (ΣA, Σ(Ax), Σ(Ay)) for a UNIFORM disk of radius r_eff centered at (y1,y2)."""
        SA = SAx = SAy = 0.0
        for img in self.VBM.ImageContours(self.s, self.q, y1, y2, r_eff):
            x = np.asarray(img[0]); y = np.asarray(img[1])
            A = self._poly_area(x, y)
            if A > 0:
                cx, cy = np.mean(x), np.mean(y)
                SA  += A
                SAx += A * cx
                SAy += A * cy
        return SA, SAx, SAy

    def _centroid_at(self, y1, y2):
        """Images centroid (cx, cy) at one source position. Uses LD rings if a1>0."""
        # Uniform, fast path
        if self.a1 <= 0:
            SA, SAx, SAy = self._uniform_cumulants(y1, y2, self.rho)
            return (SAx/SA, SAy/SA) if SA > 0 else (np.nan, np.nan)

        radii = np.linspace(0.0, self.rho, self.n_rings + 1)
        cum = [self._uniform_cumulants(y1, y2, r) for r in radii]

        Nx = Ny = D = 0.0
        for k in range(1, len(radii)):
            r_in, r_out = radii[k-1], radii[k]
            SAo, SAxo, SAyo = cum[k]
            SAi, SAxi, SAyi = cum[k-1]
            dSA  = SAo  - SAi
            if dSA == 0:
                continue
            dSAx = SAxo - SAxi
            dSAy = SAyo - SAyi

            rmid = 0.5 * (r_in + r_out)
            mu   = np.sqrt(max(0.0, 1.0 - (rmid / self.rho)**2))
            I    = 1.0 - self.a1 * (1.0 - mu)   # linear LD intensity

            # Correct: weight only by intensity I (no ΔS factor)
            Nx += I * dSAx
            Ny += I * dSAy
            D  += I * dSA

        return (Nx/D, Ny/D) if D > 0 else (np.nan, np.nan)
    # -----------------------------

    def _prepare_systems(self):
        systems = []

        for u0, color in zip(self.u0_list, self.colors):
            # source trajectories
            x_src    = self.tau    * np.cos(self.theta) - u0 * np.sin(self.theta)
            y_src    = self.tau    * np.sin(self.theta) + u0 * np.cos(self.theta)
            x_src_hr = self.tau_hr * np.cos(self.theta) - u0 * np.sin(self.theta)
            y_src_hr = self.tau_hr * np.sin(self.theta) + u0 * np.cos(self.theta)
            x_src_lc = self.tau_lc * np.cos(self.theta) - u0 * np.sin(self.theta)
            y_src_lc = self.tau_lc * np.sin(self.theta) + u0 * np.cos(self.theta)

            # LD-aware centroids
            cent_x    = [self._centroid_at(x, y)[0] for x, y in zip(x_src,    y_src)]
            cent_y    = [self._centroid_at(x, y)[1] for x, y in zip(x_src,    y_src)]
            cent_x_hr = [self._centroid_at(x, y)[0] for x, y in zip(x_src_hr, y_src_hr)]
            cent_y_hr = [self._centroid_at(x, y)[1] for x, y in zip(x_src_hr, y_src_hr)]
            cent_x_lc = [self._centroid_at(x, y)[0] for x, y in zip(x_src_lc, y_src_lc)]
            cent_y_lc = [self._centroid_at(x, y)[1] for x, y in zip(x_src_lc, y_src_lc)]

            # Photometry at light-curve times (uses same a1 inside VBM)
            mag, *_ = self.VBM.BinaryLightCurve(
                [math.log(self.s), math.log(self.q), u0, self.theta,
                 math.log(self.rho), math.log(self.tE), self.t0],
                self.t_lc
            )

            systems.append({
                "u0": u0,
                "color": color,
                "mag": np.asarray(mag),
                "x_src": x_src, "y_src": y_src,
                "cent_x": np.asarray(cent_x), "cent_y": np.asarray(cent_y),
                "x_src_hr": x_src_hr, "y_src_hr": y_src_hr,
                "cent_x_hr": np.asarray(cent_x_hr), "cent_y_hr": np.asarray(cent_y_hr),
                "x_src_lc": x_src_lc, "y_src_lc": y_src_lc,
                "cent_x_lc": np.asarray(cent_x_lc), "cent_y_lc": np.asarray(cent_y_lc),
            })

        return systems
    
    def plot_caustic_critical_curves(self):
        caustics = self.VBM.Caustics(self.s, self.q)
        criticalcurves = self.VBM.Criticalcurves(self.s, self.q)

        lens_handle = Line2D([0], [0], marker='o', color='k', linestyle='None', label='Lens', markersize=6)
        caustic_handle = Line2D([0], [0], color='r', lw=1.2, label='Caustic')
        crit_curve_handle = Line2D([0], [0], color='k', linestyle='--', lw=0.8, label='Critical Curve')
        q_handle = Line2D([0], [0], color='k', linestyle='None', label=fr"$q$ = {self.q}")
        s_handle = Line2D([0], [0], color='k', linestyle='None', label=fr"$s$ = {self.s}")
                          

        plt.figure(figsize=(6, 6))

        for cau in caustics:
            plt.plot(cau[0], cau[1], 'r', lw=1.2)
        for crit in criticalcurves:
            plt.plot(crit[0], crit[1], 'k--', lw=0.8)

        x1 = -self.s * self.q / (1 + self.q)
        x2 = self.s / (1 + self.q)
        plt.plot([x1, x2], [0, 0], 'ko')

        for system in self.systems:
            plt.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.6)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel(r"$\theta_x$ ($\theta_E$)")
        plt.ylabel(r"$\theta_y$ ($\theta_E$)")
        plt.title("2L1S Lensing Event")
        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.legend(handles=[lens_handle, caustic_handle, crit_curve_handle, q_handle, s_handle], loc='upper right', prop={'size': 8})
        plt.tight_layout()
        plt.show()

    def plot_light_curve(self):
        plt.figure(figsize=(6, 4))
        
        for system in self.systems:
            plt.plot(self.tau_lc, system['mag'], color=system['color'], label=fr"$u_0$ = {system['u0']}")
        
        plt.xlabel(r"Time ($\tau$)")
        plt.ylabel("Magnification")
        plt.title("Light Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()    

    def animate(self):
        caustics = self.VBM.Caustics(self.s, self.q)
        criticalcurves = self.VBM.Criticalcurves(self.s, self.q)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(wspace=0.4)

        lens_handle = Line2D([0], [0], marker='o', color='k', linestyle='None', label='Lens', markersize=6)
        caustic_handle = Line2D([0], [0], color='r', lw=1.2, label='Caustic')
        crit_curve_handle = Line2D([0], [0], color='k', linestyle='--', lw=0.8, label='Critical Curve')
        q_handle = Line2D([0], [0], color='k', linestyle='None', label=fr"$q$ = {self.q}")
        s_handle = Line2D([0], [0], color='k', linestyle='None', label=fr"$s$ = {self.s}")

        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_xlabel(r"$\theta_x$ ($\theta_E$)")
        ax1.set_ylabel(r"$\theta_y$ ($\theta_E$)")
        ax1.set_title("2L1S Microlensing Event")
        ax1.set_aspect("equal")
        ax1.grid(True)

        for cau in caustics:
            ax1.plot(cau[0], cau[1], 'r', lw=1.2)
        for crit in criticalcurves:
            ax1.plot(crit[0], crit[1], 'k--', lw=0.8)

        x1 = -self.s * self.q / (1 + self.q)
        x2 = self.s / (1 + self.q)

        ax1.plot(x1, 0, 'ko', markersize=4)
        ax1.plot(x2, 0, 'ko', markersize=4)

        ax1.legend(handles=[lens_handle, caustic_handle, crit_curve_handle, q_handle, s_handle], loc='upper right', prop={'size': 8})

        source_dots, centroid_dots = [], []
        for system in self.systems:
            ax1.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.4)
            src_dot, = ax1.plot([], [], '*', color=system['color'], markersize=10)
            #cen_dot, = ax1.plot([], [], 'x', color=system['color'], markersize=6, label=f"$u_0$ = {system['u0']}")
            source_dots.append(src_dot)
            #centroid_dots.append(cen_dot)
        ax1.legend(handles=[lens_handle, caustic_handle, crit_curve_handle, q_handle, s_handle], loc='upper right', prop={'size': 8})


        ax2.set_xlim(self.tau_lc[0], self.tau_lc[-1])
        all_mag = np.concatenate([s['mag'] for s in self.systems])
        ax2.set_ylim(float(all_mag.min())*0.95, float(all_mag.max())*1.05)
        ax2.set_xlabel(r"Time ($\tau$)")
        ax2.set_ylabel("Magnification")
        ax2.set_title("Light Curve")

        tracer_dots = []
        for system in self.systems:
            ax2.plot(self.tau_lc, system['mag'], color=system['color'], label=f"$u_0$ = {system['u0']}")
            dot, = ax2.plot([], [], 'o', color=system['color'], markersize=6)
            tracer_dots.append(dot)
        ax2.legend()

        # Precompute frame count to match LC sampling
        n_frames = len(self.tau_lc)

        image_dots = []
        for system in self.systems:
            system_dots = []
            for _ in range(5):
                img_dot, = ax1.plot([], [], '.', color=system['color'], alpha=0.6, markersize=4)
                system_dots.append(img_dot)
            image_dots.append(system_dots)

        def update(i):
            artists = []
            # advance along source track on the SAME index as LC
            for j, system in enumerate(self.systems):
                # find source position at the i-th LC time
                x_s = system['x_src_lc'][i]
                y_s = system['y_src_lc'][i]

                source_dots[j].set_data([x_s], [y_s])
                tracer_dots[j].set_data([self.tau_lc[i]], [system['mag'][i]])
                artists.extend([source_dots[j], tracer_dots[j]])

                images = self.VBM.ImageContours(self.s, self.q, x_s, y_s, self.rho)
                for k, img_dot in enumerate(image_dots[j]):
                    if k < len(images):
                        img_dot.set_data(images[k][0], images[k][1])
                        img_dot.set_alpha(0.6)
                    else:
                        img_dot.set_data([], [])
                        img_dot.set_alpha(0)
                    artists.append(img_dot)

            return artists

        ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
        plt.close(fig)
        self.last_animation = ani
        return HTML(ani.to_jshtml())
    
    def plot_centroid_trajectory(self):
        plt.figure(figsize=(6, 6))
        for system in self.systems:
            delta_x = system['cent_x_hr'] - system['x_src_hr']
            delta_y = system['cent_y_hr'] - system['y_src_hr']
            plt.plot(delta_x, delta_y, color=system['color'], label=fr"$u_0$ = {system['u0']}")
        #plt.xlim(-0.4, .8)    
        #plt.ylim(-0.4, 0.5)
        plt.xlabel(r"$\delta \Theta_1$")
        plt.ylabel(r"$\delta \Theta_2$")
        plt.gca().set_aspect("equal")
        plt.title("Centroid Trajectory")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_centroid_shift(self):
        plt.figure(figsize=(6, 4))
        for system in self.systems:
            delta_x = system['cent_x_hr'] - system['x_src_hr']
            delta_y = system['cent_y_hr'] - system['y_src_hr']
            delta_theta = np.sqrt(delta_x**2 + delta_y**2)
            plt.plot(self.tau_hr, delta_theta, color=system['color'], label=fr"$u_0$ = {system['u0']}")
        
        plt.xlabel(r"Time ($\tau$)")
        plt.ylabel(r"$|\delta \vec{\Theta}|$")
        plt.title(r"Centroid Shift over Time ($\tau$)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_all(self):

        fig = plt.figure(figsize=(9, 9), constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig)

        # --- Top Left: Lensing Animation ---
        ax1 = fig.add_subplot(gs[0, 0])
        caustics = self.VBM.Caustics(self.s, self.q)
        criticalcurves = self.VBM.Criticalcurves(self.s, self.q)

        lens_handle = Line2D([0], [0], marker='o', color='k', linestyle='None', label='Lens', markersize=6)
        caustic_handle = Line2D([0], [0], color='r', lw=1.2, label='Caustic')
        crit_curve_handle = Line2D([0], [0], color='k', linestyle='--', lw=0.8, label='Critical Curve')
        q_handle = Line2D([0], [0], color='k', linestyle='None', label=fr"$q$ = {self.q}")
        s_handle = Line2D([0], [0], color='k', linestyle='None', label=fr"$s$ = {self.s}")

        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_aspect("equal")
        ax1.grid(True)
        ax1.set_title("2L1S Lensing Event")
        for cau in caustics:
            ax1.plot(cau[0], cau[1], 'r', lw=1.2)
        for crit in criticalcurves:
            ax1.plot(crit[0], crit[1], 'k--', lw=0.8)
        x1 = -self.s * self.q / (1 + self.q)
        x2 = self.s / (1 + self.q)
        ax1.plot([x1, x2], [0, 0], 'ko')
        ax1.set_ylabel(r"Y ($\theta_E$)")
        ax1.set_xlabel(r"X ($\theta_E$)")
        ax1.legend(handles=[lens_handle, caustic_handle, crit_curve_handle, q_handle, s_handle], loc='upper right', prop={'size': 8})

        source_dots, tracer_dots, image_dots = [], [], []
        for system in self.systems:
            ax1.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.4)
            src_dot, = ax1.plot([], [], '*', color=system['color'], markersize=10)
            source_dots.append(src_dot)

            dots = []
            for _ in range(5):
                dot, = ax1.plot([], [], '.', color=system['color'], alpha=0.6, markersize=4)
                dots.append(dot)
            image_dots.append(dots)

        # --- Top Right: Light Curve ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(self.tau[0], self.tau[-1])
        all_mag = np.concatenate([s['mag'] for s in self.systems])
        ax2.set_ylim(min(all_mag)*0.95, max(all_mag)*1.05)
        ax2.set_ylabel("Magnification")
        ax2.set_title("Light Curve")
        ax2.set_xlabel(r"Time ($\tau$)")

        for system in self.systems:
            ax2.plot(self.tau, system['mag'], color=system['color'], label=fr"$u_0$ = {system['u0']}")
            dot, = ax2.plot([], [], 'o', color=system['color'], markersize=6)
            tracer_dots.append(dot)
            ax2.legend(prop={'size': 8})

        # --- Bottom Left: Centroid Trajectory ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_box_aspect(1)

        for system in self.systems:
            dx = system['cent_x_hr'] - system['x_src_hr']
            dy = system['cent_y_hr'] - system['y_src_hr']
            ax3.plot(dx, dy, color=system['color'], label=fr"$\rho$ = {self.rho}")
        #ax3.set_xlim(-1, 1)
        #ax3.set_ylim(-1, 1)
        ax3.set_title("Centroid Shift Trajectory")
        ax3.set_xlabel(r"$\delta \Theta_1$")
        ax3.set_ylabel(r"$\delta \Theta_2$")
        ax3.grid(True)
        ax3.set_aspect("equal")
        ax3.legend(prop={'size': 8})

        # --- Bottom Right: Centroid Shift vs Tau ---
        ax4 = fig.add_subplot(gs[1, 1])
        for system in self.systems:
            dx = system['cent_x_hr'] - system['x_src_hr']
            dy = system['cent_y_hr'] - system['y_src_hr']
            dtheta = np.sqrt(dx**2 + dy**2)
            ax4.plot(self.tau_hr, dtheta, color=system['color'])
        ax4.set_xlabel(r"Time ($\tau$)")
        ax4.set_ylabel(r"$|\delta \vec{\Theta}|$")
        ax4.set_title(r"Centroid Shift over Time ($\tau$)")
        ax4.grid(True)
    
        #fig.subplots_adjust(hspace=0.2, wspace=0.2)    

        # --- Animate function ---
        def update(i):
            artists = []
            for j, system in enumerate(self.systems):
                x_s = system['x_src'][i]
                y_s = system['y_src'][i]
                source_dots[j].set_data([x_s], [y_s])
                tracer_dots[j].set_data([self.tau[i]], [system['mag'][i]])
                artists.extend([source_dots[j], tracer_dots[j]])

                images = self.VBM.ImageContours(self.s, self.q, x_s, y_s, self.rho)
                for k, dot in enumerate(image_dots[j]):
                    if k < len(images):
                        dot.set_data(images[k][0], images[k][1])
                        dot.set_alpha(0.6)
                    else:
                        dot.set_data([], [])
                        dot.set_alpha(0)
                    artists.append(dot)
            return artists

        ani = animation.FuncAnimation(fig, update, frames=len(self.t), interval=50, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())
