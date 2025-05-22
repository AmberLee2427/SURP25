import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML, display
import VBMicrolensing
import math
from matplotlib.lines import Line2D

class OneL1S:
    def __init__(self, t0, tE, rho, u0_list):

        self.t0 = t0
        self.tE = tE
        self.t = np.linspace(t0-tE, t0+tE, 50)
        self.rho = rho
        self.u0_list = u0_list
        self.tau = (t - t0) / tE

        self.VBM = VBMicrolensing.VBMicrolensing()
        self.VBM.RelTol = 1e-3
        self.VBM.Tol = 1e-3
        self.VBM.astrometry = True

    def plot_light_curve_on_ax(self, ax):
        cmap_es = plt.colormaps['BuPu']
        colors_es = [cmap_es(i) for i in np.linspace(0.5, 1.0, len(self.u0_list))]
        cmap_ps = plt.colormaps['binary']
        colors_ps = [cmap_ps(i) for i in np.linspace(0.5, 1.0, len(self.u0_list))]

        for idx, u0 in enumerate(self.u0_list):
            color_es = colors_es[idx]
            color_ps = colors_ps[idx]
            u = np.sqrt(u0**2 + self.tau**2)
            pspl_mag = [self.VBM.PSPLMag(ui) for ui in u]
            espl_mag = [self.VBM.ESPLMag2(ui, self.rho) for ui in u]

            ax.plot(self.tau, espl_mag, '-', color=color_es, label=f'ESPL $u_0$ = {u0}')
            ax.plot(self.tau, pspl_mag, '--', color=color_ps, label=f'PSPL $u_0$ = {u0}', alpha=0.7)

        ax.set_xlabel(r"Time ($\tau$)")
        ax.set_ylabel("Magnification")
        ax.set_title("Single-Lens Magnification")
        ax.grid(True)
        ax.legend()

    def plot_centroid_shift_on_ax(self, ax):
        cmap_cs = plt.colormaps['BuPu']
        colors_cs = [cmap_cs(i) for i in np.linspace(0.5, 1.0, len(self.u0_list))]

        for idx, u0 in enumerate(self.u0_list):
            color_cs = colors_cs[idx]
            u = np.sqrt(u0**2 + self.tau**2)
            centroid_shift = [self.VBM.astrox1 - ui for ui in u]
            ax.plot(self.tau, centroid_shift, color=color_cs, label=f'$u_0$ = {u0}')

        ax.set_xlabel(r"Time ($\tau$)")
        ax.set_ylabel(r"Centroid Shift ($\Delta \theta$)")
        ax.set_title("Astrometric Centroid Shift")
        ax.grid(True)
        ax.legend()

    def plot_light_curve(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        self.plot_light_curve_on_ax(ax)
        plt.show()

    def plot_centroid_shift(self):
        fig, ax = plt.subplots(figsize=(8, 4))
        self.plot_centroid_shift_on_ax(ax)
        plt.show()

    def animate(self):
        return self._create_animation(figsize=(6, 6), layout='single')

    def show_all(self):
        return self._create_animation(figsize=(14, 6), layout='grid')

    def _create_animation(self, figsize=(6, 6), layout='single'):
        tau = self.tau
        n = len(self.t)
        colors = [plt.colormaps['BuPu'](i) for i in np.linspace(0.5, 1.0, len(self.u0_list))]

        systems = []
        for u0, color in zip(self.u0_list, colors):
            x_source = tau
            y_source = np.full_like(tau, u0)
            u = np.sqrt(x_source**2 + y_source**2)
            espl_mag = [self.VBM.ESPLMag2(ui, self.rho) for ui in u]
            systems.append({'u0': u0, 'x': x_source, 'y': y_source, 'mag': espl_mag, 'color': color})

        if layout == 'grid':
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(2, 2, width_ratios=[1, 1])
            ax_anim = fig.add_subplot(gs[:, 0])
            ax_light = fig.add_subplot(gs[0, 1])
            ax_centroid = fig.add_subplot(gs[1, 1])
            self.plot_light_curve_on_ax(ax_light)
            self.plot_centroid_shift_on_ax(ax_centroid)
        else:
            fig, ax_anim = plt.subplots(figsize=figsize)

        ax_anim.set_xlim(-2, 2)
        ax_anim.set_ylim(-2, 2)
        ax_anim.set_xlabel(r"X ($\theta_E$)")
        ax_anim.set_ylabel(r"Y ($\theta_E$)")
        ax_anim.set_title("Single-Lens Microlensing Events")
        ax_anim.grid(True)
        ax_anim.set_aspect("equal")
        ax_anim.plot([0], [0], 'ko', label="Lens")
        einstein_ring = plt.Circle((0, 0), 1, color='green', fill=False, linestyle='--', linewidth=1.5)
        ax_anim.add_patch(einstein_ring)

        source_dots, img_dots, trails_1, trails_2, trail_data = [], [], [], [], []

        for system in systems:
            s_dot, = ax_anim.plot([], [], '*', color=system['color'], label=f"$u_0$ = {system['u0']}")
            i_dot = ax_anim.scatter([], [], color=system['color'], s=20)
            t1 = ax_anim.scatter([], [], color=system['color'], alpha=0.3)
            t2 = ax_anim.scatter([], [], color=system['color'], alpha=0.3)
            source_dots.append(s_dot)
            img_dots.append(i_dot)
            trails_1.append(t1)
            trails_2.append(t2)
            trail_data.append({'x1': [], 'y1': [], 's1': [], 'x2': [], 'y2': [], 's2': []})

        ax_anim.legend(loc='lower left')

        def update(frame):
            for i, system in enumerate(systems):
                x_s = system['x'][frame]
                y_s = system['y'][frame]
                u = np.sqrt(x_s**2 + y_s**2)
                theta = np.arctan2(y_s, x_s)
                r_plus = (u + np.sqrt(u**2 + 4)) / 2
                r_minus = (u - np.sqrt(u**2 + 4)) / 2

                x1 = r_plus * np.cos(theta)
                y1 = r_plus * np.sin(theta)
                x2 = r_minus * np.cos(theta)
                y2 = r_minus * np.sin(theta)

                source_dots[i].set_data([x_s], [y_s])
                img_dots[i].set_offsets([[x1, y1], [x2, y2]])
                mag = system['mag'][frame]
                size = 20 * mag
                img_dots[i].set_sizes([size, size])

                trail_data[i]['x1'].append(x1)
                trail_data[i]['y1'].append(y1)
                trail_data[i]['s1'].append(size)
                trail_data[i]['x2'].append(x2)
                trail_data[i]['y2'].append(y2)
                trail_data[i]['s2'].append(size)

                trails_1[i].set_offsets(np.column_stack([trail_data[i]['x1'], trail_data[i]['y1']]))
                trails_1[i].set_sizes(trail_data[i]['s1'])
                trails_2[i].set_offsets(np.column_stack([trail_data[i]['x2'], trail_data[i]['y2']]))
                trails_2[i].set_sizes(trail_data[i]['s2'])

            return source_dots + img_dots + trails_1 + trails_2

        ani = animation.FuncAnimation(fig, update, frames=n, interval=50, blit=True)
        plt.tight_layout()
        return HTML(ani.to_jshtml())
    
class TwoLens1S:
    def __init__(self, t0, tE, rho, u0_list, q, s, alpha):
        self.t0 = t0
        self.tE = tE
        self.rho = rho
        self.u0_list = u0_list
        self.q = q
        self.s = s
        self.alpha = alpha
        self.tau = np.linspace(-4, 4, 200)
        self.t = self.t0 + self.tau * self.tE
        self.theta = np.radians(self.alpha)

        self.VBM = VBMicrolensing.VBMicrolensing()
        self.VBM.RelTol = 1e-3
        self.VBM.Tol = 1e-3
        self.VBM.astrometry = True
        self.colors = [plt.colormaps['BuPu'](i) for i in np.linspace(0.4, 1.0, len(u0_list))]
        self.systems = self._prepare_systems()

    def _prepare_systems(self):
        systems = []
        for u0, color in zip(self.u0_list, self.colors):
            pr = [math.log(self.s), math.log(self.q), u0, self.theta, math.log(self.rho), math.log(self.tE), self.t0]
            mag, cent_x, cent_y = self.VBM.BinaryLightCurve(pr, self.t)
            x_src = self.tau * np.cos(self.theta) - u0 * np.sin(self.theta)
            y_src = self.tau * np.sin(self.theta) + u0 * np.cos(self.theta)

            systems.append({
                'u0': u0,
                'color': color,
                'mag': mag,
                'x_src': x_src,
                'y_src': y_src,
                'cent_x': cent_x,
                'cent_y': cent_y
            })
        return systems

    def animate(self):
        caustics = self.VBM.Caustics(self.s, self.q)
        criticalcurves = self.VBM.Criticalcurves(self.s, self.q)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(wspace=0.4)

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
        ax1.plot([x1, x2], [0, 0], 'ko', label="Lenses")

        source_dots, centroid_dots = [], []
        for system in self.systems:
            ax1.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.4)
            src_dot, = ax1.plot([], [], '*', color=system['color'], markersize=6)
            #cen_dot, = ax1.plot([], [], 'x', color=system['color'], markersize=6, label=f"$u_0$ = {system['u0']}")
            source_dots.append(src_dot)
            #centroid_dots.append(cen_dot)
        ax1.legend(loc='lower right')

        
        ax2.set_xlim(self.tau[0], self.tau[-1])
        all_mag = np.concatenate([s['mag'] for s in self.systems])
        ax2.set_ylim(min(all_mag)*0.95, max(all_mag)*1.05)
        ax2.set_xlabel(r"Time ($\tau$)")
        ax2.set_ylabel("Magnification")
        ax2.set_title("Light Curve")

        tracer_dots = []
        for system in self.systems:
            ax2.plot(self.tau, system['mag'], color=system['color'], label=f"$u_0$ = {system['u0']}")
            dot, = ax2.plot([], [], 'o', color=system['color'], markersize=6)
            tracer_dots.append(dot)
        ax2.legend()

        image_dots = []
        for system in self.systems:
            system_dots = []
            for _ in range(5): 
                img_dot, = ax1.plot([], [], '.', color=system['color'], alpha=0.6, markersize=4)
                system_dots.append(img_dot)
            image_dots.append(system_dots)

        def update(i):
            artists = []
            for j, system in enumerate(self.systems):
                x_src = system['x_src'][i]
                y_src = system['y_src'][i]

                source_dots[j].set_data([x_src], [y_src])
                tracer_dots[j].set_data([self.tau[i]], [system['mag'][i]])
                artists.extend([source_dots[j], tracer_dots[j]])
                
                images = self.VBM.ImageContours(self.s, self.q, x_src, y_src, self.rho)

                for k, img_dot in enumerate(image_dots[j]):
                    if k < len(images):
                        img_dot.set_data(images[k][0], images[k][1])
                        img_dot.set_alpha(0.6)
                    else:
                        img_dot.set_data([], [])
                        img_dot.set_alpha(0)
                    artists.append(img_dot)

            return artists

        ani = animation.FuncAnimation(fig, update, frames=len(self.t), interval=50, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())
    
    def plot_centroid_shift(self):
        
        plt.figure(figsize=(10, 5))
        cmap_cs = plt.colormaps['BuPu']
        colors_cs = [cmap_cs(i) for i in np.linspace(0.4, 1.0, len(self.u0_list))]

        for idx, system in enumerate(self.systems):
            color = colors_cs[idx]
            x_src = system['x_src']
            y_src = system['y_src']
            cent_x = system['cent_x']
            cent_y = system['cent_y']

            shift = np.sqrt((cent_x - x_src) ** 2 + (cent_y - y_src) ** 2)
            plt.plot(self.tau, shift, color=color, label=f"$u_0$ = {system['u0']}")

        plt.xlabel(r"Time ($\tau$)")
        plt.ylabel(r"Centroid Shift ($\Delta \theta$)")
        plt.title("Astrometric Centroid Shift for 2L1S")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def show_all(self):

        caustics = self.VBM.Caustics(self.s, self.q)
        criticalcurves = self.VBM.Criticalcurves(self.s, self.q)

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, height_ratios=[2, 1])

        ax1 = fig.add_subplot(gs[0, 0])  # lensing event
        ax2 = fig.add_subplot(gs[0, 1])  # light curve
        ax3 = fig.add_subplot(gs[1, :])  # centroid shift 

        #lensing event
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_title("2L1S Lensing Event")
        ax1.set_xlabel(r"$\theta_x$ ($\theta_E$)")
        ax1.set_ylabel(r"$\theta_y$ ($\theta_E$)")
        ax1.set_aspect("equal")
        ax1.grid(True)

        for cau in caustics:
            ax1.plot(cau[0], cau[1], 'r', lw=1.2)
        for crit in criticalcurves:
            ax1.plot(crit[0], crit[1], 'k--', lw=0.8)

        m1 = 1.0 / (1 + self.q)
        m2 = self.q / (1 + self.q)
        x1 = -self.s * m2
        x2 = self.s * m1
        ax1.plot([x1, x2], [0, 0], 'ko', label="Lenses")

        source_dots = []
        tracer_dots = []

        #light curve
        ax2.set_xlim(self.tau[0], self.tau[-1])
        all_mag = np.concatenate([s['mag'] for s in self.systems])
        ax2.set_ylim(min(all_mag)*0.95, max(all_mag)*1.05)
        ax2.set_xlabel(r"Time ($\tau$)")
        ax2.set_ylabel("Magnification")
        ax2.set_title("Microlensing Light Curve")
        ax2.grid(True)

        #centroid shift
        ax3.set_title("Centroid Shift")
        ax3.set_xlabel(r"Time ($\tau$)")
        ax3.set_ylabel(r"$\Delta \theta$")
        ax3.grid(True)

        rho_legend = Line2D([0], [0], color='none', label=fr"$\rho$ = {self.rho}")
        for system in self.systems:
            
            ax1.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.4)
            src_dot, = ax1.plot([], [], '*', color=system['color'], markersize=6)
            source_dots.append(src_dot)
            
            ax2.plot(self.tau, system['mag'], color=system['color'], label=f"$u_0$ = {system['u0']}")
            dot, = ax2.plot([], [], 'o', color=system['color'], markersize=6)
            tracer_dots.append(dot)

            shift = np.sqrt((system['cent_x'] - system['x_src'])**2 + (system['cent_y'] - system['y_src'])**2)
            ax3.plot(self.tau, shift, color=system['color'], label=f"$u_0$ = {system['u0']}")

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles + [rho_legend])
        ax2.legend()
        ax3.legend()

        # --- Animation update ---
        def update(i):
            for j, system in enumerate(self.systems):
                source_dots[j].set_data([system['x_src'][i]], [system['y_src'][i]])
                tracer_dots[j].set_data([self.tau[i]], [system['mag'][i]])
            return source_dots + tracer_dots

        ani = animation.FuncAnimation(fig, update, frames=len(self.t), interval=50, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())



        

