import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML, display
import VBMicrolensing
import TripleLensing
import math
from matplotlib.lines import Line2D

class OneL1S:
    def __init__(self, t0, tE, rho, u0_list):

        self.t0 = t0
        self.tE = tE
        self.t = np.linspace(t0-tE, t0+tE, 50)
        self.rho = rho
        self.u0_list = u0_list
        self.tau = (self.t - t0) / tE

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
        self.tau = np.linspace(-4, 4, 100)
        self.t = self.t0 + self.tau * self.tE
        self.theta = np.radians(self.alpha)

        self.tau_hr = np.linspace(-4, 4, 1000)
        self.t_hr = self.t0 + self.tau_hr * self.tE

        self.VBM = VBMicrolensing.VBMicrolensing()
        self.VBM.RelTol = 1e-3
        self.VBM.Tol = 1e-3
        self.VBM.astrometry = True
        self.colors = [plt.colormaps['BuPu'](i) for i in np.linspace(1.0, 0.4, len(u0_list))]
        self.systems = self._prepare_systems()

    def _prepare_systems(self):
        systems = []

        def polygon_area(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        for u0, color in zip(self.u0_list, self.colors): 
            x_src = self.tau * np.cos(self.theta) - u0 * np.sin(self.theta) #for animation (lower resolution) 
            y_src = self.tau * np.sin(self.theta) + u0 * np.cos(self.theta)

            cent_x = []
            cent_y = []

            for x_s, y_s in zip(x_src, y_src):
                images = self.VBM.ImageContours(self.s, self.q, x_s, y_s, self.rho)

                image_fluxes = []
                image_cx = []
                image_cy = []

                for img in images:
                    x = np.array(img[0])
                    y = np.array(img[1])
                    flux = polygon_area(x, y)

                    if flux > 0:
                        cx = np.mean(x)
                        cy = np.mean(y)
                        image_fluxes.append(flux)
                        image_cx.append(cx)
                        image_cy.append(cy)

                total_flux = np.sum(image_fluxes)

                if total_flux > 0:
                    cx_weighted = np.sum(np.array(image_cx) * image_fluxes) / total_flux
                    cy_weighted = np.sum(np.array(image_cy) * image_fluxes) / total_flux
                else:
                    cx_weighted = np.nan
                    cy_weighted = np.nan

                cent_x.append(cx_weighted)
                cent_y.append(cy_weighted)

            x_src_hr = self.tau_hr * np.cos(self.theta) - u0 * np.sin(self.theta) #for centroid shift, higher resolution
            y_src_hr = self.tau_hr * np.sin(self.theta) + u0 * np.cos(self.theta)

            cent_x_hr = []
            cent_y_hr = []

            for x_s, y_s in zip(x_src_hr, y_src_hr):
                images = self.VBM.ImageContours(self.s, self.q, x_s, y_s, self.rho)

                image_fluxes, image_cx, image_cy = [], [], []

                for img in images:
                    x = np.array(img[0])
                    y = np.array(img[1])
                    flux = polygon_area(x, y)

                    if flux > 0:
                        cx = np.mean(x)
                        cy = np.mean(y)
                        image_fluxes.append(flux)
                        image_cx.append(cx)
                        image_cy.append(cy)

                total_flux = np.sum(image_fluxes)

                if total_flux > 0:
                    cx_weighted = np.sum(np.array(image_cx) * image_fluxes) / total_flux
                    cy_weighted = np.sum(np.array(image_cy) * image_fluxes) / total_flux
                else:
                    cx_weighted = np.nan
                    cy_weighted = np.nan

                cent_x_hr.append(cx_weighted)
                cent_y_hr.append(cy_weighted)

            mag, *_ = self.VBM.BinaryLightCurve(
                [math.log(self.s), math.log(self.q), u0, self.theta, math.log(self.rho), math.log(self.tE), self.t0],
                self.t)

            systems.append({
                'u0': u0,
                'color': color,
                'mag': mag,
                'x_src': x_src,
                'y_src': y_src,
                'cent_x': np.array(cent_x),
                'cent_y': np.array(cent_y),
                'x_src_hr': x_src_hr,
                'y_src_hr': y_src_hr,
                'cent_x_hr': np.array(cent_x_hr),
                'cent_y_hr': np.array(cent_y_hr),
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
            plt.plot(self.tau, system['mag'], color=system['color'], label=fr"$u_0$ = {system['u0']}")
        
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
        ax1.legend(handles=[lens_handle, caustic_handle, crit_curve_handle, q_handle, s_handle], loc='upper right', prop={'size': 8})

        source_dots, centroid_dots = [], []
        for system in self.systems:
            ax1.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.4)
            src_dot, = ax1.plot([], [], '*', color=system['color'], markersize=10)
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
    
    def plot_centroid_trajectory(self):
        plt.figure(figsize=(6, 6))
        for system in self.systems:
            delta_x = system['cent_x_hr'] - system['x_src_hr']
            delta_y = system['cent_y_hr'] - system['y_src_hr']
            plt.plot(delta_x, delta_y, color=system['color'], label=fr"$u_0$ = {system['u0']}")
        plt.xlim(-0.4, .8)    
        plt.ylim(-0.4, 0.5)
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
    
class ThreeLens1S:
    def __init__(self, t0, tE, rho, u0_list, q2, q3, s12, s23, alpha, psi):
        self.t0 = t0
        self.tE = tE
        self.rho = rho
        self.u0_list = u0_list
        self.q2 = q2
        self.q3 = q3
        self.s12 = s12
        self.s23 = s23
        self.alpha = alpha
        self.psi = psi
        self.tau = np.linspace(-4, 4, 100)
        self.t = self.t0 + self.tau * self.tE
        self.theta = np.radians(self.alpha)
        self.psi_rad = np.radians(self.psi)

        self.VBM = VBMicrolensing.VBMicrolensing()
        self.VBM.RelTol = 1e-3
        self.VBM.Tol = 1e-3
        self.VBM.astrometry = True
        self.VBM.SetMethod(self.VBM.Method.Nopoly)

        self.colors = [plt.colormaps['BuPu'](i) for i in np.linspace(1.0, .4, len(u0_list))]
        self.systems = self._prepare_systems()

    def _prepare_systems(self):
        systems = []
        for u0, color in zip(self.u0_list, self.colors):
            param_vec = [
                np.log(self.s12), np.log(self.q2), u0, self.alpha,
                np.log(self.rho), np.log(self.tE), self.t0,
                np.log(self.s23), np.log(self.q3), self.psi
            ]

            mag, *_ = self.VBM.TripleLightCurve(param_vec, self.t)

            x_src = self.tau * np.cos(self.theta) - u0 * np.sin(self.theta)
            y_src = self.tau * np.sin(self.theta) + u0 * np.cos(self.theta)

            systems.append({
                'u0': u0,
                'color': color,
                'mag': mag,
                'x_src': x_src,
                'y_src': y_src
            })

        return systems

    def _setting_parameters(self):
        """
        """
        param = [
            np.log(self.s12), np.log(self.q2), self.u0_list[0], self.alpha,
            np.log(self.rho), np.log(self.tE), self.t0,
            np.log(self.s23), np.log(self.q3), self.psi
        ]
        _ = self.VBM.TripleLightCurve(param, self.t)

    def _compute_lens_positions(self):
        x1, y1 = 0, 0
        x2, y2 = x1 + self.s12, y1
        x3 = x2 + self.s23 * np.cos(self.psi_rad)
        y3 = y2 + self.s23 * np.sin(self.psi_rad)
        return [(x1, y1), (x2, y2), (x3, y3)]
    
    def _calculate_image_positions(self, xs, ys):
        TRIL = TripleLensing.TripleLensing()
        mlens = [1 - self.q2 - self.q3, self.q2, self.q3]
        zlens = self._compute_lens_positions()
        zlens_cpp_format = [coord for pair in zlens for coord in pair] 
        nlens = len(mlens)

        zrxy_flat = TRIL.solv_lens_equation(mlens, zlens_cpp_format, xs, ys, nlens)
        degree = nlens * nlens + 1
        real_parts = zrxy_flat[:degree]
        imag_parts = zrxy_flat[degree:2 * degree]

        return [complex(re, im) for re, im in zip(real_parts, imag_parts)]
    
    def _true_solution(self, z_image, xs, ys, so_leps=1e-10):
        mlens = [1 - self.q2 - self.q3, self.q2, self.q3]
        zlens = [complex(x, y) for x, y in self._compute_lens_positions()]
        zs = complex(xs, ys)
        dzs = zs - z_image
        for m, zl in zip(mlens, zlens):
            dzs += m / np.conj(z_image - zl)
        return abs(dzs) < so_leps

    def plot_caustic_critical_curves(self):
        self._setting_parameters()
        caustics = self.VBM.Multicaustics()
        criticalcurves = self.VBM.Multicriticalcurves()

        plt.figure(figsize=(6, 6))
        lens_handle = Line2D([0], [0], marker='o', color='k', linestyle='None', label='Lens', markersize=6)
        caustic_handle = Line2D([0], [0], color='r', lw=1.2, label='Caustic')
        crit_curve_handle = Line2D([0], [0], color='k', linestyle='--', lw=0.8, label='Critical Curve')

        for cau in caustics:
            plt.plot(cau[0], cau[1], 'r', lw=1.2)
        for crit in criticalcurves:
            plt.plot(crit[0], crit[1], 'k--', lw=0.8)

        for system in self.systems:
            plt.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.6)

        lens_positions = self._compute_lens_positions()
        for x, y in lens_positions:
            plt.plot(x, y, 'ko', label='Lens')

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel(r"$\theta_x$ ($\theta_E$)")
        plt.ylabel(r"$\theta_y$ ($\theta_E$)")
        plt.title("3L1S Lensing Event")
        plt.gca().set_aspect("equal")
        plt.legend(handles=[lens_handle, caustic_handle, crit_curve_handle], loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_light_curve(self):
        plt.figure(figsize=(6, 4))
        for system in self.systems:
            plt.plot(self.tau, system['mag'], color=system['color'], label=fr"$u_0$ = {system['u0']}")
        plt.xlabel(r"Time ($\tau$)")
        plt.ylabel("Magnification")
        plt.title("Triple Lens Light Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def animate(self):
        self._setting_parameters()
        caustics = self.VBM.Multicaustics()
        criticalcurves = self.VBM.Multicriticalcurves()
        lens_positions = self._compute_lens_positions()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(wspace=0.4)

        lens_handle = Line2D([0], [0], marker='o', color='k', linestyle='None', label='Lens', markersize=6)
        caustic_handle = Line2D([0], [0], color='r', lw=1.2, label='Caustic')
        crit_curve_handle = Line2D([0], [0], color='k', linestyle='--', lw=0.8, label='Critical Curve')

        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        ax1.set_xlabel(r"$\theta_x$ ($\theta_E$)")
        ax1.set_ylabel(r"$\theta_y$ ($\theta_E$)")
        ax1.set_title("3L1S Microlensing Event")
        ax1.set_aspect("equal")
        ax1.grid(True)

        for cau in caustics:
            ax1.plot(cau[0], cau[1], 'r', lw=1.2)
        for crit in criticalcurves:
            ax1.plot(crit[0], crit[1], 'k--', lw=0.8)
        for x, y in lens_positions:
            ax1.plot(x, y, 'ko')

        source_dots = []
        image_dots = []
        for system in self.systems:
            ax1.plot(system['x_src'], system['y_src'], '--', color=system['color'], alpha=0.4)
            src_dot, = ax1.plot([], [], '*', color=system['color'], markersize=10)
            dots = [ax1.plot([], [], 'o', color='gray', markersize=4)[0] for _ in range(10)]
            source_dots.append(src_dot)
            image_dots.append(dots)

        ax1.legend(handles=[lens_handle, caustic_handle, crit_curve_handle], loc='upper right', prop={'size': 8})

        ax2.set_xlim(self.tau[0], self.tau[-1])
        all_mag = np.concatenate([s['mag'] for s in self.systems])
        ax2.set_ylim(min(all_mag) * 0.95, max(all_mag) * 1.05)
        ax2.set_xlabel(r"Time ($\tau$)")
        ax2.set_ylabel("Magnification")
        ax2.set_title("Light Curve")

        tracer_dots = []
        for system in self.systems:
            ax2.plot(self.tau, system['mag'], color=system['color'], label=f"$u_0$ = {system['u0']}")
            dot, = ax2.plot([], [], 'o', color=system['color'], markersize=6)
            tracer_dots.append(dot)
        ax2.legend()

        def update(i):
            artists = []
            for j, system in enumerate(self.systems):
                x_src = system['x_src'][i]
                y_src = system['y_src'][i]
                source_dots[j].set_data([x_src], [y_src])
                artists.append(source_dots[j])

                images = self._calculate_image_positions(x_src, y_src)
                verified_images = [img for img in images if self._true_solution(img, x_src, y_src)]

                for k in range(len(image_dots[j])):
                    if k < len(verified_images):
                        image_dots[j][k].set_data([verified_images[k].real], [verified_images[k].imag])
                    else:
                        image_dots[j][k].set_data([], [])
                    artists.append(image_dots[j][k])

                tracer_dots[j].set_data([self.tau[i]], [system['mag'][i]])
                artists.append(tracer_dots[j])

                print(f"Frame {i}: Source = ({x_src:.3f}, {y_src:.3f})")
                print(f"  Total images: {len(images)}")
                print(f"  Verified images: {len(verified_images)}")

            return artists
        
        ani = animation.FuncAnimation(fig, update, frames=len(self.t), interval=50, blit=True)
        plt.close(fig)
        return HTML(ani.to_jshtml())

    

    
    def plot_different_q3_lc(self, q3_values, reference_q3=None, colormap='RdPu'): 
        """
        """
        colors = [plt.colormaps[colormap](i) for i in np.linspace(.5, 1, len(q3_values))]
        
        plt.figure(figsize=(8, 6))
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)

        ref_q3 = reference_q3 if reference_q3 is not None else q3_values[0]
        ref_param = [
            np.log(self.s12), np.log(self.q2), self.u0_list[0], self.alpha,
            np.log(self.rho), np.log(self.tE), self.t0,
            np.log(self.s23), np.log(ref_q3), self.psi
        ]
        ref_mag, *_ = self.VBM.TripleLightCurve(ref_param, self.t)

        for idx, q3 in enumerate(q3_values):
            color = colors[idx]
            param_vec = [
                np.log(self.s12), np.log(self.q2), self.u0_list[0], self.alpha,
                np.log(self.rho), np.log(self.tE), self.t0,
                np.log(self.s23), np.log(q3), self.psi
            ]
            mag, *_ = self.VBM.TripleLightCurve(param_vec, self.t)

            label = fr"$q_3$ = {q3:.2e}"
            ax1.plot(self.tau, mag, label=label, color=color)
            residual = np.array(ref_mag) - np.array(mag)
            ax2.plot(self.tau, residual, color=color)

        ax1.set_ylabel("Magnification")
        ax1.set_title("Light Curve for Varying $q_3$")
        ax1.grid(True)
        ax1.legend()

        ax2.set_xlabel(r"Time ($\tau$)")
        ax2.set_ylabel("Residuals")
        ax2.axhline(0, color='gray', lw=0.5, ls='--')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()




