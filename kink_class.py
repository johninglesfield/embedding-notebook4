# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 08:44:36 2016

@author: johninglesfield
"""
import cmath
import numpy as np
from numpy import sqrt, sin, cos, pi, trace
from scipy.integrate import quad, dblquad
from scipy.linalg import inv, eig
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

def spline(y):
    """
    This spline defines a bell function for 0<y<2.
    """
    if 0 <= y <= 0.5:
        curve = 8.0*y**3
    elif 0.5 < y <= 1:
        curve = 24.0*((1-y)**3 - (1-y)**2) + 4.0
    elif 1 < y <= 1.5:
        curve = 24.0*((y-1)**3 - (y-1)**2) + 4.0
    else:
        curve = 8.0*(2-y)**3
    return 0.25*curve
    
def dspline(y):
    """
    Derivative of spline function.
    """
    if 0 <= y <= 0.5:
        deriv = 24.0*y**2
    elif 0.5 < y <= 1.0:
        deriv = 24.0*(-3.0*(1.0-y)**2 + 2.0*(1.0-y))
    elif 1.0 < y <= 1.5:
        deriv = 24.0*(3.0*(y-1)**2 - 2.0*(y-1.0))
    else:
        deriv = -24.0*(2.0-y)**2
    return 0.25*deriv

class Kink:
    """
    Calculates embedding for kinks of different geometries.
    """
    
    def __init__(self, a, D):
        """
        Sets up geometry of kink.
        """
        self.a = a
        self.fac = pi/D
        self.displace = 0.5*(D - a)
        self.n_plot = 50
                
    def kink_define(self, ifn, param):
        """
        Defines the kink.
        """
        g_zero = param[0]
        h_zero = param[1]
        if ifn == 0:
            g_slope = param[2]
            h_slope = param[3]
            self.g = lambda x: g_zero + g_slope*x
            self.h = lambda x: h_zero + h_slope*x
            self.dg = lambda x: g_slope
            self.dh = lambda x: h_slope
        elif ifn == 1:
            width = param[2]
            disp = param[3]
            dlg = 0.5*(self.a+disp)
            dlh = 0.5*(self.a-disp)
            self.g = lambda x: g_zero + 0.5*width*(np.tanh(2.0*(x-dlg))+1.0)
            self.h = lambda x: h_zero + 0.5*width*(np.tanh(2.0*(x-dlh))+1.0)
            self.dg = lambda x: width/np.cosh(2.0*(x-dlg))**2
            self.dh = lambda x: width/np.cosh(2.0*(x-dlh))**2
        else:
            g_width = param[2]
            h_width = param[3]
            self.g = lambda x: g_width*spline(2.0*x/self.a) + g_zero
            self.h = lambda x: h_width*spline(2.0*x/self.a) + h_zero
            self.dg = lambda x: g_width*dspline(2.0*x/self.a)*2.0/self.a
            self.dh = lambda x: g_width*dspline(2.0*x/self.a)*2.0/self.a
        self.kink_plot()

    def kink_plot(self):
        """
        Plots the kink, and if the kink has an unacceptable geometry the program
        stops with an assertion error.
        """
        test = True
        self.xy = np.linspace(0.0, self.a, self.n_plot)
        self.g_plot = np.zeros(len(self.xy))
        self.h_plot = np.zeros(len(self.xy))
        for i in xrange(len(self.xy)):
            self.g_plot[i] = self.g(self.xy[i])
            self.h_plot[i] = self.h(self.xy[i])
            if (self.g_plot[i] < 0 or self.g_plot[i] > self.h_plot[i] or
            self.h_plot[i] > self.a):
                test = False
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.set(aspect='equal', xlim=[0.0, self.a], ylim=[0.0, self.a])
        ax.minorticks_on()
        ax.grid(linestyle='--')
        ax.grid(which='minor', linestyle=':')
        ax.plot(self.xy, self.g_plot, linewidth=2.0, color='black')
        ax.plot(self.xy, self.h_plot, linewidth=2.0, color='black')
        ax.plot([0.0, -0.3], [self.g_plot[0], self.g_plot[0]], clip_on=False,
            linewidth=2.0, color='black')
        ax.plot([self.a, self.a+0.3], [self.g_plot[self.n_plot-1], 
            self.g_plot[self.n_plot-1]], clip_on=False, linewidth=2.0, 
            color='black')
        ax.plot([0.0, -0.3], [self.h_plot[0], self.h_plot[0]], clip_on=False,
            linewidth=2.0, color='black')
        ax.plot([self.a, self.a+0.3], [self.h_plot[self.n_plot-1], 
            self.h_plot[self.n_plot-1]], clip_on=False, linewidth=2.0, 
            color='black')
        ax.set_title('Shape of kink', fontsize='x-large')               
        assert (test), "kink shape outside range - try again!"
                
    def kink_matrices(self, N, V):
        """
        Evaluates matrices for kink geometry.
        """
        self.N = N
        sigma = sqrt(0.5*V).real
        ab_array = np.zeros((N,N), dtype=int)
        for i in range(self.N):
            ab_array[i,] = range(N)
        self.al = ab_array.flatten(order='C')
        self.be = ab_array.flatten(order='F')
        self.ovlp = np.zeros((N*N, N*N))
        self.ham = np.zeros((N*N, N*N))
        for i in range(N*N):
            m = [self.al[i], self.be[i]]
            for j in range(N*N):
                n = [self.al[j], self.be[j]]
                self.ovlp[i, j] = dblquad(self.kink_ovlp_int, 0., self.a, 
                                     self.g, self.h, args=(m, n))[0]
                kinen = dblquad(self.kink_kinen_int, 0., self.a,
                                self.g, self.h, args=(m, n))[0]
                confine = sigma*(quad(self.kink_low_embed, 0., self.a, 
                                      args=(m, n))[0] +
                                      quad(self.kink_top_embed, 0., self.a,
                                      args=(m, n))[0])
                self.ham[i, j] = kinen + confine
            if (i+1)%5 == 0:
                print '%s %3d %s %3d' % ('Number of integrals done =', i+1,
                                         'x', N*N) 
        print '%s %3d %s %3d' % ('Number of integrals done =', i+1, 'x', N*N)
        return 
    
    def kink_ovlp_int(self, y0, x0, m, n):
        """
        Evaluates overlap integrand for kink geometry.
        """
        x = x0 + self.displace
        y = y0 + self.displace
        chi_i = cos(self.fac*m[0]*x) * cos(self.fac*m[1]*y)
        chi_j = cos(self.fac*n[0]*x) * cos(self.fac*n[1]*y)
        return chi_i*chi_j

    def kink_kinen_int(self, y0, x0, m, n):
        """
        Evaluates kinetic energy integrand for kink geometry.
        """
        x = x0 + self.displace
        y = y0 + self.displace
        grad_chi_i = np.array([-m[0]*sin(self.fac*m[0]*x)*cos(self.fac*m[1]*y),
                              -m[1]*cos(self.fac*m[0]*x)*sin(self.fac*m[1]*y)])
        grad_chi_j = np.array([-n[0]*sin(self.fac*n[0]*x)*cos(self.fac*n[1]*y),
                              -n[1]*cos(self.fac*n[0]*x)*sin(self.fac*n[1]*y)])
        return 0.5*self.fac*self.fac*np.dot(grad_chi_i,grad_chi_j)
        
    def kink_low_embed(self, x0, m, n):
        """
        Line integral along lower side of wave-guide.
        """
        y0 = self.g(x0)
        gp = self.dg(x0)
        x = x0 + self.displace
        y = y0 + self.displace
        chi_i = cos(self.fac*m[0]*x) * cos(self.fac*m[1]*y)
        chi_j = cos(self.fac*n[0]*x) * cos(self.fac*n[1]*y)
        return sqrt(1.0+gp*gp)*chi_i*chi_j
        
    def kink_top_embed(self, x0, m, n):
        """
        Line integral along upper side of wave-guide.
        """
        y0 = self.h(x0)
        hp = self.dh(x0)
        x = x0 + self.displace
        y = y0 + self.displace
        chi_i = cos(self.fac*m[0]*x) * cos(self.fac*m[1]*y)
        chi_j = cos(self.fac*n[0]*x) * cos(self.fac*n[1]*y)
        return sqrt(1.0+hp*hp)*chi_i*chi_j    
        
    def kink_embed(self, nemb):
        """
        Integrals of channel functions times basis functions.
        """
        xl = self.displace
        xr = self.displace + self.a
        csl = [cos(self.fac*i*xl) for i in range(self.N)]
        csr = [cos(self.fac*i*xr) for i in range(self.N)]
        self.w_left = self.h(0) - self.g(0)
        self.w_right = self.h(self.a) - self.g(self.a)
        self.nemb = nemb
        left_int = np.zeros(self.N)
        right_int = np.zeros(self.N)
        self.left_ovlp = np.zeros((self.nemb, self.N*self.N))
        self.right_ovlp = np.zeros((self.nemb, self.N*self.N))
        for p in range(self.nemb):
            for i in range(self.N):
                left_int[i] = quad(self.kink_lemb, 0.0, self.w_left,
                                            args=(p+1, i))[0]
                right_int[i] = quad(self.kink_remb, 0.0, self.w_right,
                                            args=(p+1, i))[0]
            for i in range(self.N*self.N):
                self.left_ovlp[p, i] = left_int[self.be[i]]*csl[self.al[i]]
                self.right_ovlp[p, i] = right_int[self.be[i]]*csr[self.al[i]]
        return
                
    def kink_lemb(self, yp, p, i):
        """
        Product of left channel function times basis function.
        """
        y = yp + self.g(0.0) + self.displace
        return sin(pi*p*yp/self.w_left)*cos(self.fac*i*y)
                
    def kink_remb(self, yp, p, i):
        """
        Product of right channel function times basis function.
        """
        y = yp + self.g(self.a) + self.displace
        return sin(pi*p*yp/self.w_right)*cos(self.fac*i*y)
        
    def kink_green(self, energy):
        """
        Calculates the Green function and density of states in kink.
        """
        emb_left = np.zeros((self.N*self.N, self.N*self.N), complex)
        emb_right = np.zeros((self.N*self.N,self.N*self.N), complex)
        for p in range(self.nemb):
            sig_left = -1j*(cmath.sqrt(2.0*energy-((p+1)*pi/self.w_left)**2)/
                        self.w_left)
            sig_right = -1j*(cmath.sqrt(2.0*energy-((p+1)*pi/self.w_right)**2)/
                        self.w_right)
            emb_left = emb_left + (sig_left*(np.outer(self.left_ovlp[p,],
                                        self.left_ovlp[p,])))
            emb_right = emb_right + (sig_right*(np.outer(self.right_ovlp[p,],
                                        self.right_ovlp[p,])))
        self.green = self.ham + emb_left + emb_right - energy*self.ovlp
        self.green = -inv(self.green)
        dos = -trace(np.dot(self.green, self.ovlp)).imag/pi
        return dos
                          
    def kink_continuum_wf(self, energy, input_channel):
        """
        Calculates the continuum wave-function for a given energy and channel.
        """
        self.energy = energy
        self.inchann = input_channel
        E_channel = 0.5*((input_channel+1)*pi/self.w_left)**2
        kz = sqrt(2.0*(energy - E_channel))
        sigma = -1j*kz/self.w_left
        psi_input = 0.5*self.w_left*self.left_ovlp[input_channel, :]
        self.kink_green(energy)
        self.psi = -2.0j*sigma.imag*np.dot(self.green, psi_input)
        
    def kink_transmission(self, energy, input_channel, output_channel):
        """
        Calculates transmission probability for given input and output channels
        and energy.
        """
        E_input = 0.5*((input_channel+1)*pi/self.w_left)**2
        E_output = 0.5*((output_channel+1)*pi/self.w_right)**2
        if energy < E_input or energy < E_output:
            T = 0.0
        else:
            kz_input = sqrt(2.0*(energy-E_input))
            kz_output = sqrt(2.0*(energy-E_output))
            input_current = 0.5*self.w_left*kz_input
            sigma_input = -1j*kz_input/self.w_left
            sigma_output = -1j*kz_output/self.w_right
            psi_input = 0.5*self.w_left*self.left_ovlp[input_channel, :]
            self.kink_green(energy)
            psi = -2.0j*sigma_input.imag*np.dot(self.green, psi_input)
            Psi = np.dot(psi, self.right_ovlp[output_channel, ])
            output_current = -2.0*(np.conj(Psi)*Psi*sigma_output.imag).real
            T = output_current/input_current
        return T

    def kink_transmit_output_range(self, energy):
        """
        Gives the number of possible output channels for a given energy.
        """
        range = 1
        while energy > 0.5 * ((range+1)*pi/self.w_right)**2:
            range += 1
        return range
        
    def kink_current(self, xp, yp):
        """
        Calculates the wave-function and current density at a given point.
        """
        x = xp + self.displace
        y = yp + self.displace
        phi = 0.0
        dphi_x = 0.0
        dphi_y = 0.0
        for i in range(self.N*self.N):
            phi = phi + (self.psi[i]*cos(self.fac*self.al[i]*x)*
            cos(self.fac*self.be[i]*y))
            dphi_x = dphi_x - (self.psi[i]*self.fac*self.al[i]*
            sin(self.fac*self.al[i]*x)*cos(self.fac*self.be[i]*y))
            dphi_y = dphi_y - (self.psi[i]*self.fac*self.be[i]*
            cos(self.fac*self.al[i]*x)*sin(self.fac*self.be[i]*y))
        current_x=(np.conj(phi)*dphi_x).imag
        current_y=(np.conj(phi)*dphi_y).imag
        return phi, current_x, current_y
        

    def kink_total_transmission(self, energy):
        """
        Calculates total transmission at a given energy using Meir-Wingreen.
        """
        self.kink_green(energy)
        Sigma_in = np.zeros((self.N**2, self.N**2))
        Sigma_out = np.zeros((self.N**2, self.N**2))
        for p in range(self.nemb):
            sigma_in = -1j*cmath.sqrt(2.0*energy - 
                ((p+1)*pi/self.w_left)**2)/self.w_left
            sigma_out = -1j*cmath.sqrt(2.0*energy -
                ((p+1)*pi/self.w_right)**2)/self.w_right
            Sigma_in = Sigma_in + sigma_in.imag*np.outer(
                                self.left_ovlp[p, ], self.left_ovlp[p, ])
            Sigma_out = Sigma_out + sigma_out.imag*np.outer(
                                self.right_ovlp[p,], self.right_ovlp[p, ])
        sigin_green = np.dot(Sigma_in, np.conj(self.green))
        sigout_green = np.dot(Sigma_out, np.conj(self.green))
        green_sigin_green = np.dot(self.green, sigin_green).real
        green_sigout_green = np.dot(self.green, sigout_green).real
        green_sigin_green_sigout = np.dot(green_sigin_green, Sigma_out)
        dos = -np.trace(np.dot(self.green, self.ovlp)).imag/pi
        left_dos = -np.trace(np.dot(green_sigin_green, self.ovlp))/pi
        right_dos = -np.trace(np.dot(green_sigout_green, self.ovlp))/pi
        tot_trans = 4.0*np.trace(green_sigin_green_sigout)
        return dos, left_dos, right_dos, tot_trans
                  
    def kink_current_plot(self, size, arrow):
        """
        Plots wave-function and current density.
        """
        self.X, self.Y = np.meshgrid(self.xy, self.xy)
        Z, Jx, Jy = self.kink_current(self.X, self.Y)
        for i in range(self.n_plot):
            for j in range(self.n_plot):
                if (self.Y[i, j] > self.h(self.X[i, j]) or 
                self.Y[i, j] < self.g(self.X[i, j])):
                    Z[i, j] = 0.0
                    Jx[i, j] = 0.0
                    Jy[i, j] = 0.0
        Z_mod = np.zeros((self.n_plot, self.n_plot))
        Z_phase = np.zeros((self.n_plot, self.n_plot))
        for i in range(self.n_plot):
            for j in range(self.n_plot):
                Z_mod[i, j], Z_phase[i, j] = cmath.polar(Z[i, j])
        ps_max = np.amax(np.absolute(Z.real))
        ps_max = max(ps_max, 1.0)
        md_max = np.amax(Z_mod)
        md_max = max(md_max, 1.0)
        fig = plt.figure(figsize=(size, size))
        fig.suptitle('Continuum wavefunction at E = ' + str(self.energy) + 
                     ' a.u., channel ' + str(self.inchann), fontsize=16,
                     y=0.97)
        ax = fig.add_axes([0.05, 0.525, 0.375, 0.375])
        ax.plot(self.xy, self.g_plot, linewidth=1.5, color='black')
        ax.plot(self.xy, self.h_plot, linewidth=1.5, color='black')
        cs = ax.contourf(self.X, self.Y, Z.real, 50, vmin=-ps_max, vmax=ps_max)
        cax = fig.add_axes([0.425, 0.525, 0.03, 0.375])
        fig.colorbar(cs, cax)
        cs = ax.contour(self.X, self.Y, Z.imag, 10, colors='black')
        ax.clabel(cs, color='black')
        ax.set_title('Real and imaginary parts', fontsize=14)
        ax = fig.add_axes([0.54, 0.525, 0.375, 0.375])
        ax.plot(self.xy, self.g_plot, linewidth=1.5, color='black')
        ax.plot(self.xy, self.h_plot, linewidth=1.5, color='black')
        cs = ax.contourf(self.X, self.Y, Z_mod, 50, vmin=-md_max, vmax=md_max)
        cax = fig.add_axes([0.915, 0.525, 0.03, 0.375])
        fig.colorbar(cs, cax)
        cs = ax.contour(self.X, self.Y, Z_phase, 10, colors='black')
        ax.clabel(cs, color='black')
        ax.set_title('Modulus and phase', fontsize=14)
        ax = fig.add_axes([0.3, 0.075, 0.375, 0.375])
        ax.plot(self.xy, self.g_plot, linewidth=1.5, color='black')
        ax.plot(self.xy, self.h_plot, linewidth=1.5, color='black')
        ax.quiver(self.X[::2, ::2], self.Y[::2, ::2], Jx[::2, ::2], Jy[::2, ::2],                  scale_units='xy', scale=arrow)
        ax.set_title('Current density', fontsize=14)
