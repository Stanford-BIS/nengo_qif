from __future__ import division

import numpy as np

import nengo
from nengo.builder import Builder
from nengo.builder.signal import Signal
from nengo.builder.neurons import SimNeurons
from nengo.params import NumberParam


# Neuron types must subclass `nengo.Neurons`
class QIFRate(nengo.neurons.NeuronType):
    """A rate-based quadratic integrate-and-fire neuron model"""
    tau_rc = NumberParam(low=0, low_open=True)
    tau_ref = NumberParam(low=0)
    probeable = ['rates']
    threshold = 10.0  # spike threshold

    def __init__(self, tau_rc=.01, tau_ref=.002, gb_tol=1e-6):
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.gb_tol = gb_tol  # tolerance in gain_bias binary search

    def gain_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to satisfy max_rates, intercepts

        Returns gain (alpha) and offset (J_bias) values of neurons.

        Parameters
        ----------
        max_rates : array of floats
            Maximum firing rates of neurons
        intercepts : array of floats
            x-intercepts of neurons
        """
        inv_tau_ref = 1./self.tau_ref if self.tau_ref > 0. else np.inf
        if (max_rates > inv_tau_ref).any():
            raise ValueError(
                "Max rates must be below the inverse refractory period (%0.3f)"
                % (inv_tau_ref))

        tspk_min = 1./max_rates - self.tau_ref

        J0 = np.zeros_like(tspk_min) + 0.5
        J1 = J0.copy()
        tspk = np.zeros_like(tspk_min) + np.inf
        J_max = np.zeros_like(tspk_min)

        # First binary search to find the correct range of J_max
        idx = np.ones(tspk_min.shape, dtype=bool)
        while idx.any():
            J0[idx] = J1[idx]
            J1[idx] *= 2
            tspk[idx] = self._tspk(J1[idx])
            idx[tspk < tspk_min] = False

        # Second binary search to find J_max
        idx[:] = True
        while idx.any():
            J_max[idx] = 0.5*(J0[idx]+J1[idx])
            tspk[idx] = self._tspk(J_max[idx])

            idx[np.abs(tspk - tspk_min) < self.gb_tol] = False
            high_idx = np.logical_and(idx, tspk < tspk_min)
            low_idx = np.logical_and(idx, tspk > tspk_min)

            J1[high_idx] = J_max[high_idx]
            J0[low_idx] = J_max[low_idx]

        # compute gain and bias from J_max and intercepts
        gain = (0.5 - J_max) / (intercepts - 1.)
        bias = 0.5 - gain*intercepts
        return gain, bias

    def _tspk(self, J):
        xt = self.threshold
        tspk = (
            2*self.tau_rc *
            (np.arctan((xt-1)/np.sqrt(2*J-1)) +
             np.arctan(1./np.sqrt(2*J-1))) /
            np.sqrt(2*J-1))
        return tspk

    def step_math(self, dt, J, output):
        idx = J > .5
        output[idx] = (self.tau_ref + self._tspk(J[idx]))**-1
        output[~idx] = 0.

    def rates(self, x, gain, bias):
        J = gain * x + bias
        out = np.zeros_like(J)
        QIFRate.step_math(self, dt=1, J=J, output=out)
        return out


class QIF(QIFRate):
    """A spiking quadratic leaky-integrate-and-fire neuron model"""
    probeable = ['spikes', 'voltage', 'refractory_time']

    def _vdot(self, J, voltage):
        dvdt = (-voltage + voltage**2/2 + J) / self.tau_rc
        return dvdt

    def rates(self, x, gain, bias):
        return QIFRate.rates(self, x, gain, bias)

    def step_math(self, dt, J, spiked, voltage, refractory_time):
        # Use Heun's method
        vv = voltage
        dvdt = self._vdot(J, vv)
        v = vv + dt*dvdt
        dvdt2 = self._vdot(J, v)
        dvdt_avg = (dvdt+dvdt2)/2.
        dv = dt*dvdt_avg

        # Use the Runge-Kutta method
#         k1 = self._vdot(J, voltage)
#         k2 = self._vdot(J, voltage + dt/2*k1)
#         k3 = self._vdot(J, voltage + dt/2*k2)
#         k4 = self._vdot(J, voltage + dt*k3)
#         dv = dt/6.*(k1 + 2*k2 + 2*k3 + k4)

        voltage += dv

        # update refractory period assuming no spikes for now
        refractory_time -= dt

        # set voltages of neurons still in their refractory period to 0
        # and reduce voltage of neurons partway out of their ref. period
        voltage *= (1 - refractory_time / dt).clip(0, 1)

        # determine which neurons spike
        spiked[:] = np.where(voltage > QIFRate.threshold, 1.0/dt, 0.0)

        # linearly approximate the time since neuron crossed spike threshold
        overshoot = (voltage[spiked > 0] - QIFRate.threshold) / dv[spiked > 0]
        spiketime = dt * (1 - overshoot)

        # set spiking neurons' voltages to zero, and ref. time to tau_ref
        voltage[spiked > 0] = 0.
        refractory_time[spiked > 0] = self.tau_ref + spiketime


# Register the QIFRate model with Nengo
@Builder.register(QIFRate)
def build_qifrate(model, qifrate, neurons):
    model.add_op(SimNeurons(
        neurons=qifrate,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out']))


# Register the QIF model with Nengo
@Builder.register(QIF)
def build_qif(model, qif, neurons):
    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.add_op(SimNeurons(
        neurons=qif,
        J=model.sig[neurons]['in'],
        output=model.sig[neurons]['out'],
        states=[model.sig[neurons]['voltage'],
                model.sig[neurons]['refractory_time']]))
