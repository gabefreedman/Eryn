# -*- coding: utf-8 -*-

"""
Adaptive HMC move variant.

This module provides an HMCMove subclass with a `tune()` method that runs a
short warmup phase and adapts the integration step size using the dual
averaging algorithm (Hoffman & Gelman, 2014) commonly used in NUTS/HMC
implementations.

The implementation keeps changes minimal by subclassing the existing
`HMCMove` from `hmc.py` and only adding the adaptation routine.
"""

import numpy as np
from copy import deepcopy
from ..state import State
from .hmc import HMCBase

__all__ = ["AdaptHMCMove"]


class AdaptHMCMove(HMCBase):
    """Hamiltonian Monte Carlo (HMC) proposal move.

    This class implements Hamiltonian Monte Carlo sampling, which uses gradient
    information to generate informed proposals. The move performs a series of
    leapfrog integration steps through a fictitious Hamiltonian dynamics to
    explore the parameter space more efficiently than random walk proposals.

    Args:
        grad_fn (callable, optional): A function that computes the gradient of the
            log-posterior with respect to the parameters. Should accept coordinates
            dictionary and return a dictionary of gradients with the same structure.
            If ``None``, numerical gradients will be computed. (default: ``None``)
        step_size (float or dict, optional): Step size for leapfrog integration.
            Can be a scalar (applied to all branches) or a dictionary with branch
            names as keys and scalar values. Smaller values are more accurate but
            require more steps. Typical values are in the range [0.01, 0.1].
            (default: ``0.1``)
        num_steps (int or dict, optional): Number of leapfrog integration steps.
            Can be an integer (applied to all branches) or a dictionary with branch
            names as keys and integer values. More steps lead to larger proposals
            but higher computational cost. Typical values are in the range [10, 100].
            (default: ``20``)
        inverse_metric (dict, optional): The inverse metric (mass matrix) for the
            kinetic energy. Keys are branch names and values are the inverse metric
            information. This information can be provided as a scalar, vector, or
            matrix and will be assumed isotropic, axis-aligned, or general, respectively.
            If not provided, uses identity matrix (unit mass). (default: ``None``)
        **kwargs (dict, optional): Kwargs for parent class :class:`MHMove`.
            (default: ``{}``)

    Raises:
        ValueError: If proposal parameters are inconsistent or invalid.

    Attributes:
        grad_fn (callable): Function to compute log-posterior gradients.
        step_size (dict): Step size for each branch.
        num_steps (dict): Number of integration steps for each branch.
        inverse_metric (dict): Inverse metric for each branch.

    """

    def __init__(
        self,
        grad_fn=None,
        step_size=0.1,
        num_steps=20,
        sim_length=0.5,
        inverse_metric=None,
        return_gpu=False,
        random_seed=None,
        **kwargs,
    ):
        """Initialize HMCMove.

        Args:
            grad_fn (callable, optional): Function to compute log-posterior gradients.
            step_size (float or dict, optional): Leapfrog step size.
            num_steps (int or dict, optional): Number of leapfrog steps.
            inverse_metric (dict, optional): Inverse metric (mass matrix) per branch.
            **kwargs: Additional arguments for parent class.

        """
        self.grad_fn = grad_fn

        # control whether outputs remain on gpu when use_gpu==True
        self.return_gpu = return_gpu

        # Will be populated with actual structure in setup
        self.step_size = step_size
        self.num_steps = num_steps
        self.sim_length = sim_length
        self.inverse_metric = inverse_metric if inverse_metric is not None else {}

        # Store for later conversion to per-branch dicts
        self._step_size_input = step_size
        self._num_steps_input = num_steps
        self._metric_setup_done = False

        # Flag for initial setup
        self._setup_complete = False

        super(AdaptHMCMove, self).__init__(**kwargs)

        # set the random seed of the array library if desired
        if random_seed is not None:
            self.xp.random.seed(random_seed)

    def setup(self, branches_coords):
        """Setup HMC proposal parameters for the given branch structure.

        This method initializes and validates the step size, number of steps, and
        inverse metric for each branch based on the dimensions of the coordinates.

        Args:
            branches_coords (dict): Keys are branch names. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the
                current coordinates for all the walkers.

        """
        # Convert step_size to per-branch dict if needed
        if isinstance(self._step_size_input, dict):
            self.step_size = self._step_size_input.copy()
        else:
            self.step_size = {name: self._step_size_input for name in branches_coords}

        # Convert num_steps to per-branch dict if needed
        if isinstance(self._num_steps_input, dict):
            self.num_steps = self._num_steps_input.copy()
        else:
            self.num_steps = {name: self._num_steps_input for name in branches_coords}

        # Setup inverse metric (mass matrix) for each branch
        if not self._metric_setup_done:
            self._metric_setup_done = True
            for name, coords in branches_coords.items():
                if name not in self.inverse_metric or self.inverse_metric[name] is None:
                    # Default to identity (unit mass)
                    ndim = coords.shape[-1]
                    self.inverse_metric[name] = np.eye(ndim)
                else:
                    # Validate and process provided metric
                    metric = self.inverse_metric[name]
                    try:
                        float(metric)
                        # Scalar metric - create identity scaled by value
                        ndim = coords.shape[-1]
                        self.inverse_metric[name] = metric * np.eye(ndim)
                    except TypeError:
                        metric = np.atleast_1d(metric)
                        if len(metric.shape) == 1:
                            # Diagonal metric
                            self.inverse_metric[name] = np.diag(metric)
                        elif (
                            len(metric.shape) == 2
                            and metric.shape[0] == metric.shape[1]
                        ):
                            # Full matrix metric
                            self.inverse_metric[name] = metric
                        else:
                            raise ValueError(
                                f"Invalid inverse metric dimensions for branch {name}"
                            )
        self._setup_complete = True

    def _gradient(self, model, coords, inds, branch_name):
        """Compute gradient of log-posterior.

        If grad_fn is provided, uses that. Otherwise computes numerical gradient.

        Args:
            model (Model): The model containing compute_log_prior_fn and compute_log_like_fn.
            coords (np.ndarray): Coordinates with shape (ntemps, nwalkers, nleaves_max, ndim)
                or subset thereof.
            inds (np.ndarray): Boolean array indicating which coordinates are valid.
            branch_name (str): Name of the branch.

        Returns:
            np.ndarray: Gradient of log-posterior with same shape as coords.

        """
        if self.grad_fn is not None:
            # Use provided gradient function. The expected signature for
            # user-provided grad_fn is fn(theta) where theta is a 1D array of
            # parameters (or it can accept a 2D array of positions and return
            # a 2D array of gradients). We therefore collect the active
            # coordinates, call the user's function in a vectorized way if
            # possible, otherwise fall back to per-row calls. Handle GPU
            # arrays by moving to CPU if needed and converting results back.
            xp = self.xp
            grad = xp.zeros_like(coords)

            inds_where = np.where(inds)
            q_active = coords[inds_where]

            # Move to numpy for user function if it's a GPU array
            if hasattr(q_active, "get"):
                q_active_np = q_active.get()
            else:
                q_active_np = np.asarray(q_active)

            # Try vectorized call first
            try:
                grad_active_np = self.grad_fn(q_active_np)
            except Exception:
                # Fallback to per-row calls
                grad_list = [self.grad_fn(theta) for theta in q_active_np]
                grad_active_np = np.vstack(grad_list)

            grad_active_np = np.asarray(grad_active_np)

            # Ensure shape matches
            if grad_active_np.ndim == 1:
                # Single gradient vector for all active points
                grad_active_np = np.tile(grad_active_np, (q_active_np.shape[0], 1))

            # Convert back to xp array if necessary
            if xp is not np and not isinstance(grad_active_np, xp.ndarray):
                try:
                    grad_active = xp.asarray(grad_active_np)
                except Exception:
                    grad_active = grad_active_np
            else:
                grad_active = grad_active_np

            grad[inds_where] = grad_active
            # Ensure no NaNs/Infs propagate from user gradients
            try:
                grad = xp.asarray(grad)
                mask = xp.isfinite(grad)
                if not mask.all():
                    grad = xp.where(mask, grad, xp.zeros_like(grad))
            except Exception:
                # If xp conversion fails, fall back to numpy sanitization
                grad = np.asarray(grad)
                mask = np.isfinite(grad)
                grad[~mask] = 0.0

            # Clip extreme gradient values to avoid overflow in momentum updates
            max_grad = getattr(self, "max_grad_clip", 1e6)
            try:
                grad = xp.clip(grad, -max_grad, max_grad)
            except Exception:
                grad = np.clip(np.asarray(grad), -max_grad, max_grad)

            return grad
        else:
            # Compute numerical gradient
            return self._numerical_gradient(model, coords, inds, branch_name)

    def _numerical_gradient(self, model, coords, inds, branch_name):
        """Compute numerical gradient using finite differences.

        Args:
            model (Model): The model.
            coords (np.ndarray): Coordinates.
            inds (np.ndarray): Valid coordinate indices.
            branch_name (str): Branch name.

        Returns:
            np.ndarray: Numerical gradient.

        """
        epsilon = 1e-5
        xp = self.xp
        grad = xp.zeros_like(coords)

        ntemps, nwalkers, nleaves_max, ndim = coords.shape
        inds_where = np.where(inds)

        # Evaluate at current point
        logp_raw = model.compute_log_prior_fn(
            {branch_name: coords}, inds={branch_name: inds}
        )
        if isinstance(logp_raw, dict):
            logp_curr = logp_raw[branch_name]
        else:
            logp_curr = logp_raw
        logl_curr = model.compute_log_like_fn(
            {branch_name: coords},
            inds={branch_name: inds},
            logp=logp_raw,
        )[0]
        if isinstance(logl_curr, dict):
            logl_curr = logl_curr[branch_name]
        # Index log arrays using the full (ntemps, nwalkers, nleaves) indices
        if hasattr(logl_curr, "ndim") and logl_curr.ndim == 3:
            post_curr = logl_curr[inds_where] + logp_curr[inds_where]
        else:
            post_curr = (
                logl_curr[inds_where[0], inds_where[1]]
                + logp_curr[inds_where[0], inds_where[1]]
            )

        # Finite difference for each dimension
        for d in range(ndim):
            coords_plus = coords.copy()
            coords_plus[inds_where + (d,)] += epsilon

            logp_plus_raw = model.compute_log_prior_fn(
                {branch_name: coords_plus}, inds={branch_name: inds}
            )
            if isinstance(logp_plus_raw, dict):
                logp_plus = logp_plus_raw[branch_name]
            else:
                logp_plus = logp_plus_raw
            logl_plus = model.compute_log_like_fn(
                {branch_name: coords_plus},
                inds={branch_name: inds},
                logp=logp_plus_raw,
            )[0]
            if isinstance(logl_plus, dict):
                logl_plus = logl_plus[branch_name]
            if hasattr(logl_plus, "ndim") and logl_plus.ndim == 3:
                post_plus = logl_plus[inds_where] + logp_plus[inds_where]
            else:
                post_plus = (
                    logl_plus[inds_where[0], inds_where[1]]
                    + logp_plus[inds_where[0], inds_where[1]]
                )

            # Safely compute finite-difference gradient: ignore non-finite values
            delta = post_plus - post_curr
            finite_mask = np.isfinite(delta)
            grad_vals = xp.zeros_like(post_curr)
            if finite_mask.any():
                grad_vals[finite_mask] = delta[finite_mask] / epsilon
            grad[inds_where + (d,)] = grad_vals

        # Ensure numerical gradient is finite and clipped
        try:
            grad = xp.asarray(grad)
            mask = xp.isfinite(grad)
            if not mask.all():
                grad = xp.where(mask, grad, xp.zeros_like(grad))
        except Exception:
            grad = np.asarray(grad)
            grad[~np.isfinite(grad)] = 0.0

        max_grad = getattr(self, "max_grad_clip", 1e6)
        try:
            grad = xp.clip(grad, -max_grad, max_grad)
        except Exception:
            grad = np.clip(np.asarray(grad), -max_grad, max_grad)

        return grad

    def _leapfrog(self, q, p, grad_q, epsilon, model, inds, name):
        pprime = p + 0.5 * epsilon * grad_q
        qprime = q + epsilon * pprime
        gradprime = self._gradient(model, qprime, inds, name)
        pprime = pprime + 0.5 * epsilon * gradprime
        return qprime, pprime, gradprime

    def _kinetic_energy(self, p):
        # get number of temperature and walkers
        ntemps, nwalkers, nleaves_max, ndim = p.shape

        kinetic_term = np.zeros((ntemps, nwalkers))
        kinetic_temp = 0.5 * np.sum(p**2, axis=-1)

        # vectorized because everything is rectangular (no groups to indicate model difference)
        kinetic_term += kinetic_temp.sum(axis=-1)

        return kinetic_term

    def find_initial_step_size(self, model, state, initial_eps=1.0, verbose=False):
        # Only run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(state.branches_coords)

        branch_names = list(state.branches_coords.keys())

        # get log prior and likelihood if not provided in the initial state
        if state.log_prior is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_prior = model.compute_log_prior_fn(coords, inds=inds)
        if state.log_like is None:
            coords = state.branches_coords
            inds = state.branches_inds
            state.log_like, state.blobs = model.compute_log_like_fn(
                coords,
                inds=inds,
                logp=state.log_prior,
                supps=state.supplemental,
                branch_supps=state.branches_supplemental,
            )

        for name in branch_names:
            coords = state.branches_coords[name]
            inds = state.branches_inds[name]

            p0 = model.random.randn(*coords.shape)
            if verbose:
                print(f"Finding reasonable step size for branch '{name}'...")

            coords = state.branches_coords[name]

            # Get initial log prior and log likelihood
            logp = state.log_prior
            logl = state.log_like
            # get log posterior
            logP = self.compute_log_posterior(logl, logp)

            # get current gradient
            grad0 = self._gradient(model, coords, inds, name)

            # Backoff loop: shrink step until logp and gradient are finite
            epsilon = initial_eps
            qprime, pprime, gradprime = self._leapfrog(
                coords, p0, grad0, epsilon, model, inds, name
            )
            logpprime = model.compute_log_prior_fn(
                {name: qprime}, inds=state.branches_inds
            )
            loglprime, _ = model.compute_log_like_fn(
                {name: qprime}, inds=state.branches_inds, logp=logpprime
            )
            logPprime = self.compute_log_posterior(loglprime, logpprime)
            while np.isinf(logPprime).any() or np.isinf(gradprime).any():
                epsilon = epsilon * 0.5
                qprime, pprime, gradprime = self._leapfrog(
                    coords, p0, grad0, epsilon, model, inds, name
                )
                logpprime = model.compute_log_prior_fn(
                    {name: qprime}, inds=state.branches_inds
                )
                loglprime, _ = model.compute_log_like_fn(
                    {name: qprime}, inds=state.branches_inds, logp=logpprime
                )
                logPprime = self.compute_log_posterior(loglprime, logpprime)

            a = logPprime - logP
            b = self._kinetic_energy(pprime) - self._kinetic_energy(p0)

            acceptprob = np.exp(a - b)
            accept = np.median(np.minimum(1.0, acceptprob))

            direction = 2.0 * float((accept > 0.5)) - 1.0
            while (accept**direction) > (2.0 ** (-direction)):
                epsilon = epsilon * (2.0**direction)
                qprime, pprime, gradprime = self._leapfrog(
                    coords, p0, grad0, epsilon, model, inds, name
                )
                logpprime = model.compute_log_prior_fn(
                    {name: qprime}, inds=state.branches_inds
                )
                loglprime, _ = model.compute_log_like_fn(
                    {name: qprime}, inds=state.branches_inds, logp=logpprime
                )
                logPprime = self.compute_log_posterior(loglprime, logpprime)
                a = logPprime - logP
                b = self._kinetic_energy(pprime) - self._kinetic_energy(p0)
                acceptprob = np.exp(a - b)
                accept = np.median(np.minimum(1.0, acceptprob))

            self.step_size[name] = epsilon

        return

    def tune(
        self,
        model,
        state,
        nburnin=50,
        delta=0.65,
        gamma=0.05,
        t0=10.0,
        kappa=0.75,
        Hbar=0.0,
    ):
        # Only run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(state.branches_coords)

        all_branch_names = list(state.branches_coords.keys())

        # get initial shape information
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        # initialize adaptation settings
        Hbar = {name: Hbar for name in all_branch_names}
        logepsilon_bar = {name: 0.0 for name in all_branch_names}
        mu = {name: np.log(10.0 * self.step_size[name]) for name in all_branch_names}

        for m in range(1, nburnin + 1):  # adaptation algorithm not valid for m=0
            for name in all_branch_names:
                # setup supplemental information
                if not np.all(
                    np.asarray(list(state.branches_supplemental.values())) is None
                ):
                    new_branch_supps = deepcopy(state.branches_supplemental)
                else:
                    new_branch_supps = None

                if state.supplemental is not None:
                    new_supps = deepcopy(state.supplemental)
                else:
                    new_supps = None

                # make proposal
                q, deltaK = self.get_proposal(
                    state.branches_coords,
                    model.random,
                    branches_inds=state.branches_inds,
                    supps=new_supps,
                    branch_supps=new_branch_supps,
                    model=model,
                )

                # if not wrapping with mutliple try (normal route)
                if not hasattr(self, "mt_ll") or not hasattr(self, "mt_lp"):
                    # Compute prior of the proposed position
                    logp = model.compute_log_prior_fn(q, inds=state.branches_inds)

                    # Compute the lnprobs of the proposed position.
                    # Can adjust supplementals in place
                    logl, new_blobs = model.compute_log_like_fn(
                        q,
                        inds=state.branches_inds,
                        logp=logp,
                        supps=new_supps,
                        branch_supps=new_branch_supps,
                    )

                else:
                    # if already computed in multiple try
                    logl = self.mt_ll
                    logp = self.mt_lp
                    new_blobs = None

                # get log posterior
                logP = self.compute_log_posterior(logl, logp)

                # get previous information
                prev_logp = state.log_prior
                prev_logl = state.log_like

                # takes care of tempering
                prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

                # determine acceptance
                delta_H = deltaK - (logP - prev_logP)
                Hdiff = -delta_H

                # draw against acceptance fraction
                accept_frac = np.median(np.minimum(1.0, np.exp(Hdiff)))
                accepted = Hdiff > np.log(model.random.rand(ntemps, nwalkers))

                # Update the parameters
                new_state = State(
                    q,
                    log_like=logl,
                    log_prior=logp,
                    blobs=new_blobs,
                    inds=state.branches_inds,
                    supplemental=new_supps,
                    branch_supplemental=new_branch_supps,
                )
                state = self.update(state, new_state, accepted)

                # add to move-specific accepted information
                self.accepted += accepted
                self.num_proposals += 1

                # do adaptation
                eta = 1.0 / (m + t0)
                mkappa = m ** (-kappa)
                Hbar[name] = (1.0 - eta) * Hbar[name] + eta * (delta - accept_frac)
                logepsilon = mu[name] - (np.sqrt(m) / gamma) * Hbar[name]
                logepsilon_bar[name] = (
                    mkappa * logepsilon + (1.0 - mkappa) * logepsilon_bar[name]
                )
                self.step_size[name] = np.exp(logepsilon)

        print(
            f"Final tuned epsilons: {[name + ': ' + str(np.exp(logepsilon_bar[name])) for name in all_branch_names]}"
        )
        self.step_size = {
            name: np.exp(logepsilon_bar[name]) for name in all_branch_names
        }

        # temperature swaps
        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        # return final state
        return state

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        # only run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(branches_coords)

        # get model form kwargs if available (set in MHMove.propose)
        model = kwargs.get("model", None)
        if model is None:
            # if model not in kwargs, try to use current_model
            if hasattr(self, "current_model"):
                model = self.current_model
            else:
                raise RuntimeError("Model must be provided for HMC proposals")

        # initialize output
        q = {}
        deltaK = {}

        # proposal per branch
        for name, coords in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            rnd = random if not self.use_gpu else self.xp.random
            p0 = rnd.randn(*coords.shape)

            num_steps = np.maximum(1, int(self.sim_length / self.step_size[name]))

            coords = branches_coords[name].copy()

            # get current gradient
            grad0 = self._gradient(model, coords, inds, name)

            # make copies before starting leapfrog integration
            qprime = coords.copy()
            pprime = p0.copy()
            gradprime = grad0.copy()

            for _ in range(1, int(num_steps) + 1):
                qprime, pprime, gradprime = self._leapfrog(
                    qprime, pprime, gradprime, self.step_size[name], model, inds, name
                )

            # flip momentum sign to make proposal explicitly symmetric
            pprime = -pprime

            # calculate change in kinetic energy
            q[name] = qprime
            deltaK = self._kinetic_energy(pprime) - self._kinetic_energy(p0)

        # Handle periodic parameters
        if self.periodic is not None:
            q = self.periodic.wrap(
                {
                    name: tmp.reshape((ntemps * nwalkers,) + tmp.shape[-2:])
                    for name, tmp in q.items()
                },
                xp=self.xp,
            )

            q = {
                name: tmp.reshape(
                    (
                        ntemps,
                        nwalkers,
                    )
                    + tmp.shape[-2:]
                )
                for name, tmp in q.items()
            }

        # If running on GPU but user requested CPU returns, transfer arrays back
        if self.use_gpu and not getattr(self, "return_gpu", False):
            for name, arr in list(q.items()):
                if hasattr(arr, "get"):
                    q[name] = arr.get()
            if hasattr(deltaK, "get"):
                deltaK = deltaK.get()

        return q, deltaK
