# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
from ..state import State
from .hmc import HMCBase

__all__ = ["AdaptHMCMove"]


class AdaptHMCMove(HMCBase):
    """Hamiltonian Monte Carlo proposal with dual-averaging step size adaptation.

    Algorithm 5 from `Hoffman & Gelman (2014) <https://arxiv.org/abs/1111.4246>`.

    Step size is tuned by calling ``tune()`` method after initialization for a
    set number of iterations. This runs separately and before the main sampling
    loop.

    Args:
        grad_fn (callable, optional): Likelihood gradient function. If
            ``None``, a central finite difference method is used.
            (default: ``None``)
        step_size (double or dict, optional): Leapfrog step size. If a dict,
            keys are branch names and values are per-branch step sizes.
            (default: ``0.1``)
        sim_length (double or dict, optional): Approximate trajectory length
            for each proposal. The effective number of leapfrog steps is
            ``max(1, int(sim_length / step_size))`` per branch. If a dict,
            keys are branch names and values are per-branch lengths.
            (default: ``0.5``)
        inverse_metric (ndarray or dict, optional): Inverse mass-matrix for
            momenta. If a dict, keys are branch names and values are per-branch
            inverse mass matrices. If ``None``, identity matrices are used.
            (default: ``None``)
        return_gpu (bool, optional): If ``use_gpu == True`` and
            ``return_gpu == True``, returned arrays remain on GPU. (default:
            ``False``)
        kwargs (dict, optional): Additional keyword arguments passed through
            :class:`HMCBase`.

    Attributes:
        step_size (dict): Per-branch leapfrog step sizes.
        sim_length (dict): Per-branch target trajectory lengths.
        inverse_metric (dict): Per-branch inverse mass matrices.
        return_gpu (bool): Whether to return array in ``Cupy`` or ``NumPy``.

    """

    def __init__(
        self,
        grad_fn=None,
        step_size=0.1,
        sim_length=0.5,
        inverse_metric=None,
        return_gpu=False,
        **kwargs,
    ):
        # control whether outputs remain on gpu when use_gpu==True
        self.return_gpu = return_gpu

        # will be populated with per-branch structure during setup
        self._step_size_input = step_size
        self._sim_length_input = sim_length
        self._inverse_metric_input = inverse_metric

        super(AdaptHMCMove, self).__init__(grad_fn=grad_fn, **kwargs)

    def setup(self, branches_coords):
        """Setup per-branch integration settings.

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]``.

        Raises:
            ValueError: If a provided ``inverse_metric`` has shape
                incompatible with a branch dimension.

        """
        # leapfrog integration step size
        if isinstance(self._step_size_input, dict):
            self.step_size = self._step_size_input.copy()
        else:
            self.step_size = {name: self._step_size_input for name in branches_coords}

        # leapfrog integration trajectory length
        if isinstance(self._sim_length_input, dict):
            self.sim_length = self._sim_length_input.copy()
        else:
            self.sim_length = {name: self._sim_length_input for name in branches_coords}

        # inverse mass metric for kinetic energy term
        if isinstance(self._inverse_metric_input, dict):
            self.inverse_metric = self._inverse_metric_input.copy()
        else:
            self.inverse_metric = {}
            for name, coords in branches_coords.items():
                diag_metric = np.eye(coords.shape[-1])
                if self._inverse_metric_input is not None:
                    if (
                        np.asarray(self._inverse_metric_input).shape
                        != diag_metric.shape
                    ):
                        raise ValueError(
                            "Incorrect input dimension for inverse metric: "
                            f"received {np.asarray(self._inverse_metric_input).shape}, "
                            f"expected {diag_metric.shape}"
                        )
                    else:
                        self.inverse_metric[name] = self._inverse_metric_input
                else:
                    self.inverse_metric[name] = diag_metric

        self._setup_complete = True

    def find_initial_step_size(self, model, state, initial_eps=1.0):
        """Heuristic to find a reasonable initial per-branch step size.

        Algorithm 4 from `Hoffman & Gelman (2014) <https://arxiv.org/abs/1111.4246>`.

        This follows a doubling/halving strategy based on the median
        one-step acceptance statistic.

        Args:
            model (:class:`eryn.model.Model`): Model object used by sampler.
            state (:class:`eryn.state.State`): Initial sampler state.
            initial_eps (double, optional): Starting trial step size for
                heuristic. (default: ``1.0``)

        Returns:
            None

        """
        # run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(state.branches_coords)

        # get log prior and likelihood if not provided in the initial state
        if state.log_prior is None:
            state.log_prior = model.compute_log_prior_fn(
                state.branches_coords, inds=state.branches_inds
            )
        if state.log_like is None:
            state.log_like, state.blobs = model.compute_log_like_fn(
                state.branches_coords,
                inds=state.branches_inds,
                logp=state.log_prior,
                supps=state.supplemental,
                branch_supps=state.branches_supplemental,
            )

        for name, q0 in state.branches_coords.items():
            inds = state.branches_inds[name]

            # draw momenta
            rnd = model.random if not self.use_gpu else self.xp.random
            p0 = rnd.randn(*q0.shape)

            # get initial log posterior
            logp = state.log_prior
            logl = state.log_like
            logP = self.compute_log_posterior(logl, logp)

            # get current gradient
            grad0 = self._gradient(q0, model, inds, name)

            # shrink step until logp and gradient are finite
            epsilon = initial_eps
            qprime, pprime = self._leapfrog_once(
                q0, p0, grad0, epsilon, model, inds, name
            )
            gradprime = self._gradient(qprime, model, inds, name)
            logPprime = self._get_logpost_q(qprime, model, inds, name)
            while np.isinf(logPprime).any() or np.isinf(gradprime).any():
                epsilon = epsilon * 0.5
                qprime, pprime = self._leapfrog_once(
                    q0, p0, grad0, epsilon, model, inds, name
                )
                gradprime = self._gradient(qprime, model, inds, name)
                logPprime = self._get_logpost_q(qprime, model, inds, name)

            a = logPprime - logP
            b = self._kinetic_energy(pprime, name) - self._kinetic_energy(p0, name)

            acceptprob = np.exp(a - b)
            accept = np.median(np.minimum(1.0, acceptprob))

            # main doubling/halving loop
            direction = 2.0 * float((accept > 0.5)) - 1.0
            while (accept**direction) > (2.0 ** (-direction)):
                epsilon = epsilon * (2.0**direction)
                qprime, pprime = self._leapfrog_once(
                    q0, p0, grad0, epsilon, model, inds, name
                )
                logPprime = self._get_logpost_q(qprime, model, inds, name)

                a = logPprime - logP
                b = self._kinetic_energy(pprime, name) - self._kinetic_energy(p0, name)

                acceptprob = np.exp(a - b)
                accept = np.median(np.minimum(1.0, acceptprob))

            # save final tuned epsilon
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
        """Run dual-averaging tuning warmup to adapt per-branch step sizes.

        Args:
            model (:class:`eryn.model.Model`): Model object used by sampler.
            state (:class:`eryn.state.State`): Initial sampler state.
            nburnin (int, optional): Number of adaptation iterations.
                (default: ``50``)
            delta (double, optional): Target acceptance statistic.
                (default: ``0.65``)
            gamma (double, optional): Dual-averaging regularization scale.
                (default: ``0.05``)
            t0 (double, optional): Iteration offset controlling early update
                stability. (default: ``10.0``)
            kappa (double, optional): Exponent controlling shrinkage toward
                the running average. (default: ``0.75``)
            Hbar (double, optional): Initial dual-averaging error accumulator.
                (default: ``0.0``)

        Returns:
            :class:`eryn.state.State`: Final warmup state after adaptation.

        """
        # Only run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(state.branches_coords)

        # get initial shape information
        all_branch_names = list(state.branches_coords.keys())
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
                    model=model
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
                prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

                # determine acceptance (-delta_H = delta_logP - deltaK)
                Hdiff = (logP - prev_logP) - deltaK

                # draw against acceptance fraction
                accept_frac = np.median(np.minimum(1.0, np.exp(Hdiff)))
                accepted = Hdiff > np.log(model.random.rand(ntemps, nwalkers))

                # update the parameters
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
        """Generate adaptive-length leapfrog HMC proposals.

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]``.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and
                values are ``np.ndarray[ntemps, nwalkers, nleaves_max]``
                indicating active leaves. (default: ``None``)
            **kwargs (dict, optional): Extra keyword arguments for proposal.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, ndarray). The
                factors are ``deltaK`` which is a
                ``ndarray[ntemps, nwalkers]`` of the change in kinetic energy
                between the proposed and current states.

        """
        # only run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(branches_coords)

        q = {}

        # model object needed for some functions (find some way to refactor out?)
        model = kwargs.get("model", None)

        for name, q0 in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = q0.shape

            step_size = self.step_size[name]
            sim_length = self.sim_length[name]
            num_steps = np.maximum(1, int(sim_length / step_size))

            # setup indices
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # draw momenta
            rnd = random if not self.use_gpu else self.xp.random
            p0 = rnd.randn(*q0.shape)

            # get current gradient
            grad0 = self._gradient(q0, model, inds, name)

            qprime, pprime = self._leapfrog_n(
                q0,
                p0,
                grad0,
                step_size,
                num_steps,
                model,
                inds,
                name,
            )

            # flip momentum sign to make proposal explicitly symmetric
            pprime = -pprime

            # calculate change in kinetic energy
            deltaK = self._kinetic_energy(pprime, name) - self._kinetic_energy(p0, name)

            # add proposed coordinates back in
            q[name] = qprime

        # handle periodic parameters
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

        # if running on GPU but user requested CPU returns, transfer arrays back
        if self.use_gpu and not self.return_gpu:
            for name, arr in list(q.items()):
                q[name] = arr.get()
            deltaK = deltaK.get()

        return q, deltaK
