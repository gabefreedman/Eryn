# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
from ..state import State
from .move import Move

__all__ = ["HMCBase", "HMCMove"]


class HMCBase(Move):
    """Base class for Hamiltonian Monte Carlo and other gradient-aware proposals.

    Subclasses must have a ``get_proposal`` method. Current implementations include:
    - :class:`moves.HMCMove`: Fixed step-size HMC
    - :class:`moves.AdaptHMCMove`: HMC with dual averaging step-size adaptation

    Analytic gradients can be supplied explicitly via ``grad_fn`` or calculated
    numerically using central finite differences.

    Args:
        grad_fn (callable, optional): User-supplied gradient function for the
            active parameters. If ``None``, central finite difference gradients
            are used. (default: ``None``)
        priors (:class:`eryn.prior.ProbDistContainer`, optional): Prior
            container used to initialize finite-difference step sizes when
            ``grad_fn`` is ``None``. (default: ``None``)
        grad_clip (double or ``None``, optional): Clipping limit for
            gradient to improve numerical stability. If ``None``,
            no clipping is applied. (default: ``1e6``)
        **kwargs (dict, optional): Kwargs for parent classes. (default: ``{}``)

    Raises:
        AttributeError: If both ``grad_fn`` and ``priors`` are ``None``.

    Attributes:
        grad_fn (callable or ``None``): Gradient callback function.
        grad_clip (double or ``None``): Gradient clipping threshold.
        finite_diff_eps (ndarray[ndim] or None): Per-dimension finite-difference
            step sizes.

    """

    def __init__(self, grad_fn=None, priors=None, grad_clip=1e6, **kwargs):
        Move.__init__(self, **kwargs)
        self.grad_fn = grad_fn
        self.grad_clip = grad_clip
        if grad_fn is None:
            try:
                self._init_finite_diff_eps(priors)
            except Exception:
                raise AttributeError(
                    "Must provide priors to initialize finite-difference step sizes when grad_fn is None."
                )

        # setup is required for HMC moves, define flag
        self._setup_complete = False

    def _init_finite_diff_eps(self, priors):
        """Initialize finite-difference step sizes from prior definitions.

        Per-dimension perturbation scales are estimated from prior width when available.

        Args:
            priors (:class:`eryn.prior.ProbDistContainer`): Prior container object

        """
        self.finite_diff_eps = self.xp.zeros(priors.ndim)
        for i, prior_in in priors.priors_in.items():
            # set step size to small fraction of prior width
            if hasattr(prior_in, "max_val") and hasattr(prior_in, "min_val"):
                # uniform distribution
                width = prior_in.max_val - prior_in.min_val
                self.finite_diff_eps[i] = width * np.sqrt(np.finfo(float).eps)
            elif hasattr(prior_in, "sigma"):
                # normal distribution
                self.finite_diff_eps[i] = prior_in.sigma * np.sqrt(np.finfo(float).eps)
            else:
                # other prior types (can update conditional as more priors are added)
                self.finite_diff_eps[i] = np.sqrt(np.finfo(float).eps)

    def _leapfrog_once(self, q, p, grad_q, epsilon, model, inds, name):
        """Perform a single leapfrog integration step.

        Args:
            q (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Position state.
            p (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Momentum state.
            grad_q (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Log posterior
                gradient evaluated at ``q``.
            epsilon (double): Leapfrog step size.
            model (:class:`eryn.model.Model`): Model object used by sampler.
            inds (ndarray[ntemps, nwalkers, nleaves_max]): Boolean mask
                indicating active leaves.
            name (str): Branch name.

        Returns:
            tuple: (Updated coordinates, Updated momenta) -> (ndarray, ndarray)

        """
        pprime = p + 0.5 * epsilon * grad_q
        qprime = q + epsilon * pprime
        gradprime = self._gradient(qprime, model, inds, name)
        pprime = pprime + 0.5 * epsilon * gradprime
        return qprime, pprime

    def _leapfrog_n(self, q, p, grad_q, epsilon, n_steps, model, inds, name):
        """Perform ``n_steps`` leapfrog integration steps.

        Args:
            q (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Position state.
            p (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Momentum state.
            grad_q (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Log posterior
                gradient evaluated at ``q``.
            epsilon (double): Leapfrog step size.
            n_steps (int): Number of leapfrog steps.
            model (:class:`eryn.model.Model`): Model object used by sampler.
            inds (ndarray[ntemps, nwalkers, nleaves_max]): Boolean mask
                indicating active leaves.
            name (str): Branch name.

        Returns:
            tuple: (Updated coordinates, Updated momenta) -> (ndarray, ndarray)

        """
        pprime = p + 0.5 * epsilon * grad_q
        qprime = q + epsilon * pprime
        gradprime = self._gradient(qprime, model, inds, name)
        for _ in range(n_steps - 1):
            pprime = pprime + epsilon * gradprime
            qprime = qprime + epsilon * pprime
            gradprime = self._gradient(qprime, model, inds, name)
        pprime = pprime + 0.5 * epsilon * gradprime
        return qprime, pprime

    def _kinetic_energy(self, p, name):
        """Compute kinetic energy from momentum array and inverse metric.

        Computes ``0.5 * p^T * M^-1 * p`` per leaf, then sums over
        leaves for each temperature/walker pair.

        Args:
            p (ndarray[ntemps, nwalkers, nleaves_max, ndim]): Momentum state.
            name (str): Branch name used to look up inverse metric.

        Returns:
            ndarray[ntemps, nwalkers]: Kinetic energy term.

        """
        kinetic_tmp = 0.5 * self.xp.einsum(
            "...i,ij,...j->...", p, self.inverse_metric[name], p
        )

        return kinetic_tmp.sum(axis=-1)

    def _compute_logjoint(self, q, p, model, inds, name):
        """Compute log joint probability, equivalent to the negative Hamiltonian.

        Args:
            q (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Position state.
            p (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Momentum state.
            model (:class:`eryn.model.Model`): Model object used by sampler.
            inds (ndarray[ntemps, nwalkers, nleaves_max]): Boolean mask
                indicating active leaves.
            name (str): Branch name.

        Returns:
            ndarray[ntemps, nwalkers]: Log joint probability.

        """
        logp = model.compute_log_prior_fn({name: q}, {name: inds})
        logl, _ = model.compute_log_like_fn({name: q}, {name: inds}, logp=logp)
        logjoint = self.compute_log_posterior(logl, logp) - self._kinetic_energy(
            p, name
        )
        return logjoint

    def _gradient(self, q, model, inds, name):
        """Compute gradient of the log posterior.

        Uses ``self.grad_fn`` if defined. Otherwise, central finite differences
        are used per dimension. Non-finite masking and optional clipping is
        applied.

        Args:
            q (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Position state.
            model (:class:`eryn.model.Model`): Model object used by sampler.
            inds (ndarray[ntemps, nwalkers, nleaves_max]): Boolean mask
                indicating active leaves.
            name (str): Branch name.

        Returns:
            ndarray[ntemps, nwalkers, nlwaves_max, ndim]: Gradient array.

        """
        grad = self.xp.zeros_like(q)

        ndim = q.shape[-1]
        inds_active = self.xp.nonzero(inds)

        # use analytic gradient if available
        if self.grad_fn is not None:
            q_active = q[inds_active]

            try:
                # try vectorized call
                grad_active = self.grad_fn(q_active)
            except Exception:
                # fallback to per-row calls
                grad_list = [self.grad_fn(theta) for theta in q_active]
                grad_active = self.xp.vstack(grad_list)

            grad[inds_active] = grad_active

            # check for NaNs/infs propagating from user gradients
            mask = self.xp.isfinite(grad)
            if not mask.all():
                grad = self.xp.where(mask, grad, self.xp.zeros_like(grad))

            # optional gradient clipping to avoid overflow in leapfrog updates
            if self.grad_clip is not None:
                grad = self.xp.clip(grad, -self.grad_clip, self.grad_clip)

        # central finite difference per dimension
        else:
            for d in range(ndim):
                epsilon = self.finite_diff_eps[d]
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[inds_active + (d,)] += epsilon
                q_minus[inds_active + (d,)] -= epsilon

                logp_plus = model.compute_log_prior_fn(
                    {name: q_plus}, inds={name: inds}
                )
                logp_minus = model.compute_log_prior_fn(
                    {name: q_minus}, inds={name: inds}
                )

                logl_plus, _ = model.compute_log_like_fn(
                    {name: q_plus},
                    inds={name: inds},
                    logp=logp_plus,
                )
                logl_minus, _ = model.compute_log_like_fn(
                    {name: q_minus},
                    inds={name: inds},
                    logp=logp_minus,
                )

                post_plus = (
                    logl_plus[inds_active[0], inds_active[1]]
                    + logp_plus[inds_active[0], inds_active[1]]
                )
                post_minus = (
                    logl_minus[inds_active[0], inds_active[1]]
                    + logp_minus[inds_active[0], inds_active[1]]
                )

                delta = post_plus - post_minus
                grad[inds_active + (d,)] = delta / (2.0 * epsilon)

            # check for NaNs/infs propagating from numerical gradients
            mask = self.xp.isfinite(grad)
            if not mask.all():
                grad = self.xp.where(mask, grad, self.xp.zeros_like(grad))

            # optional gradient clipping to avoid overflow in leapfrog updates
            if self.grad_clip is not None:
                grad = self.xp.clip(grad, -self.grad_clip, self.grad_clip)

        return grad

    def _get_logpost_q(self, q, model, inds, name):
        """Compute log posterior of q.

        Args:
            q (ndarray[ntemps, nwalkers, nlwaves_max, ndim]): Position state.
            model (:class:`eryn.model.Model`): Model object used by sampler.
            inds (ndarray[ntemps, nwalkers, nleaves_max]): Boolean mask
                indicating active leaves.
            name (str): Branch name.

        Returns:
            ndarray[ntemps, nwalkers]: Log joint probability..

        """
        logp = model.compute_log_prior_fn({name: q}, {name: inds})
        logl, _ = model.compute_log_like_fn({name: q}, {name: inds}, logp=logp)
        logpost = self.compute_log_posterior(logl, logp)
        return logpost

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Build an HMC proposal for the requested branches.

        Concrete subclasses must implement this method and return proposed
        coordinates together with the kinetic-energy change term used in the
        acceptance ratio.

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]``.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and
                values are ``np.ndarray[ntemps, nwalkers, nleaves_max]``
                indicating active leaves. (default: ``None``)
            **kwargs (dict, optional): Extra keyword arguments for subclass
                implementations.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, ndarray). The
                factors are ``deltaK`` which is a ``ndarray[ntemps, nwalkers]``
                of the change in kinetic energy between the proposed and
                current states.

        Raises:
            NotImplementedError: If proposal is not implemented in subclass.

        """
        raise NotImplementedError("HMC proposal must be implemented by " "subclasses")

    def setup(self, branches_coords):
        """Setup branch-dependent HMC configuration.

        Must be defined in subclass. Creates per-branch dictionaries for
        settings such as integration step sizes, inverse mass metrics, or
        other hyperparameters.

        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                ``np.ndarray[ntemps, nwalkers, nleaves_max, ndim]``.

        Raises:
            NotImplementedError: If setup is not implemented in a subclass.

        """
        raise NotImplementedError("HMC setup must be implemented by " "subclasses")

    def propose(self, model, state):
        """Generate HMC proposals and apply Metropolis-Hastings acceptance.

        This routine handles Gibbs split iteration, proposal generation,
        likelihood/prior evaluation, Hamiltonian acceptance decisions, and
        state updates. Proposal details (integration length, stepping strategy,
        etc.) are delegated to :meth:`get_proposal`.

        Args:
            model (:class:`eryn.model.Model`): Carrier of sampler functions.
            state (:class:`eryn.state.State`): Current sampler state.

        Returns:
            tuple: (state, accepted) -> (ndarray, ndarray) where ``state`` is
                the updated sampler state and ``accepted`` is the accepted
                count array.

        """
        # run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(state.branches_coords)

        # get all branch names for gibbs setup
        all_branch_names = list(state.branches.keys())

        # get initial shape information
        ntemps, nwalkers, _, _ = state.branches[all_branch_names[0]].shape

        # in case there are no leaves yet
        accepted = np.zeros((ntemps, nwalkers), dtype=bool)

        # iterate through gibbs setup
        for branch_names_run, inds_run in self.gibbs_sampling_setup_iterator(
            all_branch_names
        ):
            # setup supplemental information
            if not all(val is None for val in state.branches_supplemental.values()):
                new_branch_supps = deepcopy(state.branches_supplemental)
            else:
                new_branch_supps = None

            if state.supplemental is not None:
                new_supps = deepcopy(state.supplemental)
            else:
                new_supps = None

            # setup information according to gibbs info
            (
                coords_going_for_proposal,
                inds_going_for_proposal,
                at_least_one_proposal,
            ) = self.setup_proposals(
                branch_names_run, inds_run, state.branches_coords, state.branches_inds
            )

            # if no walkers are actually being proposed
            if not at_least_one_proposal:
                continue

            # Get the move-specific proposal.
            q, deltaK = self.get_proposal(
                coords_going_for_proposal,
                model.random,
                branches_inds=inds_going_for_proposal,
                supps=new_supps,
                branch_supps=new_branch_supps,
                model=model
            )

            # account for gibbs sampling
            self.cleanup_proposals_gibbs(
                branch_names_run, inds_run, q, state.branches_coords
            )

            # order everything properly
            q, _, new_branch_supps = self.ensure_ordering(
                list(state.branches.keys()), q, state.branches_inds, new_branch_supps
            )

            # if not wrapping with mutliple try (normal route)
            if not hasattr(self, "mt_ll") or not hasattr(self, "mt_lp"):
                # Compute prior of the proposed position
                logp = model.compute_log_prior_fn(q, inds=state.branches_inds)

                self.fix_logp_gibbs(
                    branch_names_run, inds_run, logp, state.branches_inds
                )

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
            prev_logl = state.log_like
            prev_logp = state.log_prior
            prev_logP = self.compute_log_posterior(prev_logl, prev_logp)

            # determine acceptance (-delta_H = delta_logP - deltaK)
            Hdiff = (logP - prev_logP) - deltaK

            # draw against acceptance fraction
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

        # temperature swaps
        if self.temperature_control is not None:
            state = self.temperature_control.temper_comps(state)

        return state, accepted


class HMCMove(HMCBase):
    """Fixed step size Hamiltonian Monte Carlo proposal.

    Algorithm 1 from `Hoffman & Gelman (2014) <https://arxiv.org/abs/1111.4246>`.

    Each branch evolves using leapfrog integration with a fixed step size and
    fixed number of integration steps.

    Args:
        grad_fn (callable, optional): Likelihood gradient function. If
            ``None``, a central finite difference method is used.
            (default: ``None``)
        step_size (double or dict, optional): Leapfrog step size. If a dict,
            keys are branch names and values are per-branch step sizes.
            (default: ``0.1``)
        num_steps (int or dict, optional): Number of leapfrog steps. If a dict,
            keys are branch names and values are per-branch step counts.
            (default: ``20``)
        inverse_metric (ndarray or dict, optional): Inverse mass-matrix for
            momenta. If a dict, keys are branch names and values are per-branch
            inverse mass matrices. If ``None``, identity matrices are used.
            (default: ``None``)
        return_gpu (bool, optional): If ``use_gpu == True`` and
            ``return_gpu == True``, returned arrays remain on GPU. (default: ``False``)
        kwargs (dict, optional): Additional keyword arguments passed through
            :class:`HMCBase`.

    Attributes:
        step_size (dict): Per-branch leapfrog step sizes.
        num_steps (dict): Per-branch leapfrog step counts.
        inverse_metric (dict): Per-branch inverse mass matrices.
        return_gpu (bool): Whether to return array in ``Cupy`` or ``NumPy``.

    """

    def __init__(
        self,
        grad_fn=None,
        step_size=0.1,
        num_steps=20,
        inverse_metric=None,
        return_gpu=False,
        **kwargs,
    ):

        # control whether outputs remain on gpu when use_gpu==True
        self.return_gpu = return_gpu

        # will be populated with per-branch structure during setup
        self._step_size_input = step_size
        self._num_steps_input = num_steps
        self._inverse_metric_input = inverse_metric

        super(HMCMove, self).__init__(grad_fn=grad_fn, **kwargs)

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

        # leapfrog integration number of steps
        if isinstance(self._num_steps_input, dict):
            self.num_steps = self._num_steps_input.copy()
        else:
            self.num_steps = {name: self._num_steps_input for name in branches_coords}

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

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """Generate fixed-step leapfrog HMC proposals.

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
                factors are ``deltaK`` which is a ``ndarray[ntemps, nwalkers]``
                of the change in kinetic energy between the proposed and
                current states.

        """
        # only run setup if it hasn't been run already
        if not self._setup_complete:
            self.setup(branches_coords)

        q = {}

        model = self.current_model

        for name, q0 in branches_coords.items():
            ntemps, nwalkers, nleaves_max, ndim = q0.shape

            step_size = self.step_size[name]
            num_steps = self.num_steps[name]

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
            deltaK = self._kinetic_energy(pprime) - self._kinetic_energy(p0, name)

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

        # if running on GPU but requested CPU returns, trasfer arrays back
        if self.use_gpu and not self.return_gpu:
            for name, arr in list(q.items()):
                q[name] = arr.get()
            deltaK = deltaK.get()

        return q, deltaK
