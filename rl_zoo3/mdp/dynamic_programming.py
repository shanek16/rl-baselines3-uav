from multiprocessing import Array, Process, Value, cpu_count
from time import time
from sys import stdout
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import bicgstab, lgmres, spsolve

__all__ = ["ValueIteration", "PolicyIteration"]

class Verbose:
    def __init__(self, *args):
        self.verbose = args[0]
        self.lenStr = 0

    def __call__(self, string):
        if self.verbose:
            stdout.write("\r" + " " * self.lenStr + "\r")
            stdout.flush()
            self.lenStr = 0 if string[-2:] == "\n" else len(string)
            stdout.write(string)
            stdout.flush()

class ValueIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.values = np.max(self.mdp.rewards.toarray(), axis=1)

    def solve(
        self,
        sigma,
        n_r,
        n_alpha,
        max_iteration=1e3,
        tolerance=1e-8,
        earlystop=100,
        verbose=True,
        callback=None,
        parallel=True,
    ):

        self.verbose = Verbose(verbose)
        self.verbose("solving with Value Iteration...")
        best_iter = 0
        min_val_diff = 0.001
        start_time = time()
        last_time = time()
        current_time = time()

        if parallel:
            self.shared = np.frombuffer(
                Array(
                    np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype),
                    int(self.mdp.states.num_states * self.mdp.actions.num_actions),
                ).get_obj(),
                dtype=self.mdp.rewards.dtype,
            )
            self.values = np.frombuffer(
                Array(
                    np.ctypeslib.as_ctypes_type(self.mdp.rewards.dtype), self.values
                ).get_obj(),
                dtype=self.mdp.rewards.dtype,
            )

            def _worker(P, key, flag):
                while True:
                    if flag.value > 0:
                        self.shared[key] = P.dot(self.values)
                        flag.value = 0
                    elif flag.value < 0:
                        break

            chunksize = np.ceil(
                self.mdp.states.num_states * self.mdp.actions.num_actions / cpu_count()
            )
            self.workers = []
            for pid in range(cpu_count()):
                key = slice(
                    int(chunksize * pid),
                    int(
                        min(
                            chunksize * (pid + 1),
                            self.mdp.states.num_states * self.mdp.actions.num_actions,
                        )
                    ),
                    None,
                )
                flag = Value("i", 0)
                self.workers.append(
                    (
                        Process(
                            target=_worker,
                            args=(
                                self.mdp.state_transition_probability[key],
                                key,
                                flag,
                            ),
                        ),
                        flag,
                    )
                )
                self.workers[-1][0].start()

        for iter in range(int(max_iteration)):
            value_diff = self.update(parallel=parallel)
            if value_diff <= 0.001:
                min_val_diff = min(value_diff, min_val_diff)
                if value_diff <= min_val_diff:
                    best_iter = iter
                    # save the best result
                    result_filename = "result_" + str(sigma)[:3]
                    self.save(result_filename)

                    value_filename = "value_" + str(sigma)[:3] + ".png"
                    value_plot = self.values.reshape((n_r, n_alpha))
                    value_tips = np.flipud(
                        value_plot.T
                    )  # value plot of Tips paper format
                    plt.imsave(value_filename, value_tips, cmap="gray")

                    policy_filename = "policy_" + str(sigma)[:3] + ".png"
                    policy_plot = self.mdp.policy.toarray().reshape((n_r, n_alpha))
                    policy_tips = np.flipud(
                        policy_plot.T
                    )  # policy plot of Tips paper format
                    plt.imsave(
                        policy_filename,
                        policy_tips,
                        cmap="gray",
                    )
                    self.verbose("possible best model saved...")

                delta = iter - best_iter
                if delta >= earlystop:
                    self.verbose(
                        f"Stopping training early as no improvement observed in last {earlystop} epochs. "
                        f"Best results observed at iter {best_iter+1}, best model saved as result_xx.npz.\n"
                    )
                    break

            current_time = time()
            self.verbose(
                "Iter.: %d, Value diff.: %f, Step time: %f (sec).\n"
                % (iter + 1, value_diff, current_time - last_time)
            )
            last_time = current_time

            if callback is not None:
                callback(self)

            if value_diff is np.nan or value_diff is np.inf:
                raise OverflowError("Divergence detected.")

            if value_diff < tolerance:
                break

        for p, flag in self.workers:
            flag.value = -1
            p.join()
        self.verbose("Time elapsed: %f (sec).\n" % (current_time - start_time))
        del self.verbose

    def update(self, parallel=True):

        self.verbose("Computing action values...")
        if parallel:
            for _, flag in self.workers:
                flag.value = 1
            for _, flag in self.workers:
                while True:
                    if flag.value == 0:
                        break
            q = self.mdp.rewards.toarray() + np.multiply(
                self.mdp.discount, self.shared.reshape(self.mdp.rewards.shape)
            )
        else:
            q = self.mdp.rewards.toarray() + np.multiply(
                self.mdp.discount,
                self.mdp.state_transition_probability.dot(self.values).reshape(
                    self.mdp.rewards.shape
                ),
            )

        self.verbose("Updating policy...")
        policy = np.argmax(q, axis=1).astype(self.mdp.policy.dtype)
        new_values = np.take_along_axis(q, policy[:, np.newaxis], axis=1).ravel()
        self.mdp.policy.update(policy)

        value_diff = self.values[:] - new_values[:]
        value_diff = np.sqrt(
            np.dot(value_diff, value_diff) / self.mdp.states.num_states
        )

        self.values[:] = new_values.copy()

        return value_diff

    def save(self, filename):
        np.savez(filename, values=self.values, policy=self.mdp.policy.toarray())


class PolicyIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.values = None
        # Identity matrix $I_{|s|}$ and $I_{|a|}$ for computation
        self.terminal_state = False
        self.__I = sp.identity(
            self.mdp.states.num_states, dtype=np.float32, format="csr"
        )
        self.__innerloop_maxiter = max(int(np.sqrt(self.mdp.states.num_states)), 100)
        self.__innerloop_maxiter = int(self.mdp.states.num_states)

    def save(self, filename):
        np.savez(filename, values=self.values, policy=self.mdp.policy.toarray())

    def solve(
        self,
        sigma,
        n_r,
        n_alpha,
        max_iteration=1e3,
        tolerance=1e-8,
        earlystop=100,
        verbose=True,
        callback=None,
    ):

        self.verbose = Verbose(verbose)
        self.verbose("solving with Policy Iteration...")
        best_iter = 0
        min_val_diff = 0.001
        start_time = time()
        last_time = time()
        current_time = time()

        for iter in range(int(max_iteration)):

            value_diff = self.update()
            min_val_diff = min(value_diff, min_val_diff)
            if value_diff <= min_val_diff:
                best_iter = iter
                # save the best result
                result_filename = "result_" + str(sigma)[:3]
                self.save(result_filename)

                value_filename = "value_" + str(sigma)[:3] + ".png"
                value_plot = self.values.reshape((n_r, n_alpha))
                value_tips = np.flipud(value_plot.T)  # value plot of Tips paper format
                plt.imsave(value_filename, value_tips, cmap="gray")

                policy_filename = "policy_" + str(sigma)[:3] + ".png"
                policy_plot = self.mdp.policy.toarray().reshape((n_r, n_alpha))
                policy_tips = np.flipud(
                    policy_plot.T
                )  # policy plot of Tips paper format
                plt.imsave(
                    policy_filename,
                    policy_tips,
                    cmap="gray",
                )
                self.verbose("possible best model saved...")

            delta = iter - best_iter
            if delta >= earlystop:
                self.verbose(
                    f"Stopping training early as no improvement observed in last {earlystop} epochs. "
                    f"Best results observed at iter {best_iter+1}, best model saved as result_xx.npz.\n"
                )
                break

            current_time = time()
            self.verbose(
                "Iter.: %d, Value diff.: %f, Step time: %f (sec).\n"
                % (iter + 1, value_diff, current_time - last_time)
            )
            last_time = current_time

            if callback is not None:
                callback(self)
            if iter > 0:
                if value_diff is np.nan or value_diff is np.inf:
                    raise OverflowError("Divergence detected.")

            if value_diff < tolerance:
                break

        current_time = time()
        self.verbose("Time elapsed: %f (sec).\n" % (current_time - start_time))
        del self.verbose

    def update(self, direct_method=False):

        # Compute the value difference $|\V_{k}-V_{k+1}|\$ for check the convergence
        if self.values is None:
            self.verbose("Computing initial values...")
            policy = np.argmax(self.mdp.rewards.toarray(), axis=1).astype(
                self.mdp.policy.dtype
            )
            self.mdp.policy.update(
                np.argmax(self.mdp.rewards.toarray(), axis=1).astype(
                    self.mdp.policy.dtype
                )
            )
            self.values = np.take_along_axis(
                self.mdp.rewards.toarray(), policy[:, np.newaxis], axis=1
            ).ravel()
            self.mdp.policy.update(policy)
            value_diff = np.inf
        else:
            # Compute the value $V(s)$ via solving the linear system $(I-\gamma P^{\pi}), R^{\pi}$
            self.verbose("Constructing linear system...")
            A = self.__I - self.mdp.discount * sp.vstack(
                [
                    self.mdp.state_transition_probability[s, a, :]
                    for s, a in enumerate(self.mdp.policy)
                ],
                format="csr",
            )
            if np.all(A.diagonal()):
                b = self.mdp.rewards[self.mdp.policy.one_hot()]
                if self.mdp.rewards.issparse:
                    b = b.T
                if direct_method:
                    self.verbose("Solving linear system (SuperLU)...")
                    new_values = spsolve(A, b)
                else:
                    self.verbose("Solving linear system (BiCGstab)...")
                    new_values, info = bicgstab(
                        A, b, x0=self.values, tol=1e-8, maxiter=self.__innerloop_maxiter
                    )
                    if info < 0:
                        self.verbose("BiCGstab failed. Call LGMRES...")
                        new_values, info = lgmres(
                            A,
                            b,
                            x0=new_values,
                            tol=1e-8,
                            maxiter=int(max(np.sqrt(self.__innerloop_maxiter), 10)),
                        )

                self.verbose("Updating policy...")
                self.mdp.policy.update(
                    np.argmax(
                        self.mdp.rewards.toarray()
                        + np.multiply(
                            self.mdp.discount,
                            self.mdp.state_transition_probability.dot(new_values),
                        ).reshape(self.mdp.rewards.shape),
                        axis=1,
                    ).astype(self.mdp.policy.dtype)
                )

            else:
                self.verbose("det(A) is zero. Use value iteration update instead...")
                q = self.mdp.rewards.toarray() + np.multiply(
                    self.mdp.discount,
                    self.mdp.state_transition_probability.dot(self.values),
                ).reshape(self.mdp.rewards.shape)
                self.verbose("Updating policy...")
                policy = np.argmax(q, axis=1).astype(self.mdp.policy.dtype)
                new_values = np.take_along_axis(
                    q, policy[:, np.newaxis], axis=1
                ).ravel()
                self.mdp.policy.update(policy)

            value_diff = self.values[:] - new_values[:]
            # if np.mean(value_diff)>0:
            #  self.values = spsolve(A, b) # use spsolve if value function does not improved
            value_diff = np.sqrt(
                np.dot(value_diff, value_diff) / self.mdp.states.num_states
            )

            self.values = new_values.copy()

        return value_diff
