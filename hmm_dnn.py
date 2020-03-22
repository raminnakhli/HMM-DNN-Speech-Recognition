from hmmlearn.base import _BaseHMM, ConvergenceMonitor
from hmmlearn.utils import iter_from_X_lengths, normalize
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

update_dnn = True
fast_update = False
forward_backward = False


class hmm_dnn(_BaseHMM):

    def __init__(self, mlp, aucoustic_model, observation_count, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):

        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.aucoustic_model = aucoustic_model
        self.observation_count = observation_count
        self.mlp = mlp
        self.mlp.info()

    def _compute_log_likelihood(self, X):
        prob = self.mlp.log_probablity(X).astype(type(X[0, 0]))

        prob = prob - np.log(self.observation_count)
        prob = prob - np.log(self.aucoustic_model + (self.aucoustic_model == 0))

        return prob

    def _accumulate_sufficient_statistics(self, stats, X, epsilon, gamma, path, bwdlattice):

        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += gamma[0]
        if 't' in self.params:
            n_samples = X.shape[0]

            if n_samples <= 1:
                return

            a = np.zeros((self.n_components, self.n_components))

            for i in range(self.n_components):
                for j in range(self.n_components):
                    a[i, j] = np.sum(epsilon[:, i, j]) / (np.sum(gamma[:, i]) + (np.sum(gamma[:, i]) == 0))

            stats['trans'] += a

    def fit(self, X, lengths=None):

        X = check_array(X)
        self._init(X, lengths=lengths)
        self._check()

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        for iter in range(self.n_iter):
            print('iteration: {}'.format(iter))
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            tt = 0
            path_list = list()

            for i, j in iter_from_X_lengths(X, lengths):
                logprob, state_sequence = self.decode(X[i:j], algorithm="viterbi")

                curr_logprob += logprob

                epsilon = np.zeros((state_sequence.shape[0] - 1, self.n_components, self.n_components))
                gamma = np.zeros((state_sequence.shape[0], self.n_components))

                for t in range(state_sequence.shape[0] - 1):
                    epsilon[t, state_sequence[t], state_sequence[t + 1]] = 1

                for t in range(state_sequence.shape[0]):
                    for i in range(self.n_components):
                        if t != (state_sequence.shape[0] - 1):
                            gamma[t, i] = np.sum(epsilon[t, i])
                        else:
                            gamma[t, i] = gamma[t-1, i]

                path_list.append(state_sequence)
                self._accumulate_sufficient_statistics(stats, X[i:j], epsilon, gamma, state_sequence, None)
                tt += 1

            print('average loss: {}'.format(curr_logprob / tt))

            if not fast_update:
                stats['start'] /= tt
                stats['trans'] /= tt

                self._do_mstep(stats)
                if update_dnn:
                    temp_path = np.zeros((0, 1))
                    for k, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
                        temp_path = np.vstack([temp_path, np.array(path_list[k]).reshape(-1, 1)])
                    self.mlp.train(X, temp_path, 20)

                acoustic_model = np.zeros(self.n_components)
                for i, j in iter_from_X_lengths(X, lengths):
                    logprob, state_sequence = self.decode(X[i:j], algorithm="viterbi")
                    for state in state_sequence:
                        acoustic_model[state] += 1
                self.aucoustic_model = acoustic_model / np.sum(acoustic_model)

            self.monitor_.report(curr_logprob)
            if self.monitor_.iter == self.monitor_.n_iter or \
                    (len(self.monitor_.history) == 2 and
                     abs(self.monitor_.history[1] - self.monitor_.history[0]) < self.monitor_.tol * abs(
                                self.monitor_.history[1])):
                break

        print('----------------------------------------------')
        return self

    def _do_mstep(self, stats):
        if 's' in self.params:
            startprob_ = stats['start']
            self.startprob_ = np.where(self.startprob_ == 0.0,
                                       self.startprob_, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = stats['trans']
            self.transmat_ = np.where(self.transmat_ == 0.0,
                                      self.transmat_, transmat_)
            normalize(self.transmat_, axis=1)

            for i, row in enumerate(self.transmat_):
                if not np.any(row):
                    self.transmat_[i][i] = 1
