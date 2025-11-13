class FFRPrep:
    def make_folds(self, subject, num_folds=5):
        subject.fold(num_folds=num_folds)
        return subject.folds

    # ---------------- Internal helper ----------------
    def _trials_to_np(self, trials, adjust_labels=True):
        import numpy as np

        X = np.stack([t.data for t in trials]).astype("float32")
        y = np.asarray([int(t.raw_label) for t in trials], dtype=np.int64)
        idx = np.asarray([t.trial_index for t in trials], dtype=np.int64)

        if adjust_labels:
            y -= 1  # NOTE: adjusts labels to 0-3 (assumes labels are 1,2,3,4)

        return X, y, idx  # NOTE: default shape of dataloaders is (N,T)

    # ---------------- PyTorch versions ----------------
    def make_train_val_loaders(
        self,
        folds,
        fold_idx,
        val_frac=0.2,
        batch_size=256,
    ):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import StratifiedShuffleSplit
        import numpy as np

        pool = [t for i, f in enumerate(folds) if i != fold_idx for t in f]
        X, y, idx = self._trials_to_np(pool)

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=42 + fold_idx
        )
        tr_idx, val_idx = next(sss.split(np.arange(len(y)), y))

        def make_dl(idxs, shuffle):
            X_t = torch.from_numpy(X[idxs])
            y_t = torch.from_numpy(y[idxs])
            i_t = torch.from_numpy(idx[idxs])
            return DataLoader(
                TensorDataset(X_t, y_t, i_t), batch_size=batch_size, shuffle=shuffle
            )

        return make_dl(tr_idx, True), make_dl(val_idx, False)

    def make_test_loader(
        self,
        folds,
        fold_idx,
        batch_size=256,
        add_channel_dim=False,
        adjust_labels=False,
    ):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X, y, idx = self._trials_to_np(folds[fold_idx])
        ds = TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(idx)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    # ---------------- NumPy versions ----------------
    def get_train_val_nparrays(
        self, folds, fold_idx, val_frac=0.2, add_channel_dim=False, adjust_labels=False
    ):
        import numpy as np
        from sklearn.model_selection import StratifiedShuffleSplit

        pool = [t for i, f in enumerate(folds) if i != fold_idx for t in f]
        X, y, idx = self._trials_to_np(pool)

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=42 + fold_idx
        )
        tr_idx, val_idx = next(sss.split(np.arange(len(y)), y))

        return (X[tr_idx], y[tr_idx], idx[tr_idx]), (
            X[val_idx],
            y[val_idx],
            idx[val_idx],
        )

    def get_test_nparrays(
        self, folds, fold_idx, add_channel_dim=False, adjust_labels=False
    ):
        X, y, idx = self._trials_to_np(folds[fold_idx])
        return X, y, idx
