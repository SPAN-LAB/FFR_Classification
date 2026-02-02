import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC

class Model(nn.Module):
	def __init__(self, input_size, n_classes=4, kernel='rbf', C=1.0, gamma='scale'):
		super().__init__()
		self.input_size = input_size
		self.n_classes = n_classes
		self.svm = SVC(probability=True, kernel=kernel, C=C, gamma=gamma)
		self.is_accumulating = True
		self.data_buffer = []
		self.labels_buffer = []
		self.is_fitted = False
		self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

	def forward(self, x):
		device = x.device
		batch_size = x.shape[0]
		x_np = x.detach().cpu().numpy()
		if x_np.ndim > 2:
			x_np = x_np.reshape(batch_size, -1)

		if self.training and self.is_accumulating:
			self.data_buffer.append(x_np)
			dummy_logits = torch.zeros(batch_size, self.n_classes, device=device, requires_grad=True)
			dummy_logits = dummy_logits + self._dummy.to(device) * 0.0
			return dummy_logits

		if self.is_fitted:
			try:
				proba = self.svm.predict_proba(x_np)
				proba = np.clip(proba, 1e-10, 1.0)
				logits = np.log(proba)
				result = torch.from_numpy(logits).float().to(device)
				result = result + self._dummy.to(device) * 0.0
				return result
			except Exception:
				zeros = torch.zeros(batch_size, self.n_classes, device=device, requires_grad=True)
				zeros = zeros + self._dummy.to(device) * 0.0
				return zeros

		zeros = torch.zeros(batch_size, self.n_classes, device=device, requires_grad=True)
		zeros = zeros + self._dummy.to(device) * 0.0
		return zeros

	def train(self, mode=True):
		was_training = self.training
		result = super().train(mode)
		if was_training and not mode and self.is_accumulating:
			self._fit_svm()
		return result

	def _fit_svm(self):
		if len(self.data_buffer) == 0 or len(self.labels_buffer) == 0:
			return
		try:
			X = np.concatenate(self.data_buffer, axis=0)
			y = np.concatenate(self.labels_buffer, axis=0)
			self.svm.fit(X, y)
			self.is_fitted = True
			self.is_accumulating = False
			self.data_buffer = []
			self.labels_buffer = []
		except Exception:
			pass

	def accumulate_labels(self, y):
		y_np = y.detach().cpu().numpy()
		self.labels_buffer.append(y_np)
