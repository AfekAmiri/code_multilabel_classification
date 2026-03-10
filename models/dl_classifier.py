import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DLClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_labels,
        hidden_dims=(256, 128),
        dropout=0.2,
        loss_type="bce",
        focal_alpha=1,
        focal_gamma=2,
    ):
        super().__init__()
        layers = []
        input_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, num_labels))
        self.classifier = nn.Sequential(*layers)
        self.loss_type = loss_type
        if loss_type == "bce":
              self.loss_fn = None  # sera défini lors du fit
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, embeddings, labels=None):
        logits = self.classifier(embeddings)
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        return logits

    def fit(self, embeddings, labels, num_epochs=500, lr=1e-3, verbose=True):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        # Ajout du calcul de pos_weight pour BCE
        if self.loss_type == "bce":
            # Calcul du poids pour chaque classe
            # pos_weight = (N - P) / P
            P = labels.sum(dim=0)
            N = labels.shape[0] - P
            pos_weight = N / (P + 1e-8)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss, logits = self.forward(embeddings, labels)
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")

    def predict(self, embeddings):
        self.eval()
        with torch.no_grad():
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            logits = self.classifier(embeddings)
            probs = torch.sigmoid(logits)
            return (probs > 0.5).cpu().numpy()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        self.eval()
