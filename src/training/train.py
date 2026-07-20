import torch
from tqdm import tqdm

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    for sar, optical in tqdm(dataloader):

        sar = sar.to(device)
        optical = optical.to(device)

        optimizer.zero_grad()

        sar_emb, optical_emb = model(
            sar,
            optical
        )

        loss = criterion(
            sar_emb,
            optical_emb
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    for sar, optical in dataloader:

        sar = sar.to(device)
        optical = optical.to(device)

        sar_emb, optical_emb = model(
            sar,
            optical
        )

        loss = criterion(
            sar_emb,
            optical_emb
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)
def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=1
):

    best_loss = float("inf")

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print("Best model saved")